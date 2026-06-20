"""A*-FREE maze navigation with a LEARNED subgoal generator (no A* steering at eval).

Closed loop: encode the obs -> the SubgoalPredictor proposes the next waypoint ->
a low-level reacher scores the 4 cardinals by a K-step FINE-world-model rollout
(distance of the predicted endpoint to the waypoint). A* is used ONLY to size the
per-episode step budget (a difficulty-proportional clock), NEVER to steer.

EXECUTION MEMORY (honest accounting)
------------------------------------
The low-level reacher may use three pieces of hand-coded *execution memory*. None of
them use privileged information (no A*, no goal cell, no maze grid) -- they are
standard closed-loop bookkeeping -- but they are NOT learned, so the success number
must not silently credit the learned policy for what they do. They are therefore
exposed as flags, and ``--ablation`` measures how much each contributes:

  --use-last-rev   don't immediately reverse the last committed move (anti-oscillation)
  --use-blocked    remember (cell, dir) pairs that physically didn't move (wall memory)
  --revisit-pen P  penalize stepping onto already-visited cells (count-based exploration)

Defaults (blocked + last_rev ON) reproduce the legacy ``eval_subgoal`` behavior.
The memoryless reactive floor -- which replaces the deleted ``eval_strict.py`` -- is
``--no-blocked --no-last-rev --revisit-pen 0``. NOTE: even this floor still relies on
the simulator's ground-truth collision result within a step (the reacher tries
candidates until one physically moves); it removes only the cross-step memory.

Run:
  # single config (legacy positional args still work)
  python -m examples.ac_video_jepa.maze.eval_subgoal FINE_CKPT SG_CKPT OUT \
      [num_ep] [lookahead] [revisit_pen] [n_gifs] [budget_factor] [budget_margin]

  # leave-one-out ablation table on a FIXED maze set (same --seed for every row)
  python -m examples.ac_video_jepa.maze.eval_subgoal FINE_CKPT SG_CKPT OUT \
      32 4 --ablation --revisit-pen 1.0 --n-gifs 4
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.datasets.maze.maze_solver import solve_a_star
from eb_jepa.hierarchical import CARDINALS, SubgoalPredictor, fine_kstep_target
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_checkpoint
from eb_jepa.vis_utils import save_gif
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine
from omegaconf import OmegaConf

# Env's hard step cap. Our per-episode budget (budget_factor*A* + margin) stays the
# binding limit; this is just a high ceiling so the env never truncates first.
N_ALLOWED = 800
OPP = {0: 1, 1: 0, 2: 3, 3: 2}  # opposite cardinal (D<->U, R<->L)


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="A*-free maze eval with a learned subgoal predictor.")
    p.add_argument("fine_ckpt")
    p.add_argument("sg_ckpt")
    p.add_argument("rdir")
    # Legacy positional knobs (kept so existing docs/commands keep working).
    p.add_argument("num_ep", nargs="?", type=int, default=16)
    p.add_argument("lookahead", nargs="?", type=int, default=1)
    p.add_argument("revisit_pen_pos", nargs="?", type=float, default=0.0,
                   metavar="revisit_pen")
    p.add_argument("n_gifs_pos", nargs="?", type=int, default=0, metavar="n_gifs")
    p.add_argument("budget_factor", nargs="?", type=float, default=4.0)
    p.add_argument("budget_margin", nargs="?", type=int, default=10)
    # Execution-memory flags (default ON = legacy eval_subgoal behavior).
    p.add_argument("--use-blocked", "--blocked", action=argparse.BooleanOptionalAction,
                   default=True, dest="use_blocked",
                   help="remember (cell,dir) pairs that didn't move (wall memory); "
                        "disable with --no-blocked")
    p.add_argument("--use-last-rev", "--last-rev", action=argparse.BooleanOptionalAction,
                   default=True, dest="use_last_rev",
                   help="forbid immediately reversing the last committed move; "
                        "disable with --no-last-rev")
    p.add_argument("--revisit-pen", type=float, default=None,
                   help="override the positional revisit_pen")
    p.add_argument("--n-gifs", type=int, default=None, help="override positional n_gifs")
    p.add_argument("--seed", type=int, default=0,
                   help="env seed; identical for every ablation row -> same mazes")
    p.add_argument("--ablation", action="store_true",
                   help="run the leave-one-out memory ablation table")
    return p.parse_args(argv)


@torch.no_grad()
def main():
    args = parse_args(sys.argv[1:])
    os.makedirs(args.rdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(args.fine_ckpt).parent / "config.yaml")
    _, _, env_config, _ = init_data(env_name=cfg.data.env_name,
                                    cfg_data=OmegaConf.to_container(cfg.data, resolve=True),
                                    device=device)
    cell_size = float(env_config.cell_size)
    off = (cell_size - 1) / 2.0

    # --- models (loaded once, shared by every config) ---
    jepa, f = build_fine(cfg, env_config, device)
    info = load_checkpoint(Path(args.fine_ckpt), jepa, optimizer=None, scheduler=None,
                           device=device, strict=False)
    jepa.eval()
    sck = torch.load(args.sg_ckpt, map_location=device, weights_only=False)
    subgoal = SubgoalPredictor(f).to(device)
    subgoal.load_state_dict(sck["subgoal"])
    subgoal.eval()

    def build_env():
        # Fresh, identically-seeded env -> every config sees the SAME maze sequence.
        return create_env(cfg.data.env_name, config=env_config, n_allowed_steps=N_ALLOWED,
                          n_steps=N_ALLOWED, max_step_norm=1.5,
                          rng=np.random.default_rng(args.seed))

    norm = build_env().normalizer  # normalizer depends only on img_size -> stable
    xy_head = MLPXYHead(input_shape=f, normalizer=norm).to(device)
    if "xy_head_state_dict" in info:
        xy_head.load_state_dict(info["xy_head_state_dict"])
    xy_head.eval()

    def obs_tensor(o):
        return norm.normalize_state(
            o.to(dtype=torch.float32, device=device)).unsqueeze(0).unsqueeze(2)

    def probe_xy(z):  # latent -> [2] normalized position
        return xy_head(z.float()).permute(0, 2, 1)[0, 0]

    def pred_cell(z):  # latent -> predicted maze cell (via learned probe, not env truth)
        xy = norm.unnormalize_location(xy_head(z.float()).permute(0, 2, 1)[:, 0])[0]
        return (int(round((float(xy[0]) - off) / cell_size)),
                int(round((float(xy[1]) - off) / cell_size)))

    def run_config(tag, use_blocked, use_last_rev, revisit_pen, n_gifs, verbose_first):
        """Run num_ep episodes under one memory configuration; return metrics dict."""
        env = build_env()
        successes, spls = [], []
        for ep in range(args.num_ep):
            obs, info_e = env.reset()
            obs, _, _, _, info_e = env.step(np.zeros(env.action_space.shape[0]))
            goal_xy = norm.normalize_location(
                info_e["target_position"].to(dtype=torch.float32, device=device).unsqueeze(0))[0]
            goal_img = info_e.get("target_obs")
            # A* path length only sizes the per-episode budget; it never steers.
            grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
            solved = solve_a_star(grid, tuple(int(c) for c in env.agent_cell),
                                  tuple(int(c) for c in env.goal_cell))
            astar_len = (len(solved[0]) - 1) if solved else 100
            max_steps = min(int(args.budget_factor * astar_len + args.budget_margin), N_ALLOWED)
            frames = [obs]
            n_moves = 0
            success = False
            blocked, visit, last_rev = {}, {}, -1
            verbose = verbose_first and ep == 0
            for step in range(max_steps):
                ot = obs_tensor(obs)
                z = jepa.encode(ot)
                sg = subgoal(z, goal_xy.unsqueeze(0))[0]      # [2] normalized waypoint
                cell = tuple(int(c) for c in env.agent_cell)
                visit[cell] = visit.get(cell, 0) + 1
                # Score cardinals by a K-step fine-WM rollout: distance of the predicted
                # endpoint to the waypoint (+ optional revisit penalty on the predicted
                # cell). A blocked dir's endpoint stays put -> far from waypoint.
                dist = []
                for dd in range(4):
                    zf = fine_kstep_target(jepa, ot, torch.tensor([dd], device=device),
                                           args.lookahead, cell_size)
                    d = float(torch.norm(probe_xy(zf) - sg).item())
                    if revisit_pen > 0:
                        d += revisit_pen * visit.get(pred_cell(zf), 0)
                    dist.append(d)
                order = sorted(range(4), key=lambda dd: dist[dd])
                # Apply execution-memory filters (each independently toggleable). If all
                # directions get filtered out, fall back to the full greedy order.
                cand = [d for d in order
                        if (not use_blocked or d not in blocked.get(cell, set()))
                        and (not use_last_rev or d != last_rev)]
                cand += [d for d in order if d not in cand]
                if verbose and step < 12:
                    print(f"   [s{step}] cell={list(cell)} goal={env.goal_cell.tolist()} "
                          f"dist[D,U,R,L]={[round(x, 2) for x in dist]}", flush=True)
                moved = done = False
                for d in cand:
                    prev = env.agent_cell.copy()
                    obs, _, done, trunc, info_e = env.step(
                        (CARDINALS[d] * cell_size).cpu().numpy())
                    if not np.array_equal(env.agent_cell, prev):
                        moved = True
                        if use_last_rev:
                            last_rev = OPP[d]
                        frames.append(obs)
                        n_moves += 1
                        break
                    if use_blocked:
                        blocked.setdefault(cell, set()).add(d)
                    if done or trunc:
                        break
                if done:
                    success = True
                    break
                if not moved:
                    break
            successes.append(float(success))
            spls.append((astar_len / max(n_moves, astar_len)) if success else 0.0)
            if verbose:
                print(f"   [{tag}] ep0: A*_len={astar_len} budget={max_steps} "
                      f"moves={n_moves}", flush=True)
            if ep < n_gifs and len(frames) > 1:
                label = "succ" if success else "fail"
                prefix = f"{tag}_" if tag else ""
                try:
                    save_gif(torch.stack([fr.to(torch.float32) for fr in frames]),
                             os.path.join(args.rdir, f"{prefix}ep{ep}_{label}.gif"), fps=8,
                             show_frame_numbers=True, goal_frame=goal_img)
                except Exception as e:
                    print(f"   [gif {tag} ep{ep}] skipped: {e}", flush=True)
            print(f"[{tag or 'eval'}] ep {ep}: {'SUCCESS' if success else 'fail'}", flush=True)
        sr, spl = float(np.mean(successes)), float(np.mean(spls))
        return {"tag": tag, "use_blocked": use_blocked, "use_last_rev": use_last_rev,
                "revisit_pen": revisit_pen, "success_rate": sr, "spl": spl}

    revisit_pen = args.revisit_pen if args.revisit_pen is not None else args.revisit_pen_pos
    n_gifs = args.n_gifs if args.n_gifs is not None else args.n_gifs_pos
    print(f"[eval] A*-FREE | N={sck['N']} | {args.num_ep} mazes | seed={args.seed} | "
          f"lookahead={args.lookahead} | budget={args.budget_factor}xA*+{args.budget_margin}",
          flush=True)

    if args.ablation:
        # Ablation needs a non-zero penalty to make the -revisit row meaningful.
        p = revisit_pen if revisit_pen > 0 else 1.0
        rows_spec = [
            ("full", True, True, p),
            ("-revisit", True, True, 0.0),
            ("-last_rev", True, False, p),
            ("-blocked", False, True, p),
            ("memoryless", False, False, 0.0),
        ]
        print(f"[ablation] revisit_pen={p} | rows: "
              f"{', '.join(r[0] for r in rows_spec)}", flush=True)
        rows = [run_config(tag, ub, ulr, rp, n_gifs if tag == "full" else 0, tag == "full")
                for (tag, ub, ulr, rp) in rows_spec]
        on = lambda b: "on" if b else "off"
        header = "| config | blocked | last_rev | revisit_pen | success | SPL |"
        sep = "|---|---|---|---|---|---|"
        body = [f"| {r['tag']} | {on(r['use_blocked'])} | {on(r['use_last_rev'])} | "
                f"{r['revisit_pen']:g} | {r['success_rate']*100:.2f}% | {r['spl']:.3f} |"
                for r in rows]
        table = "\n".join([header, sep, *body])
        print("\n[ablation] memory-heuristic leave-one-out "
              f"({args.num_ep} mazes, seed {args.seed}):\n" + table, flush=True)
        with open(os.path.join(args.rdir, "ablation.md"), "w") as fh:
            fh.write(f"# A*-free memory ablation ({args.num_ep} mazes, seed {args.seed})\n\n"
                     + table + "\n")
        json.dump({"num_episodes": args.num_ep, "seed": args.seed, "N": sck["N"],
                   "astar_free": True, "budget": f"{args.budget_factor}xA*+{args.budget_margin}",
                   "rows": rows}, open(os.path.join(args.rdir, "ablation.json"), "w"), indent=2)
        return

    res = run_config("", args.use_blocked, args.use_last_rev, revisit_pen, n_gifs, True)
    json.dump({**res, "num_episodes": args.num_ep, "seed": args.seed, "N": sck["N"],
               "astar_free": True, "budget": f"{args.budget_factor}xA*+{args.budget_margin}"},
              open(os.path.join(args.rdir, "subgoal_eval.json"), "w"), indent=2)
    print(f"[eval] A*-FREE success={res['success_rate']*100:.2f}%  SPL={res['spl']:.3f}  "
          f"over {args.num_ep} mazes  (blocked={args.use_blocked} last_rev={args.use_last_rev} "
          f"revisit_pen={revisit_pen})", flush=True)


if __name__ == "__main__":
    main()
