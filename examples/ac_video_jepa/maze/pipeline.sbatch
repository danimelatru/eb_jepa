#!/bin/bash
# End-to-end maze pipeline: fine WM -> subgoal predictor -> memory ablation table.
# wandb is disabled so no API key / internet is needed. One submit, one log to tail.
#SBATCH --job-name=maze_pipeline
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/work/vivatech-hacktheworld/zraounak/logs/maze_pipeline_%j.out
#SBATCH --error=/lustre/work/vivatech-hacktheworld/zraounak/logs/maze_pipeline_%j.out
set -euo pipefail

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
export WANDB_MODE=disabled          # no wandb login required
cd "$REPO"
uv sync --project "$REPO" >/dev/null 2>&1 || true

# --- knobs (override at submit time with --export=ALL,CFG=...,N=...) ---
CFG=${CFG:-examples/ac_video_jepa/cfgs/train/maze/train_maze_small.yaml}  # 11x11, fast (~15m)
FINE_DIR=${FINE_DIR:-$EBJEPA_CKPTS/maze/fine_small}
SG_DIR=${SG_DIR:-$EBJEPA_CKPTS/maze/subgoal_small}
OUT=${OUT:-$EBJEPA_WORK/results/maze/ablation_small}
N=${N:-8}              # subgoal horizon (waypoint ~N cells ahead). NOT 512 (would collapse to goal)
EPOCHS=${EPOCHS:-10}
NUM_EP=${NUM_EP:-32}
LOOKAHEAD=${LOOKAHEAD:-4}
REVISIT_PEN=${REVISIT_PEN:-1.0}
SEED=${SEED:-0}
N_GIFS=${N_GIFS:-4}

echo "=== [1/3] fine world model | cfg=$CFG | $(hostname) $(date) ==="
uv run --project "$REPO" python -m examples.ac_video_jepa.main \
    --fname "$CFG" --meta.model_folder="$FINE_DIR" --meta.load_model=False

echo "=== [2/3] subgoal predictor | N=$N epochs=$EPOCHS | $(date) ==="
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.main_subgoal \
    "$FINE_DIR/latest.pth.tar" "$SG_DIR" "$N" "$EPOCHS"

echo "=== [3/3] memory ablation | $NUM_EP mazes | $(date) ==="
mkdir -p "$OUT"
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_subgoal \
    "$FINE_DIR/latest.pth.tar" "$SG_DIR/subgoal.pth.tar" "$OUT" \
    "$NUM_EP" "$LOOKAHEAD" --ablation --revisit-pen "$REVISIT_PEN" --seed "$SEED" --n-gifs "$N_GIFS"

echo "=== DONE $(date) — table below and at $OUT/ablation.md ==="
cat "$OUT/ablation.md"
