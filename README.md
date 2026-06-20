<h1 align="center">
    <p>⚡ <b>EB-JEPA</b></p>
</h1>

<h2 align="center">
    <p><i>Energy-Based Joint-Embedding Predictive Architectures</i></p>
</h2>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/Trick5t3r/eb_jepa" target="_blank" style="margin: 2px;"><img alt="Github" src="https://img.shields.io/badge/Github-Trick5t3r/eb__jepa-black?logo=github" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://arxiv.org/abs/2602.03604" target="_blank" style="margin: 2px;"><img alt="ArXiv" src="https://img.shields.io/badge/arXiv-2602.03604-b5212f?logo=arxiv" style="display: inline-block; vertical-align: middle;"/></a>
</div>

<br>

<p align="center">
  <b><a href="https://ai.facebook.com/research/">Meta AI Research, FAIR</a></b>
</p>

<p align="center">
  <a href="https://x.com/BasileTerv987">Basile Terver</a>,
  Randall Balestriero,
  Megi Dervishi,
  David Fan,
  Quentin Garrido,
  Tushar Nagarajan,
  <br>
  Koustuv Sinha,
  Wancong Zhang,
  Mike Rabbat,
  Yann LeCun,
  Amir Bar
</p>

<p align="center">
  An open source library and tutorial for learning representations for<br>
  prediction and planning using joint embedding predictive architectures.
</p>

<p align="center">
  <img src="docs/archi-schema-eb-jepa.png" alt="EB-JEPA Architecture" width="800">
</p>

> Each example is (almost) self-contained and training takes up to a few hours on a single GPU card.

---

## 🏆 VivaTech Hackathon 2026 — Team HackTheWorld

### Hierarchical World Model Co-Training: A Scientific Pivot

**Track:** Hierarchical Maze World Model — A*-Free Navigation

Initially, we hypothesized that the catastrophic forgetting during Co-Training could be solved via **Latent Distillation**. However, systematic hyperparameter sweeps on the cluster revealed two massive discoveries that forced us to pivot:

#### 1. Co-Training is Actively Harmful
Our best result (81.25% A*-Free Success) was achieved at Epoch 4 of Co-Training. However, mathematical analysis of our training logs revealed that during the first 5 "warmup" epochs, the encoder's learning rate was exactly `0.0`. 
This proves that the performance spike was **not** due to Latent Distillation—it was simply the result of training the High-Level Subgoal Predictor for 5 extra epochs on top of the **perfectly frozen physics representations** from Stage 1. 

As soon as Epoch 5 hit and the encoder unfroze (activating Latent Distillation), the performance crashed to **34%**.

**Conclusion:** The mathematical optimum for Hierarchical World Models is strict compartmentalization. Co-training the shared latent space actively destroys the low-level physics representations.

#### 2. Hand-Coded Execution Memory vs. the Learned Policy
The 66% A\*-free number leans on three pieces of hand-coded *execution memory* in the
reacher loop (`eval_subgoal.py`). None of them use privileged information (no A\*, no
goal cell, no maze grid) — they are standard closed-loop bookkeeping — but they are
**not learned**, so the success number must not silently credit the learned policy for
what they do:
- `blocked`: cross-step memory of `(cell, direction)` pairs that physically didn't move (wall memory).
- `last_rev`: forbids immediately reversing the last committed move (anti-oscillation).
- `revisit_pen`: count-based penalty for stepping onto already-visited cells (exploration drive).

The honest charge is **attributional, not goal-peeking**: with these stripped, a purely
reactive memoryless policy oscillates in place (~0%), so the heuristics — not the learned
world-model — are doing much of the load-bearing navigation. The fix is not a separate
"strict" script but an **ablation table** that quantifies each heuristic's contribution.

### Honest Evaluation: One Merged Evaluator + Ablation Table

`eval_subgoal.py` is now the single A\*-free evaluator. Each memory heuristic is a flag,
and `--ablation` runs a leave-one-out table on a **fixed maze set** (same `--seed` for
every row, so rows are directly comparable). The deleted `eval_strict.py` is just the
`--no-blocked --no-last-rev --revisit-pen 0` row.

1. **Strict Isolation:** Co-Training (Stage 3) is disabled — we run only Stage 1 (Fine Model) and Stage 2 (Subgoal Predictor).
2. **Honest Evaluation:** The ablation reports `full` vs. each heuristic removed vs. the `memoryless` reactive floor, so the learned policy's true contribution is visible.

```bash
# 1. Train the Fine World Model on 11x11 — ~15 min
uv run python -m examples.ac_video_jepa.maze.main examples/ac_video_jepa/cfgs/train/maze/train_maze_small.yaml --folder=./maze_fine_small

# 2. Train the Subgoal Predictor on 11x11 — ~10 min
uv run python -m examples.ac_video_jepa.maze.main_subgoal \
  ./maze_fine_small/latest.pth.tar ./maze_subgoal_small 512 10 0.001

# 3. A*-Free memory ablation table (full -> memoryless floor) on a fixed maze set
uv run python -m examples.ac_video_jepa.maze.eval_subgoal \
  ./maze_fine_small/latest.pth.tar \
  ./maze_subgoal_small/subgoal.pth.tar \
  ./eval_results_ablation 32 4 --ablation --revisit-pen 1.0 --n-gifs 4 --seed 0
```

This writes `ablation.md` / `ablation.json`:

| config | blocked | last_rev | revisit_pen | success | SPL |
|---|---|---|---|---|---|
| full | on | on | 1.0 | … | … |
| -revisit | on | on | 0 | … | … |
| -last_rev | on | off | 1.0 | … | … |
| -blocked | off | on | 1.0 | … | … |
| memoryless | off | off | 0 | … | … |

To run a single config instead (legacy positional args still work):

```bash
# memoryless reactive floor (the old eval_strict.py)
uv run python -m examples.ac_video_jepa.maze.eval_subgoal \
  ./maze_fine_small/latest.pth.tar ./maze_subgoal_small/subgoal.pth.tar \
  ./eval_results_strict 32 4 --no-blocked --no-last-rev --revisit-pen 0
```

### Files Modified

| File | Description |
|---|---|
| `examples/ac_video_jepa/maze/main_subgoal.py` | Bug fix: added `data_pipeline.warm_up()` and `.float()` casting |
| `examples/ac_video_jepa/maze/eval_subgoal.py` | Merged evaluator: per-heuristic flags (`--use-blocked`/`--use-last-rev`/`--revisit-pen`) + `--ablation` table; seeded mazes; absorbs and replaces `eval_strict.py` |
| `presentation.tex` | LaTeX Beamer presentation with TikZ diagrams and our scientific pivot |
| `hackathon_paper.md` | Detailed research-paper-style analysis |

---

## 📚 Examples

### [Image JEPA](examples/image_jepa/README.md)

Self-supervised representations from unlabeled images on CIFAR-10, evaluated on classification.

![Image JEPA Architecture](examples/image_jepa/assets/arch_figure.png)

### [Video JEPA](examples/video_jepa/README.md)

Predict next image representation in a sequence.

![Moving MNIST](examples/video_jepa/assets/viz.png)

### [AC Video JEPA](examples/ac_video_jepa/README.md)

JEPA for world modeling + planning in Two Rooms environment.

| Planning Episode | Task Definition |
|------------------|-----------------|
| <img src="examples/ac_video_jepa/assets/top_randw_agent_steps_succ.gif" alt="Successful planning episode" width="155" /> | <img src="examples/ac_video_jepa/assets/top_randw_state.png" alt="Episode task definition" width="300" /> |
| *Successful planning episode* | *From init to goal state* |

---

## 🚀 Installation

### HTW cluster — quick start (hackathon only)

> Skip this section unless you are on the HTW hackathon cluster — the generic install below is all you need locally.

Please follow the [setup instructions](setup.md) before starting the project.

---

### Local / generic (start here)

We use [uv](https://docs.astral.sh/uv/guides/projects/) for package management.

```bash
# Install dependencies
uv sync
# Option 1: Activate virtual environment
source .venv/bin/activate
python -m examples.image_jepa.main
# Option 2: Run directly with uv
uv run python -m examples.image_jepa.main
```
If you need conda-specific packages, you can use **Conda + uv**

```bash
# Create conda environment with Python 3.12
conda create -n eb_jepa python=3.12 -y
conda activate eb_jepa
# Install package in editable mode with dev dependencies (pytest, black, isort, autoflake)
uv pip install -e . --group dev
```

Add these to your `~/.bashrc` for persistent configuration.

```bash
# Where datasets are stored / looked up
export EBJEPA_DSETS=/path/to/eb_jepa/datasets
# Optional: Directory for checkpoints and logs
export EBJEPA_CKPTS=/path/to/checkpoints
```

Verify the install with `uv run pytest tests/`.

## 🏋️ Training

### Quick Start

```bash
# Local training
python -m examples.{image_jepa,video_jepa,ac_video_jepa}.main
```
> Our default configs are tuned for H100 GPUs. With older GPUs (e.g., A100, V100), you may need to reduce batch size to fit in memory.

### 📂 Folder Structure

All experiments use a unified folder structure:

```
checkpoints/
└── {example_name}/
    ├── dev_2026-01-16_00-10/                 # Single/local runs (dev_ prefix)
    │   └── {exp_name}_seed1/
    │
    ├── sweep_2026-01-16_00-10/         # Auto-named 3-seed sweep
    │   ├── {exp_name}_seed1/
    │   ├── {exp_name}_seed1000/
    │   └── {exp_name}_seed10000/
    │
    └── sweep_my_experiment/            # Custom-named sweep
        └── ...
```

`{exp_name}` encodes key hyperparameters to avoid folder collisions, e.g.:
- **image_jepa**: `resnet_vicreg_proj_bs256_ep300_ph2048_po2048_std1.0_cov80.0`
- **video_jepa**: `resnet_bs64_lr0.001_std10.0_cov100.0`
- **ac_video_jepa**: `impala_cov8_std16_simt12_idm1`

<details>
<summary><span style="font-size: 1.17em; font-weight: bold;">🖥️ SLURM Launcher (optional)</span></summary>

| Command | Description |
|---------|-------------|
| `--example {name}` | Choose: `image_jepa`, `video_jepa`, `ac_video_jepa`, `maze`, `fintime`, `ltsf`, `eeg`, `audio`, `pointcloud`, `gray_scott`, `intuitive_physics`, `factors_of_variation` |
| `--fname {path}` | Run the sweep specified in the config at `{path}` |
| `--single` | Launch single job (dev mode) |
| `--sweep {name}` | Custom sweep name |
| `--array-parallelism {N}` | Limits the maximum number of concurrent jobs to `N` |
| `--full-sweep` | Full hyperparameter sweep from config |
| `--use-wandb-sweep` | Enable wandb sweep UI |

```bash
# 3 seeds with wandb averaging (recommended)
python -m examples.launch_sbatch --example image_jepa --fname examples/image_jepa/cfgs/default.yaml

# Custom sweep name
python -m examples.launch_sbatch --example image_jepa --fname examples/image_jepa/cfgs/default.yaml --sweep my_experiment

# Single job
python -m examples.launch_sbatch --example image_jepa --fname examples/image_jepa/cfgs/default.yaml --single

# Full hyperparameter sweep
python -m examples.launch_sbatch --example image_jepa --fname examples/image_jepa/cfgs/default.yaml --full-sweep

# With wandb sweep UI for hyperparameter analysis
python -m examples.launch_sbatch --example image_jepa --fname examples/image_jepa/cfgs/default.yaml --use-wandb-sweep
```

Replace `image_jepa` with `ac_video_jepa`, `video_jepa`, or `maze` for other examples.

**Full Sweep Configuration:** The `--full-sweep` flag reads the `sweep.param_grid` section from the example's YAML config file (e.g., `examples/image_jepa/cfgs/default.yaml`). Without this flag, only a 3-seed sweep is launched. To customize sweep parameters, edit the `sweep` section in the config:

```yaml
# Example: examples/image_jepa/cfgs/default.yaml
sweep:
  param_grid:
    loss.cov_coeff: [0.1, 1.0, 10.0, 100.0]
    loss.std_coeff: [1.0, 10.0]
    meta.seed: [1, 1000, 10000]
```

### Wandb Seed Averaging

Runs with the same hyperparameters but different seeds share the same wandb run name, enabling automatic averaging:

1. Go to wandb web UI → Runs table
2. Click **"Group by"** → select **"Name"**
   → Groups runs with identical hyperparameters (different seeds) together

To filter runs from a specific sweep:
3. Click **"Filter"** → **"Group"** → select your sweep name

For detailed wandb sweep analysis (parallel coordinates, hyperparameter importance):
1. Use `--use-wandb-sweep` flag when launching
2. Go to wandb web UI → left pane → **"Sweeps"** → click your sweep name

**SLURM Configuration:** SLURM parameters default to the HTW cluster and are read from `EBJEPA_SLURM_*` env vars (set by `env.sh`, which also auto-detects your account/QOS per user). Override per launch with the CLI flags `--partition`/`--account`/`--cpus-per-task`/`--time-min`/`--gpus-per-node`, or export the matching `EBJEPA_SLURM_*` var. The `SLURM_DEFAULTS` dictionary at the top of `examples/launch_sbatch.py` holds the fallbacks.

</details>

## 🧪 Running test cases

Libraries added to eb_jepa [must have their own test cases](/tests/). To run the tests:

```bash
# With uv sync installation
uv run pytest tests/
# With conda + uv installation (no .venv created)
pytest tests/
```

## 👩‍💻 Development

Before contributing, please format your code with the following tools:

```bash
# Remove unused imports
autoflake --remove-all-unused-imports -r --in-place .
# Sort imports
python -m isort eb_jepa examples tests
# Format code
python -m black eb_jepa examples tests
```

## 📚 Citing EB-JEPA

If you find this repository useful, please consider giving a ⭐ and citing:

```bibtex
@misc{terver2026lightweightlibraryenergybasedjointembedding,
      title={A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures},
      author={Basile Terver and Randall Balestriero and Megi Dervishi and David Fan and Quentin Garrido and Tushar Nagarajan and Koustuv Sinha and Wancong Zhang and Mike Rabbat and Yann LeCun and Amir Bar},
      year={2026},
      eprint={2602.03604},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.03604},
}
```

## 📄 License

EB-JEPA is Apache licensed. See [LICENSE](LICENSE.md).
