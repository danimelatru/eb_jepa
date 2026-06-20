#!/bin/bash
#SBATCH --job-name=maze_ablation
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/work/vivatech-hacktheworld/zraounak/logs/maze_ablation_%j.out
# NOTE: #SBATCH lines cannot expand $VARS, so the output path is hardcoded to
# $EBJEPA_WORK/logs. Adjust if your $EBJEPA_WORK differs.

set -euo pipefail

REPO=/lustre/work/vivatech-hacktheworld/zraounak/eb_jepa
source "$REPO/env.sh"
module load python312
cd "$REPO"
uv sync

# Override any of these at submit time, e.g.:
#   sbatch --export=ALL,NUM_EP=64,SEED=1 examples/ac_video_jepa/maze/ablation.sbatch
FINE=${FINE:-$EBJEPA_WORK/maze_fine_small/latest.pth.tar}
SG=${SG:-$EBJEPA_WORK/maze_subgoal_small/subgoal.pth.tar}
OUT=${OUT:-$EBJEPA_WORK/results/maze/ablation}
NUM_EP=${NUM_EP:-32}
LOOKAHEAD=${LOOKAHEAD:-4}
REVISIT_PEN=${REVISIT_PEN:-1.0}
SEED=${SEED:-0}
N_GIFS=${N_GIFS:-4}

mkdir -p "$OUT"

uv run python -m examples.ac_video_jepa.maze.eval_subgoal \
  "$FINE" "$SG" "$OUT" "$NUM_EP" "$LOOKAHEAD" \
  --ablation --revisit-pen "$REVISIT_PEN" --seed "$SEED" --n-gifs "$N_GIFS"

echo "[ablation] table written to $OUT/ablation.md and $OUT/ablation.json"
