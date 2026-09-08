#!/usr/bin/env bash
# Run scripts/accept_s1.py with the environment it needs.
#
#   scripts/eval.sh <TaskId> <checkpoint> <label> [extra accept_s1 flags...]
#
# Environment:
#   CONDA_ENV   Conda environment name     (default: livingtwin)
#   GPU         cuda device index          (default: 0)
#   NUM_ENVS    parallel environments      (default: 512)
#   STEPS       control steps              (default: 2400)
#   SEED        rollout seed               (default: 20260823)
#   OUT         directory for the JSON     (default: results/eval/baseline)
#
# 512 x 2400 is the protocol every number in docs/history/nominal_results_2026-08.md was measured
# under: 409.6 arm-minutes, about 22k object instances.  Smaller samples read
# 1-2% optimistic across the board, so they are not comparable and are not
# mixed in.
set -Eeuo pipefail

if [ $# -lt 3 ]; then
  sed -n '2,17p' "$0" >&2
  exit 2
fi

TASK=$1; CKPT=$2; LABEL=$3; shift 3

CONDA_ENV=${CONDA_ENV:-${MJLAB_ENV:-livingtwin}}
GPU=${GPU:-0}
NUM_ENVS=${NUM_ENVS:-512}
STEPS=${STEPS:-2400}
SEED=${SEED:-20260823}
OUT=${OUT:-results/eval/baseline}

cd "$(dirname "$0")/.."
ROOT=$PWD
source "$ROOT/scripts/conda_env.sh"

# The camera sensors go through mujoco_warp's rasteriser and need no GL at
# all; importing mujoco still initialises a backend, which fails on a box
# without glvnd's libEGL.so.1.
mkdir -p "$OUT"
exec python scripts/accept_s1.py \
  "$TASK" "$CKPT" \
  --num-envs "$NUM_ENVS" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --device "cuda:$GPU" \
  --label "$LABEL" \
  --json "$OUT/$LABEL.json" \
  "$@"
