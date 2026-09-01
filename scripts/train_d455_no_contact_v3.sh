#!/usr/bin/env bash
# Adapt the proven nominal state teacher to the full D455 DR distribution.
# Simulated table contacts are audited but never used as reward/termination.
set -Eeuo pipefail

ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
TAG=${TAG:-d455_heavy_dr_v3_no_contact}
BASE=${BASE:-logs/rsl_rl/piperx_pick_place/2026-08-27_15-50-47_d455_heavy_dr_v2_nominal_teacher/model_3499.pt}
NUM_ENVS=${NUM_ENVS:-8192}
ADAPT_ITERS=${ADAPT_ITERS:-2000}
GPU=${GPU:-0}
OUT=${OUT:-results/d455_heavy_dr/$TAG}

cd "$ROOT"
test -f "$BASE"
mkdir -p "$OUT"

start_name=$(basename "$BASE")
start_iter=${start_name#model_}
start_iter=${start_iter%.pt}
bootstrap="logs/rsl_rl/piperx_pick_place_robust/${TAG}_bootstrap"
mkdir -p "$bootstrap"
if [ ! -f "$bootstrap/$start_name" ]; then
  cp "$BASE" "$bootstrap/$start_name"
fi

printf '%s\n' "$BASE" >"$OUT/base_teacher_checkpoint.txt"
printf '%s\n' "$bootstrap/$start_name" >"$OUT/bootstrap_checkpoint.txt"
printf 'adapting from iteration %s for %s additional iterations with full D455 DR\n' \
  "$start_iter" "$ADAPT_ITERS"

TASK=Mjlab-Pick-Place-PiperX-Robust NUM_ENVS="$NUM_ENVS" \
ITERS="$ADAPT_ITERS" GPUS="[$GPU]" RUN_NAME="${TAG}_robust_teacher" \
  bash scripts/train.sh \
  --agent.resume True \
  --agent.load-run "$(basename "$bootstrap")" \
  --agent.load-checkpoint "$start_name" \
  --agent.logger tensorboard
