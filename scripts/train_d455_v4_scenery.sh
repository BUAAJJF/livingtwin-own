#!/usr/bin/env bash
# Adapt the v3 robust teacher to the v4 distribution, whose one new axis is
# what lies beyond the task's own sector.
#
# The simulated world has a single piece of scenery -- an infinite MuJoCo
# PLANE -- so everything outside the working area is that plane receding
# smoothly to the far clip.  No deployment scene looks like that.  Feeding one
# recorded deployment channel 0 into the environment, with the mask left
# untouched and everything else the simulator's own, took the trained policy
# from 170 objects placed to zero; flattening the deployment image onto the
# calibrated plane first brought it to 37, above a frozen-frame control of 15.
#
# That correction works and it is a patch: it can only remove what it knows
# about, and it silently erased the top third of the arm the first time
# because a height threshold is not a model of a robot.  Randomising the
# region instead is what removes the deployment's need to correct anything.
set -Eeuo pipefail

ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
TAG=${TAG:-d455_heavy_dr_v4_scenery}
# The v3 robust teacher, which already carries the no-contact task and the
# broad plant distribution.  Only the scenery is new, so this adapts rather
# than restarts.
BASE=${BASE:-logs/rsl_rl/piperx_pick_place_robust/2026-08-28_03-05-25_d455_heavy_dr_v3_no_contact_robust_teacher/model_5498.pt}
NUM_ENVS=${NUM_ENVS:-8192}
ADAPT_ITERS=${ADAPT_ITERS:-1500}
GPU=${GPU:-0}
OUT=${OUT:-results/d455_heavy_dr/$TAG}

cd "$ROOT"
test -f "$BASE"
mkdir -p "$OUT"

start_name=$(basename "$BASE")
bootstrap="logs/rsl_rl/piperx_pick_place_robust/${TAG}_bootstrap"
mkdir -p "$bootstrap"
[ -f "$bootstrap/$start_name" ] || cp "$BASE" "$bootstrap/$start_name"

printf '%s\n' "$BASE" >"$OUT/base_teacher_checkpoint.txt"
printf 'adapting from %s for %s additional iterations with scenery DR\n' \
  "$start_name" "$ADAPT_ITERS"

TASK=Mjlab-Pick-Place-PiperX-Robust NUM_ENVS="$NUM_ENVS" \
ITERS="$ADAPT_ITERS" GPUS="[$GPU]" RUN_NAME="${TAG}_robust_teacher" \
  bash scripts/train.sh \
  --agent.resume True \
  --agent.load-run "$(basename "$bootstrap")" \
  --agent.load-checkpoint "$start_name" \
  --agent.logger tensorboard
