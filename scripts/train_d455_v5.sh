#!/usr/bin/env bash
# Adapt the v4 robust teacher to the v5 distribution.
#
# Three changes, all forced by measurements taken on the rig on 2026-08-31:
#
#  * the target mask now disappears.  Tracing the arm's sphere cover against
#    the camera-to-object line on two recordings: the arm blocked the line 52%
#    of the time on the fast run and 16% on the slow one, and with a CLEAR line
#    the segmenter still found the object only 28% and 50% of the time.  The
#    simulator called the same scene visible 99%.  A policy trained on that has
#    never had to act without seeing its target, and the v3 policy that scored
#    1.45 placements per environment here scores 0.38 once the measured
#    visibility is applied -- which is the bench failure, reproduced.
#
#  * the command rate ceiling drops from 0.62 to 0.50 of the joint trip.  The
#    0.62 was derived from kp 125 / kd 6.5; the arm measured 280-379 with
#    kd/kp = 56 ms, so the derivation behind the old number no longer holds.
#
#  * the progress term that paid per step for closing on the bin is off.  Its
#    own comment records that at weight 25 the trip did not pay for itself, so
#    if the policy stops transporting, raise HEAVY_DR_PROFILE["task"]
#    ["transport_progress_weight"] before changing anything else.
#
# The teacher is state-based and never sees the mask, so it is adapting only to
# the rate ceiling and the reward.  The vision stages are where the mask
# dropout does its work.
set -Eeuo pipefail

ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
TAG=${TAG:-d455_v5_slow_blind}
BASE=${BASE:-logs/rsl_rl/piperx_pick_place_robust/2026-08-30_11-29-11_d455_heavy_dr_v4_scenery_robust_teacher/model_6997.pt}
NUM_ENVS=${NUM_ENVS:-8192}
ADAPT_ITERS=${ADAPT_ITERS:-2000}
GPU=${GPU:-3}
OUT=${OUT:-results/d455_heavy_dr/$TAG}

cd "$ROOT"
test -f "$BASE"
mkdir -p "$OUT"

start_name=$(basename "$BASE")
bootstrap="logs/rsl_rl/piperx_pick_place_robust/${TAG}_bootstrap"
mkdir -p "$bootstrap"
[ -f "$bootstrap/$start_name" ] || cp "$BASE" "$bootstrap/$start_name"

printf '%s\n' "$BASE" >"$OUT/base_teacher_checkpoint.txt"
printf 'adapting from %s for %s additional iterations: slower command rate, no
progress reward, and (for the vision stages) a target mask that vanishes\n' \
  "$start_name" "$ADAPT_ITERS"

TASK=Mjlab-Pick-Place-PiperX-Robust NUM_ENVS="$NUM_ENVS" \
ITERS="$ADAPT_ITERS" GPUS="[$GPU]" RUN_NAME="${TAG}_robust_teacher" \
  bash scripts/train.sh \
  --agent.resume True \
  --agent.load-run "$(basename "$bootstrap")" \
  --agent.load-checkpoint "$start_name" \
  --agent.logger tensorboard
