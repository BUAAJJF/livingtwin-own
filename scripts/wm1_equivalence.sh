#!/usr/bin/env bash
# Re-measure the WM1 target domain after moving observation delay onto mjlab's
# DelayBuffer.
#
#   scripts/wm1_equivalence.sh [gpu]
#
# Every Phase WM1 gate threshold is inherited from Phase WM0:
#
#   J_zero_shot  42.19 obj/min   (repeats 41.85, 42.32, 42.40)
#   J_oracle     49.72
#   G2 pass      47.46 = 42.19 + 0.70 * (49.72 - 42.19)
#
# Those were measured through a ring buffer that no longer exists.  The unit
# test pins the two implementations to the same output sequence on a synthetic
# stream, but they differ in a three-step transient after an episode boundary
# -- mjlab ramps the served lag up as its history refills, the ring buffer held
# the frame from the reset -- and no unit test can say what that is worth in
# objects per minute.  This measures it, under the same protocol and the same
# three seeds, so the report can either carry the WM0 thresholds forward or say
# out loud that it cannot.
set -Eeuo pipefail

GPU=${1:-0}
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-disable}
PREFIX="$(micromamba env list | awk '$1=="mjlab" {print $NF}')"
[ -n "$PREFIX" ] && export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

BASE=logs/rsl_rl/piperx_pick_place_vision/2026-08-22_17-15-09_f3/model_1500.pt
TASK=Mjlab-Pick-Place-PiperX-Vision
OUT=results/wm1_latency/equivalence
mkdir -p "$OUT"

i=0
for SEED in 20260823 31415926 27182818; do
  for TAG in zeroshot nominal; do
    FLAG=""
    [ "$TAG" = zeroshot ] && FLAG="--obs-latency-steps 3"
    micromamba run -n mjlab python scripts/accept_s1.py "$TASK" "$BASE" \
      --num-envs 512 --steps 2400 --seed "$SEED" --device "cuda:$GPU" \
      --label "${TAG}__r$i" --json "$OUT/${TAG}__r$i.json" \
      $FLAG > "$OUT/${TAG}__r$i.log" 2>&1
    echo "[$TAG r$i done]"
  done
  i=$((i + 1))
done
echo "=== EQUIVALENCE DONE ==="
