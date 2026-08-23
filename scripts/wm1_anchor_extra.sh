#!/usr/bin/env bash
# Three more repeats of each anchor, at evaluation seeds nobody has used.
#
#   scripts/wm1_anchor_extra.sh [gpu]
#
# Throughput reproduced to a tenth of an object per minute between Phase WM0
# and this phase's re-measurement.  The safety-trip rate did not: the nominal
# condition, whose code path did not change at all, moved from 2.29/h to
# 3.15/h.  That is 47 events against 43, a rate ratio of 1.37 with a 95%
# interval of [0.91, 2.08] -- consistent with noise, and not something three
# repeats can settle either way.  G3 is a threshold on this quantity, so the
# anchors get six repeats instead of three rather than the report guessing.
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

i=3
for SEED in 16180339 14142135 17320508; do
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
echo "=== ANCHOR EXTRA DONE ==="
