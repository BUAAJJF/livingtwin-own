#!/usr/bin/env bash
# Long calibration sessions, for choosing score weights and temperature.
#
#   scripts/wm1_collect_cal.sh <cuda_device> <shard> <n_shards> [out_dir]
#
# The `val`/`valh` splits are 30 s per session, which is enough to pick a
# dynamics model but not to calibrate a posterior at the 180 s and 300 s
# budgets: fitting a temperature at 30 s and applying it at 300 s is a
# ten-fold extrapolation of exactly the quantity that decides how confident
# the report claims to be.  These are the same length as the test sessions --
# 300 s -- and are simulation domains with no reward attached, so everything
# the posterior stage tunes is tuned away from the target.
#
# Two shape populations: `cal` uses the training classes, `calh` the held-out
# ones.  A weight chosen on `cal` and checked on `calh` cannot be a weight that
# only works on the objects the model was fitted to.
set -Eeuo pipefail

GPU=${1:?usage: wm1_collect_cal.sh <cuda_device> <shard> <n_shards> [out]}
SHARD=${2:?}; NSHARD=${3:?}; OUT=${4:-results/wm1_latency/data}

cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-disable}
PREFIX="$(micromamba env list | awk '$1=="mjlab" {print $NF}')"
[ -n "$PREFIX" ] && export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

JOBS=()
for LAG in 0 1 2 3 4; do
  JOBS+=("--lag $LAG --seed $((5050 + LAG)) --num-envs 16 --steps 15000 --split cal  --shapes train")
  JOBS+=("--lag $LAG --seed $((6060 + LAG)) --num-envs 16 --steps 15000 --split calh --shapes holdout")
done

mkdir -p "$OUT" logs/wm1_collect
i=100
for J in "${JOBS[@]}"; do
  if [ $(((i - 100) % NSHARD)) -eq "$SHARD" ]; then
    echo "=== [gpu $GPU shard $SHARD] job $i: $J"
    micromamba run -n mjlab python scripts/wm_dataset.py $J \
      --device "cuda:$GPU" --out "$OUT" \
      > "logs/wm1_collect/job${i}.log" 2>&1 \
      || { echo "!!! job $i FAILED" >&2; exit 5; }
    tail -2 "logs/wm1_collect/job${i}.log"
  fi
  i=$((i + 1))
done
echo "=== CAL SHARD $SHARD DONE ==="
