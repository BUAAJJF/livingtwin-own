#!/usr/bin/env bash
# Build the whole WM1-A dataset.  One shard per GPU.
#
#   scripts/wm1_collect.sh <cuda_device> <shard> <n_shards> [out_dir]
#
# Jobs are listed once, below, and shard k takes every n_shards-th one, so
# running the same command with k = 0..n_shards-1 covers the list exactly once
# and a shard that dies can be re-run on its own without re-doing the others.
# The device is separate from the shard index because the free GPUs on this
# box are not 0..n-1.
#
# The splits, and why each exists:
#
#   train  128 envs x 1500 steps, two generation seeds, TRAIN SHAPES.
#          What the dynamics model and the history classifier fit.
#   val     64 envs x 1500 steps, a third seed, TRAIN SHAPES.
#          Where every hyper-parameter, score weight and temperature is
#          chosen.  Simulation only: it is not the target domain and carries
#          no reward.
#   valh    64 envs x 1500 steps, a fourth seed, HELD-OUT SHAPES.
#          The same job on objects the model never saw, so that a weight
#          chosen on `val` can be checked for shape-specific overfitting
#          before anything touches the target.
#   test   32 envs x 15000 steps, HELD-OUT SHAPES, one run per candidate lag.
#          The evaluation sessions.  15000 control steps is 300 s of one arm,
#          so the 10/30/60/180/300 s budgets are nested prefixes of the same
#          session rather than five differently-sampled datasets.  Lag 3 is
#          the hidden target; the other four are the benign domains a method
#          must not mistake for it.
#
# Every split is a different generation seed AND a different environment
# population, so no window in one is a near-copy of a window in another.
set -Eeuo pipefail

GPU=${1:?usage: wm1_collect.sh <cuda_device> <shard> <n_shards> [out]}
SHARD=${2:?usage: wm1_collect.sh <cuda_device> <shard> <n_shards> [out]}
NSHARD=${3:?usage: wm1_collect.sh <cuda_device> <shard> <n_shards> [out]}
OUT=${4:-results/wm1_latency/data}

cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-disable}
PREFIX="$(micromamba env list | awk '$1=="mjlab" {print $NF}')"
[ -n "$PREFIX" ] && export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

JOBS=()
for LAG in 0 1 2 3 4; do
  for SEED in 101 202; do
    JOBS+=("--lag $LAG --seed $((SEED + LAG)) --num-envs 128 --steps 1500 --split train  --shapes train")
  done
  JOBS+=("--lag $LAG --seed $((303 + LAG)) --num-envs 64 --steps 1500 --split val   --shapes train")
  JOBS+=("--lag $LAG --seed $((404 + LAG)) --num-envs 64 --steps 1500 --split valh  --shapes holdout")
  JOBS+=("--lag $LAG --seed $((909 + LAG)) --num-envs 32 --steps 15000 --split test   --shapes holdout")
done

mkdir -p "$OUT" logs/wm1_collect
i=0
for J in "${JOBS[@]}"; do
  if [ $((i % NSHARD)) -eq "$SHARD" ]; then
    echo "=== [gpu $GPU shard $SHARD] job $i: $J"
    micromamba run -n mjlab python scripts/wm_dataset.py $J \
      --device "cuda:$GPU" --out "$OUT" \
      > "logs/wm1_collect/job${i}.log" 2>&1 \
      || { echo "!!! job $i FAILED, see logs/wm1_collect/job${i}.log" >&2; exit 5; }
    tail -2 "logs/wm1_collect/job${i}.log"
  fi
  i=$((i + 1))
done
echo "=== COLLECT SHARD $SHARD DONE ($i jobs listed) ==="
