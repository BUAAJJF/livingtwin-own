#!/usr/bin/env bash
# Build the WM1-B dataset: servo damping at 0.75, 1.0 and 1.5.
#
#   scripts/wm1b_collect.sh <cuda_device> <shard> <n_shards> [out_dir]
#
# Same splits and same discipline as Phase WM1-A -- train and val on shape
# classes 0-2, valh/calh/test on 3-4, one generation seed per split, budgets
# nested inside one 300 s session -- with two differences that the axis forces.
#
# **Larger training rollouts.**  The risk head is a rare-event classifier: at
# nominal damping the safety shell fires about twice an arm-hour, so a training
# split sized for a dynamics model would carry a few dozen positives.  256
# environments x 1500 steps per file gives 21 arm-hours per generation seed.
#
# **Three values, not five.**  0.75 is the target, 1.0 is nominal, and 1.5 is
# the counter-direction control the specification asks for, so that a method
# cannot score by always answering "less damped than nominal".
set -Eeuo pipefail

GPU=${1:?usage: wm1b_collect.sh <cuda_device> <shard> <n_shards> [out]}
SHARD=${2:?}; NSHARD=${3:?}; OUT=${4:-results/wm1_damping/data}

cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-disable}
PREFIX="$(micromamba env list | awk '$1=="mjlab" {print $NF}')"
[ -n "$PREFIX" ] && export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

AXIS=servo_damping_scale
JOBS=()
i=0
for V in 0.75 1.0 1.5; do
  for SEED in 101 202; do
    JOBS+=("--value $V --seed $((SEED + i)) --num-envs 256 --steps 1500 --split train  --shapes train")
  done
  JOBS+=("--value $V --seed $((303 + i)) --num-envs 128 --steps 1500 --split val   --shapes train")
  JOBS+=("--value $V --seed $((404 + i)) --num-envs 128 --steps 1500 --split valh  --shapes holdout")
  JOBS+=("--value $V --seed $((5050 + i)) --num-envs 16 --steps 15000 --split cal  --shapes train")
  JOBS+=("--value $V --seed $((6060 + i)) --num-envs 16 --steps 15000 --split calh --shapes holdout")
  JOBS+=("--value $V --seed $((909 + i)) --num-envs 32 --steps 15000 --split test  --shapes holdout")
  i=$((i + 1))
done

mkdir -p "$OUT" logs/wm1b_collect
j=0
for J in "${JOBS[@]}"; do
  if [ $((j % NSHARD)) -eq "$SHARD" ]; then
    echo "=== [gpu $GPU shard $SHARD] job $j: $J"
    micromamba run -n mjlab python scripts/wm_dataset.py --axis "$AXIS" $J \
      --device "cuda:$GPU" --out "$OUT" \
      > "logs/wm1b_collect/job${j}.log" 2>&1 \
      || { echo "!!! job $j FAILED, see logs/wm1b_collect/job${j}.log" >&2; exit 5; }
    tail -3 "logs/wm1b_collect/job${j}.log"
  fi
  j=$((j + 1))
done
echo "=== WM1-B COLLECT SHARD $SHARD DONE ($j jobs listed) ==="
