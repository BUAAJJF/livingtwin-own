#!/usr/bin/env bash
# Run one shard of a posterior-guided adaptation plan, then evaluate it.
#
#   scripts/wm1_adapt.sh <cuda_device> <shard> <n_shards> <plan.json>
#
# Each job is 600 PPO iterations from f3/model_1500 under the adaptation
# distribution the plan carries, followed by six evaluations: the target domain
# and the nominal domain, three process repeats each, under the same
# 512 x 2400 protocol every other number in this repository was measured with.
#
# Nothing about a job differs from any other except p_adapt.  Same starting
# checkpoint, same iteration budget, same hyper-parameters, same evaluation
# seeds.  A method that got more training steps than another would not be a
# comparison.
set -Eeuo pipefail

GPU=${1:?usage: wm1_adapt.sh <cuda_device> <shard> <n_shards> <plan.json>}
SHARD=${2:?}; NSHARD=${3:?}; PLAN=${4:?}

cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-disable}
PREFIX="$(micromamba env list | awk '$1=="mjlab" {print $NF}')"
[ -n "$PREFIX" ] && export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

BASE=logs/rsl_rl/piperx_pick_place_vision/2026-08-22_17-15-09_f3/model_1500.pt
TASK=Mjlab-Pick-Place-PiperX-Vision
OUT=results/wm1_latency/adapt
mkdir -p "$OUT" logs/wm1_adapt

# The env's interpreter directly, not `micromamba run`: that wrapper merges
# stderr into stdout, and a single import warning captured into a variable that
# is then used as a loop bound is how Phase WM0 got a sweep that reported
# success over zero jobs.
PY="$PREFIX/bin/python"
[ -x "$PY" ] || { echo "!!! no interpreter at $PY" >&2; exit 2; }

N=$("$PY" -c "import json; print(len(json.load(open('$PLAN'))['jobs']))")
case "$N" in
  ''|*[!0-9]*) echo "!!! job count is not a number: '$N'" >&2; exit 3 ;;
  0) echo "!!! plan $PLAN has no jobs" >&2; exit 3 ;;
esac
echo "=== $PLAN: $N jobs, shard $SHARD of $NSHARD on cuda:$GPU"

for ((i = 0; i < N; i++)); do
  [ $((i % NSHARD)) -eq "$SHARD" ] || continue
  read -r TAG PROBS SEED ITERS METHODS < <("$PY" -c "
import json
j = json.load(open('$PLAN'))['jobs'][$i]
print(j['tag'], j['probs_arg'], j['seed'], j['iterations'],
      '+'.join(j['methods']))")
  [ -n "$TAG" ] && [ -n "$PROBS" ] && [ -n "$SEED" ] \
    || { echo "!!! job $i did not parse: tag='$TAG' probs='$PROBS'" >&2; exit 4; }
  echo "=== [gpu $GPU] job $i: $TAG  probs=$PROBS  seed=$SEED  ($METHODS)"

  if [ ! -e "$OUT/$TAG.ckpt" ]; then
    micromamba run -n mjlab python scripts/finetune.py \
      --task "$TASK" --resume "$BASE" \
      --num-envs 512 --iterations "$ITERS" \
      --latency-probs "$PROBS" --latency-seed "$SEED" --seed "$SEED" \
      --run-name "wm1_$TAG" --device "cuda:$GPU" --logger tensorboard \
      > "logs/wm1_adapt/$TAG.train.log" 2>&1 \
      || { echo "!!! $TAG training FAILED" >&2; exit 5; }
    ADAPTED=$(ls -1v logs/rsl_rl/piperx_pick_place_vision/*wm1_"$TAG"/model_*.pt \
              2>/dev/null | tail -1)
    [ -n "$ADAPTED" ] || { echo "!!! $TAG produced no checkpoint" >&2; exit 6; }
    echo "$ADAPTED" > "$OUT/$TAG.ckpt"
  fi
  ADAPTED=$(cat "$OUT/$TAG.ckpt")
  echo "=== adapted: $ADAPTED"

  r=0
  for SEED_E in 20260823 31415926 27182818; do
    # target: the hidden domain, 60 ms of observation delay
    micromamba run -n mjlab python scripts/accept_s1.py "$TASK" "$ADAPTED" \
      --num-envs 512 --steps 2400 --seed "$SEED_E" --device "cuda:$GPU" \
      --obs-latency-steps 3 \
      --label "${TAG}_target__r$r" --json "$OUT/${TAG}_target__r$r.json" \
      > "$OUT/${TAG}_target__r$r.log" 2>&1
    # retention: back in the domain the policy was deployed from
    micromamba run -n mjlab python scripts/accept_s1.py "$TASK" "$ADAPTED" \
      --num-envs 512 --steps 2400 --seed "$SEED_E" --device "cuda:$GPU" \
      --label "${TAG}_retention__r$r" --json "$OUT/${TAG}_retention__r$r.json" \
      > "$OUT/${TAG}_retention__r$r.log" 2>&1
    echo "[$TAG repeat $r done]"
    r=$((r + 1))
  done
done
echo "=== ADAPT SHARD $SHARD DONE ==="
