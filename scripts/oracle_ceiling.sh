#!/usr/bin/env bash
# Measure the recoverability ceiling for one target domain.
#
#   scripts/oracle_ceiling.sh <axis> <value> <tag> [gpu] [iterations]
#
# Four quantities, and the point of the exercise is the ratio between them:
#
#   J_zero_shot   the deployed policy in the target domain, unadapted
#   J_oracle      the same policy fine-tuned in the target domain with the
#                 parameter KNOWN, then measured there
#   J_retention   that fine-tuned policy back in the nominal domain
#   J_nominal     the deployed policy in the nominal domain (the starting point)
#
# A learned calibration is later scored as
#
#   recovery = (J_adapted - J_zero_shot) / (J_oracle - J_zero_shot)
#
# so if J_oracle is not meaningfully above J_zero_shot there is no ceiling to
# aim at and the whole direction is answered in the negative for that domain.
#
# This is the ORACLE: it is given the answer.  It is not a method, it is the
# bound a method would be trying to reach.
set -uo pipefail

if [ $# -lt 3 ]; then sed -n '2,25p' "$0" >&2; exit 2; fi
AXIS=$1; VALUE=$2; TAG=$3; GPU=${4:-0}; ITERS=${5:-2100}

cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-disable}
PREFIX="$(micromamba env list | awk '$1=="mjlab" {print $NF}')"
[ -n "$PREFIX" ] && export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

BASE=logs/rsl_rl/piperx_pick_place_vision/2026-08-22_17-15-09_f3/model_1500.pt
TASK=Mjlab-Pick-Place-PiperX-Vision
OUT=results/sim2real_sweep/oracle
FLAG="--${AXIS//_/-} $VALUE"
mkdir -p "$OUT"

echo "=== oracle ceiling: $AXIS = $VALUE  (tag=$TAG, gpu=$GPU, to iter $ITERS)"

# --- 1. fine-tune with the target parameter known -------------------------
# --resume, not --student: f3/model_1500 is a PPO checkpoint and already
# carries a trained critic, so there is no random value function to destroy
# the actor with.  --iterations is a TARGET TOTAL, so 2100 from 1500 is 600
# more.  Hyper-parameters are finetune.py's fine-tuning defaults, which exist
# because the from-scratch ones destroyed a distilled policy in one step.
micromamba run -n mjlab python scripts/finetune.py \
  --task "$TASK" --resume "$BASE" \
  --num-envs 512 --iterations "$ITERS" \
  --run-name "oracle_$TAG" --device "cuda:$GPU" \
  --logger tensorboard $FLAG 2>&1 | tail -3

ADAPTED=$(ls -1v logs/rsl_rl/piperx_pick_place_vision/*oracle_"$TAG"/model_*.pt \
          2>/dev/null | tail -1)
if [ -z "$ADAPTED" ]; then
  echo "!!! no adapted checkpoint produced; not writing any result" >&2
  exit 4
fi
echo "=== adapted: $ADAPTED"

# --- 2. the four measurements, three repeats each --------------------------
i=0
for SEED in 20260823 31415926 27182818; do
  # zero-shot in the target domain
  micromamba run -n mjlab python scripts/accept_s1.py "$TASK" "$BASE" \
    --num-envs 512 --steps 2400 --seed "$SEED" --device "cuda:$GPU" \
    --label "${TAG}_zeroshot__r$i" --json "$OUT/${TAG}_zeroshot__r$i.json" \
    $FLAG > "$OUT/${TAG}_zeroshot__r$i.log" 2>&1
  # oracle-adapted, in the target domain
  micromamba run -n mjlab python scripts/accept_s1.py "$TASK" "$ADAPTED" \
    --num-envs 512 --steps 2400 --seed "$SEED" --device "cuda:$GPU" \
    --label "${TAG}_oracle__r$i" --json "$OUT/${TAG}_oracle__r$i.json" \
    $FLAG > "$OUT/${TAG}_oracle__r$i.log" 2>&1
  # oracle-adapted, back in the nominal domain: did adapting cost anything?
  micromamba run -n mjlab python scripts/accept_s1.py "$TASK" "$ADAPTED" \
    --num-envs 512 --steps 2400 --seed "$SEED" --device "cuda:$GPU" \
    --label "${TAG}_retention__r$i" --json "$OUT/${TAG}_retention__r$i.json" \
    > "$OUT/${TAG}_retention__r$i.log" 2>&1
  echo "[repeat $i done]"
  i=$((i + 1))
done

echo "=== ORACLE DONE ($TAG) ==="
