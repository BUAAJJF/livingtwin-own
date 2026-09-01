#!/usr/bin/env bash
# Retrain the calibrated D455 pick policy without permitting robot/table contact.
#
# Two independent branches are launched:
#   1. adapt the last good rotated D405 vision policy to D455 + table guard;
#   2. train a new safe state teacher, distil it through the measured D455
#      model, then PPO fine-tune the resulting vision policy.
#
# The second branch is the release candidate.  The first branch is an earlier
# diagnostic/backup and is never promoted without the same contact audit.
set -Eeuo pipefail

ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
ENV_PREFIX=${ENV_PREFIX:-/home/yunfan/micromamba/envs/mjlab}
TAG=${TAG:-d455_table_safe_v1}

TEACHER_GPU=${TEACHER_GPU:-0}
QUICK_GPU=${QUICK_GPU:-1}
DISTILL_GPU=${DISTILL_GPU:-2}
FINETUNE_GPU=${FINETUNE_GPU:-3}
AUDIT_GPU=${AUDIT_GPU:-4}

TEACHER_ENVS=${TEACHER_ENVS:-8192}
TEACHER_ITERS=${TEACHER_ITERS:-3500}
VISION_ENVS=${VISION_ENVS:-512}
DISTILL_ITERS=${DISTILL_ITERS:-3000}
FINETUNE_ITERS=${FINETUNE_ITERS:-3000}
QUICK_ITERS=${QUICK_ITERS:-3000}
RUN_QUICK=${RUN_QUICK:-1}

OLD_VISION=${OLD_VISION:-logs/rsl_rl/piperx_pick_place_vision/2026-08-25_17-02-31_calib_rot90_d405_finetune_v1/model_1400.pt}
OUT=${OUT:-results/d455_table_safe/${TAG}}

cd "$ROOT"
mkdir -p "$OUT"
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

latest_run() {
  local experiment=$1 name=$2
  find "logs/rsl_rl/$experiment" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${name}" -printf '%T@ %p\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

latest_checkpoint() {
  local run=$1 file
  file=$(find "$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
    | sort -V | tail -n 1)
  test -n "$file"
  printf '%s/%s\n' "$run" "$file"
}

audit_vision() {
  local checkpoint=$1 label=$2 device=$3
  "$MM" run -n "$ENV_NAME" python -u scripts/audit_table_contact.py \
    "$checkpoint" --num-envs 256 --steps 600 --seed 20260827 \
    --device "cuda:$device" --out "$OUT/${label}_contact.json"
}

quick_branch() {
  local name="${TAG}_quick"
  test -f "$OLD_VISION"
  "$MM" run -n "$ENV_NAME" python -u scripts/finetune.py \
    --resume "$OLD_VISION" --num-envs "$VISION_ENVS" \
    --iterations "$QUICK_ITERS" --run-name "$name" \
    --device "cuda:$QUICK_GPU" --seed 42 --cadence object \
    --logger tensorboard
  local run checkpoint
  run=$(latest_run piperx_pick_place_vision "$name")
  checkpoint=$(latest_checkpoint "$run")
  printf '%s\n' "$checkpoint" > "$OUT/quick_checkpoint.txt"
  audit_vision "$checkpoint" quick "$QUICK_GPU"
}

echo "[$(date -Is)] campaign=$TAG root=$ROOT"
QUICK_PID=""
if [ "$RUN_QUICK" = 1 ]; then
  echo "[$(date -Is)] starting quick adaptation on cuda:$QUICK_GPU"
  quick_branch > "$OUT/quick.log" 2>&1 &
  QUICK_PID=$!
  echo "$QUICK_PID" > "$OUT/quick.pid"
fi

TEACHER_NAME="${TAG}_teacher"
echo "[$(date -Is)] training safe state teacher on cuda:$TEACHER_GPU"
TASK=Mjlab-Pick-Place-PiperX NUM_ENVS="$TEACHER_ENVS" \
ITERS="$TEACHER_ITERS" GPUS="[$TEACHER_GPU]" RUN_NAME="$TEACHER_NAME" \
  bash scripts/train.sh --agent.logger tensorboard \
  > "$OUT/teacher.log" 2>&1

TEACHER_RUN=$(latest_run piperx_pick_place "$TEACHER_NAME")
TEACHER_CKPT=$(latest_checkpoint "$TEACHER_RUN")
printf '%s\n' "$TEACHER_CKPT" > "$OUT/teacher_checkpoint.txt"
echo "[$(date -Is)] teacher=$TEACHER_CKPT"

# Validate the teacher's task success before spending camera-rendering time.
GPU="$AUDIT_GPU" NUM_ENVS=512 STEPS=2400 SEED=20260827 \
OUT="$OUT" bash scripts/eval.sh Mjlab-Pick-Place-PiperX \
  "$TEACHER_CKPT" teacher > "$OUT/teacher_eval.log" 2>&1

DISTILL_NAME="${TAG}_distill"
echo "[$(date -Is)] distilling measured D455 student on cuda:$DISTILL_GPU"
"$MM" run -n "$ENV_NAME" python -u scripts/distill.py \
  --teacher "$TEACHER_CKPT" --num-envs "$VISION_ENVS" \
  --iterations "$DISTILL_ITERS" --run-name "$DISTILL_NAME" \
  --device "cuda:$DISTILL_GPU" --seed 42 --cadence object \
  --sensor measured --logger tensorboard \
  > "$OUT/distill.log" 2>&1

DISTILL_RUN=$(latest_run piperx_pick_place_distill "$DISTILL_NAME")
DISTILL_CKPT=$(latest_checkpoint "$DISTILL_RUN")
printf '%s\n' "$DISTILL_CKPT" > "$OUT/distill_checkpoint.txt"
echo "[$(date -Is)] student=$DISTILL_CKPT"

FINETUNE_NAME="${TAG}_finetune"
echo "[$(date -Is)] PPO fine-tuning D455 student on cuda:$FINETUNE_GPU"
"$MM" run -n "$ENV_NAME" python -u scripts/finetune.py \
  --student "$DISTILL_CKPT" --critic "$TEACHER_CKPT" \
  --num-envs "$VISION_ENVS" --iterations "$FINETUNE_ITERS" \
  --run-name "$FINETUNE_NAME" --device "cuda:$FINETUNE_GPU" \
  --seed 42 --cadence object --logger tensorboard \
  > "$OUT/finetune.log" 2>&1

FINETUNE_RUN=$(latest_run piperx_pick_place_vision "$FINETUNE_NAME")
FINETUNE_CKPT=$(latest_checkpoint "$FINETUNE_RUN")
printf '%s\n' "$FINETUNE_CKPT" > "$OUT/final_checkpoint.txt"
echo "[$(date -Is)] final=$FINETUNE_CKPT"

audit_vision "$FINETUNE_CKPT" final "$AUDIT_GPU" \
  > "$OUT/final_contact.log" 2>&1
GPU="$AUDIT_GPU" NUM_ENVS=512 STEPS=2400 SEED=20260827 \
OUT="$OUT" bash scripts/eval.sh Mjlab-Pick-Place-PiperX-Vision \
  "$FINETUNE_CKPT" final > "$OUT/final_eval.log" 2>&1

if [ -n "$QUICK_PID" ]; then
  if wait "$QUICK_PID"; then
    echo "[$(date -Is)] quick branch completed"
  else
    echo "[$(date -Is)] quick branch failed; full branch remains valid" >&2
  fi
fi
echo "[$(date -Is)] campaign complete"
