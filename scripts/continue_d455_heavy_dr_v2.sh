#!/usr/bin/env bash
# Continue the v2 campaign after the valid nominal teacher and the robust
# adaptation (seeded from nominal model_600) finish.
set -Eeuo pipefail

ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
ENV_PREFIX=${ENV_PREFIX:-/home/yunfan/micromamba/envs/mjlab}
NOMINAL_PID=${NOMINAL_PID:-299826}
ROBUST_PID=${ROBUST_PID:-318757}
TAG=d455_heavy_dr_v2
VISION_ENVS=${VISION_ENVS:-512}
DISTILL_ITERS=${DISTILL_ITERS:-3000}
FINETUNE_ITERS=${FINETUNE_ITERS:-3000}
EVAL_ENVS=${EVAL_ENVS:-512}
EVAL_STEPS=${EVAL_STEPS:-2400}
OUT=${OUT:-results/d455_heavy_dr/$TAG}

cd "$ROOT"
mkdir -p "$OUT"
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

latest_run() {
  local experiment=$1 name=$2
  find "logs/rsl_rl/$experiment" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${name}" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}
latest_checkpoint() {
  local run=$1 file
  file=$(find "$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
    | sort -V | tail -n 1)
  test -n "$file"; printf '%s/%s\n' "$run" "$file"
}
run_pair() {
  local p1=$1 p2=$2 s1 s2
  set +e; wait "$p1"; s1=$?; wait "$p2"; s2=$?; set -e
  if [ "$s1" -ne 0 ] || [ "$s2" -ne 0 ]; then
    echo "paired stage failed: statuses $s1 $s2" >&2; return 1
  fi
}
trap 'jobs -pr | xargs -r kill' EXIT

echo "[$(date -Is)] waiting for teacher PIDs $NOMINAL_PID and $ROBUST_PID"
while kill -0 "$NOMINAL_PID" 2>/dev/null || kill -0 "$ROBUST_PID" 2>/dev/null; do
  sleep 60
done

NTR=$(latest_run piperx_pick_place "${TAG}_nominal_teacher")
RTR=$(latest_run piperx_pick_place_robust "${TAG}_robust_from_nominal600")
NT=$(latest_checkpoint "$NTR"); RT=$(latest_checkpoint "$RTR")
printf '%s\n' "$NT" >"$OUT/nominal_teacher_checkpoint.txt"
printf '%s\n' "$RT" >"$OUT/robust_teacher_checkpoint.txt"
echo "[$(date -Is)] teachers ready: nominal=$NT robust=$RT"

"$MM" run -n "$ENV_NAME" python -u scripts/distill.py \
  --task Mjlab-Pick-Place-PiperX-Distill --teacher "$NT" \
  --num-envs "$VISION_ENVS" --iterations "$DISTILL_ITERS" \
  --run-name "${TAG}_nominal_distill" --device cuda:3 --seed 42 \
  --cadence object --sensor measured --logger tensorboard \
  >"$OUT/nominal_distill.log" 2>&1 & P1=$!
"$MM" run -n "$ENV_NAME" python -u scripts/distill.py \
  --task Mjlab-Pick-Place-PiperX-Distill-Robust --teacher "$RT" \
  --num-envs "$VISION_ENVS" --iterations "$DISTILL_ITERS" \
  --run-name "${TAG}_robust_distill" --device cuda:4 --seed 42 \
  --cadence object --sensor measured --logger tensorboard \
  >"$OUT/robust_distill.log" 2>&1 & P2=$!
run_pair "$P1" "$P2"

NDR=$(latest_run piperx_pick_place_distill "${TAG}_nominal_distill")
RDR=$(latest_run piperx_pick_place_distill_robust "${TAG}_robust_distill")
ND=$(latest_checkpoint "$NDR"); RD=$(latest_checkpoint "$RDR")
printf '%s\n' "$ND" >"$OUT/nominal_distill_checkpoint.txt"
printf '%s\n' "$RD" >"$OUT/robust_distill_checkpoint.txt"

"$MM" run -n "$ENV_NAME" python -u scripts/finetune.py \
  --task Mjlab-Pick-Place-PiperX-Vision --student "$ND" --critic "$NT" \
  --num-envs "$VISION_ENVS" --iterations "$FINETUNE_ITERS" \
  --run-name "${TAG}_nominal_finetune" --device cuda:5 --seed 42 \
  --cadence object --logger tensorboard >"$OUT/nominal_finetune.log" 2>&1 & P1=$!
"$MM" run -n "$ENV_NAME" python -u scripts/finetune.py \
  --task Mjlab-Pick-Place-PiperX-Vision-Robust --student "$RD" --critic "$RT" \
  --num-envs "$VISION_ENVS" --iterations "$FINETUNE_ITERS" \
  --run-name "${TAG}_robust_finetune" --device cuda:6 --seed 42 \
  --cadence object --logger tensorboard >"$OUT/robust_finetune.log" 2>&1 & P2=$!
run_pair "$P1" "$P2"

NFR=$(latest_run piperx_pick_place_vision "${TAG}_nominal_finetune")
RFR=$(latest_run piperx_pick_place_vision_robust "${TAG}_robust_finetune")
NF=$(latest_checkpoint "$NFR"); RF=$(latest_checkpoint "$RFR")
printf '%s\n' "$NF" >"$OUT/nominal_final_checkpoint.txt"
printf '%s\n' "$RF" >"$OUT/robust_final_checkpoint.txt"

eval_one() {
  local task=$1 ckpt=$2 label=$3 gpu=$4
  GPU="$gpu" NUM_ENVS="$EVAL_ENVS" STEPS="$EVAL_STEPS" SEED=20260828 OUT="$OUT" \
    bash scripts/eval.sh "$task" "$ckpt" "$label" >"$OUT/${label}.log" 2>&1
}
eval_one Mjlab-Pick-Place-PiperX-Vision "$NF" nominal_on_nominal 3 & P1=$!
eval_one Mjlab-Pick-Place-PiperX-Vision-Robust "$NF" nominal_on_stress 4 & P2=$!
run_pair "$P1" "$P2"
eval_one Mjlab-Pick-Place-PiperX-Vision "$RF" robust_on_nominal 3 & P1=$!
eval_one Mjlab-Pick-Place-PiperX-Vision-Robust "$RF" robust_on_stress 4 & P2=$!
run_pair "$P1" "$P2"

"$MM" run -n "$ENV_NAME" python scripts/report_dr_degrade.py "$OUT" \
  >"$OUT/report.log" 2>&1
"$MM" run -n "$ENV_NAME" python -u scripts/audit_table_contact.py "$RF" \
  --task Mjlab-Pick-Place-PiperX-Vision-Robust --num-envs 256 --steps 600 \
  --seed 20260828 --device cuda:7 --out "$OUT/robust_contact.json" \
  >"$OUT/robust_contact.log" 2>&1
echo "[$(date -Is)] v2 continuation complete"
