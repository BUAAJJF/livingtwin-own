#!/usr/bin/env bash
# v10c: a teacher trained from random weights under the cold-start curriculum,
# evaluated on the full -Robust domain with three seeds, and frozen as a
# baseline only if it meets the criteria declared in config_snapshot.json
# BEFORE training starts.
#
#   TAG=v10c_coldstart_sight   SIGHT=1 GPU=6 nohup setsid bash scripts/run_v10c.sh >results/d455_heavy_dr/v10c_coldstart_sight/watcher.log 2>&1 &
#   TAG=v10c_coldstart_nosight SIGHT=0 GPU=7 nohup setsid bash scripts/run_v10c.sh >results/d455_heavy_dr/v10c_coldstart_nosight/watcher.log 2>&1 &
#
# No resume, no warm start, no checkpoint is loaded before training: the only
# checkpoint this script ever loads is the one it produced, for evaluation.
# If the GPU is busy the script waits (and says so) rather than kill anything.
# The result directory is never overwritten: an existing one is an error.
set -Eeuo pipefail
ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
TAG=${TAG:?set TAG}
GPU=${GPU:?set GPU (physical index as nvidia-smi lists it)}
SIGHT=${SIGHT:?set SIGHT=1 or 0}
OUT=${OUT:-results/d455_heavy_dr/$TAG}
SEED=${SEED:-42}
TEACHER_ENVS=${TEACHER_ENVS:-8192}
TEACHER_ITERS=${TEACHER_ITERS:-9000}
EVAL_TASK=Mjlab-Pick-Place-PiperX-Robust
EVAL_ENVS=${EVAL_ENVS:-256}
EVAL_STEPS=${EVAL_STEPS:-2400}
EVAL_SEEDS=${EVAL_SEEDS:-"101 202 303"}
APPROACH=${APPROACH:-0}   # 1: the v10d variant with the approach terms (-Cold2 ids)
if [ "$APPROACH" = "1" ]; then BASE_TASK=Mjlab-Pick-Place-PiperX-Robust-Cold2; else BASE_TASK=Mjlab-Pick-Place-PiperX-Robust-Cold; fi
if [ "$SIGHT" = "1" ]; then TASK=$BASE_TASK; else TASK=${BASE_TASK}-NoSight; fi

cd "$ROOT"
ENV_PREFIX=$("$MM" env list | awk -v e="$ENV_NAME" '$1==e {print $NF}')
[ -n "$ENV_PREFIX" ] || { echo "no micromamba env named $ENV_NAME" >&2; exit 2; }
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say()  { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date -Is)" "$*" >&2; exit 1; }
marker() { printf '%s\n' "$OUT/stage_$1.done"; }
stage_done() { [ -f "$(marker "$1")" ]; }
mark_done()  { date -Is >"$(marker "$1")"; }
latest_checkpoint() {
  local run=$1 file
  file=$(find "$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -n 1)
  [ -n "$file" ] || fail "no model_*.pt under $run"
  printf '%s/%s\n' "$run" "$file"
}
run_dir_from_log() {
  local log=$1 dir
  dir=$(grep -aE 'Logging experiment in directory: |\[INFO\] logging to ' "$log" | tail -n 1 | sed -E 's/.*(directory: |logging to )//')
  [ -n "$dir" ] || fail "no run directory recorded in $log"
  dir=${dir#$ROOT/}
  [ -d "$dir" ] || fail "run directory recorded in $log does not exist: $dir"
  printf '%s\n' "$dir"
}
assert_complete() {
  local log=$1 line reached target
  line=$(grep -a 'Learning iteration' "$log" | tail -1)
  reached=$(printf '%s' "$line" | sed -E 's#.*iteration ([0-9]+)/([0-9]+).*#\1#')
  target=$(printf '%s' "$line" | sed -E 's#.*iteration ([0-9]+)/([0-9]+).*#\2#')
  [ "$reached" -ge 0 ] 2>/dev/null && [ "$target" -ge 1 ] 2>/dev/null || fail "cannot read the iteration counter out of $log"
  [ "$((reached + 1))" -ge "$target" ] || fail "$log stopped at $reached of $target -- killed or crashed"
}
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -lt 1024 ]; }

if [ -n "${SELFCHECK:-}" ]; then
  say "SELFCHECK: nothing is launched"
  say "tag $TAG  task $TASK  gpu $GPU  seed $SEED  envs $TEACHER_ENVS  iters $TEACHER_ITERS"
  say "gpu $GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i "$GPU")"
  # tests/test_cold_curriculum.py asserts this script carries no resume/warm-start flag
  "$MM" run -n "$ENV_NAME" python scripts/v10c_verdict.py --snapshot-only --task "$TASK" --sight "$SIGHT" --seed "$SEED" \
    --envs "$TEACHER_ENVS" --iters "$TEACHER_ITERS" --gpu "$GPU" --out /dev/stdout 2>&1 | grep -v "^\[" | head -60
  say "SELFCHECK passed"
  exit 0
fi

[ -e "$OUT" ] && [ -z "${ALLOW_RESUME_OF_THIS_RUN:-}" ] && fail "$OUT exists; a v10c run is never overwritten (set a new TAG)"
mkdir -p "$OUT"
export PIPER_U_TELEMETRY="$OUT/u_telemetry.jsonl" PIPER_U_TELEMETRY_TAG=teacher
export PIPER_CURRICULUM_LOG="$OUT/curriculum.jsonl"
say "v10c ($TAG) task $TASK on physical GPU $GPU, seed $SEED, budget $TEACHER_ITERS iterations"

# --- the declaration, before anything trains --------------------------------
TRAIN_CMD="TASK=$TASK NUM_ENVS=$TEACHER_ENVS ITERS=$TEACHER_ITERS GPUS=[$GPU] RUN_NAME=${TAG}_teacher bash scripts/train.sh --agent.logger tensorboard --agent.seed $SEED"
"$MM" run -n "$ENV_NAME" python scripts/v10c_verdict.py --snapshot-only --task "$TASK" --sight "$SIGHT" --seed "$SEED" \
  --envs "$TEACHER_ENVS" --iters "$TEACHER_ITERS" --gpu "$GPU" --command "$TRAIN_CMD" --out "$OUT/config_snapshot.json" \
  >"$OUT/config_snapshot.log" 2>&1 || fail "config snapshot -- see $OUT/config_snapshot.log"
say "wrote $OUT/config_snapshot.json"

# --- queue for the GPU, never kill ------------------------------------------
until gpu_free; do
  say "GPU $GPU is busy ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU") used); queued, checking again in 120 s"
  sleep 120
done

# --- 1. the teacher, from random weights --------------------------------------
if ! stage_done teacher; then
  say "teacher: $TRAIN_CMD"
  if ! TASK="$TASK" NUM_ENVS="$TEACHER_ENVS" ITERS="$TEACHER_ITERS" GPUS="[$GPU]" RUN_NAME="${TAG}_teacher" \
       bash scripts/train.sh --agent.logger tensorboard --agent.seed "$SEED" >"$OUT/teacher.log" 2>&1; then
    fail "teacher -- see $OUT/teacher.log"
  fi
  mark_done teacher
fi
TEACHER_RUN=$(run_dir_from_log "$OUT/teacher.log")
assert_complete "$OUT/teacher.log"
RT=$(latest_checkpoint "$TEACHER_RUN")   # declared: the FINAL checkpoint, never a picked one
printf '%s\n' "$RT" >"$OUT/teacher_checkpoint.txt"
say "teacher $RT"

# --- 2. fixed three-seed evaluation on the full -Robust domain ---------------
if ! stage_done eval; then
  for s in $EVAL_SEEDS; do
    "$MM" run -n "$ENV_NAME" python -u scripts/accept_s1.py "$EVAL_TASK" "$RT" \
      --num-envs "$EVAL_ENVS" --steps "$EVAL_STEPS" --seed "$s" --device "cuda:$GPU" --sensor measured \
      --label "teacher_robust_s$s" --json "$OUT/accept_teacher_robust_s$s.json" >"$OUT/accept_teacher_robust_s$s.log" 2>&1 \
      || say "accept seed $s failed (see log)"
    "$MM" run -n "$ENV_NAME" python -u scripts/eval_endurance.py --checkpoint "$RT" --task "$EVAL_TASK" \
      --num-envs 128 --steps 1200 --seed "$s" --device "cuda:$GPU" --sensor measured \
      --out "$OUT/endurance_teacher_s$s.json" >"$OUT/endurance_teacher_s$s.log" 2>&1 \
      || say "endurance seed $s failed (see log)"
  done
  "$MM" run -n "$ENV_NAME" python -u scripts/eval_occlusion.py --checkpoint "$RT" --task "$EVAL_TASK" \
    --num-envs 128 --steps 300 --seed 101 --device "cuda:$GPU" --sensor measured \
    --out "$OUT/occlusion_teacher.json" >"$OUT/occlusion_teacher.log" 2>&1 || say "occlusion failed (see log)"
  mark_done eval
fi

# --- 3. the verdict against the declaration, and the freeze ----------------------
"$MM" run -n "$ENV_NAME" python scripts/v10c_verdict.py --out-dir "$OUT" --checkpoint "$RT" \
  >"$OUT/verdict.log" 2>&1 || true
cat "$OUT/verdict.log" | tail -30
chmod -R a-w "$OUT" 2>/dev/null || true
say "v10c ($TAG) done; verdict in $OUT/verdict.json (directory is now read-only)"
