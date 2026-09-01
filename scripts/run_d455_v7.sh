#!/usr/bin/env bash
# v7 overnight: teacher and distillation are the deliverables; PPO is a bonus.
#
#   nohup setsid bash scripts/run_d455_v7.sh \
#     >results/d455_heavy_dr/d455_v7_visible/watcher.log 2>&1 &
#
# What is different from v6b, and why:
#
#   * The reward changed, so the teacher is retrained rather than reused.  The
#     five new terms -- premature_touch, jaws_ready, sight_arm, sight_hand,
#     wrist_side_on, table_touch -- are all state quantities, which is the
#     whole point: the student imitates the teacher's ACTIONS, so a habit the
#     teacher never formed is one the student cannot copy.
#
#   * It bootstraps from v5's teacher instead of starting cold.  That
#     checkpoint placed 5.78 objects and its value function is wrong for the
#     new rewards, but the curriculum ramps the new terms in from a third of
#     their final weight over 600 iterations, which is the same shape as
#     adaptation.  Cold would cost three hours this run does not have.
#
#   * It is DEADLINE AWARE.  The stages are ordered by what is worth having in
#     the morning: teacher, then distillation, then PPO only if the clock
#     allows, then the evaluations, which are cheap and always run.  A stage
#     that cannot finish before the deadline is skipped and said so in the log,
#     rather than started and killed halfway.
#
# Stage markers make a re-run resume rather than repeat.
set -Eeuo pipefail

ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
TAG=${TAG:-d455_v7_visible}
OUT=${OUT:-results/d455_heavy_dr/$TAG}
GPU=${GPU:?set GPU to the card this run owns}

# v5's robust teacher: 8997 iterations, 5.78 objects placed, mean reward 19.50.
BASE=${BASE:-logs/rsl_rl/piperx_pick_place_robust/2026-08-31_16-54-18_d455_v5_slow_blind_robust_teacher/model_8996.pt}

TEACHER_ENVS=${TEACHER_ENVS:-4096}
TEACHER_ITERS=${TEACHER_ITERS:-3000}
VISION_ENVS=${VISION_ENVS:-512}
DISTILL_ITERS=${DISTILL_ITERS:-3000}
FINETUNE_ITERS=${FINETUNE_ITERS:-3000}
EVAL_ENVS=${EVAL_ENVS:-512}
EVAL_STEPS=${EVAL_STEPS:-2400}
EVAL_SEED=${EVAL_SEED:-20260902}

# Hours from launch after which no new long stage may start.  The evaluations
# and the occlusion report are not long stages and run regardless.
DEADLINE_H=${DEADLINE_H:-10}
# What each stage has historically cost, used only to decide whether to start.
# Over-estimates on purpose: skipping a stage that would have fitted is a
# morning's lost work, starting one that will not is the same loss plus a
# corrupt-looking log.
FINETUNE_COST_H=${FINETUNE_COST_H:-3.5}

cd "$ROOT"
mkdir -p "$OUT"
START_EPOCH=$(date +%s)
DEADLINE_EPOCH=$(( START_EPOCH + $(printf '%.0f' "$(echo "$DEADLINE_H * 3600" | bc)") ))

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

hours_left() {
  python3 -c "print(max(0.0, ($DEADLINE_EPOCH - $(date +%s)) / 3600.0))"
}
fits() {
  python3 -c "import sys; sys.exit(0 if $(hours_left) >= $1 else 1)"
}

latest_checkpoint() {
  local run=$1 file
  file=$(find "$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
    | sort -V | tail -n 1)
  [ -n "$file" ] || fail "no model_*.pt under $run"
  printf '%s/%s\n' "$run" "$file"
}

run_dir_from_log() {
  local log=$1 dir
  dir=$(grep -aE 'Logging experiment in directory: |\[INFO\] logging to ' "$log" \
        | tail -n 1 | sed -E 's/.*(directory: |logging to )//')
  [ -n "$dir" ] || fail "no run directory recorded in $log"
  dir=${dir#$ROOT/}
  [ -d "$dir" ] || fail "run directory recorded in $log does not exist: $dir"
  printf '%s\n' "$dir"
}

# rsl_rl prints "Learning iteration X/Y" with X zero-indexed against a
# one-indexed Y, so a clean run ends at X = Y-1.  Comparing against Y is off by
# one on every success, which is how this gate once stopped a finished run.
assert_complete() {
  local log=$1 line reached target
  line=$(grep -a 'Learning iteration' "$log" | tail -1)
  reached=$(printf '%s' "$line" | sed -E 's#.*iteration ([0-9]+)/([0-9]+).*#\1#')
  target=$(printf '%s' "$line" | sed -E 's#.*iteration ([0-9]+)/([0-9]+).*#\2#')
  [ "$reached" -ge 0 ] 2>/dev/null && [ "$target" -ge 1 ] 2>/dev/null \
    || fail "cannot read the iteration counter out of $log"
  say "$(basename "$log") ended at iteration $reached of $target"
  [ "$((reached + 1))" -ge "$target" ] \
    || fail "$log stopped at $reached of $target -- killed or crashed"
}

run_stage() {
  local name=$1; shift
  if stage_done "$name"; then say "stage $name already done, skipping"; return 0; fi
  say "stage $name ($(hours_left) h left): $*"
  if ! "$@" >"$OUT/$name.log" 2>&1; then fail "stage $name -- see $OUT/$name.log"; fi
  mark_done "$name"
  say "stage $name complete"
}

# accept_s1.py returns `0 if verdict else 1`: its exit code is the PASS/FAIL
# verdict, NOT whether the evaluation ran.  A cell succeeded when it wrote a
# parseable JSON; the verdict is a result to report, not an error to stop on.
eval_cell() {
  local task=$1 ckpt=$2 label=$3
  GPU="$GPU" NUM_ENVS="$EVAL_ENVS" STEPS="$EVAL_STEPS" SEED="$EVAL_SEED" \
    OUT="$OUT" bash scripts/eval.sh "$task" "$ckpt" "$label" || true
  "$MM" run -n "$ENV_NAME" python -c "
import json, sys
d = json.load(open(sys.argv[1])); m = d['metrics']
print('%s  verdict %s  success %.4f  %.2f/min  p95 %.2f s'
      % (sys.argv[2], d.get('verdict', d.get('pass', '?')),
         m['success'], m['throughput_per_min'], m['p95_s']))
" "$OUT/$label.json" "$label" || say "could not parse $OUT/$label.json"
}

occlusion() {
  local ckpt=$1 task=$2 label=$3
  "$MM" run -n "$ENV_NAME" python -u scripts/eval_occlusion.py \
    --checkpoint "$ckpt" --task "$task" --num-envs 64 --steps 600 \
    --device "cuda:$GPU" --seed "$EVAL_SEED" \
    --out "$OUT/occlusion_$label.json" 2>&1 | tail -20
}

if [ -n "${SELFCHECK:-}" ]; then
  say "SELFCHECK: nothing is launched"
  say "tag $TAG   gpu $GPU   deadline in $DEADLINE_H h"
  say "base teacher $BASE"; [ -f "$BASE" ] || fail "base checkpoint missing"
  for f in scripts/train.sh scripts/distill.py scripts/finetune.py \
           scripts/eval.sh scripts/eval_occlusion.py; do
    [ -f "$f" ] || fail "missing $f"
  done
  say "markers: $(ls "$OUT"/stage_*.done 2>/dev/null | xargs -n1 basename 2>/dev/null | paste -sd' ' || echo none)"
  say "SELFCHECK passed"
  exit 0
fi

say "v7 starting on GPU $GPU, deadline in $DEADLINE_H h"
say "new this version: premature_touch, jaws_ready, sight_arm, sight_hand,"
say "wrist_side_on, table_touch; transport_progress restored to 40"

# --- 1. the state teacher, adapted from v5 ---------------------------------

if ! stage_done teacher; then
  [ -f "$BASE" ] || fail "base checkpoint missing: $BASE"
  start_name=$(basename "$BASE")
  boot="logs/rsl_rl/piperx_pick_place_robust/${TAG}_bootstrap"
  mkdir -p "$boot"
  [ -f "$boot/$start_name" ] || cp "$BASE" "$boot/$start_name"
  printf '%s\n' "$boot/$start_name" >"$OUT/bootstrap_checkpoint.txt"
  say "adapting the teacher from $start_name for $TEACHER_ITERS more iterations"
  if ! TASK=Mjlab-Pick-Place-PiperX-Robust NUM_ENVS="$TEACHER_ENVS" \
       ITERS="$TEACHER_ITERS" GPUS="[$GPU]" RUN_NAME="${TAG}_teacher" \
       bash scripts/train.sh \
         --agent.resume True \
         --agent.load-run "$(basename "$boot")" \
         --agent.load-checkpoint "$start_name" \
         --agent.logger tensorboard >"$OUT/teacher.log" 2>&1; then
    fail "teacher -- see $OUT/teacher.log"
  fi
  mark_done teacher
fi
TEACHER_RUN=$(run_dir_from_log "$OUT/teacher.log")
assert_complete "$OUT/teacher.log"
RT=$(latest_checkpoint "$TEACHER_RUN")
printf '%s\n' "$RT" >"$OUT/teacher_checkpoint.txt"
say "teacher $RT"

# The one number that says whether the new penalties broke it.  A teacher that
# stopped picking things up makes every later stage meaningless, and the log is
# long enough that this is worth pulling to the top.
say "teacher final: $(grep -a 'objects_placed' "$OUT/teacher.log" | tail -1 | sed 's/.*: //') objects placed"
occlusion "$RT" Mjlab-Pick-Place-PiperX-Robust teacher || say "occlusion (teacher) failed"

# --- 2. distillation -------------------------------------------------------

run_stage distill \
  "$MM" run -n "$ENV_NAME" python -u scripts/distill.py \
    --task Mjlab-Pick-Place-PiperX-Distill-Robust --teacher "$RT" \
    --num-envs "$VISION_ENVS" --iterations "$DISTILL_ITERS" \
    --run-name "${TAG}_distill" --device "cuda:$GPU" --seed 42 \
    --cadence object --sensor measured --logger tensorboard

DISTILL_RUN=$(run_dir_from_log "$OUT/distill.log")
RD=$(latest_checkpoint "$DISTILL_RUN")
printf '%s\n' "$RD" >"$OUT/distill_checkpoint.txt"
say "student $RD"
say "distill final: $(grep -a 'objects_placed' "$OUT/distill.log" | tail -1 | sed 's/.*: //') objects placed"

# --- 3. PPO, only if the clock allows --------------------------------------

FINAL=$RD
FINAL_TASK=Mjlab-Pick-Place-PiperX-Distill-Robust
if stage_done finetune; then
  say "fine-tuning already done"
elif fits "$FINETUNE_COST_H"; then
  run_stage finetune \
    "$MM" run -n "$ENV_NAME" python -u scripts/finetune.py \
      --task Mjlab-Pick-Place-PiperX-Vision-Robust --student "$RD" --critic "$RT" \
      --num-envs "$VISION_ENVS" --iterations "$FINETUNE_ITERS" \
      --run-name "${TAG}_finetune" --device "cuda:$GPU" --seed 42 \
      --cadence object --logger tensorboard
else
  say "SKIPPING fine-tuning: $(hours_left) h left, it needs about $FINETUNE_COST_H"
  say "the distilled student is the deliverable; re-run this script to continue"
fi

if stage_done finetune; then
  FINETUNE_RUN=$(run_dir_from_log "$OUT/finetune.log")
  assert_complete "$OUT/finetune.log"
  FINAL=$(latest_checkpoint "$FINETUNE_RUN")
  FINAL_TASK=Mjlab-Pick-Place-PiperX-Vision-Robust
fi
printf '%s\n' "$FINAL" >"$OUT/final_checkpoint.txt"
say "final policy $FINAL"

# --- 4. evaluation and the occlusion report, always ------------------------

if ! stage_done eval; then
  eval_cell Mjlab-Pick-Place-PiperX-Vision "$FINAL" v7_on_nominal \
    >"$OUT/v7_on_nominal.log" 2>&1 || say "nominal evaluation failed"
  eval_cell Mjlab-Pick-Place-PiperX-Vision-Robust "$FINAL" v7_on_stress \
    >"$OUT/v7_on_stress.log" 2>&1 || say "stress evaluation failed"
  mark_done eval
fi
occlusion "$FINAL" "$FINAL_TASK" final || say "occlusion (final) failed"

cat >"$OUT/manifest.json" <<JSON
{
  "tag": "$TAG",
  "finished_utc": "$(date -u -Is)",
  "commit": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "base_teacher": "$BASE",
  "teacher_checkpoint": "$RT",
  "distill_checkpoint": "$RD",
  "final_checkpoint": "$FINAL",
  "finetune_ran": $(stage_done finetune && echo true || echo false),
  "iterations": {"teacher": $TEACHER_ITERS, "distill": $DISTILL_ITERS,
                 "finetune": $FINETUNE_ITERS},
  "eval_protocol": {"num_envs": $EVAL_ENVS, "steps": $EVAL_STEPS, "seed": $EVAL_SEED}
}
JSON

say "v7 complete after $(python3 -c "print(round(($(date +%s)-$START_EPOCH)/3600, 2))") h"
say "final: $FINAL"
