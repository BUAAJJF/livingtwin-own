#!/usr/bin/env bash
# v6: the same teacher, distilled with a clear mask and then made blind by
# degrees.  One script, two variants, launched twice against different GPUs.
#
#   VARIANT=plain GPU=0 nohup setsid bash scripts/run_d455_v6.sh \
#     >results/d455_heavy_dr/d455_v6_ramped/watcher.log 2>&1 &
#   VARIANT=wrist GPU=2 nohup setsid bash scripts/run_d455_v6.sh \
#     >results/d455_heavy_dr/d455_v6_wrist/watcher.log 2>&1 &
#
# Why this exists rather than another copy of continue_d455_v5.sh:
#
#   * There is no teacher to wait for.  The state teacher never looks at an
#     image, so neither the dropout schedule nor a second camera changes its
#     problem, and both variants load the SAME v5 checkpoint.  That is not an
#     economy -- it is what makes the two branches comparable, because the
#     only difference between them is the one being tested.
#   * Fine-tuning is two stages, not one.  Measured 2026-08-31: distilling
#     under the full measured dropout took the student's behaviour loss from
#     0.226 to 0.541 and its placements from 2.06 to 0.31, and 2400 iterations
#     of PPO on top of it recovered nothing (0.35, flat from iteration 200).
#     DAgger regresses the student onto a teacher that can see what the
#     student cannot; on a blind frame the label is not a function of the
#     student's observation, the loss has an irreducible floor, and its
#     minimiser is the conditional mean -- a policy that reached for an object
#     0.46 times an episode against 5.94.  PPO has no such defect: its critic
#     is privileged and its objective is return, which is the asymmetric
#     actor-critic setting partial observability actually calls for.  So the
#     blindness is ramped in AFTER the imitation, across two PPO stages.
#   * The GPU is named, not claimed.  Two instances of this script run at once
#     and "the first free card" would hand both of them the same one.
#
# Stages write a marker when they finish, so a re-run resumes rather than
# repeating hours of work.
set -Eeuo pipefail

ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
VARIANT=${VARIANT:?set VARIANT=plain or VARIANT=wrist}
GPU=${GPU:?set GPU to the card index this variant owns}

case "$VARIANT" in
  plain)
    TAG=${TAG:-d455_v6_ramped}
    T_DISTILL=Mjlab-Pick-Place-PiperX-Distill-Robust-Clear
    T_HALF=Mjlab-Pick-Place-PiperX-Vision-Robust-Half
    T_FULL=Mjlab-Pick-Place-PiperX-Vision-Robust
    T_EVAL_NOMINAL=Mjlab-Pick-Place-PiperX-Vision
    T_EVAL_STRESS=Mjlab-Pick-Place-PiperX-Vision-Robust
    ;;
  wrist)
    TAG=${TAG:-d455_v6_wrist}
    T_DISTILL=Mjlab-Pick-Place-PiperX-Distill-Robust-Wrist
    T_HALF=Mjlab-Pick-Place-PiperX-Vision-Robust-Wrist-Half
    T_FULL=Mjlab-Pick-Place-PiperX-Vision-Robust-Wrist
    T_EVAL_NOMINAL=Mjlab-Pick-Place-PiperX-Vision-Wrist
    T_EVAL_STRESS=Mjlab-Pick-Place-PiperX-Vision-Robust-Wrist
    ;;
  *) echo "VARIANT must be plain or wrist, got '$VARIANT'" >&2; exit 2 ;;
esac

OUT=${OUT:-results/d455_heavy_dr/$TAG}

# The v5 state teacher: 8997 iterations, 5.78 objects placed, mean reward 19.50.
# Better than v4's (5.00 / 13.50), and it already carries the slower command
# derate and the removed transport-progress term.
TEACHER=${TEACHER:-logs/rsl_rl/piperx_pick_place_robust/2026-08-31_16-54-18_d455_v5_slow_blind_robust_teacher/model_8996.pt}

VISION_ENVS=${VISION_ENVS:-512}
DISTILL_ITERS=${DISTILL_ITERS:-3000}
# Absolute targets, not increments: finetune.py's --iterations is where the
# run STOPS, and the full stage resumes from the half stage's checkpoint.
HALF_ITERS=${HALF_ITERS:-1500}
FULL_ITERS=${FULL_ITERS:-3000}
EVAL_ENVS=${EVAL_ENVS:-512}
EVAL_STEPS=${EVAL_STEPS:-2400}
EVAL_SEED=${EVAL_SEED:-20260901}

cd "$ROOT"
mkdir -p "$OUT"

ENV_PREFIX=$("$MM" env list | awk -v e="$ENV_NAME" '$1==e {print $NF}')
[ -n "$ENV_PREFIX" ] || { echo "no micromamba env named $ENV_NAME" >&2; exit 2; }
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

say()  { printf '[%s] %s %s\n' "$(date -Is)" "$VARIANT" "$*"; }
fail() { printf '[%s] %s FAILED: %s\n' "$(date -Is)" "$VARIANT" "$*" >&2; exit 1; }
done_marker() { printf '%s\n' "$OUT/stage_$1.done"; }
stage_done()  { [ -f "$(done_marker "$1")" ]; }
mark_done()   { date -Is >"$(done_marker "$1")"; }

latest_checkpoint() {
  local run=$1 file
  file=$(find "$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
    | sort -V | tail -n 1)
  [ -n "$file" ] || fail "no model_*.pt under $run"
  printf '%s/%s\n' "$run" "$file"
}

# The directory the process actually logged to.  Taking the newest directory
# whose name matches is how a stale run becomes the next stage's input.
run_dir_from_log() {
  local log=$1 dir
  dir=$(grep -aE 'Logging experiment in directory: |\[INFO\] logging to ' "$log" \
        | tail -n 1 | sed -E 's/.*(directory: |logging to )//')
  [ -n "$dir" ] || fail "no run directory recorded in $log"
  dir=${dir#$ROOT/}
  [ -d "$dir" ] || fail "run directory recorded in $log does not exist: $dir"
  printf '%s\n' "$dir"
}

# A stage that stopped early is a different experiment.  rsl_rl prints
# "Learning iteration X/Y" with X zero-indexed against a one-indexed Y, so a
# clean run ends at X = Y-1; comparing against Y is off by one on every
# success, which is how this gate once stopped a run that had finished.
assert_ran_to_completion() {
  local log=$1 line reached target
  line=$(grep -a 'Learning iteration' "$log" | tail -1)
  reached=$(printf '%s' "$line" | sed -E 's#.*iteration ([0-9]+)/([0-9]+).*#\1#')
  target=$(printf '%s' "$line" | sed -E 's#.*iteration ([0-9]+)/([0-9]+).*#\2#')
  [ "$reached" -ge 0 ] 2>/dev/null && [ "$target" -ge 1 ] 2>/dev/null \
    || fail "cannot read the iteration counter out of $log"
  say "$(basename "$log") ends at iteration $reached of $target"
  [ "$((reached + 1))" -ge "$target" ] \
    || fail "$log stopped at $reached of $target -- killed or crashed"
}

run_stage() {
  local name=$1; shift
  if stage_done "$name"; then say "stage $name already done, skipping"; return 0; fi
  say "stage $name: $*"
  if ! "$@" >"$OUT/$name.log" 2>&1; then fail "stage $name -- see $OUT/$name.log"; fi
  mark_done "$name"
  say "stage $name complete"
}

# accept_s1.py returns `0 if verdict else 1`: its exit code is the PASS/FAIL
# verdict, NOT whether the evaluation ran.  A cell succeeded when it wrote a
# parseable metrics JSON; the verdict is a result to report, not an error.
eval_cell() {
  local task=$1 ckpt=$2 label=$3 gpu=$4
  GPU="$gpu" NUM_ENVS="$EVAL_ENVS" STEPS="$EVAL_STEPS" SEED="$EVAL_SEED" \
    OUT="$OUT" bash scripts/eval.sh "$task" "$ckpt" "$label" || true
  "$MM" run -n "$ENV_NAME" python -c "
import json, sys
d = json.load(open(sys.argv[1])); m = d['metrics']
for k in ('success', 'throughput_per_min', 'p95_s'): float(m[k])
print('%s  verdict %s  success %.4f  %.2f/min  p95 %.2f s'
      % (sys.argv[2], d.get('verdict', d.get('pass', '?')),
         m['success'], m['throughput_per_min'], m['p95_s']))
" "$OUT/$label.json" "$label"
}

# --- everything that can be wrong before six hours pass ---------------------

if [ -n "${SELFCHECK:-}" ]; then
  say "SELFCHECK: not launching anything"
  say "tag            $TAG"
  say "gpu            $GPU"
  say "env prefix     $ENV_PREFIX"
  say "teacher        $TEACHER"
  [ -f "$TEACHER" ] || fail "teacher checkpoint missing"
  for t in "$T_DISTILL" "$T_HALF" "$T_FULL" "$T_EVAL_NOMINAL" "$T_EVAL_STRESS"; do
    say "task           $t"
  done
  for f in scripts/distill.py scripts/finetune.py scripts/eval.sh \
           scripts/report_dr_degrade.py; do
    [ -f "$f" ] || fail "missing $f"
  done
  say "stage markers  $(ls "$OUT"/stage_*.done 2>/dev/null | paste -sd' ' || echo none)"
  say "SELFCHECK passed"
  exit 0
fi

[ -f "$TEACHER" ] || fail "teacher checkpoint missing: $TEACHER"
printf '%s\n' "$TEACHER" >"$OUT/teacher_checkpoint.txt"
say "starting on GPU $GPU from teacher $TEACHER"

# --- 1. distillation, with the mask clear -----------------------------------

run_stage distill \
  "$MM" run -n "$ENV_NAME" python -u scripts/distill.py \
    --task "$T_DISTILL" --teacher "$TEACHER" \
    --num-envs "$VISION_ENVS" --iterations "$DISTILL_ITERS" \
    --run-name "${TAG}_distill" --device "cuda:$GPU" --seed 42 \
    --cadence object --sensor measured --logger tensorboard

DISTILL_RUN=$(run_dir_from_log "$OUT/distill.log")
RD=$(latest_checkpoint "$DISTILL_RUN")
printf '%s\n' "$RD" >"$OUT/distill_checkpoint.txt"
say "distilled student $RD"

# --- 2. PPO at half the measured dropout ------------------------------------

run_stage finetune_half \
  "$MM" run -n "$ENV_NAME" python -u scripts/finetune.py \
    --task "$T_HALF" --student "$RD" --critic "$TEACHER" \
    --num-envs "$VISION_ENVS" --iterations "$HALF_ITERS" \
    --run-name "${TAG}_finetune_half" --device "cuda:$GPU" --seed 42 \
    --cadence object --logger tensorboard

HALF_RUN=$(run_dir_from_log "$OUT/finetune_half.log")
assert_ran_to_completion "$OUT/finetune_half.log"
RH=$(latest_checkpoint "$HALF_RUN")
printf '%s\n' "$RH" >"$OUT/finetune_half_checkpoint.txt"
say "half-dropout policy $RH"

# --- 3. PPO at the measured dropout -----------------------------------------

run_stage finetune_full \
  "$MM" run -n "$ENV_NAME" python -u scripts/finetune.py \
    --task "$T_FULL" --resume "$RH" --critic "$TEACHER" \
    --num-envs "$VISION_ENVS" --iterations "$FULL_ITERS" \
    --run-name "${TAG}_finetune_full" --device "cuda:$GPU" --seed 42 \
    --cadence object --logger tensorboard

FULL_RUN=$(run_dir_from_log "$OUT/finetune_full.log")
assert_ran_to_completion "$OUT/finetune_full.log"
RF=$(latest_checkpoint "$FULL_RUN")
printf '%s\n' "$RF" >"$OUT/final_checkpoint.txt"
say "final policy $RF"

# --- 4. evaluation ----------------------------------------------------------

if ! stage_done eval; then
  say "evaluations on GPU $GPU, one after the other -- the card is shared with
nothing else in this variant but the other variant owns its own"
  eval_cell "$T_EVAL_NOMINAL" "$RF" "${VARIANT}_on_nominal" "$GPU" \
    >"$OUT/${VARIANT}_on_nominal.log" 2>&1 || \
    fail "nominal evaluation failed -- see $OUT/${VARIANT}_on_nominal.log"
  eval_cell "$T_EVAL_STRESS" "$RF" "${VARIANT}_on_stress" "$GPU" \
    >"$OUT/${VARIANT}_on_stress.log" 2>&1 || \
    fail "stress evaluation failed -- see $OUT/${VARIANT}_on_stress.log"
  mark_done eval
fi

# The half-dropout checkpoint is evaluated too.  Without it a failure at the
# end cannot be told apart from a failure that was already there at half, and
# that distinction is the whole point of ramping.
if ! stage_done eval_half; then
  eval_cell "$T_EVAL_STRESS" "$RH" "${VARIANT}_half_on_stress" "$GPU" \
    >"$OUT/${VARIANT}_half_on_stress.log" 2>&1 || \
    fail "half-stage evaluation failed"
  mark_done eval_half
fi

cat >"$OUT/manifest.json" <<JSON
{
  "tag": "$TAG",
  "variant": "$VARIANT",
  "finished_utc": "$(date -u -Is)",
  "commit": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "teacher_checkpoint": "$TEACHER",
  "distill_checkpoint": "$RD",
  "half_checkpoint": "$RH",
  "final_checkpoint": "$RF",
  "dropout_schedule": {"distill": 0.0, "finetune_half": 0.5, "finetune_full": 1.0},
  "iterations": {"distill": $DISTILL_ITERS, "half": $HALF_ITERS, "full": $FULL_ITERS},
  "eval_protocol": {"num_envs": $EVAL_ENVS, "steps": $EVAL_STEPS, "seed": $EVAL_SEED}
}
JSON

say "complete -- final policy $RF"
