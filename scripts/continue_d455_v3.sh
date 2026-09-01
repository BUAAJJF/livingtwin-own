#!/usr/bin/env bash
# Robust-only v3 continuation: wait for the running teacher, then distil,
# fine-tune and evaluate without anyone sitting in front of it.
#
#   nohup bash scripts/continue_d455_v3.sh >results/.../watcher.log 2>&1 &
#
# Deliberately not scripts/continue_d455_heavy_dr_v2.sh with the names
# changed.  That one assumes a *pair* of teachers, two hardcoded PIDs, run
# names that no longer exist and fixed GPU indices, and it takes "the newest
# directory whose name matches" as proof that the stage it is waiting for is
# the stage that produced it.  Four things are different here:
#
#   * one teacher, not two.  The nominal control is the already-finished v2
#     vision policy, so its two evaluation cells are launched immediately and
#     run alongside the teacher instead of behind it.
#   * the teacher's run directory is read out of its own log, not guessed
#     from mtime.  A stale directory with the right suffix cannot be picked.
#   * the checkpoint is checked against the iteration the run was launched to
#     reach, and against the exit status of the process that wrote it.  A
#     teacher that died at 4200 does not silently become a distillation input.
#   * GPUs are chosen from what is free at the moment the stage starts.
#
# Stages write a marker when they finish, so re-running after a failure picks
# up where it stopped rather than repeating four hours of work.
set -Eeuo pipefail

ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
TAG=${TAG:-d455_heavy_dr_v3_no_contact}
OUT=${OUT:-results/d455_heavy_dr/$TAG}

# The iteration the teacher was launched to reach.  train_d455_no_contact_v3.sh
# resumes from 3499 and adds 2000; rsl_rl's resume semantics make that the
# *displayed* 5499, and getting this wrong is the failure that already
# happened once on this campaign.
EXPECT_ITER=${EXPECT_ITER:-5499}

# The v2 nominal vision policy, used as the unchanged control column of the
# DR-degradation matrix.  It is not retrained here.
NOMINAL_FINAL=${NOMINAL_FINAL:-logs/rsl_rl/piperx_pick_place_vision/2026-08-27_22-05-10_d455_heavy_dr_v2_nominal_finetune/model_2999.pt}

VISION_ENVS=${VISION_ENVS:-512}
DISTILL_ITERS=${DISTILL_ITERS:-3000}
FINETUNE_ITERS=${FINETUNE_ITERS:-3000}
EVAL_ENVS=${EVAL_ENVS:-512}
EVAL_STEPS=${EVAL_STEPS:-2400}
EVAL_SEED=${EVAL_SEED:-20260828}
POLL_S=${POLL_S:-60}

cd "$ROOT"
mkdir -p "$OUT"

ENV_PREFIX=$("$MM" env list | awk -v e="$ENV_NAME" '$1==e {print $NF}')
[ -n "$ENV_PREFIX" ] || { echo "no micromamba env named $ENV_NAME" >&2; exit 2; }
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

say() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date -Is)" "$*" >&2; exit 1; }
done_marker() { printf '%s\n' "$OUT/stage_$1.done"; }
stage_done() { [ -f "$(done_marker "$1")" ]; }
mark_done() { date -Is >"$(done_marker "$1")"; }

# Whatever is idle *now*.  Reserving indices up front is what made the v2
# script unable to run on a box whose free set had changed.
free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '$2 < 1024 {print $1}'
}
claim_gpus() {
  local want=$1 got
  got=$(free_gpus | head -n "$want" | paste -sd' ')
  [ "$(printf '%s\n' $got | grep -c .)" -eq "$want" ] \
    || fail "wanted $want free GPUs, got '${got:-none}'"
  printf '%s\n' "$got"
}

latest_checkpoint() {
  local run=$1 file
  file=$(find "$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
    | sort -V | tail -n 1)
  [ -n "$file" ] || fail "no model_*.pt under $run"
  printf '%s/%s\n' "$run" "$file"
}
checkpoint_iter() {
  local name; name=$(basename "$1"); name=${name#model_}; printf '%s\n' "${name%.pt}"
}

# The run directory the process actually logged to, not the newest one whose
# name happens to match.  mjlab's own train.py and this repo's distill.py /
# finetune.py word the same fact differently, so both spellings are read.
run_dir_from_log() {
  local log=$1 dir
  dir=$(grep -aE 'Logging experiment in directory: |\[INFO\] logging to ' "$log" \
        | tail -n 1 | sed -E 's/.*(directory: |logging to )//')
  [ -n "$dir" ] || fail "no run directory recorded in $log"
  dir=${dir#$ROOT/}
  [ -d "$dir" ] || fail "run directory recorded in $log does not exist: $dir"
  printf '%s\n' "$dir"
}

run_stage() {
  local name=$1; shift
  if stage_done "$name"; then say "stage $name already done, skipping"; return 0; fi
  say "stage $name: $*"
  if ! "$@" >"$OUT/$name.log" 2>&1; then
    fail "stage $name -- see $OUT/$name.log"
  fi
  mark_done "$name"
  say "stage $name complete"
}

# accept_s1.py returns `0 if verdict else 1`: its exit code is the S1 PASS/FAIL
# verdict, NOT whether the evaluation ran.  Treating nonzero as a stage failure
# aborted this watcher after the whole 4-hour fine-tune had already finished,
# because the *control* policy -- a known-mediocre v2 checkpoint kept precisely
# so there is something to compare against -- did not pass the gate.  A cell
# succeeded when it wrote a parseable metrics JSON; the verdict is a result to
# report, not an error to stop on.
eval_cell() {
  local task=$1 ckpt=$2 label=$3 gpu=$4
  GPU="$gpu" NUM_ENVS="$EVAL_ENVS" STEPS="$EVAL_STEPS" SEED="$EVAL_SEED" \
    OUT="$OUT" bash scripts/eval.sh "$task" "$ckpt" "$label" || true
  "$MM" run -n "$ENV_NAME" python -c "
import json, sys
d = json.load(open(sys.argv[1]))
m = d['metrics']
for k in ('success', 'throughput_per_min', 'p95_s'):
    float(m[k])
print('%s  verdict %s  success %.4f  %.2f/min  p95 %.2f s'
      % (sys.argv[2], d.get('verdict', d.get('pass', '?')),
         m['success'], m['throughput_per_min'], m['p95_s']))
" "$OUT/$label.json" "$label"
}

# --- a dry check of everything that can be wrong before four hours pass -----

if [ -n "${SELFCHECK:-}" ]; then
  say "SELFCHECK: not launching anything"
  say "env prefix     $ENV_PREFIX"
  say "teacher log    $OUT/robust_teacher.log"
  [ -f "$OUT/robust_teacher.log" ] || fail "teacher log missing"
  say "teacher run    $(run_dir_from_log "$OUT/robust_teacher.log")"
  if [ -f "$OUT/robust_teacher.pid" ]; then
    tp=$(cat "$OUT/robust_teacher.pid")
    if kill -0 "$tp" 2>/dev/null; then say "teacher pid    $tp (running)"
    else say "teacher pid    $tp (already exited)"; fi
  fi
  say "expected iter  $EXPECT_ITER"
  say "control ckpt   $NOMINAL_FINAL"
  [ -f "$NOMINAL_FINAL" ] || fail "nominal control checkpoint missing"
  say "free GPUs      $(free_gpus | paste -sd' ')"
  for t in Mjlab-Pick-Place-PiperX-Distill-Robust \
           Mjlab-Pick-Place-PiperX-Vision-Robust \
           Mjlab-Pick-Place-PiperX-Vision; do
    say "task           $t"
  done
  for f in scripts/distill.py scripts/finetune.py scripts/eval.sh \
           scripts/report_dr_degrade.py scripts/audit_table_contact.py; do
    [ -f "$f" ] || fail "missing $f"
  done
  say "stage markers  $(ls "$OUT"/stage_*.done 2>/dev/null | paste -sd' ' || echo none)"
  say "SELFCHECK passed"
  exit 0
fi

# --- the nominal control column, which does not wait for anything -----------

NOMINAL_PIDS=()
if stage_done nominal_control; then
  say "nominal control evaluations already done"
else
  [ -f "$NOMINAL_FINAL" ] || fail "nominal control checkpoint missing: $NOMINAL_FINAL"
  read -r g1 g2 <<<"$(claim_gpus 2)"
  say "nominal control: $NOMINAL_FINAL on GPUs $g1/$g2 (alongside the teacher)"
  ( eval_cell Mjlab-Pick-Place-PiperX-Vision "$NOMINAL_FINAL" \
      nominal_on_nominal "$g1" >"$OUT/nominal_on_nominal.log" 2>&1 ) & NOMINAL_PIDS+=($!)
  ( eval_cell Mjlab-Pick-Place-PiperX-Vision-Robust "$NOMINAL_FINAL" \
      nominal_on_stress "$g2" >"$OUT/nominal_on_stress.log" 2>&1 ) & NOMINAL_PIDS+=($!)
fi

# --- wait for the teacher ---------------------------------------------------

TEACHER_LOG=$OUT/robust_teacher.log
[ -f "$TEACHER_LOG" ] || fail "no teacher log at $TEACHER_LOG"

if [ -f "$OUT/robust_teacher.pid" ]; then
  TPID=$(cat "$OUT/robust_teacher.pid")
  say "waiting for robust teacher pid $TPID"
  while kill -0 "$TPID" 2>/dev/null; do sleep "$POLL_S"; done
  say "teacher pid $TPID has exited"
else
  say "no pid file; assuming the teacher has already finished"
fi

TEACHER_RUN=$(run_dir_from_log "$TEACHER_LOG")
[ -d "$TEACHER_RUN" ] || fail "teacher run directory does not exist: $TEACHER_RUN"
RT=$(latest_checkpoint "$TEACHER_RUN")
RT_ITER=$(checkpoint_iter "$RT")
say "teacher run $TEACHER_RUN, newest checkpoint $RT (iteration $RT_ITER)"

# A teacher that stopped early is a different experiment, and its checkpoint is
# not a distillation input just because it is the newest file present.  What
# counts as "finished" comes from the run's own log rather than from a number
# asserted here: rsl_rl prints "Learning iteration X/Y" with X zero-indexed
# against a one-indexed Y, so a run that completes ends at X = Y-1 and writes
# model_(Y-1).pt.  Comparing the checkpoint against Y is therefore off by one
# on *every* successful run, which is how this gate first stopped a teacher
# that had finished cleanly.
last_line=$(grep -a 'Learning iteration' "$TEACHER_LOG" | tail -1)
reached=$(printf '%s' "$last_line" | sed -E 's#.*iteration ([0-9]+)/([0-9]+).*#\1#')
target=$(printf '%s' "$last_line" | sed -E 's#.*iteration ([0-9]+)/([0-9]+).*#\2#')
if ! [ "$reached" -ge 0 ] 2>/dev/null || ! [ "$target" -ge 1 ] 2>/dev/null; then
  fail "cannot read the iteration counter out of $TEACHER_LOG"
fi
say "teacher log ends at iteration $reached of $target"
if [ "$((reached + 1))" -lt "$target" ]; then
  fail "the teacher stopped at iteration $reached of $target -- it was killed \
or it crashed.  Read $TEACHER_LOG.  Continuing from a short teacher is a \
deliberate decision, not a default."
fi
if [ "$RT_ITER" -lt "$reached" ]; then
  fail "the teacher reached iteration $reached but the newest checkpoint is \
$RT (iteration $RT_ITER).  The final save did not happen; do not distil from \
a stale one."
fi
if [ -n "${EXPECT_ITER:-}" ] && [ "$((reached + 1))" -lt "$EXPECT_ITER" ]; then
  fail "the teacher was expected to reach $EXPECT_ITER and its log ends at \
$((reached + 1))"
fi
printf '%s\n' "$RT" >"$OUT/robust_teacher_checkpoint.txt"

# --- distillation -----------------------------------------------------------

if ! stage_done robust_distill; then
  read -r g1 <<<"$(claim_gpus 1)"
  say "robust distillation on GPU $g1"
  run_stage robust_distill \
    "$MM" run -n "$ENV_NAME" python -u scripts/distill.py \
      --task Mjlab-Pick-Place-PiperX-Distill-Robust --teacher "$RT" \
      --num-envs "$VISION_ENVS" --iterations "$DISTILL_ITERS" \
      --run-name "${TAG}_robust_distill" --device "cuda:$g1" --seed 42 \
      --cadence object --sensor measured --logger tensorboard
fi

DISTILL_RUN=$(run_dir_from_log "$OUT/robust_distill.log")
RD=$(latest_checkpoint "$DISTILL_RUN")
printf '%s\n' "$RD" >"$OUT/robust_distill_checkpoint.txt"
say "distilled student $RD"

# --- vision PPO fine-tuning -------------------------------------------------

if ! stage_done robust_finetune; then
  read -r g1 <<<"$(claim_gpus 1)"
  say "robust fine-tuning on GPU $g1"
  run_stage robust_finetune \
    "$MM" run -n "$ENV_NAME" python -u scripts/finetune.py \
      --task Mjlab-Pick-Place-PiperX-Vision-Robust --student "$RD" --critic "$RT" \
      --num-envs "$VISION_ENVS" --iterations "$FINETUNE_ITERS" \
      --run-name "${TAG}_robust_finetune" --device "cuda:$g1" --seed 42 \
      --cadence object --logger tensorboard
fi

FINETUNE_RUN=$(run_dir_from_log "$OUT/robust_finetune.log")
RF=$(latest_checkpoint "$FINETUNE_RUN")
printf '%s\n' "$RF" >"$OUT/robust_final_checkpoint.txt"
say "final robust vision policy $RF"

# --- evaluation -------------------------------------------------------------

if [ "${#NOMINAL_PIDS[@]}" -gt 0 ]; then
  say "collecting the nominal control evaluations"
  ok=1
  for p in "${NOMINAL_PIDS[@]}"; do wait "$p" || ok=0; done
  [ "$ok" -eq 1 ] || fail "a nominal control evaluation failed -- see \
$OUT/nominal_on_nominal.log and $OUT/nominal_on_stress.log"
  mark_done nominal_control
fi

if ! stage_done robust_eval; then
  read -r g1 g2 <<<"$(claim_gpus 2)"
  say "robust evaluations on GPUs $g1/$g2"
  ( eval_cell Mjlab-Pick-Place-PiperX-Vision "$RF" \
      robust_on_nominal "$g1" >"$OUT/robust_on_nominal.log" 2>&1 ) & P1=$!
  ( eval_cell Mjlab-Pick-Place-PiperX-Vision-Robust "$RF" \
      robust_on_stress "$g2" >"$OUT/robust_on_stress.log" 2>&1 ) & P2=$!
  ok=1; wait "$P1" || ok=0; wait "$P2" || ok=0
  [ "$ok" -eq 1 ] || fail "a robust evaluation failed -- see $OUT/robust_on_*.log"
  mark_done robust_eval
fi

for cell in nominal_on_nominal nominal_on_stress robust_on_nominal robust_on_stress; do
  [ -s "$OUT/$cell.json" ] || fail "evaluation matrix incomplete: $OUT/$cell.json"
done

run_stage dr_degrade "$MM" run -n "$ENV_NAME" python scripts/report_dr_degrade.py "$OUT"

if ! stage_done contact_audit; then
  read -r g1 <<<"$(claim_gpus 1)"
  run_stage contact_audit \
    "$MM" run -n "$ENV_NAME" python -u scripts/audit_table_contact.py "$RF" \
      --task Mjlab-Pick-Place-PiperX-Vision-Robust --num-envs 256 --steps 600 \
      --seed "$EVAL_SEED" --device "cuda:$g1" --out "$OUT/robust_contact.json"
fi

cat >"$OUT/v3_manifest.json" <<JSON
{
  "tag": "$TAG",
  "finished_utc": "$(date -u -Is)",
  "commit": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "teacher_checkpoint": "$RT",
  "teacher_iteration": $RT_ITER,
  "distill_checkpoint": "$RD",
  "final_checkpoint": "$RF",
  "nominal_control_checkpoint": "$NOMINAL_FINAL",
  "eval_protocol": {"num_envs": $EVAL_ENVS, "steps": $EVAL_STEPS, "seed": $EVAL_SEED}
}
JSON

say "v3 continuation complete"
say "final policy: $RF"
say "pull it with: scripts/pull_results.sh   (or scp the checkpoint directly)"
