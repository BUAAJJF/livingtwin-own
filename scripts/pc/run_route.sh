#!/usr/bin/env bash
# yf/pc phase 3: one vision route end to end on one GPU.
#
#   TAG=pc_P1A_20260905T19 ROUTE=P1A GPU=5 TEACHER=logs/.../model_7400.pt COMMIT=<sha> \
#     nohup setsid bash scripts/pc/run_route.sh >results/pc/routes/$TAG/watcher.log 2>&1 &
#
# Stages, each behind a marker so a re-run resumes: smoke -> distill -> distill_eval
# -> finetune -> eval -> initiation -> viewer -> export.  The result directory
# must not already exist.  Every stage's wall time goes to timing.jsonl, and
# the budget in environment steps and optimizer updates is in manifest.json.
# No legacy loading anywhere; the teacher is checked against the task's action
# spec by every loader.  Nothing here edits a config after launch: every knob is
# an environment variable recorded in manifest.json before the first stage.
set -Eeuo pipefail
ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
TAG=${TAG:?set TAG}
ROUTE=${ROUTE:?set ROUTE (P0 P1A P1B P2)}
GPU=${GPU:?set GPU}
TEACHER=${TEACHER:?set TEACHER checkpoint}
COMMIT=${COMMIT:-unknown}
SEED=${SEED:-42}
EVAL_SEEDS=${EVAL_SEEDS:-"101 202 303"}
VISION_ENVS=${VISION_ENVS:-512}
DISTILL_ITERS=${DISTILL_ITERS:-1500}
FINETUNE_ITERS=${FINETUNE_ITERS:-1000}
EPISODE_S=${EPISODE_S:-36.0}
EVAL_ENVS=${EVAL_ENVS:-256}
OUT=${OUT:-results/pc/routes/$TAG}
DISTILL_TASK=Mjlab-Pick-Place-PiperX-PC-$ROUTE-Distill
VISION_TASK=Mjlab-Pick-Place-PiperX-PC-$ROUTE-Vision
HELDOUT_TASK=Mjlab-Pick-Place-PiperX-PC-$ROUTE-Vision-Heldout
LONG_STEPS=${LONG_STEPS:-9000}
LONG_SEEDS=${LONG_SEEDS:-"101 202 303"}
VIEWER_STEPS=${VIEWER_STEPS:-3000}
VIEWER_SEEDS=${VIEWER_SEEDS:-"101 202"}
cd "$ROOT"
[ -e "$OUT" ] && { echo "refusing to reuse $OUT" >&2; exit 2; }
mkdir -p "$OUT"
ORACLE=$("$MM" run -n "$ENV_NAME" python -c "from piper_push.pc import routes; print('true' if routes.is_oracle('$ROUTE') else 'false')" 2>/dev/null || echo unknown)
[ "$ORACLE" = true ] || [ "$ORACLE" = false ] || { echo "route $ROUTE unknown to piper_push.pc.routes" >&2; exit 2; }
ENV_PREFIX=$("$MM" env list | awk -v e="$ENV_NAME" '$1==e {print $NF}')
[ -n "$ENV_PREFIX" ] || { echo "no micromamba env named $ENV_NAME" >&2; exit 2; }
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PIPER_U_TELEMETRY="$OUT/u_telemetry.jsonl"
unset PIPER_ALLOW_LEGACY_ACTION_API RESET_FULL_RANGE
say()  { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date -Is)" "$*" >&2; date -Is >"$OUT/FAILED"; exit 1; }
marker() { printf '%s\n' "$OUT/stage_$1.done"; }
stage_done() { [ -f "$(marker "$1")" ]; }
STAGE_T0=0
stage_begin() { STAGE_T0=$(date +%s); }
mark_done()  {
  local now; now=$(date +%s)
  date -Is >"$(marker "$1")"
  printf '{"stage": "%s", "started": %d, "finished": %d, "seconds": %d, "gpu": %d}\n' "$1" "$STAGE_T0" "$now" "$((now - STAGE_T0))" "$GPU" >>"$OUT/timing.jsonl"
}
PY="$MM run -n $ENV_NAME python -u"
latest_checkpoint() {
  local run=$1 file
  file=$(find "$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -n 1)
  [ -n "$file" ] || fail "no model_*.pt under $run"
  printf '%s/%s\n' "$run" "$file"
}
run_dir_from_log() {
  local log=$1 dir
  dir=$(grep -aE '\[INFO\] logging to ' "$log" | tail -n 1 | sed -E 's/.*logging to //')
  [ -n "$dir" ] || fail "no run directory recorded in $log"
  dir=${dir#$ROOT/}
  [ -d "$dir" ] || fail "run directory recorded in $log does not exist: $dir"
  printf '%s\n' "$dir"
}
[ -f "$TEACHER" ] || fail "no teacher at $TEACHER"
TSHA=$(sha256sum "$TEACHER" | cut -d' ' -f1)
# The budget, written down as what it is: distillation rolls 32 steps per
# environment per iteration and takes one optimizer step every gradient_length
# (16; 8 for P0) of them; PPO rolls 32 steps per environment per iteration and
# takes num_learning_epochs (5) x num_mini_batches (4) optimizer steps.
STEPS_PER_ITER=32
GRAD_LEN=16; [ "$ROUTE" = P0 ] && GRAD_LEN=8
DISTILL_ENV_STEPS=$((VISION_ENVS * DISTILL_ITERS * STEPS_PER_ITER))
FINETUNE_ENV_STEPS=$((VISION_ENVS * FINETUNE_ITERS * STEPS_PER_ITER))
DISTILL_UPDATES=$((DISTILL_ITERS * STEPS_PER_ITER / GRAD_LEN))
FINETUNE_UPDATES=$((FINETUNE_ITERS * 5 * 4))
cat >"$OUT/manifest.json" <<JSON
{"tag": "$TAG", "route": "$ROUTE", "oracle_only": $ORACLE, "gpu": $GPU, "teacher": "$TEACHER", "teacher_sha256": "$TSHA",
 "code_commit_local": "$COMMIT", "remote_git_head": "$(git rev-parse HEAD)", "seed": $SEED,
 "vision_envs": $VISION_ENVS, "distill_iters": $DISTILL_ITERS, "finetune_iters": $FINETUNE_ITERS,
 "episode_s": $EPISODE_S, "eval_envs": $EVAL_ENVS, "eval_seeds": "$EVAL_SEEDS",
 "long_steps": $LONG_STEPS, "long_seeds": "$LONG_SEEDS", "viewer_steps": $VIEWER_STEPS, "viewer_seeds": "$VIEWER_SEEDS",
 "budget": {"steps_per_env_per_iter": $STEPS_PER_ITER, "distill_env_steps": $DISTILL_ENV_STEPS, "finetune_env_steps": $FINETUNE_ENV_STEPS,
            "distill_gradient_length": $GRAD_LEN, "distill_optimizer_updates": $DISTILL_UPDATES,
            "finetune_optimizer_updates": $FINETUNE_UPDATES, "finetune_epochs_x_minibatches": "5x4"},
 "distill_task": "$DISTILL_TASK", "vision_task": "$VISION_TASK", "heldout_task": "$HELDOUT_TASK",
 "gpu_name": "$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" 2>/dev/null | head -n 1)",
 "started": "$(date -Is)", "host": "$(hostname)"}
JSON
say "route $ROUTE  tag $TAG  gpu $GPU  oracle_only $ORACLE  teacher $TEACHER  sha256 $TSHA  commit $COMMIT"

# -- smoke -------------------------------------------------------------------
if ! stage_done smoke; then
  stage_begin
  say "smoke: 8 envs, 2 distillation iterations"
  $PY scripts/pc/smoke.py --route "$ROUTE" --teacher "$TEACHER" --num-envs 8 --steps 40 \
    --device "cuda:$GPU" --out "$OUT/smoke.json" >"$OUT/smoke.log" 2>&1 || fail "smoke (see $OUT/smoke.log)"
  mark_done smoke
fi

# -- distill -----------------------------------------------------------------
if ! stage_done distill; then
  stage_begin
  say "distill: $VISION_ENVS envs x $DISTILL_ITERS iterations, episode ${EPISODE_S}s"
  PIPER_U_TELEMETRY_TAG=distill $PY scripts/distill.py --task "$DISTILL_TASK" --teacher "$TEACHER" \
    --num-envs "$VISION_ENVS" --iterations "$DISTILL_ITERS" --episode-length-s "$EPISODE_S" \
    --run-name "${TAG}_distill" --device "cuda:$GPU" --seed "$SEED" --logger tensorboard \
    >"$OUT/distill.log" 2>&1 || fail "distill (see $OUT/distill.log)"
  run_dir_from_log "$OUT/distill.log" >"$OUT/distill_run.txt"
  latest_checkpoint "$(cat "$OUT/distill_run.txt")" >"$OUT/distill_checkpoint.txt"
  mark_done distill
fi
DS=$(cat "$OUT/distill_checkpoint.txt")
say "distilled student: $DS"

# -- distill_eval: one seed, the student on its own task ---------------------
if ! stage_done distill_eval; then
  stage_begin
  s=101
  $PY scripts/accept_s1.py "$VISION_TASK" "$DS" --num-envs "$EVAL_ENVS" --steps 2400 --seed $s \
    --device "cuda:$GPU" --sensor measured --label "${TAG}_student" --json "$OUT/accept_student_s$s.json" \
    >"$OUT/accept_student_s$s.log" 2>&1 || say "student accept exited non-zero"
  $PY scripts/eval_endurance.py --checkpoint "$DS" --task "$VISION_TASK" --num-envs "$EVAL_ENVS" --steps 1200 \
    --seed $s --device "cuda:$GPU" --sensor measured --out "$OUT/endurance_student_s$s.json" \
    >"$OUT/endurance_student_s$s.log" 2>&1 || say "student endurance failed"
  mark_done distill_eval
fi

# -- finetune ----------------------------------------------------------------
if ! stage_done finetune; then
  stage_begin
  say "finetune: $VISION_ENVS envs x $FINETUNE_ITERS iterations from $DS, critic from the teacher"
  PIPER_U_TELEMETRY_TAG=finetune $PY scripts/finetune.py --task "$VISION_TASK" --student "$DS" --critic "$TEACHER" \
    --num-envs "$VISION_ENVS" --iterations "$FINETUNE_ITERS" --episode-length-s "$EPISODE_S" \
    --run-name "${TAG}_finetune" --device "cuda:$GPU" --seed "$SEED" --logger tensorboard \
    >"$OUT/finetune.log" 2>&1 || fail "finetune (see $OUT/finetune.log)"
  run_dir_from_log "$OUT/finetune.log" >"$OUT/finetune_run.txt"
  latest_checkpoint "$(cat "$OUT/finetune_run.txt")" >"$OUT/final_checkpoint.txt"
  mark_done finetune
fi
FT=$(cat "$OUT/final_checkpoint.txt")
say "final: $FT"

# -- eval: three seeds, held-out objects, actions, occlusion -----------------
if ! stage_done eval; then
  stage_begin
  for s in $EVAL_SEEDS; do
    say "final accept seed $s"
    $PY scripts/accept_s1.py "$VISION_TASK" "$FT" --num-envs "$EVAL_ENVS" --steps 2400 --seed "$s" \
      --device "cuda:$GPU" --sensor measured --label "${TAG}_final" --json "$OUT/accept_final_s$s.json" \
      >"$OUT/accept_final_s$s.log" 2>&1 || say "final accept seed $s exited non-zero"
    say "final endurance seed $s"
    $PY scripts/eval_endurance.py --checkpoint "$FT" --task "$VISION_TASK" --num-envs "$EVAL_ENVS" --steps 1200 \
      --seed "$s" --device "cuda:$GPU" --sensor measured --out "$OUT/endurance_final_s$s.json" \
      >"$OUT/endurance_final_s$s.log" 2>&1 || say "final endurance seed $s failed"
  done
  say "held-out objects, seed 101"
  $PY scripts/accept_s1.py "$HELDOUT_TASK" "$FT" --num-envs "$EVAL_ENVS" --steps 2400 --seed 101 \
    --device "cuda:$GPU" --sensor measured --label "${TAG}_heldout" --json "$OUT/accept_heldout_s101.json" \
    >"$OUT/accept_heldout_s101.log" 2>&1 || say "held-out accept exited non-zero"
  $PY scripts/pc/eval_actions.py --checkpoint "$FT" --task "$VISION_TASK" --num-envs "$EVAL_ENVS" --steps 600 \
    --seed 101 --device "cuda:$GPU" --sensor measured --out "$OUT/actions_final_s101.json" \
    >"$OUT/actions_final_s101.log" 2>&1 || say "actions failed"
  $PY scripts/eval_occlusion.py --checkpoint "$FT" --task "$VISION_TASK" --num-envs "$EVAL_ENVS" --steps 300 \
    --seed 101 --device "cuda:$GPU" --sensor measured --out "$OUT/occlusion_final_s101.json" \
    >"$OUT/occlusion_final_s101.log" 2>&1 || say "occlusion failed"
  mark_done eval
fi

# -- initiation: why it stops starting; three seeds at the training horizon,
# then the long no-reset run past it (time-out off, the table refilled by the task)
if ! stage_done initiation; then
  stage_begin
  for s in $EVAL_SEEDS; do
    say "initiation seed $s (36 s)"
    $PY scripts/pc/eval_initiation.py --checkpoint "$FT" --task "$VISION_TASK" --num-envs "$EVAL_ENVS" --steps 1800 \
      --seed "$s" --device "cuda:$GPU" --sensor measured --out "$OUT/initiation_final_s$s.json" \
      >"$OUT/initiation_final_s$s.log" 2>&1 || say "initiation seed $s failed"
  done
  for s in $LONG_SEEDS; do
    say "long no-reset run seed $s ($LONG_STEPS steps)"
    $PY scripts/pc/eval_initiation.py --checkpoint "$FT" --task "$VISION_TASK" --num-envs "$EVAL_ENVS" --steps "$LONG_STEPS" \
      --episode-length-s 1000000 --seed "$s" --device "cuda:$GPU" --sensor measured --out "$OUT/long_final_s$s.json" \
      >"$OUT/long_final_s$s.log" 2>&1 || say "long run seed $s failed"
  done
  say "initiation, distilled student, seed 101"
  $PY scripts/pc/eval_initiation.py --checkpoint "$DS" --task "$VISION_TASK" --num-envs "$EVAL_ENVS" --steps 1800 \
    --seed 101 --device "cuda:$GPU" --sensor measured --out "$OUT/initiation_student_s101.json" \
    >"$OUT/initiation_student_s101.log" 2>&1 || say "student initiation failed"
  mark_done initiation
fi

# -- viewer: frame-by-frame pages (untracked; the README names the frames to look at)
if ! stage_done viewer; then
  stage_begin
  for s in $VIEWER_SEEDS; do
    say "viewer seed $s ($VIEWER_STEPS steps)"
    $PY scripts/pc/viewer.py --checkpoint "$FT" --task "$VISION_TASK" --steps "$VIEWER_STEPS" --seed "$s" \
      --device "cuda:$GPU" --sensor measured --out "$OUT/viewer_final_s$s.html" \
      >"$OUT/viewer_final_s$s.log" 2>&1 || say "viewer seed $s failed"
  done
  mark_done viewer
fi

# -- export ------------------------------------------------------------------
if ! stage_done export; then
  stage_begin
  say "export"
  $PY scripts/check_export.py "$FT" --task "$VISION_TASK" --out "$OUT/export" --device "cuda:$GPU" \
    >"$OUT/export.log" 2>&1 || say "export failed (see $OUT/export.log)"
  if [ "$ORACLE" = true ]; then
    # The graph check above is a consistency check only; nothing from here may
    # become a bundle.  scripts/pc/bundle.py refuses the manifest as well.
    printf 'oracle-only route %s: this export is a graph-consistency check, never a deployment candidate\n' "$ROUTE" >"$OUT/export/ORACLE_ONLY"
  fi
  mark_done export
fi
say "route $ROUTE complete"
date -Is >"$OUT/all.done"
