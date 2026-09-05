#!/usr/bin/env bash
# yf/pc phase 3: one vision route end to end on one GPU.
#
#   TAG=pc_P1A_20260905T19 ROUTE=P1A GPU=5 TEACHER=logs/.../model_7400.pt COMMIT=<sha> \
#     nohup setsid bash scripts/pc/run_route.sh >results/pc/routes/$TAG/watcher.log 2>&1 &
#
# Stages, each behind a marker so a re-run resumes: smoke -> distill -> distill_eval
# -> finetune -> eval -> export.  The result directory must not already exist.
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
cd "$ROOT"
[ -e "$OUT" ] && { echo "refusing to reuse $OUT" >&2; exit 2; }
mkdir -p "$OUT"
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
mark_done()  { date -Is >"$(marker "$1")"; }
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
cat >"$OUT/manifest.json" <<JSON
{"tag": "$TAG", "route": "$ROUTE", "gpu": $GPU, "teacher": "$TEACHER", "teacher_sha256": "$TSHA",
 "code_commit_local": "$COMMIT", "remote_git_head": "$(git rev-parse HEAD)", "seed": $SEED,
 "vision_envs": $VISION_ENVS, "distill_iters": $DISTILL_ITERS, "finetune_iters": $FINETUNE_ITERS,
 "episode_s": $EPISODE_S, "eval_envs": $EVAL_ENVS, "eval_seeds": "$EVAL_SEEDS",
 "distill_task": "$DISTILL_TASK", "vision_task": "$VISION_TASK", "heldout_task": "$HELDOUT_TASK",
 "started": "$(date -Is)", "host": "$(hostname)"}
JSON
say "route $ROUTE  tag $TAG  gpu $GPU  teacher $TEACHER  sha256 $TSHA  commit $COMMIT"

# -- smoke -------------------------------------------------------------------
if ! stage_done smoke; then
  say "smoke: 8 envs, 2 distillation iterations"
  $PY scripts/pc/smoke.py --route "$ROUTE" --teacher "$TEACHER" --num-envs 8 --steps 40 \
    --device "cuda:$GPU" --out "$OUT/smoke.json" >"$OUT/smoke.log" 2>&1 || fail "smoke (see $OUT/smoke.log)"
  mark_done smoke
fi

# -- distill -----------------------------------------------------------------
if ! stage_done distill; then
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

# -- export ------------------------------------------------------------------
if ! stage_done export; then
  say "export"
  $PY scripts/check_export.py "$FT" --task "$VISION_TASK" --out "$OUT/export" --device "cuda:$GPU" \
    >"$OUT/export.log" 2>&1 || say "export failed (see $OUT/export.log)"
  mark_done export
fi
say "route $ROUTE complete"
date -Is >"$OUT/all.done"
