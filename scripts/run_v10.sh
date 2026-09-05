#!/usr/bin/env bash
# v10 overnight: the first teacher -> student -> PPO pipeline under the bounded
# action convention (tanh head, a = +-1 is the safe clip; 2026-09-05).
#
#   TAG=v10b_sight   SIGHT=1 GPU=4 nohup setsid bash scripts/run_v10.sh >results/d455_heavy_dr/v10b_sight/watcher.log 2>&1 &
#   TAG=v10b_nosight SIGHT=0 GPU=5 nohup setsid bash scripts/run_v10.sh >results/d455_heavy_dr/v10b_nosight/watcher.log 2>&1 &
#
# Two arms, one knob between them: SIGHT=1 is the v4 lineage (the current
# -Robust task) WITH the visibility reward family (sight_arm, sight_hand,
# wrist_side_on at their default weights and curriculum); SIGHT=0 is the same
# task with those three weights at zero, the control.  Nothing else differs.
#
# Every checkpoint before this date used the unbounded head, so there is no
# bootstrap: the teacher starts cold.  The vision stages run in the measured
# detector domain -- the rig's target-visibility process (session spread and
# gap lengths as measured, TARGET_GAP_SCALE 1.0), the fitted D455 sensor, the
# 43 ms latency mixture -- plus the deployment's geometric rebuild of the held
# object (HELD_PROXY_M=0.045), which every earlier student trained without and
# no robot can do without.
#
# Deadline aware like v7: a long stage that cannot finish before DEADLINE_H
# hours after launch is skipped and said so; the evaluations always run on
# whatever the last finished stage produced.  Stage markers make a re-run
# resume, so a watchdog loop can restart it after a crash.
set -Eeuo pipefail
ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
TAG=${TAG:?set TAG}
GPU=${GPU:?set GPU to the card this run owns}
SIGHT=${SIGHT:?set SIGHT=1 (visibility rewards) or SIGHT=0 (control)}
OUT=${OUT:-results/d455_heavy_dr/$TAG}
TEACHER_ENVS=${TEACHER_ENVS:-8192}
TEACHER_ITERS=${TEACHER_ITERS:-3000}
VISION_ENVS=${VISION_ENVS:-512}
DISTILL_ITERS=${DISTILL_ITERS:-2000}
FINETUNE_ITERS=${FINETUNE_ITERS:-2000}
EPISODE_S=${EPISODE_S:-36.0}
HELD_PROXY_M=${HELD_PROXY_M:-0.045}
EVAL_ENVS=${EVAL_ENVS:-256}
EVAL_STEPS=${EVAL_STEPS:-2400}
EVAL_SEEDS=${EVAL_SEEDS:-"101 202 303"}
DEADLINE_H=${DEADLINE_H:-7.0}
DISTILL_COST_H=${DISTILL_COST_H:-1.8}
FINETUNE_COST_H=${FINETUNE_COST_H:-2.6}

cd "$ROOT"
mkdir -p "$OUT"
START_EPOCH=$(date +%s)
DEADLINE_EPOCH=$(( START_EPOCH + $(python3 -c "print(int($DEADLINE_H * 3600))") ))
ENV_PREFIX=$("$MM" env list | awk -v e="$ENV_NAME" '$1==e {print $NF}')
[ -n "$ENV_PREFIX" ] || { echo "no micromamba env named $ENV_NAME" >&2; exit 2; }
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# The one knob.  Unset = the task's defaults (arm -2, hand -4, wrist 0.8,
# ramped); the control zeroes all three.
if [ "$SIGHT" = "0" ]; then
  export SIGHT_ARM_W=0 SIGHT_HAND_W=0 WRIST_W=0
fi
# The detector domain for the vision stages (distill, finetune, evals).
VISION_ENV=(env HELD_PROXY_M="$HELD_PROXY_M")
# Per-dimension telemetry of the pre-squash action, one JSON line per window,
# tagged by stage (piper_push.squashed.UTelemetry).
export PIPER_U_TELEMETRY="$OUT/u_telemetry.jsonl"

say()  { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date -Is)" "$*" >&2; exit 1; }
marker() { printf '%s\n' "$OUT/stage_$1.done"; }
stage_done() { [ -f "$(marker "$1")" ]; }
mark_done()  { date -Is >"$(marker "$1")"; }
hours_left() { python3 -c "print(max(0.0, ($DEADLINE_EPOCH - $(date +%s)) / 3600.0))"; }
fits() { python3 -c "import sys; sys.exit(0 if $(hours_left) >= $1 else 1)"; }
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
  say "$(basename "$log") ended at iteration $reached of $target"
  [ "$((reached + 1))" -ge "$target" ] || fail "$log stopped at $reached of $target -- killed or crashed"
}
run_stage() {
  local name=$1; shift
  if stage_done "$name"; then say "stage $name already done, skipping"; return 0; fi
  say "stage $name ($(hours_left) h left): $*"
  if ! "$@" >"$OUT/$name.log" 2>&1; then fail "stage $name -- see $OUT/$name.log"; fi
  mark_done "$name"
  say "stage $name complete"
}
write_domain() {
  local dir=$1
  {
    echo "# written by scripts/run_v10.sh $(date -Is); exported by accept_student.sh"
    echo "HELD_PROXY_M=$HELD_PROXY_M"
    [ "$SIGHT" = "0" ] && echo "SIGHT_ARM_W=0" && echo "SIGHT_HAND_W=0" && echo "WRIST_W=0"
    true
  } >"$dir/domain.env"
}
endurance() {  # ckpt task label seeds...
  local ckpt=$1 task=$2 label=$3; shift 3
  for s in "$@"; do
    "${VISION_ENV[@]}" "$MM" run -n "$ENV_NAME" python -u scripts/eval_endurance.py \
      --checkpoint "$ckpt" --task "$task" --num-envs 128 --steps 1200 --seed "$s" \
      --device "cuda:$GPU" --sensor measured --out "$OUT/endurance_${label}_s$s.json" \
      >"$OUT/endurance_${label}_s$s.log" 2>&1 || say "endurance $label seed $s failed"
    grep -aE "^placed/min|^early|^final jaw" "$OUT/endurance_${label}_s$s.log" | sed "s/^/  [$label s$s] /" || true
  done
}
occlusion() {
  local ckpt=$1 task=$2 label=$3
  "${VISION_ENV[@]}" "$MM" run -n "$ENV_NAME" python -u scripts/eval_occlusion.py \
    --checkpoint "$ckpt" --task "$task" --num-envs 128 --steps 300 --seed 101 \
    --device "cuda:$GPU" --sensor measured --out "$OUT/occlusion_$label.json" \
    >"$OUT/occlusion_$label.log" 2>&1 || say "occlusion $label failed"
}
accept() {  # ckpt task label
  local ckpt=$1 task=$2 label=$3
  for s in $EVAL_SEEDS; do
    "${VISION_ENV[@]}" "$MM" run -n "$ENV_NAME" python -u scripts/accept_s1.py "$task" "$ckpt" \
      --num-envs "$EVAL_ENVS" --steps "$EVAL_STEPS" --seed "$s" --device "cuda:$GPU" \
      --sensor measured --label "${label}_s$s" --json "$OUT/accept_${label}_s$s.json" \
      >"$OUT/accept_${label}_s$s.log" 2>&1 || say "accept $label seed $s failed"
  done
}

if [ -n "${SELFCHECK:-}" ]; then
  say "SELFCHECK: nothing is launched"
  say "tag $TAG  gpu $GPU  sight $SIGHT  deadline ${DEADLINE_H}h  out $OUT"
  say "teacher: cold, $TEACHER_ENVS envs x $TEACHER_ITERS it on Mjlab-Pick-Place-PiperX-Robust"
  say "vision: $VISION_ENVS envs, distill $DISTILL_ITERS, finetune $FINETUNE_ITERS, episode ${EPISODE_S}s, HELD_PROXY_M=$HELD_PROXY_M"
  say "sight env: arm ${SIGHT_ARM_W:-default(-2)}  hand ${SIGHT_HAND_W:-default(-4)}  wrist ${WRIST_W:-default(0.8)}"
  for f in scripts/train.sh scripts/distill.py scripts/finetune.py scripts/accept_s1.py \
           scripts/eval_endurance.py scripts/eval_occlusion.py src/piper_push/squashed.py; do
    [ -f "$f" ] || fail "missing $f"
  done
  "$MM" run -n "$ENV_NAME" python -c "
import mjlab.tasks, piper_push.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
c = load_env_cfg('Mjlab-Pick-Place-PiperX-Robust'); r = load_rl_cfg('Mjlab-Pick-Place-PiperX-Robust')
assert c.actions['arm'].bounded and 'Squashed' in r.actor.distribution_cfg['class_name']
rw = c.rewards
print('bounded convention OK; sight weights:', {k: rw[k].weight for k in rw if 'sight' in k or 'wrist' in k})
" 2>&1 | grep -v "^\[" | tail -2 || fail "task check"
  say "markers: $(ls "$OUT"/stage_*.done 2>/dev/null | xargs -n1 basename 2>/dev/null | paste -sd' ' || echo none)"
  say "SELFCHECK passed"
  exit 0
fi

say "v10 ($TAG) starting on GPU $GPU, sight=$SIGHT, deadline in ${DEADLINE_H}h"
say "commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
write_domain "$OUT"

# --- 1. teacher, cold ------------------------------------------------------
if ! stage_done teacher; then
  say "teacher: cold start, $TEACHER_ENVS envs x $TEACHER_ITERS iterations"
  if ! PIPER_U_TELEMETRY_TAG=teacher TASK=Mjlab-Pick-Place-PiperX-Robust NUM_ENVS="$TEACHER_ENVS" ITERS="$TEACHER_ITERS" \
       GPUS="[$GPU]" RUN_NAME="${TAG}_teacher" bash scripts/train.sh --agent.logger tensorboard \
       >"$OUT/teacher.log" 2>&1; then
    fail "teacher -- see $OUT/teacher.log"
  fi
  mark_done teacher
fi
TEACHER_RUN=$(run_dir_from_log "$OUT/teacher.log")
assert_complete "$OUT/teacher.log"
RT=$(latest_checkpoint "$TEACHER_RUN")
printf '%s\n' "$RT" >"$OUT/teacher_checkpoint.txt"
say "teacher $RT"
say "teacher final: $(grep -a 'objects_placed' "$OUT/teacher.log" | tail -1 | sed 's/.*: //') objects placed"
if ! stage_done teacher_eval; then
  endurance "$RT" Mjlab-Pick-Place-PiperX-Robust teacher 101
  occlusion "$RT" Mjlab-Pick-Place-PiperX-Robust teacher
  mark_done teacher_eval
fi

# --- 2. distillation into the measured detector domain --------------------
FINAL=""; FINAL_TASK=""
if stage_done distill; then
  say "distillation already done"
elif fits "$DISTILL_COST_H"; then
  run_stage distill "${VISION_ENV[@]}" PIPER_U_TELEMETRY_TAG=distill \
    "$MM" run -n "$ENV_NAME" python -u scripts/distill.py \
      --task Mjlab-Pick-Place-PiperX-Distill-Robust --teacher "$RT" \
      --num-envs "$VISION_ENVS" --iterations "$DISTILL_ITERS" --episode-length-s "$EPISODE_S" \
      --run-name "${TAG}_distill" --device "cuda:$GPU" --seed 42 \
      --cadence object --sensor measured --logger tensorboard
else
  say "SKIPPING distillation: $(hours_left) h left, it needs about $DISTILL_COST_H"
fi
if stage_done distill; then
  DISTILL_RUN=$(run_dir_from_log "$OUT/distill.log")
  assert_complete "$OUT/distill.log"
  RD=$(latest_checkpoint "$DISTILL_RUN")
  printf '%s\n' "$RD" >"$OUT/distill_checkpoint.txt"
  write_domain "$(dirname "$RD")"
  say "student $RD"
  FINAL=$RD; FINAL_TASK=Mjlab-Pick-Place-PiperX-Distill-Robust
  if ! stage_done distill_eval; then
    endurance "$RD" Mjlab-Pick-Place-PiperX-Distill-Robust student 101
    mark_done distill_eval
  fi
fi

# --- 3. PPO fine-tuning in the same domain, sight rewards as the teacher's --
if [ -n "$FINAL" ]; then
  if stage_done finetune; then
    say "fine-tuning already done"
  elif fits "$FINETUNE_COST_H"; then
    run_stage finetune "${VISION_ENV[@]}" PIPER_U_TELEMETRY_TAG=finetune \
      "$MM" run -n "$ENV_NAME" python -u scripts/finetune.py \
        --task Mjlab-Pick-Place-PiperX-Vision-Robust --student "$RD" --critic "$RT" \
        --num-envs "$VISION_ENVS" --iterations "$FINETUNE_ITERS" --episode-length-s "$EPISODE_S" \
        --run-name "${TAG}_finetune" --device "cuda:$GPU" --seed 42 \
        --cadence object --critic-warmup 100 --logger tensorboard
  else
    say "SKIPPING fine-tuning: $(hours_left) h left, it needs about $FINETUNE_COST_H"
  fi
  if stage_done finetune; then
    FINETUNE_RUN=$(run_dir_from_log "$OUT/finetune.log")
    assert_complete "$OUT/finetune.log"
    FINAL=$(latest_checkpoint "$FINETUNE_RUN")
    write_domain "$(dirname "$FINAL")"
    FINAL_TASK=Mjlab-Pick-Place-PiperX-Vision-Robust
  fi
fi

# --- 4. evaluations, always ------------------------------------------------
if [ -n "$FINAL" ]; then
  printf '%s\n' "$FINAL" >"$OUT/final_checkpoint.txt"
  say "final policy $FINAL ($FINAL_TASK)"
  if ! stage_done eval; then
    accept "$FINAL" Mjlab-Pick-Place-PiperX-Vision-Robust final_robust
    accept "$FINAL" Mjlab-Pick-Place-PiperX-Vision final_nominal
    endurance "$FINAL" "$FINAL_TASK" final $EVAL_SEEDS
    occlusion "$FINAL" "$FINAL_TASK" final
    mark_done eval
  fi
fi
cat >"$OUT/manifest.json" <<JSON
{
  "tag": "$TAG", "sight": $SIGHT, "finished_utc": "$(date -u -Is)",
  "commit": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "convention": "bounded", "teacher_checkpoint": "$RT",
  "distill_checkpoint": "$(cat "$OUT/distill_checkpoint.txt" 2>/dev/null || echo "")",
  "final_checkpoint": "$FINAL", "final_task": "$FINAL_TASK",
  "finetune_ran": $(stage_done finetune && echo true || echo false),
  "iterations": {"teacher": $TEACHER_ITERS, "distill": $DISTILL_ITERS, "finetune": $FINETUNE_ITERS},
  "domain": {"episode_s": $EPISODE_S, "held_proxy_m": $HELD_PROXY_M, "sensor": "measured",
             "sight": {"arm": "${SIGHT_ARM_W:-default}", "hand": "${SIGHT_HAND_W:-default}", "wrist": "${WRIST_W:-default}"}},
  "eval_protocol": {"num_envs": $EVAL_ENVS, "steps": $EVAL_STEPS, "seeds": "$EVAL_SEEDS"}
}
JSON
say "v10 ($TAG) complete after $(python3 -c "print(round(($(date +%s)-$START_EPOCH)/3600, 2))") h"
