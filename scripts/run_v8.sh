#!/usr/bin/env bash
# v8: fix the teacher's endurance, then distil into the SAM domain.
#
#   nohup setsid bash scripts/run_v8.sh >results/d455_heavy_dr/v8_horizon/watcher.log 2>&1 &
#
# Two things are different from v7 and both come from a measurement made today.
#
#   * The teacher stops working part way through a long episode -- environments
#     die one at a time with the gripper commanded shut, and `strong_teacher`
#     falls from 18 placements a minute to 6.  It is not the randomisation
#     (worse without it), not a jam (the jaw tracks its command to 0.2 mm) and
#     not an exploit (total reward falls 2.52 -> 1.85).  Stage 1 continues the
#     same teacher, same reward, at `episode_length_s = 36`, and stage 2 does
#     not start unless the endurance gate passes.
#
#   * The distillation domain is set from what the DEPLOYED perception stack
#     can see, which is now measurable: SAM2.1 reports the target in 97.7% of
#     approach frames and 100% while held, against the depth segmenter's 47%
#     and 0%.  Training the student for a blindness the shipped stack does not
#     have is what the last four distillation runs did.
#
# Stage markers make a re-run resume rather than repeat.
set -Eeuo pipefail

ROOT=${ROOT:-/home/yunfan/Project/PiperPush/LivingTwin}
MM=${MM:-micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
TAG=${TAG:-v8_horizon}
OUT=${OUT:-results/d455_heavy_dr/$TAG}
GPU=${GPU:-0}

EPISODE_S=${EPISODE_S:-36.0}
VISION_ENVS=${VISION_ENVS:-512}
DISTILL_ITERS=${DISTILL_ITERS:-2500}
GATE_ENVS=${GATE_ENVS:-256}
GATE_STEPS=${GATE_STEPS:-1200}
# Below this the horizon fix did not work and the choice between more teacher
# training and falling back to v5_baseline is a decision, not a default.
GATE_HARD=${GATE_HARD:-0.75}
GATE_WANT=${GATE_WANT:-0.90}

# The SAM domain.  Set from seven seeds of scripts/sim_perception_check.py
# --sam, not chosen: approach 97.7% detected, held 100%, and the residual
# gaps are the watchdog's five uncertain frames rather than the segmenter's
# 15-110 step blackouts.
VIS_FLOOR=${VIS_FLOOR:-0.95}
VIS_CEIL=${VIS_CEIL:-1.0}
GAP_SCALE=${GAP_SCALE:-0.3}

cd "$ROOT"
mkdir -p "$OUT"
ENV_PREFIX=$($MM env list | awk -v e="$ENV_NAME" '$1==e {print $NF}')
[ -n "$ENV_PREFIX" ] || { echo "no micromamba env named $ENV_NAME" >&2; exit 2; }
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

say()  { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date -Is)" "$*" >&2; exit 1; }
marker() { printf '%s\n' "$OUT/stage_$1.done"; }
stage_done() { [ -f "$(marker "$1")" ]; }
mark_done()  { date -Is >"$(marker "$1")"; }

run_dir_from_log() {
  local log=$1 dir
  dir=$(grep -aE 'Logging experiment in directory: |\[INFO\] logging to ' "$log" \
        | tail -n 1 | sed -E 's/.*(directory: |logging to )//')
  [ -n "$dir" ] || fail "no run directory recorded in $log"
  dir=${dir#$ROOT/}
  [ -d "$dir" ] || fail "run directory in $log does not exist: $dir"
  printf '%s\n' "$dir"
}
latest_checkpoint() {
  local run=$1 file
  file=$(find "$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
    | sort -V | tail -n 1)
  [ -n "$file" ] || fail "no model_*.pt under $run"
  printf '%s/%s\n' "$run" "$file"
}

# --- 1. wait for the teacher continuation ----------------------------------
# It is launched separately and may already be running; this waits rather than
# starting a second one on the same GPU.
while pgrep -f "train Mjlab-Pick-Place-PiperX-Robust" >/dev/null; do sleep 60; done
[ -f "$OUT/teacher.log" ] || fail "no $OUT/teacher.log -- was the teacher launched?"
TEACHER_RUN=$(run_dir_from_log "$OUT/teacher.log")
RT=$(latest_checkpoint "$TEACHER_RUN")
printf '%s\n' "$RT" >"$OUT/teacher_checkpoint.txt"
say "teacher $RT"

# --- 2. the gate the whole run exists for ----------------------------------
say "endurance gate on $RT"
set +e
$MM run -n "$ENV_NAME" python -u scripts/eval_endurance.py \
  --checkpoint "$RT" --num-envs "$GATE_ENVS" --steps "$GATE_STEPS" \
  --device "cuda:$GPU" --gate "$GATE_HARD" \
  --out "$OUT/endurance_teacher.json" 2>&1 | tee "$OUT/endurance.log"
GATE_RC=${PIPESTATUS[0]}
set -e
RATIO=$($MM run -n "$ENV_NAME" python -c \
  "import json;print(f\"{json.load(open('$OUT/endurance_teacher.json'))['late_over_early']:.3f}\")")
say "late/early = $RATIO (hard gate $GATE_HARD, wanted $GATE_WANT)"
[ "$GATE_RC" -eq 0 ] || fail "endurance gate: $RATIO < $GATE_HARD -- the horizon \
fix did not work.  Decide between more teacher iterations and falling back to \
checkpoints/v7_teachers/v5_baseline.pt (late/early 1.08, but SAM sees 83% of \
approach frames under it against 98% under strong)."

# The occlusion this teacher exists for, conditioned on the hand being in the
# working volume so that "keeps out of the way" cannot be scored by giving up.
$MM run -n "$ENV_NAME" python -u scripts/eval_occlusion.py \
  --checkpoint "$RT" --num-envs 256 --steps 300 --device "cuda:$GPU" \
  --seed 101 --out "$OUT/occlusion_teacher.json" >"$OUT/occlusion.log" 2>&1 || true
$MM run -n "$ENV_NAME" python -c "
import json; d = json.load(open('$OUT/occlusion_teacher.json'))
e = d['engaged']; print('engaged blocked %.1f%%  visible %.3f  placed/min %.1f'
  % (100*e['blocked_rate'], e['visible_mean'], d['placed_per_min']))" || true

# --- 2b. what SAM costs the perception loop, on a free GPU ------------------
# This has to happen HERE and not while anything else is running: measured
# against a training job on the same card the loop read 438 ms/frame with the
# depth stack alone, against the ~35 ms the arm has actually recorded.  A
# perception rate measured under contention is not a perception rate.
if ! stage_done replay; then
  REC=${REC:-recordings/v4_stereo_try3}
  POL=${POL:-hardware/deploy/policies/d455_v4_final}
  if [ -d "$REC" ] && [ -d "$POL" ]; then
    for mode in depth sam21; do
      say "replay benchmark: --target-tracker $mode"
      timeout 300 $MM run -n "$ENV_NAME" python -m hardware.deploy.run \
        --replay "$REC" --no-arm --policy "$POL" \
        --target-lifecycle --held-target-radius 0.045 \
        --target-tracker "$mode" --seconds 20 \
        >"$OUT/replay_$mode.log" 2>&1 || true
      grep -aE "^perception: [0-9]|frames with target|^sam:" \
        "$OUT/replay_$mode.log" | sed "s/^/  $mode  /" || true
    done
    mark_done replay
  else
    say "no replay recording or policy on disk; skipping the rate benchmark"
  fi
fi

# --- 3. distillation, into the domain the deployed stack actually gives -----
if ! stage_done distill; then
  say "distilling $DISTILL_ITERS iterations into visible=[$VIS_FLOOR,$VIS_CEIL], \
gaps x$GAP_SCALE, episode ${EPISODE_S}s"
  env SIGHT_RAMP=0 TARGET_VISIBLE_FLOOR="$VIS_FLOOR" \
      TARGET_VISIBLE_CEIL="$VIS_CEIL" TARGET_GAP_SCALE="$GAP_SCALE" \
    $MM run -n "$ENV_NAME" python -u scripts/distill.py \
      --task Mjlab-Pick-Place-PiperX-Distill-Robust --teacher "$RT" \
      --num-envs "$VISION_ENVS" --iterations "$DISTILL_ITERS" \
      --episode-length-s "$EPISODE_S" \
      --run-name "${TAG}_distill" --device "cuda:$GPU" --seed 42 \
      --cadence object --sensor measured --logger tensorboard \
      >"$OUT/distill.log" 2>&1 || fail "distill -- see $OUT/distill.log"
  mark_done distill
fi
DISTILL_RUN=$(run_dir_from_log "$OUT/distill.log")
RD=$(latest_checkpoint "$DISTILL_RUN")
printf '%s\n' "$RD" >"$OUT/distill_checkpoint.txt"
say "student $RD"

# --- 4. the same two questions, asked of the student -----------------------
$MM run -n "$ENV_NAME" python -u scripts/eval_endurance.py \
  --checkpoint "$RD" --task Mjlab-Pick-Place-PiperX-Distill-Robust \
  --num-envs 128 --steps "$GATE_STEPS" --device "cuda:$GPU" \
  --out "$OUT/endurance_student.json" 2>&1 | tee -a "$OUT/endurance.log" || true
$MM run -n "$ENV_NAME" python -u scripts/eval_occlusion.py \
  --checkpoint "$RD" --task Mjlab-Pick-Place-PiperX-Distill-Robust \
  --num-envs 128 --steps 300 --device "cuda:$GPU" --seed 101 \
  --out "$OUT/occlusion_student.json" >>"$OUT/occlusion.log" 2>&1 || true
say "v8 done: teacher $RT   student $RD"
