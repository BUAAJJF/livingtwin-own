#!/usr/bin/env bash
# yf/pc phase 2: continue a bounded (Action API v2) teacher that failed the screen.
#
#   TAG=v11_nosight_36s GPU=4 BASE=logs/.../model_7400.pt COMMIT=<sha> \
#     nohup setsid bash scripts/pc/continue_teacher.sh >results/pc/teacher/$TAG/watcher.log 2>&1 &
#
# What changes against v10c, and nothing else:
#   * episode_length_s 36 (v8: the horizon that contains the within-episode collapse)
#   * RESET_FULL_RANGE=1 (v9: start from anywhere; it bought endurance 0.82 -> 0.90)
#   * WRIST_W=0.30 (v9's wrist bonus; inert on the -NoSight task, whose wrist term is 0)
#   * the v10c capability curriculum resumes AT the stage the checkpoint had reached
#     (PIPER_COLD_START_STAGE), with the same DR schedule and the same final weights;
#     the penalties are not rescaled (v9's SMOOTH_SCALE is not copied).
# The base checkpoint is copied into a bootstrap run directory, as every warm
# start in this repository has been, and ITERS are ADDITIONAL (rsl_rl resume).
# Every ``EVAL_EVERY`` iterations the newest checkpoint gets a one-seed
# endurance read on the -Robust play domain; the full three-seed screen
# (scripts/pc/eval_teacher.sh) is run on the final checkpoint and on any
# checkpoint whose one-seed late/early clears the gate.
set -Eeuo pipefail
ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
TAG=${TAG:?set TAG}
GPU=${GPU:?set GPU}
BASE=${BASE:?set BASE checkpoint}
COMMIT=${COMMIT:-unknown}
SIGHT=${SIGHT:-0}
NUM_ENVS=${NUM_ENVS:-8192}
ITERS=${ITERS:-2000}
EPISODE_S=${EPISODE_S:-36.0}
START_STAGE=${START_STAGE:-3}
EVAL_EVERY=${EVAL_EVERY:-250}
GATE=${GATE:-0.85}
OUT=${OUT:-results/pc/teacher/$TAG}
EXP=piperx_pick_place_robust_cold
if [ "$SIGHT" = "1" ]; then TASK=Mjlab-Pick-Place-PiperX-Robust-Cold; else TASK=Mjlab-Pick-Place-PiperX-Robust-Cold-NoSight; fi
cd "$ROOT"
[ -e "$OUT" ] && { echo "refusing to reuse $OUT" >&2; exit 2; }
mkdir -p "$OUT"
ENV_PREFIX=$("$MM" env list | awk -v e="$ENV_NAME" '$1==e {print $NF}')
[ -n "$ENV_PREFIX" ] || { echo "no micromamba env named $ENV_NAME" >&2; exit 2; }
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
unset PIPER_ALLOW_LEGACY_ACTION_API
say()  { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date -Is)" "$*" >&2; date -Is >"$OUT/FAILED"; exit 1; }
PY="$MM run -n $ENV_NAME python -u"
[ -f "$BASE" ] || fail "no base checkpoint at $BASE"
BSHA=$(sha256sum "$BASE" | cut -d' ' -f1)
start_name=$(basename "$BASE")
bootstrap="logs/rsl_rl/$EXP/${TAG}_bootstrap"
mkdir -p "$bootstrap"
cp "$BASE" "$bootstrap/$start_name"
cat >"$OUT/manifest.json" <<JSON
{"tag": "$TAG", "task": "$TASK", "gpu": $GPU, "base": "$BASE", "base_sha256": "$BSHA",
 "code_commit_local": "$COMMIT", "remote_git_head": "$(git rev-parse HEAD)",
 "num_envs": $NUM_ENVS, "additional_iterations": $ITERS, "episode_s": $EPISODE_S,
 "env": {"RESET_FULL_RANGE": "1", "WRIST_W": "0.30", "PIPER_COLD_START_STAGE": "$START_STAGE", "SMOOTH_SCALE": "unset (1.0)"},
 "eval_every": $EVAL_EVERY, "gate_late_over_early": $GATE, "started": "$(date -Is)", "host": "$(hostname)"}
JSON
say "continue $TAG from $BASE (sha256 $BSHA) on $TASK, +$ITERS iterations, episode ${EPISODE_S}s, full-range reset, start stage $START_STAGE"

# -- training, in the background ----------------------------------------------
(
  export RESET_FULL_RANGE=1 WRIST_W=0.30 PIPER_COLD_START_STAGE=$START_STAGE \
         PIPER_CURRICULUM_LOG="$OUT/curriculum.jsonl" PIPER_U_TELEMETRY="$OUT/u_telemetry.jsonl" PIPER_U_TELEMETRY_TAG=teacher
  TASK=$TASK NUM_ENVS=$NUM_ENVS ITERS=$ITERS GPUS="[$GPU]" RUN_NAME="${TAG}_teacher" \
    bash scripts/train.sh --agent.resume True --agent.load-run "$(basename "$bootstrap")" \
      --agent.load-checkpoint "$start_name" --agent.logger tensorboard --agent.seed 42 \
      --env.episode-length-s "$EPISODE_S" >"$OUT/teacher.log" 2>&1
  echo $? >"$OUT/teacher.exit"
) &
TRAIN_PID=$!
echo $TRAIN_PID >"$OUT/train_wrapper.pid"
say "training wrapper pid $TRAIN_PID; log $OUT/teacher.log"

run_dir() {
  grep -aE 'Logging experiment in directory: |\[INFO\] logging to ' "$OUT/teacher.log" 2>/dev/null | tail -n 1 | sed -E 's/.*(directory: |logging to )//' | sed "s#^$ROOT/##"
}

# -- periodic one-seed endurance reads on the newest checkpoint --------------
done_evals=""
last_seen=""
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
  rd=$(run_dir)
  [ -n "$rd" ] && [ -d "$rd" ] || continue
  for ck in $(ls "$rd"/model_*.pt 2>/dev/null | sort -V); do
    it=$(basename "$ck" .pt); it=${it#model_}
    [ $((it % EVAL_EVERY)) -eq 0 ] || continue
    [ "$it" -gt 0 ] || continue
    case " $done_evals " in *" $it "*) continue;; esac
    # the trainer may still be writing it
    sleep 5
    say "periodic endurance on $ck"
    RESET_FULL_RANGE=1 $PY scripts/eval_endurance.py --checkpoint "$ck" --task Mjlab-Pick-Place-PiperX-Robust \
      --num-envs 128 --steps 1200 --seed 101 --device "cuda:$GPU" --sensor measured \
      --out "$OUT/periodic_full_$it.json" >"$OUT/periodic_full_$it.log" 2>&1 || say "periodic eval $it failed"
    $PY scripts/eval_endurance.py --checkpoint "$ck" --task Mjlab-Pick-Place-PiperX-Robust \
      --num-envs 128 --steps 1200 --seed 101 --device "cuda:$GPU" --sensor measured \
      --out "$OUT/periodic_narrow_$it.json" >"$OUT/periodic_narrow_$it.log" 2>&1 || say "periodic eval $it failed"
    done_evals="$done_evals $it"
    python3 - "$OUT/periodic_narrow_$it.json" "$OUT/periodic_full_$it.json" "$it" <<'PY' >>"$OUT/periodic.txt" 2>/dev/null || true
import json, sys
n = json.load(open(sys.argv[1])); f = json.load(open(sys.argv[2]))
print(f"iter {sys.argv[3]:>6s}  narrow early {n['early_per_min']:.1f} late {n['late_per_min']:.1f} l/e {n['late_over_early']:.3f} jaw {n['jaw_mm_stopped']:.1f}  |  full early {f['early_per_min']:.1f} late {f['late_per_min']:.1f} l/e {f['late_over_early']:.3f}")
PY
    tail -n 1 "$OUT/periodic.txt" 2>/dev/null || true
  done
done
say "training exited with $(cat "$OUT/teacher.exit" 2>/dev/null)"
rd=$(run_dir)
final=$(ls "$rd"/model_*.pt | sort -V | tail -n 1)
printf '%s\n' "$final" >"$OUT/final_checkpoint.txt"
say "final checkpoint $final; full three-seed screen"
TAG="${TAG}_final" GPU=$GPU CKPT="$final" COMMIT=$COMMIT OUT="$OUT/screen_final" bash scripts/pc/eval_teacher.sh >"$OUT/screen_final.log" 2>&1 || say "screen failed"
say "done"
date -Is >"$OUT/all.done"
