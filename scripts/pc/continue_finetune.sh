#!/usr/bin/env bash
# Continue a finished route's PPO fine-tune under the current environment and
# re-measure it: the cheapest answer to "does the new termination teach the
# policy anything" before any retraining from scratch.
#
#   OUT=results/pc/gen3/c0_<tag> RESUME=logs/rsl_rl/.../model_799.pt TASK=Mjlab-Pick-Place-PiperX-PC-P1BZ-Vision \
#     ITERS=1600 GPU=4 nohup setsid bash scripts/pc/continue_finetune.sh &
#
# ITERS is the absolute iteration target (finetune.py counts that way); the
# optimizer, the critic and the schedule resume with the checkpoint.
set -Eeuo pipefail
ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
OUT=${OUT:?set OUT}
RESUME=${RESUME:?set RESUME}
TASK=${TASK:?set TASK}
ITERS=${ITERS:?set ITERS (absolute)}
GPU=${GPU:?set GPU}
COMMIT=${COMMIT:-unknown}
SEED=${SEED:-42}
VISION_ENVS=${VISION_ENVS:-512}
EPISODE_S=${EPISODE_S:-36.0}
EVAL_ENVS=${EVAL_ENVS:-256}
EVAL_SEEDS=${EVAL_SEEDS:-"101 202 303"}
LONG_STEPS=${LONG_STEPS:-9000}
cd "$ROOT"
[ -e "$OUT" ] && { echo "refusing to reuse $OUT" >&2; exit 2; }
mkdir -p "$OUT"
exec >>"$OUT/watcher.log" 2>&1
ENV_PREFIX=$("$MM" env list | awk -v e="$ENV_NAME" '$1==e {print $NF}')
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable WANDB_MODE=offline PYTHONUNBUFFERED=1
unset PIPER_ALLOW_LEGACY_ACTION_API RESET_FULL_RANGE
TRAIN_ENV=${TRAIN_ENV:-}   # knobs for the fine-tune only; the evaluation runs the standard ruler
PY="$MM run -n $ENV_NAME python -u"
say() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
RSHA=$(sha256sum "$RESUME" | cut -d' ' -f1)
cat >"$OUT/manifest.json" <<JSON
{"resume": "$RESUME", "resume_sha256": "$RSHA", "task": "$TASK", "route": "$(echo "$TASK" | sed -nE 's/.*-PC-([A-Z0-9]+)-.*/\1/p')", "oracle_only": false, "iterations_target": $ITERS, "gpu": $GPU,
 "seed": $SEED, "vision_envs": $VISION_ENVS, "episode_s": $EPISODE_S, "eval_envs": $EVAL_ENVS, "eval_seeds": "$EVAL_SEEDS",
 "code_commit_local": "$COMMIT", "remote_git_head": "$(git rev-parse HEAD)", "started": "$(date -Is)", "host": "$(hostname)",
 "train_env": "$TRAIN_ENV", "env_knobs": {"OBJECT_ASTRAY_TERMINATE": "${OBJECT_ASTRAY_TERMINATE:-unset(default on)}"}}
JSON
say "continue $RESUME ($RSHA) on $TASK to iteration $ITERS, GPU $GPU"
T0=$(date +%s)
env $TRAIN_ENV $PY scripts/finetune.py --task "$TASK" --resume "$RESUME" --num-envs "$VISION_ENVS" --iterations "$ITERS" \
  --episode-length-s "$EPISODE_S" --run-name "$(basename "$OUT")" --device "cuda:$GPU" --seed "$SEED" --logger tensorboard \
  >"$OUT/finetune.log" 2>&1 || { say "finetune FAILED"; date -Is >"$OUT/FAILED"; exit 1; }
printf '{"stage": "finetune", "seconds": %d}\n' "$(( $(date +%s) - T0 ))" >>"$OUT/timing.jsonl"
run=$(grep -aE '\[INFO\] logging to ' "$OUT/finetune.log" | tail -n 1 | sed -E 's/.*logging to //'); run=${run#$ROOT/}
FT="$run/$(find "$run" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sort -V | tail -n 1)"
printf '%s\n' "$FT" >"$OUT/final_checkpoint.txt"
say "final: $FT"
T0=$(date +%s)
for s in $EVAL_SEEDS; do
  $PY scripts/accept_s1.py "$TASK" "$FT" --num-envs "$EVAL_ENVS" --steps 2400 --seed "$s" --device "cuda:$GPU" --sensor measured \
    --label "$(basename "$OUT")" --json "$OUT/accept_final_s$s.json" >"$OUT/accept_final_s$s.log" 2>&1 || say "accept $s exited non-zero"
  $PY scripts/eval_endurance.py --checkpoint "$FT" --task "$TASK" --num-envs "$EVAL_ENVS" --steps 1200 --seed "$s" --device "cuda:$GPU" \
    --sensor measured --out "$OUT/endurance_final_s$s.json" >"$OUT/endurance_final_s$s.log" 2>&1 || say "endurance $s failed"
  $PY scripts/pc/eval_initiation.py --checkpoint "$FT" --task "$TASK" --num-envs "$EVAL_ENVS" --steps 1800 --seed "$s" --device "cuda:$GPU" \
    --sensor measured --out "$OUT/initiation_final_s$s.json" >"$OUT/initiation_final_s$s.log" 2>&1 || say "initiation $s failed"
done
$PY scripts/pc/eval_initiation.py --checkpoint "$FT" --task "$TASK" --num-envs "$EVAL_ENVS" --steps "$LONG_STEPS" --episode-length-s 1000000 \
  --seed 101 --device "cuda:$GPU" --sensor measured --out "$OUT/long_final_s101.json" >"$OUT/long_final_s101.log" 2>&1 || say "long failed"
printf '{"stage": "eval", "seconds": %d}\n' "$(( $(date +%s) - T0 ))" >>"$OUT/timing.jsonl"
say "done"
date -Is >"$OUT/all.done"
