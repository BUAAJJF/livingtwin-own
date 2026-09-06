#!/usr/bin/env bash
# A state teacher under the initiation ruler: 36 s x three seeds, the 180 s no-reset run and the
# bin-settle diagnostic, on the state task.  What the routes' initiation and long-run numbers are
# read against.
#
#   OUT=results/pc/gen2/teacher_v11_<stamp> TEACHER=logs/.../model_9399.pt GPU=4 COMMIT=<sha> \
#     nohup setsid bash scripts/pc/teacher_ref.sh &
set -Eeuo pipefail
ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
OUT=${OUT:?set OUT}
TEACHER=${TEACHER:?set TEACHER}
GPU=${GPU:?set GPU}
COMMIT=${COMMIT:-unknown}
TASK=${TASK:-Mjlab-Pick-Place-PiperX-Robust}
SEEDS=${SEEDS:-"101 202 303"}
LONG_STEPS=${LONG_STEPS:-9000}
cd "$ROOT"
mkdir -p "$OUT"
exec >>"$OUT/watcher.log" 2>&1
ENV_PREFIX=$("$MM" env list | awk -v e="$ENV_NAME" '$1==e {print $NF}')
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable PYTHONUNBUFFERED=1
unset PIPER_ALLOW_LEGACY_ACTION_API RESET_FULL_RANGE
PY="$MM run -n $ENV_NAME python -u"
say() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
TSHA=$(sha256sum "$TEACHER" | cut -d' ' -f1)
cat >"$OUT/manifest.json" <<JSON
{"teacher": "$TEACHER", "teacher_sha256": "$TSHA", "task": "$TASK", "gpu": $GPU, "seeds": "$SEEDS",
 "long_steps": $LONG_STEPS, "code_commit_local": "$COMMIT", "remote_git_head": "$(git rev-parse HEAD)",
 "started": "$(date -Is)", "host": "$(hostname)"}
JSON
say "teacher reference: $TEACHER ($TSHA) on $TASK, GPU $GPU"
for s in $SEEDS; do
  say "initiation seed $s"
  $PY scripts/pc/eval_initiation.py --checkpoint "$TEACHER" --task "$TASK" --num-envs 256 --steps 1800 --seed "$s" \
    --device "cuda:$GPU" --sensor measured --out "$OUT/initiation_teacher_s$s.json" >"$OUT/initiation_teacher_s$s.log" 2>&1 || say "seed $s failed"
done
say "long $LONG_STEPS steps seed 101"
$PY scripts/pc/eval_initiation.py --checkpoint "$TEACHER" --task "$TASK" --num-envs 256 --steps "$LONG_STEPS" --episode-length-s 1000000 \
  --seed 101 --device "cuda:$GPU" --sensor measured --out "$OUT/long_teacher_s101.json" >"$OUT/long_teacher_s101.log" 2>&1 || say "long failed"
say "bin settle seed 101"
$PY scripts/pc/diag_bin_settle.py --checkpoint "$TEACHER" --task "$TASK" --seed 101 --device "cuda:$GPU" \
  --out "$OUT/bin_settle_teacher_s101.json" >"$OUT/bin_settle_teacher_s101.log" 2>&1 || say "bin settle failed"
say "done"
date -Is >"$OUT/all.done"
