#!/usr/bin/env bash
# yf/pc phase 1: evaluate one bounded (Action API v2) state teacher on the repo's
# standard ruler, three fixed seeds, two chains per GPU in parallel.
#
#   TAG=v10c_sight_7400 GPU=4 CKPT=logs/.../model_7400.pt COMMIT=<local sha> \
#     nohup setsid bash scripts/pc/eval_teacher.sh >results/pc/teacher_eval/$TAG/driver.log 2>&1 &
#
# Chain A (accept):    accept_s1 on -Robust, 256 x 2400, --sensor measured, seeds 101 202 303
# Chain B (endurance): eval_endurance 256 x 1200 no reset; the same under RESET_FULL_RANGE=1;
#                      eval_occlusion 256 x 300 (engaged); eval_actions 256 x 600
# Nothing is loaded through a legacy path: no -V1 id, no PIPER_ALLOW_LEGACY_ACTION_API.
set -Eeuo pipefail
ROOT=${ROOT:-/home/yunfan/work/piper-push/LivingTwin}
MM=${MM:-/home/yunfan/.local/bin/micromamba}
ENV_NAME=${ENV_NAME:-mjlab}
TAG=${TAG:?set TAG}
GPU=${GPU:?set GPU}
CKPT=${CKPT:?set CKPT}
COMMIT=${COMMIT:-unknown}
TASK=${TASK:-Mjlab-Pick-Place-PiperX-Robust}
SEEDS=${SEEDS:-"101 202 303"}
ENVS=${ENVS:-256}
OUT=${OUT:-results/pc/teacher_eval/$TAG}
cd "$ROOT"
ENV_PREFIX=$("$MM" env list | awk -v e="$ENV_NAME" '$1==e {print $NF}')
[ -n "$ENV_PREFIX" ] || { echo "no micromamba env named $ENV_NAME" >&2; exit 2; }
export PATH="$(dirname "$MM"):$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=disable WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
unset PIPER_ALLOW_LEGACY_ACTION_API RESET_FULL_RANGE
mkdir -p "$OUT"
say() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
[ -f "$CKPT" ] || { say "no checkpoint at $CKPT"; exit 1; }
sha=$(sha256sum "$CKPT" | cut -d' ' -f1)
cat >"$OUT/manifest.json" <<JSON
{"tag": "$TAG", "checkpoint": "$CKPT", "checkpoint_sha256": "$sha", "task": "$TASK", "seeds": "$SEEDS",
 "num_envs": $ENVS, "gpu": $GPU, "code_commit_local": "$COMMIT", "remote_git_head": "$(git rev-parse HEAD)",
 "started": "$(date -Is)", "host": "$(hostname)"}
JSON
say "teacher $TAG  ckpt $CKPT  sha256 $sha  gpu $GPU"
PY="$MM run -n $ENV_NAME python -u"

chain_a() {
  for s in $SEEDS; do
    say "accept seed $s"
    $PY scripts/accept_s1.py "$TASK" "$CKPT" --num-envs "$ENVS" --steps 2400 --seed "$s" \
      --device "cuda:$GPU" --sensor measured --label "$TAG" --json "$OUT/accept_s$s.json" \
      >"$OUT/accept_s$s.log" 2>&1 || say "accept seed $s exited non-zero (FAIL verdict or error)"
  done
  date -Is >"$OUT/chain_a.done"
}
chain_b() {
  for s in $SEEDS; do
    say "endurance seed $s"
    $PY scripts/eval_endurance.py --checkpoint "$CKPT" --task "$TASK" --num-envs "$ENVS" --steps 1200 \
      --seed "$s" --device "cuda:$GPU" --sensor measured --out "$OUT/endurance_s$s.json" \
      >"$OUT/endurance_s$s.log" 2>&1 || say "endurance seed $s failed"
    say "endurance full-range reset seed $s"
    RESET_FULL_RANGE=1 $PY scripts/eval_endurance.py --checkpoint "$CKPT" --task "$TASK" --num-envs "$ENVS" --steps 1200 \
      --seed "$s" --device "cuda:$GPU" --sensor measured --out "$OUT/endurance_fullreset_s$s.json" \
      >"$OUT/endurance_fullreset_s$s.log" 2>&1 || say "full-reset endurance seed $s failed"
    say "occlusion seed $s"
    $PY scripts/eval_occlusion.py --checkpoint "$CKPT" --task "$TASK" --num-envs "$ENVS" --steps 300 \
      --seed "$s" --device "cuda:$GPU" --sensor measured --out "$OUT/occlusion_s$s.json" \
      >"$OUT/occlusion_s$s.log" 2>&1 || say "occlusion seed $s failed"
    say "actions seed $s"
    $PY scripts/pc/eval_actions.py --checkpoint "$CKPT" --task "$TASK" --num-envs "$ENVS" --steps 600 \
      --seed "$s" --device "cuda:$GPU" --sensor measured --out "$OUT/actions_s$s.json" \
      >"$OUT/actions_s$s.log" 2>&1 || say "actions seed $s failed"
  done
  date -Is >"$OUT/chain_b.done"
}
chain_a & PA=$!
chain_b & PB=$!
echo "$PA $PB" >"$OUT/chains.pid"
wait $PA; wait $PB
say "all chains done"
date -Is >"$OUT/all.done"
