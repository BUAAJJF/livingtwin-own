#!/usr/bin/env bash
# Move the v8 teacher continuation to the teacher server, without touching git.
#
#   SELFCHECK=1 bash scripts/move_to_server.sh      # what it would do
#   GPU=6 bash scripts/move_to_server.sh            # do it
#
# NOT `sync_server.sh`.  That script runs `git reset -q` on the remote, and
# both trees are deliberately dirty: 105 modified files here, and the whole
# D455 campaign uncommitted there.  It would delete the remote's work and half
# of ours.  This copies files and nothing else.
#
# It does NOT stop the local run.  Two copies of the same continuation cost
# nothing but a card, and a remote launch that fails after the local one was
# killed costs the evening.  Stop the local one only once this is producing
# iterations -- the script prints the command.
set -Eeuo pipefail

HOST=${HOST:-shen-teacher}
REMOTE=${REMOTE:-/home/yunfan/work/piper-push/LivingTwin}
LOCAL=${LOCAL:-/home/yunfan/Project/PiperPush/LivingTwin}
GPU=${GPU:-6}
ENV_NAME=${ENV_NAME:-mjlab}
ITERS=${ITERS:-2000}
EPISODE_S=${EPISODE_S:-36.0}
TAG=${TAG:-v8_horizon}
# The checkpoint the continuation starts from.  Pass CKPT to override; the
# default is whatever the local pass has most recently saved, so a later
# launch does not throw away the iterations that have already run.
CKPT=${CKPT:-}

say() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date -Is)" "$*" >&2; exit 1; }

cd "$LOCAL"
if [ -z "$CKPT" ]; then
  CKPT=$(find logs/rsl_rl/piperx_pick_place_robust -maxdepth 2 -name 'model_*.pt' \
         -path '*v8_horizon36*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
fi
[ -n "$CKPT" ] && [ -f "$CKPT" ] || fail "no v8 checkpoint found; set CKPT="
say "checkpoint: $CKPT"

if [ -n "${SELFCHECK:-}" ]; then
  say "SELFCHECK -- nothing is copied or launched"
  say "would rsync src/ scripts/ hardware/ tests/ pyproject.toml -> $HOST:$REMOTE"
  say "would copy $CKPT -> $REMOTE/logs/rsl_rl/piperx_pick_place_robust/${TAG}_bootstrap/"
  say "would launch $ITERS iterations at episode_length_s=$EPISODE_S on GPU $GPU"
  say "would NOT touch git on either side, and would NOT stop the local run"
  exit 0
fi

ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" true 2>/dev/null \
  || fail "cannot reach $HOST -- the MotionPro VPN is not up.  Bring it up \
first (the client's VNC is on http://localhost:6080) and re-run."
say "reachable: $(ssh "$HOST" hostname)"

ssh "$HOST" "test -d $REMOTE" || fail "$REMOTE does not exist on $HOST"
ssh "$HOST" "micromamba env list | grep -q '^ *$ENV_NAME '" \
  || fail "no micromamba env named $ENV_NAME on $HOST"

# Source only.  No logs, no recordings, no results, no checkpoints: those are
# the two machines' separate histories and copying either way loses one of them.
say "rsync (source only, --delete OFF so the remote's own work survives)"
rsync -az --info=stats1 \
  --exclude='__pycache__' --exclude='*.pyc' \
  src scripts hardware tests pyproject.toml "$HOST:$REMOTE/" \
  || fail "rsync failed"

BOOT="logs/rsl_rl/piperx_pick_place_robust/${TAG}_bootstrap_remote"
NAME=$(basename "$CKPT")
ssh "$HOST" "mkdir -p $REMOTE/$BOOT"
rsync -az "$CKPT" "$HOST:$REMOTE/$BOOT/$NAME" || fail "checkpoint copy failed"
say "bootstrap: $BOOT/$NAME"

OUT="results/d455_heavy_dr/${TAG}_remote"
say "launching on GPU $GPU: $ITERS iterations, episode ${EPISODE_S}s"
ssh "$HOST" "cd $REMOTE && mkdir -p $OUT && \
  export MUJOCO_GL=disable WANDB_MODE=offline \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  nohup setsid env SIGHT_RAMP=0 TASK=Mjlab-Pick-Place-PiperX-Robust \
    NUM_ENVS=4096 ITERS=$ITERS GPUS='[$GPU]' RUN_NAME=${TAG}_remote \
    bash scripts/train.sh --env.episode-length-s $EPISODE_S \
      --agent.resume True --agent.load-run $(basename $BOOT) \
      --agent.load-checkpoint $NAME --agent.logger tensorboard \
    > $OUT/teacher.log 2>&1 < /dev/null & disown; sleep 3; echo launched"

say "watch it with:"
say "  ssh $HOST 'tail -f $REMOTE/$OUT/teacher.log | grep -a \"Learning iteration\"'"
say "once it is producing iterations, stop the local one with:"
say "  ps -eo pid,cmd | grep '[t]rain Mjlab-Pick-Place-PiperX-Robust' | awk '{print \$1}' | xargs kill"
