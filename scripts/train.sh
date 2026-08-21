#!/usr/bin/env bash
# Train the push policy. Run this inside tmux -- a dropped ssh connection
# should not take the run with it.
#
#   scripts/train.sh [extra mjlab train flags...]
#
# Environment:
#   TASK        mjlab task id             (default: Mjlab-Push-Cube-PiperX)
#   MJLAB_ENV   micromamba env name       (default: mjlab)
#   NUM_ENVS    parallel envs per GPU     (default: 8192)
#   ITERS       PPO iterations            (default: 500)
#   GPUS        python list of gpu ids    (default: [0])
#   RUN_NAME    wandb / log run name      (default: ppo_baseline)
set -Eeuo pipefail

TASK=${TASK:-Mjlab-Push-Cube-PiperX}
MJLAB_ENV=${MJLAB_ENV:-mjlab}
NUM_ENVS=${NUM_ENVS:-8192}
ITERS=${ITERS:-500}
GPUS=${GPUS:-"[0]"}
RUN_NAME=${RUN_NAME:-ppo_baseline}

cd "$(dirname "$0")/.."
[ -f "$HOME/.wandb_env" ] && source "$HOME/.wandb_env"

exec micromamba run -n "$MJLAB_ENV" train "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$ITERS" \
  --agent.run-name "$RUN_NAME" \
  --gpu-ids "$GPUS" \
  "$@"
