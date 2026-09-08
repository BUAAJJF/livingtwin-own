#!/usr/bin/env bash
# Train a state teacher (PPO). Run this inside tmux -- a dropped ssh connection
# should not take the run with it.
#
#   scripts/train.sh [extra mjlab train flags...]
#
# Environment:
#   TASK        mjlab task id             (default: Mjlab-Pick-Place-PiperX)
#   CONDA_ENV   Conda environment name    (default: livingtwin)
#   NUM_ENVS    parallel envs per GPU     (default: 8192)
#   ITERS       PPO iterations            (default: 500)
#   GPUS        python list of gpu ids    (default: [0])
#   RUN_NAME    wandb / log run name      (default: ppo_baseline)
set -Eeuo pipefail

TASK=${TASK:-Mjlab-Pick-Place-PiperX}
# Keep MJLAB_ENV as a compatibility alias for older command lines.
CONDA_ENV=${CONDA_ENV:-${MJLAB_ENV:-livingtwin}}
NUM_ENVS=${NUM_ENVS:-8192}
ITERS=${ITERS:-500}
GPUS=${GPUS:-"[0]"}
RUN_NAME=${RUN_NAME:-ppo_baseline}

cd "$(dirname "$0")/.."
ROOT=$PWD
source "$ROOT/scripts/conda_env.sh"
# Headless training never renders, but importing mujoco initialises a GL
# backend anyway, and on a box without glvnd's libEGL.so.1 that import
# raises before the trainer starts.  Disabling GL outright is both the fix
# and the honest description of what a training run needs.
[ -f "$HOME/.wandb_env" ] && source "$HOME/.wandb_env"

exec train "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$ITERS" \
  --agent.run-name "$RUN_NAME" \
  --gpu-ids "$GPUS" \
  "$@"
