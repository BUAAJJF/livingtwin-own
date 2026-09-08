#!/usr/bin/env bash
# Replay a checkpoint. With no argument it plays an untrained policy, which is
# the fastest way to eyeball the scene before spending GPU hours on it.
#
#   scripts/play.sh                       # zero-action policy
#   scripts/play.sh path/to/model_500.pt  # trained policy
#
# Environment:
#   TASK        mjlab task id         (default: Mjlab-Pick-Place-PiperX)
#   CONDA_ENV   Conda environment name (default: livingtwin)
#   NUM_ENVS    envs to render        (default: 4)
#   DEVICE      torch device          (default: cuda:0)
#   VIEWER      native | viser | auto (default: auto)
set -Eeuo pipefail

TASK=${TASK:-Mjlab-Pick-Place-PiperX}
CONDA_ENV=${CONDA_ENV:-${MJLAB_ENV:-livingtwin}}
NUM_ENVS=${NUM_ENVS:-4}
DEVICE=${DEVICE:-cuda:0}
VIEWER=${VIEWER:-auto}

cd "$(dirname "$0")/.."
ROOT=$PWD
source "$ROOT/scripts/conda_env.sh"

if [ $# -ge 1 ]; then
  exec play "$TASK" \
    --checkpoint-file "$1" --num-envs "$NUM_ENVS" \
    --device "$DEVICE" --viewer "$VIEWER" "${@:2}"
else
  exec play "$TASK" \
    --agent zero --num-envs "$NUM_ENVS" \
    --device "$DEVICE" --viewer "$VIEWER"
fi
