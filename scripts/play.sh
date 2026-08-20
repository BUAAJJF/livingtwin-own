#!/usr/bin/env bash
# Replay a checkpoint. With no argument it plays an untrained policy, which is
# the fastest way to eyeball the scene before spending GPU hours on it.
#
#   scripts/play.sh                       # zero-action policy
#   scripts/play.sh path/to/model_500.pt  # trained policy
#
# Environment:
#   MJLAB_ENV   micromamba env name   (default: mjlab)
#   NUM_ENVS    envs to render        (default: 4)
#   DEVICE      torch device          (default: cuda:0)
#   VIEWER      native | viser | auto (default: auto)
set -Eeuo pipefail

MJLAB_ENV=${MJLAB_ENV:-mjlab}
NUM_ENVS=${NUM_ENVS:-4}
DEVICE=${DEVICE:-cuda:0}
VIEWER=${VIEWER:-auto}

cd "$(dirname "$0")/.."

if [ $# -ge 1 ]; then
  exec micromamba run -n "$MJLAB_ENV" play Mjlab-Push-Cube-PiperX \
    --checkpoint-file "$1" --num-envs "$NUM_ENVS" \
    --device "$DEVICE" --viewer "$VIEWER" "${@:2}"
else
  exec micromamba run -n "$MJLAB_ENV" play Mjlab-Push-Cube-PiperX \
    --agent zero --num-envs "$NUM_ENVS" \
    --device "$DEVICE" --viewer "$VIEWER"
fi
