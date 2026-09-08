#!/usr/bin/env bash
# Browse every deployment recording at its full recorded frame rate.
#
#   scripts/logview.sh
#   PORT=9000 scripts/logview.sh
#   scripts/logview.sh --root /path/to/recordings
set -Eeuo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV=${CONDA_ENV:-${MJLAB_ENV:-livingtwin}}
PORT=${PORT:-8765}
ROOT=$PWD
source "$ROOT/scripts/conda_env.sh"

exec python -u \
  -m hardware.deploy.logview --port "$PORT" "$@"
