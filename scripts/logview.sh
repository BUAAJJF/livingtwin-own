#!/usr/bin/env bash
# Browse every deployment recording at its full recorded frame rate.
#
#   scripts/logview.sh
#   PORT=9000 scripts/logview.sh
#   scripts/logview.sh --root /path/to/recordings
set -Eeuo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

MM_BIN=${MM_BIN:-micromamba}
MJLAB_ENV=${MJLAB_ENV:-mjlab}
PORT=${PORT:-8765}

exec "$MM_BIN" run -a "" -n "$MJLAB_ENV" python -u \
  -m hardware.deploy.logview --port "$PORT" "$@"
