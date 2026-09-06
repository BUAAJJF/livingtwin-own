#!/usr/bin/env bash
# Run the previous best D455/v4-final hardware recipe with SAM2.1 carrying the
# target selected by the depth stack.
#
#   scripts/run_sam21_v4best.sh
#   RUN_SECONDS=20 scripts/run_sam21_v4best.sh
#   OUT=recordings/my_test scripts/run_sam21_v4best.sh [extra run.py flags]
#
# This is a REAL ARM run.  It deliberately matches v4_stereo_best_repro_try1:
# no min-grasp-height or min-table-clearance guard is added here.
set -Eeuo pipefail

cd "$(dirname "$0")/.."

RUN_SECONDS=${RUN_SECONDS:-60}
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-recordings/sam21_v4_stereo_best_${RUN_STAMP}}

if [ -n "${PYTHON_BIN:-}" ]; then
  RUNNER=("$PYTHON_BIN")
else
  MM_BIN=${MM_BIN:-micromamba}
  MJLAB_ENV=${MJLAB_ENV:-mjlab}
  RUNNER=("$MM_BIN" run -a "" -n "$MJLAB_ENV" python -u)
fi

# This repository uses a src/ package layout.  Make the checkout importable
# even when the active deployment environment was not installed with
# `pip install -e .`; keep any existing search path for vendored dependencies.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

printf 'SAM2.1 + v4 best recipe\nrecording: %s\n' "$OUT"

exec "${RUNNER[@]}" -m hardware.deploy.run \
  --policy hardware/deploy/policies/d455_v4_final \
  --camera d455 \
  --mask depth \
  --depth-source stereo \
  --target-tracker sam21 \
  --allow-legacy-action-api \
  --command-rate-scale 0.6 \
  --home-first \
  --no-record-compress \
  --review-stride 4 \
  --seconds "$RUN_SECONDS" \
  --record "$OUT" \
  "$@"
