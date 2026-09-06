#!/usr/bin/env bash
# Explicit entry point for SAM2.1 + Fast-FoundationStereo + the v4-best recipe.
#
# This is the same perception combination as run_sam21_v4best.sh.  The wrapper
# gives the experiment an unambiguous recording name and uses a shorter first
# run by default.  Environment overrides and extra run.py flags pass through:
#
#   scripts/run_sam21_foundation_stereo.sh
#   RUN_SECONDS=60 scripts/run_sam21_foundation_stereo.sh
#   OUT=recordings/my_test scripts/run_sam21_foundation_stereo.sh --view
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RUN_STAMP=$(date +%Y%m%d_%H%M%S)

export RUN_SECONDS=${RUN_SECONDS:-20}
export OUT=${OUT:-recordings/sam21_foundation_stereo_${RUN_STAMP}}

exec "$SCRIPT_DIR/run_sam21_v4best.sh" \
  --target-lifecycle \
  --held-target-radius 0.045 \
  "$@"
