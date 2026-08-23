#!/usr/bin/env bash
# Everything after the last GPU job: analyse, time, gate, render.
#
#   scripts/wm1_finish.sh
#
# One command so that the numbers in the report and the numbers in the gate
# come from one pass over one set of files, and re-running it after a late
# result lands cannot leave the two disagreeing.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-disable}
PREFIX="$(micromamba env list | awk '$1=="mjlab" {print $NF}')"
[ -n "$PREFIX" ] && export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
PY="${PREFIX:+$PREFIX/bin/}python"
R=results/wm1_latency

# Digest first, and analyse the digest rather than the originals, so that the
# committed artefacts are self-consistent: a fresh clone has runs/ and not the
# 75 MB of raw evaluations, and recomputing the analysis there gives the same
# numbers as here rather than nothing at all.
$PY scripts/wm1_digest.py
$PY scripts/analyze_wm1.py --dirs "$R/runs" \
  --baseline runs/zeroshot --json "$R/analysis.json"
$PY scripts/wm1_timings.py --json "$R/timings.json"
$PY scripts/wm1_gate.py --json "$R/gate.json"
echo
echo "=== tables ==="
$PY scripts/wm1_tables.py --section all
