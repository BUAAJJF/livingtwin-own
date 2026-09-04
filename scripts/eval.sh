#!/usr/bin/env bash
# Run scripts/accept_s1.py with the environment it needs.
#
#   scripts/eval.sh <TaskId> <checkpoint> <label> [extra accept_s1 flags...]
#
# Environment:
#   MJLAB_ENV   micromamba env name        (default: mjlab)
#   GPU         cuda device index          (default: 0)
#   NUM_ENVS    parallel environments      (default: 512)
#   STEPS       control steps              (default: 2400)
#   SEED        rollout seed               (default: 20260823)
#   OUT         directory for the JSON     (default: results/eval/baseline)
#
# 512 x 2400 is the protocol every number in docs/results.md was measured
# under: 409.6 arm-minutes, about 22k object instances.  Smaller samples read
# 1-2% optimistic across the board, so they are not comparable and are not
# mixed in.
set -Eeuo pipefail

if [ $# -lt 3 ]; then
  sed -n '2,17p' "$0" >&2
  exit 2
fi

TASK=$1; CKPT=$2; LABEL=$3; shift 3

MJLAB_ENV=${MJLAB_ENV:-mjlab}
GPU=${GPU:-0}
NUM_ENVS=${NUM_ENVS:-512}
STEPS=${STEPS:-2400}
SEED=${SEED:-20260823}
OUT=${OUT:-results/eval/baseline}

cd "$(dirname "$0")/.."

# The camera sensors go through mujoco_warp's rasteriser and need no GL at
# all; importing mujoco still initialises a backend, which fails on a box
# without glvnd's libEGL.so.1.
export MUJOCO_GL=${MUJOCO_GL:-disable}

# The env ships libicui18n.so.78, which wants CXXABI_1.3.15, and the system
# libstdc++ under /lib/x86_64-linux-gnu does not have it.  Without this the
# loader picks the system one and `import sqlite3` dies inside mjlab's own
# import of mediapy -> IPython.  `micromamba run` does not set this; an
# interactive `micromamba activate` does, which is why this only bites
# non-interactive invocations.
PREFIX="$(micromamba env list | awk -v e="$MJLAB_ENV" '$1==e {print $NF}')"
[ -n "$PREFIX" ] && export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"
exec micromamba run -n "$MJLAB_ENV" python scripts/accept_s1.py \
  "$TASK" "$CKPT" \
  --num-envs "$NUM_ENVS" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --device "cuda:$GPU" \
  --label "$LABEL" \
  --json "$OUT/$LABEL.json" \
  "$@"
