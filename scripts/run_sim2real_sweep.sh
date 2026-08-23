#!/usr/bin/env bash
# Run a Phase WM0 sweep job list across a pool of GPUs.
#
#   scripts/run_sim2real_sweep.sh "<gpu list>" <jobfile>
#
# Each jobfile line is:  <label> <task> <checkpoint> [accept_s1 flags...]
# and is produced by scripts/sweep_plan.py, never typed by hand.
#
# Environment:
#   OUT        directory for the JSONs   (default: results/sim2real_sweep/run)
#   NUM_ENVS   parallel environments     (default: 512)
#   STEPS      control steps             (default: 2400)
#   MJLAB_ENV  micromamba env name       (default: mjlab)
#
# One job per GPU at a time, dispatched through a FIFO used as a token pool,
# so a long job never blocks a free card.  A job that fails leaves its rc in
# the log and the queue carries on: one broken level should not cost the
# other sixty.
set -uo pipefail

if [ $# -lt 2 ]; then sed -n '2,17p' "$0" >&2; exit 2; fi
GPUS=($1); JOBS=$2

cd "$(dirname "$0")/.."
: "${OUT:=results/sim2real_sweep/run}"
: "${NUM_ENVS:=512}"
: "${STEPS:=2400}"
: "${MJLAB_ENV:=mjlab}"

# The camera sensors go through mujoco_warp's rasteriser and need no GL, but
# importing mujoco still initialises a backend, which fails on a box without
# glvnd's libEGL.so.1.
export MUJOCO_GL=${MUJOCO_GL:-disable}
# The env ships libicui18n.so.78, which wants CXXABI_1.3.15, and the system
# libstdc++ does not have it.  `micromamba run` does not export the env's lib
# directory but an interactive activate does, so this breaks only
# non-interactive runs -- inside mjlab's own import of mediapy -> IPython.
PREFIX="$(micromamba env list | awk -v e="$MJLAB_ENV" '$1==e {print $NF}')"
[ -n "$PREFIX" ] && export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"
n_jobs=$(grep -cvE '^\s*(#|$)' "$JOBS")
# An empty job list is a planner that failed, not a sweep with nothing to do.
# Without this the runner prints "0 jobs", exits 0, and the failure is only
# visible as a results directory that never fills up.
if [ "$n_jobs" -eq 0 ]; then
  echo "!!! $JOBS is empty -- the planner failed. Not a successful sweep." >&2
  exit 3
fi
echo "=== $n_jobs jobs, ${#GPUS[@]} gpus, ${NUM_ENVS}x${STEPS}, out=$OUT"

fifo=$(mktemp -u); mkfifo "$fifo"; exec 9<>"$fifo"; rm "$fifo"
for g in "${GPUS[@]}"; do echo "$g" >&9; done

while read -r label task ckpt rest; do
  [ -z "${label:-}" ] && continue
  case "$label" in \#*) continue;; esac
  read -r g <&9
  (
    micromamba run -n "$MJLAB_ENV" python scripts/accept_s1.py \
      "$task" "$ckpt" \
      --num-envs "$NUM_ENVS" --steps "$STEPS" --device "cuda:$g" \
      --label "$label" --json "$OUT/$label.json" \
      $rest > "$OUT/$label.log" 2>&1
    echo "[done $label gpu=$g rc=$?]"
    echo "$g" >&9
  ) &
done < "$JOBS"
wait
echo "=== SWEEP DONE ($JOBS) ==="
