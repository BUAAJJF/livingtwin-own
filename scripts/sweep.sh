#!/usr/bin/env bash
# The final overnight run for the pick-and-place task.
#
# The sweep phase is over: of five schedules the defaults won cleanly, and the
# reason none of them placed was not the schedule.  It was a guard term --
# object_astray policed the spawn sector, the bin sits 28 degrees outside it,
# and carrying the object there cost 1.83 a step against a carry paying 1.08.
# drop_error sat at 0.21 m for 3000 iterations, which is exactly that boundary.
#
# So this runs the corrected configuration across the shape curriculum, with a
# second seed for variance and one hedge on whether the release is discoverable.
set -Euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/home/yunfan/micromamba/envs/mjlab/lib:${LD_LIBRARY_PATH:-}
# Headless training never renders, but importing mujoco initialises a GL
# backend anyway, and on a box without glvnd's libEGL.so.1 that import
# raises before the trainer starts.  Disabling GL outright is both the fix
# and the honest description of what a training run needs.
export MUJOCO_GL=${MUJOCO_GL:-disable}
[ -f "$HOME/.wandb_env" ] && source "$HOME/.wandb_env"

OUT=${OUT:-/home/yunfan/work/piper-push/sweep}
mkdir -p "$OUT"
LOG="$OUT/driver.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run() {
  local gpu=$1 name=$2 task=$3 iters=$4 envs=$5; shift 5
  micromamba run -n mjlab train "$task" \
    --env.scene.num-envs "$envs" --agent.max-iterations "$iters" \
    --agent.run-name "$name" --gpu-ids "[$gpu]" "$@" \
    > "$OUT/$name.log" 2>&1
  echo "exit=$? $name" >> "$OUT/exit.log"
}
tailmean() {
  local v
  v=$(grep -oP "$2:\s*\K[-0-9.]+" "$OUT/$1.log" 2>/dev/null | tail -10 \
      | awk '{s+=$1; n++} END{if(n) printf "%.4f", s/n; else print "0"}')
  echo "${v:-0}"
}

HEDGE="--env.rewards.place.weight 600.0"
for s in 0 1 2; do
  w=$(awk -v i=$s 'BEGIN{print (i==0?5.0:(i==1?3.5:2.0))}')
  HEDGE="$HEDGE --env.curriculum.in-bin-decay.params.stages.$s.weight $w"
done

say "=== the corrected configuration, across the shape curriculum ==="
run 2 g_cube    Mjlab-Pick-Place-PiperX-Cube 3500 8192 &
run 3 g_cube_s2 Mjlab-Pick-Place-PiperX-Cube 3500 8192 --agent.seed 17 &
run 4 g_mid     Mjlab-Pick-Place-PiperX-Mid  3500 8192 &
run 5 g_full    Mjlab-Pick-Place-PiperX      3500 8192 &
run 6 g_hedge   Mjlab-Pick-Place-PiperX-Cube 3500 8192 $HEDGE &
wait
say "finished"
for n in g_cube g_cube_s2 g_mid g_full g_hedge; do
  say "  $n: placed=$(tailmean "$n" "Metrics/pick/objects_placed")" \
      "grasp=$(tailmean "$n" "Metrics/pick/grasp_rate")" \
      "attempts=$(tailmean "$n" "Metrics/pick/grasp_attempts")" \
      "knocked_in=$(tailmean "$n" "Metrics/pick/knocked_in")" \
      "drop_err=$(tailmean "$n" "Metrics/pick/drop_error")" \
      "in_bin=$(tailmean "$n" "Episode_Reward/object_in_bin")"
done
say "=== CAMPAIGN COMPLETE ==="
