#!/usr/bin/env bash
# The first campaign on the rebuilt simulation.
#
# What changed under it: a 2 ms physics step with 8 ms contacts, so the grasp
# and the landing are resolved rather than tunnelled through; objects composed
# from a box, a cylinder and a second box, with mass and inertia computed from
# the composition; a price on touching the object with the gripper body; and a
# speed penalty that starts at 85% of the safety shell instead of at the shell.
#
# So this is not a hyper-parameter sweep.  It is three points on the shape
# curriculum plus a second seed, asking whether the task still trains once the
# simulation stops flattering it -- and what the honest throughput is.
#
# Launch it detached from any session, not just inside tmux:
#
#     setsid nohup bash scripts/sweep.sh > /dev/null 2>&1 < /dev/null &
#
# tmux keeps a run alive across a dropped connection, which is what it is for,
# and it does not keep one alive across `tmux kill-server`.  This campaign has
# now lost two runs that way -- once at iteration 46 and once at 1800 -- when
# the machine was being cleared for someone else.  setsid puts the trainers in
# their own session so clearing the terminal side does not reach them.
#
# RESUME=1 picks each run up from the newest checkpoint of the newest matching
# run directory instead of starting over.
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

OUT=${OUT:-/home/yunfan/work/piper-push/h}
NUM_ENVS=${NUM_ENVS:-8192}
ITERS=${ITERS:-3500}
mkdir -p "$OUT"
LOG="$OUT/driver.log"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

run() {
  local gpu=$1 name=$2 task=$3; shift 3
  local resume=()
  if [ "${RESUME:-0}" = "1" ]; then
    local dir
    dir=$(ls -td logs/rsl_rl/*/*_"$name" 2>/dev/null | head -1)
    if [ -n "$dir" ] && ls "$dir"/model_*.pt >/dev/null 2>&1; then
      # Newest by mtime, not by name: model_900 sorts after model_1800.
      local ckpt
      ckpt=$(ls -t "$dir"/model_*.pt | head -1)
      resume=(--agent.resume True --agent.load-run "$(basename "$dir")"
              --agent.load-checkpoint "$(basename "$ckpt")")
      say "  $name resuming from $(basename "$dir")/$(basename "$ckpt")"
    else
      say "  $name has nothing to resume from; starting fresh"
    fi
  fi
  micromamba run -n mjlab train "$task" \
    --env.scene.num-envs "$NUM_ENVS" --agent.max-iterations "$ITERS" \
    --agent.run-name "$name" --gpu-ids "[$gpu]" "${resume[@]}" "$@" \
    > "$OUT/$name.log" 2>&1
  echo "exit=$? $name" >> "$OUT/exit.log"
}
tailmean() {
  local v
  v=$(grep -oP "$2:\s*\K[-0-9.]+" "$OUT/$1.log" 2>/dev/null | tail -10 \
      | awk '{s+=$1; n++} END{if(n) printf "%.4f", s/n; else print "0"}')
  echo "${v:-0}"
}

say "=== the rebuilt simulation, across the shape curriculum ==="
run 4 h_full    Mjlab-Pick-Place-PiperX      &
run 5 h_full_s2 Mjlab-Pick-Place-PiperX      --agent.seed 17 &
run 6 h_mid     Mjlab-Pick-Place-PiperX-Mid  &
run 7 h_cube    Mjlab-Pick-Place-PiperX-Cube &
wait
say "finished"
for n in h_full h_full_s2 h_mid h_cube; do
  say "  $n: placed=$(tailmean "$n" "Metrics/pick/objects_placed")" \
      "grasp=$(tailmean "$n" "Metrics/pick/grasp_rate")" \
      "attempts=$(tailmean "$n" "Metrics/pick/grasp_attempts")" \
      "knocked_in=$(tailmean "$n" "Metrics/pick/knocked_in")" \
      "palm=$(tailmean "$n" "Episode_Reward/palm_push")" \
      "over_trip=$(tailmean "$n" "Episode_Reward/over_trip")"
done
say "=== CAMPAIGN COMPLETE ==="
