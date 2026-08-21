#!/usr/bin/env bash
# An overnight campaign for the pick-and-place task.
#
# Stage 1 hedges: five reward/exploration configs on the fixed cube, in
# parallel, because the open question -- whether the policy ever discovers that
# letting go over the bin is worth anything -- has several plausible answers and
# testing them in sequence wastes the night.
# Stage 2 takes whichever placed the most and opens the shape curriculum.
# Stage 3 gives the best of those the long run.
#
# Every decision it makes is written to driver.log with the numbers behind it.
set -Euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/home/yunfan/micromamba/envs/mjlab/lib:${LD_LIBRARY_PATH:-}
[ -f "$HOME/.wandb_env" ] && source "$HOME/.wandb_env"

OUT=${OUT:-/home/yunfan/work/piper-push/sweep}
mkdir -p "$OUT"
LOG="$OUT/driver.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run() {  # run <gpu> <name> <task> <iters> <envs> [extra flags...]
  local gpu=$1 name=$2 task=$3 iters=$4 envs=$5; shift 5
  micromamba run -n mjlab train "$task" \
    --env.scene.num-envs "$envs" --agent.max-iterations "$iters" \
    --agent.run-name "$name" --gpu-ids "[$gpu]" "$@" \
    > "$OUT/$name.log" 2>&1
  echo "exit=$? $name" >> "$OUT/exit.log"
}

# The last logged value of a metric, or 0 if the run never produced one.
metric() { grep -oP "$2:\s*\K[0-9.]+" "$OUT/$1.log" 2>/dev/null | tail -1 || true; }
placed()  { local v; v=$(metric "$1" "Metrics/pick/objects_placed"); echo "${v:-0}"; }
grasped() { local v; v=$(metric "$1" "Metrics/pick/grasp_rate");     echo "${v:-0}"; }

say "=== STAGE 1: five configs on the fixed cube, 3000 iterations, 4096 envs ==="
run 2 s1_base      Mjlab-Pick-Place-PiperX-Cube 3000 4096 &
run 3 s1_bridge    Mjlab-Pick-Place-PiperX-Cube 3000 4096 \
      --env.rewards.object_in_bin.weight 8.0 &
run 4 s1_balance   Mjlab-Pick-Place-PiperX-Cube 3000 4096 \
      --env.rewards.grasp.weight 25.0 --env.rewards.transport.weight 12.0 \
      --env.rewards.lift.weight 4.0 &
run 5 s1_explore   Mjlab-Pick-Place-PiperX-Cube 3000 4096 \
      --agent.algorithm.entropy-coef 0.012 &
run 6 s1_both      Mjlab-Pick-Place-PiperX-Cube 3000 4096 \
      --env.rewards.object_in_bin.weight 8.0 --agent.algorithm.entropy-coef 0.012 &
wait
say "stage 1 finished"

BEST=""; BEST_P="-1"
for n in s1_base s1_bridge s1_balance s1_explore s1_both; do
  p=$(placed "$n"); g=$(grasped "$n")
  say "  $n: objects_placed=$p grasp_rate=$g"
  if awk -v a="$p" -v b="$BEST_P" 'BEGIN{exit !(a>b)}'; then BEST="$n"; BEST_P="$p"; fi
done
say "winner: $BEST (objects_placed=$BEST_P)"

EXTRA=""
case "$BEST" in
  s1_bridge)  EXTRA="--env.rewards.object_in_bin.weight 8.0" ;;
  s1_balance) EXTRA="--env.rewards.grasp.weight 25.0 --env.rewards.transport.weight 12.0 --env.rewards.lift.weight 4.0" ;;
  s1_explore) EXTRA="--agent.algorithm.entropy-coef 0.012" ;;
  s1_both)    EXTRA="--env.rewards.object_in_bin.weight 8.0 --agent.algorithm.entropy-coef 0.012" ;;
esac
say "carrying forward: ${EXTRA:-<defaults>}"
echo "$EXTRA" > "$OUT/winner_flags.txt"
echo "$BEST" > "$OUT/winner_name.txt"

say "=== STAGE 2: the winner, longer and across the shape curriculum ==="
run 2 s2_cube_long Mjlab-Pick-Place-PiperX-Cube 6000 8192 $EXTRA &
run 3 s2_mid       Mjlab-Pick-Place-PiperX-Mid  4000 8192 $EXTRA &
run 4 s2_full      Mjlab-Pick-Place-PiperX      4000 8192 $EXTRA &
run 5 s2_seed2     Mjlab-Pick-Place-PiperX-Cube 4000 8192 $EXTRA --agent.seed 17 &
# A control on whether the smoothness ramp was needed or merely harmless:
# the same curriculum with every stage pinned at its final weight.
run 6 s2_noramp    Mjlab-Pick-Place-PiperX-Cube 4000 8192 $EXTRA \
      --env.curriculum.action-rate-weight.params.stages.0.weight -0.15 \
      --env.curriculum.action-rate-weight.params.stages.1.weight -0.15 \
      --env.curriculum.action-acc-weight.params.stages.0.weight -0.08 \
      --env.curriculum.action-acc-weight.params.stages.1.weight -0.08 &
wait
say "stage 2 finished"
for n in s2_cube_long s2_mid s2_full s2_seed2 s2_noramp; do
  say "  $n: objects_placed=$(placed "$n") grasp_rate=$(grasped "$n")"
done

say "=== STAGE 3: the full distribution, long ==="
run 2 s3_full_long Mjlab-Pick-Place-PiperX 8000 8192 $EXTRA &
run 3 s3_mid_long  Mjlab-Pick-Place-PiperX-Mid 8000 8192 $EXTRA &
wait
say "stage 3 finished"
for n in s3_full_long s3_mid_long; do
  say "  $n: objects_placed=$(placed "$n") grasp_rate=$(grasped "$n")"
done
say "=== CAMPAIGN COMPLETE ==="
