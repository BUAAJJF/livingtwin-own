#!/usr/bin/env bash
# An overnight campaign for the pick-and-place task.
#
# The first campaign answered its question and the answer was that the reward
# ladder had a hole in it: four of five configurations spent 3000 iterations
# learning to touch the object and never close on it, because closing was a pay
# cut.  That is fixed.  What is open now is the shape of the SCHEDULE -- how
# fast the guidance should fade, whether the bin ramp helps or just gives the
# policy somewhere to sit, and whether the placement bonus is big enough.
#
# Stage 1 runs those in parallel.  Stage 2 takes the winner across the shape
# curriculum.  Stage 3 gives the best two shape levels a seed each.
# Every decision is written to driver.log with the number behind it.
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

# Mean of the last ten logged values, not the last one.  The previous campaign
# picked its winner off a single window in which one run happened to spike.
tailmean() {
  local v
  v=$(grep -oP "$2:\s*\K[0-9.]+" "$OUT/$1.log" 2>/dev/null | tail -10 \
      | awk '{s+=$1; n++} END{if(n) printf "%.4f", s/n; else print "0"}')
  echo "${v:-0}"
}
placed()   { tailmean "$1" "Metrics/pick/objects_placed"; }
grasped()  { tailmean "$1" "Metrics/pick/grasp_rate"; }
attempts() { tailmean "$1" "Metrics/pick/grasp_attempts"; }

SLOW=""
for k in reach-decay pads-decay holding-decay lift-decay transport-decay in-bin-decay; do
  SLOW="$SLOW --env.curriculum.$k.params.stages.1.step 38400 --env.curriculum.$k.params.stages.2.step 76800"
done
NOBRIDGE="--env.rewards.object_in_bin.weight 0.0"
for s in 0 1 2; do
  NOBRIDGE="$NOBRIDGE --env.curriculum.in-bin-decay.params.stages.$s.weight 0.0"
done

say "=== STAGE 1: five schedules on the fixed cube, 3000 iterations, 4096 envs ==="
run 2 r1_base      Mjlab-Pick-Place-PiperX-Cube 3000 4096 &
run 3 r1_slow      Mjlab-Pick-Place-PiperX-Cube 3000 4096 $SLOW &
run 4 r1_nobridge  Mjlab-Pick-Place-PiperX-Cube 3000 4096 $NOBRIDGE &
run 5 r1_bigplace  Mjlab-Pick-Place-PiperX-Cube 3000 4096 --env.rewards.place.weight 600.0 &
run 6 r1_explore   Mjlab-Pick-Place-PiperX-Cube 3000 4096 --agent.algorithm.entropy-coef 0.012 &
wait
say "stage 1 finished"

BEST=""; BEST_P="-1"
for n in r1_base r1_slow r1_nobridge r1_bigplace r1_explore; do
  p=$(placed "$n")
  say "  $n: placed=$p grasp_rate=$(grasped "$n") attempts=$(attempts "$n")"
  if awk -v a="$p" -v b="$BEST_P" 'BEGIN{exit !(a>b)}'; then BEST="$n"; BEST_P="$p"; fi
done
say "winner: $BEST (placed=$BEST_P)"

EXTRA=""
case "$BEST" in
  r1_slow)     EXTRA="$SLOW" ;;
  r1_nobridge) EXTRA="$NOBRIDGE" ;;
  r1_bigplace) EXTRA="--env.rewards.place.weight 600.0" ;;
  r1_explore)  EXTRA="--agent.algorithm.entropy-coef 0.012" ;;
esac
say "carrying forward: ${EXTRA:-<defaults>}"
printf '%s\n' "$BEST" > "$OUT/winner_name.txt"
printf '%s\n' "$EXTRA" > "$OUT/winner_flags.txt"

say "=== STAGE 2: the winner, longer and across the shape curriculum ==="
run 2 r2_cube  Mjlab-Pick-Place-PiperX-Cube 4000 8192 $EXTRA &
run 3 r2_mid   Mjlab-Pick-Place-PiperX-Mid  4000 8192 $EXTRA &
run 4 r2_full  Mjlab-Pick-Place-PiperX      4000 8192 $EXTRA &
run 5 r2_seed2 Mjlab-Pick-Place-PiperX-Cube 4000 8192 $EXTRA --agent.seed 17 &
# A control on whether the smoothness ramp was needed or merely harmless.
run 6 r2_noramp Mjlab-Pick-Place-PiperX-Cube 4000 8192 $EXTRA \
      --env.curriculum.action-rate-weight.params.stages.0.weight -0.15 \
      --env.curriculum.action-rate-weight.params.stages.1.weight -0.15 \
      --env.curriculum.action-acc-weight.params.stages.0.weight -0.08 \
      --env.curriculum.action-acc-weight.params.stages.1.weight -0.08 &
wait
say "stage 2 finished"
for n in r2_cube r2_mid r2_full r2_seed2 r2_noramp; do
  say "  $n: placed=$(placed "$n") grasp_rate=$(grasped "$n") attempts=$(attempts "$n")"
done

say "=== STAGE 3: both shape levels, twice each, long ==="
run 2 r3_full_a Mjlab-Pick-Place-PiperX     3200 8192 $EXTRA --agent.seed 1 &
run 3 r3_full_b Mjlab-Pick-Place-PiperX     3200 8192 $EXTRA --agent.seed 23 &
run 4 r3_mid_a  Mjlab-Pick-Place-PiperX-Mid 3200 8192 $EXTRA --agent.seed 1 &
run 5 r3_mid_b  Mjlab-Pick-Place-PiperX-Mid 3200 8192 $EXTRA --agent.seed 23 &
wait
say "stage 3 finished"
for n in r3_full_a r3_full_b r3_mid_a r3_mid_b; do
  say "  $n: placed=$(placed "$n") grasp_rate=$(grasped "$n") attempts=$(attempts "$n")"
done
say "=== CAMPAIGN COMPLETE ==="
