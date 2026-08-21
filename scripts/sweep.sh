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

FAST=""
for k in reach-decay pads-decay holding-decay lift-decay transport-decay in-bin-decay; do
  FAST="$FAST --env.curriculum.$k.params.stages.1.step 9600 --env.curriculum.$k.params.stages.2.step 25600"
done
LOWHOLD="--env.curriculum.holding-decay.params.stages.2.weight 0.05"
STAGE1="q1_base q1_hotprog q1_bigplace q1_fastdecay q1_lowhold"

say "=== STAGE 1: five schedules on the fixed cube, 3000 iterations, 4096 envs ==="
run 2 q1_base      Mjlab-Pick-Place-PiperX-Cube 3000 4096 &
run 3 q1_hotprog   Mjlab-Pick-Place-PiperX-Cube 3000 4096 \
      --env.rewards.transport_progress.weight 240.0 &
run 4 q1_bigplace  Mjlab-Pick-Place-PiperX-Cube 3000 4096 --env.rewards.place.weight 600.0 &
run 5 q1_fastdecay Mjlab-Pick-Place-PiperX-Cube 3000 4096 $FAST &
run 6 q1_lowhold   Mjlab-Pick-Place-PiperX-Cube 3000 4096 $LOWHOLD &
wait
say "stage 1 finished"

BEST=""; BEST_S="-1"
for n in $STAGE1; do
  p=$(placed "$n"); g=$(grasped "$n")
  s=$(awk -v a="$p" -v b="$g" 'BEGIN{printf "%.6f", a*1000 + b}')
  say "  $n: placed=$p grasp_rate=$g attempts=$(attempts "$n") score=$s"
  if awk -v a="$s" -v b="$BEST_S" 'BEGIN{exit !(a>b)}'; then BEST="$n"; BEST_S="$s"; fi
done
say "winner: $BEST (score=$BEST_S)"

EXTRA=""
case "$BEST" in
  q1_hotprog)   EXTRA="--env.rewards.transport_progress.weight 240.0" ;;
  q1_bigplace)  EXTRA="--env.rewards.place.weight 600.0" ;;
  q1_fastdecay) EXTRA="$FAST" ;;
  q1_lowhold)   EXTRA="$LOWHOLD" ;;
esac
say "carrying forward: ${EXTRA:-<defaults>}"
printf '%s\n' "$BEST" > "$OUT/winner_name.txt"
printf '%s\n' "$EXTRA" > "$OUT/winner_flags.txt"

say "=== STAGE 2: the winner, longer and across the shape curriculum ==="
run 2 q2_cube  Mjlab-Pick-Place-PiperX-Cube 3500 8192 $EXTRA &
run 3 q2_mid   Mjlab-Pick-Place-PiperX-Mid  3500 8192 $EXTRA &
run 4 q2_full  Mjlab-Pick-Place-PiperX      3500 8192 $EXTRA &
run 5 q2_seed2 Mjlab-Pick-Place-PiperX-Cube 3500 8192 $EXTRA --agent.seed 17 &
# A control on whether the smoothness ramp was needed or merely harmless.
run 6 q2_noramp Mjlab-Pick-Place-PiperX-Cube 4000 8192 $EXTRA \
      --env.curriculum.action-rate-weight.params.stages.0.weight -0.15 \
      --env.curriculum.action-rate-weight.params.stages.1.weight -0.15 \
      --env.curriculum.action-acc-weight.params.stages.0.weight -0.08 \
      --env.curriculum.action-acc-weight.params.stages.1.weight -0.08 &
wait
say "stage 2 finished"
for n in q2_cube q2_mid q2_full q2_seed2 q2_noramp; do
  say "  $n: placed=$(placed "$n") grasp_rate=$(grasped "$n") attempts=$(attempts "$n")"
done

say "=== CAMPAIGN COMPLETE ==="
