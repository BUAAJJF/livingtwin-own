#!/usr/bin/env bash
# Everything a new student has to answer before it is allowed near the arm.
#
#   bash scripts/accept_student.sh <student.pt> [<teacher.pt>] [OUT]
#
# Four questions, in the order they can invalidate each other:
#
#   1. endurance   does it still work at the end of the episode it started?
#                  This is the one that was missing all campaign, and the one
#                  that turned "the student decays" into "its teacher decays".
#                  A mean throughput cannot answer it.
#   2. occlusion   does it keep its own gripper out of the sight line WHILE
#                  ENGAGED?  Conditioned on the hand being within 150 mm,
#                  because a policy that stops reaching scores well on the
#                  unconditioned number for the wrong reason.
#   3. perception  what can the DEPLOYED stack actually see under it -- the
#                  depth segmenter alone, and with SAM2.1 carrying the target.
#   4. frames      the per-frame page.  Every aggregate this project has
#                  produced has hidden the structure that mattered.
set -Eeuo pipefail
ST=${1:?student checkpoint}
TE=${2:-}
OUT=${3:-results/student_accept/$(basename "${ST%.pt}")}
DEV=${DEV:-cuda:0}
SEEDS=${SEEDS:-"11 33 44 55 66"}
# Students distilled before 2026-09-05 were trained under the unbounded action
# convention and need TASK=Mjlab-Pick-Place-PiperX-Distill-Robust-V1; the
# default id raises at the first step for them.
TASK=${TASK:-Mjlab-Pick-Place-PiperX-Distill-Robust}
# The sensor to evaluate under.  'measured' is the one the student trained
# with (piper_push.evalcfg); 'clean' is the old default and says nothing about
# the robot.
SENSOR=${SENSOR:-measured}
# The domain the student was distilled under.  robust_cfg reads its knobs
# (TARGET_VISIBLE_*, TARGET_GAP_SCALE, OBS_LATENCY_PROBS, SMOOTH_SCALE, ...)
# from the environment at import time and nothing in the checkpoint records
# them, so a run script that distils under one setting has to leave them in
# a domain.env next to the checkpoint, and this
# script has to export them, or the student is measured in the wrong domain
# (v8s_sam2 was, for two days).  Anything already exported wins.
DOMAIN_ENV=${DOMAIN_ENV:-$(dirname "$ST")/domain.env}
if [ -f "$DOMAIN_ENV" ]; then
  while IFS='=' read -r k v; do
    case "$k" in ''|'#'*) continue ;; esac
    if [ -z "${!k:-}" ]; then export "$k=$v"; fi
  done <"$DOMAIN_ENV"
  echo "domain: $(grep -v '^#' "$DOMAIN_ENV" | tr '\n' ' ')  (from $DOMAIN_ENV)"
else
  echo "domain: no $DOMAIN_ENV; evaluating under the defaults of $TASK"
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CONDA_ENV=${CONDA_ENV:-${MJLAB_ENV:-livingtwin}}
source "$ROOT/scripts/conda_env.sh"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$OUT"
say() { printf '\n===== %s\n' "$*"; }
PY="python -u"

say "1. endurance (no reset, which is what the arm does)"
$PY scripts/eval_endurance.py --checkpoint "$ST" \
  --task "$TASK" --num-envs ${ENVS:-128} --steps ${STEPS:-1200} --device "$DEV" --seed 101 \
  --sensor "$SENSOR" \
  --out "$OUT/endurance.json" 2>&1 | grep -aE "^placed/min|^early |^of |^survivors|^final jaw"
# The control.  If resetting inside the training horizon restores the rate,
# the decay is the policy leaving its distribution rather than the task
# getting harder -- that distinction cost a session to establish once.
$PY scripts/eval_endurance.py --checkpoint "$ST" \
  --task "$TASK" --num-envs ${ENVS:-128} --steps ${STEPS:-1200} --reset-every 300 --device "$DEV" \
  --seed 101 --sensor "$SENSOR" --out "$OUT/endurance_reset.json" 2>&1 | grep -aE "^early "

say "2. occlusion while engaged (300 steps: before any collapse)"
$PY scripts/eval_occlusion.py --checkpoint "$ST" \
  --task "$TASK" --num-envs ${ENVS:-128} --steps 300 --device "$DEV" --seed 101 \
  --sensor "$SENSOR" --out "$OUT/occlusion.json" >/dev/null 2>&1 || true
$PY -c "
import json; d=json.load(open('$OUT/occlusion.json')); e=d['engaged']
print('engaged blocked %.1f%%  visible %.3f  engaged frames %.0f%%  placed/min %.1f'
      % (100*e['blocked_rate'], e['visible_mean'],
         100*d['reach_m']['engaged_fraction'], d['placed_per_min']))" || true

if [ -n "$TE" ]; then
  say "3. what the deployed perception stack sees under the TEACHER's behaviour"
  # Driven by the state teacher, because sim_perception_check needs a policy
  # the state task can run; the student imitates it, so this is the domain the
  # student will deploy into.
  for s in $SEEDS; do
    $PY scripts/sim_perception_check.py --steps 300 \
      --policy "$TE" --sam --seed "$s" --device "$DEV" \
      --out "$OUT/perc_seed$s.json" >/dev/null 2>&1 || true
  done
  $PY -c "
import glob, json, numpy as np
rows = [json.load(open(f)) for f in sorted(glob.glob('$OUT/perc_seed*.json'))]
for ph in ('approach','holding'):
  for v in ('depth','deploy','sam'):
    xs=[r['by_phase'][ph][v] for r in rows if ph in r.get('by_phase',{})]
    if xs: print('%-9s %-7s detected %5.1f%%  IoU %.3f'
                 % (ph, v, 100*np.mean([x['detected'] for x in xs]),
                    np.mean([x['iou_mean'] for x in xs])))" || true
fi

say "4. the per-frame page"
$PY scripts/sight_viewer.py \
  --checkpoint "student=$ST@$TASK" ${TE:+--checkpoint "teacher=$TE"} \
  --steps 300 --seed 101 --device "$DEV" --out "$OUT/frames.html" 2>&1 | tail -3

say "wrote $OUT"
