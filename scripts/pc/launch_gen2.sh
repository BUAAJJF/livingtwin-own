#!/usr/bin/env bash
# Second generation, experiments E0 (P1BZ) and E2 (P1BT), on the training host.
#
#   bash scripts/pc/launch_gen2.sh            # sync, check GPUs 4-7, launch both routes
#   DRY=1 bash scripts/pc/launch_gen2.sh      # everything but the launch
#
# Sync is rsync of named paths (never git on the remote).  A GPU is taken only
# if it has no compute process and under 1 GiB in use; with fewer than two free
# among 4-7 nothing is launched and the script says so.  Both routes get the
# same teacher (sha256 checked), seed, budget and episode length; the tag
# carries one UTC stamp so the pair is found together.
set -Eeuo pipefail
HOST=${HOST:-shen-teacher}
RROOT=${RROOT:-/home/yunfan/work/piper-push/LivingTwin}
TEACHER=${TEACHER:-logs/rsl_rl/piperx_pick_place_robust_cold/2026-09-05_18-42-17_v11_nosight_36s_teacher/model_9399.pt}
TEACHER_SHA=${TEACHER_SHA:-d685b54821593f724714c4969bda6d7ca8a11e7fe6aa85f4c5a5c4555e19367a}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M)}
ROUTES=${ROUTES:-"P1BZ P1BT"}
DISTILL_ITERS=${DISTILL_ITERS:-1500}
FINETUNE_ITERS=${FINETUNE_ITERS:-800}
VISION_ENVS=${VISION_ENVS:-512}
SEED=${SEED:-42}
COMMIT=$(git rev-parse --short HEAD)
DIRTY=$(git status --porcelain | grep -v '^??' | wc -l)
[ "$DIRTY" = 0 ] || { echo "commit first: $DIRTY tracked files modified" >&2; exit 2; }

echo "== sync code to $HOST ($COMMIT)"
rsync -az --delete src/piper_push/ "$HOST:$RROOT/src/piper_push/"
rsync -az scripts/ "$HOST:$RROOT/scripts/"
rsync -az tests/ "$HOST:$RROOT/tests/"

echo "== teacher"
ssh "$HOST" "cd $RROOT && sha256sum $TEACHER" | grep -q "^$TEACHER_SHA " || { echo "teacher sha256 mismatch on $HOST" >&2; exit 2; }

echo "== GPUs"
FREE=$(ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while IFS=", " read -r i m; do
  case "$i" in 4|5|6|7) ;; *) continue ;; esac
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i "$i" | wc -l)
  if [ "$m" -lt 1024 ] && [ "$busy" = 0 ]; then printf "%s " "$i"; fi
done')
echo "free among 4-7: [$FREE]"
set -- $FREE
N=0; for r in $ROUTES; do N=$((N + 1)); done
if [ "$#" -lt "$N" ]; then
  echo "need $N free GPUs among 4-7, have $#: not launching (never preempt another job)" >&2
  exit 3
fi
if [ "${DRY:-0}" = 1 ]; then echo "DRY: would launch $ROUTES on GPUs $*"; exit 0; fi

# The v11 teacher under the initiation ruler (36 s x 3 seeds, 180 s, the bin-settle
# diagnostic), so the routes' long runs are read against the teacher's own horizon decay.
# Evaluation only, 256 envs; it shares the first route's GPU.
TREF="results/pc/gen2/teacher_v11_${STAMP}"
ssh "$HOST" "cd $RROOT && OUT=$TREF TEACHER=$TEACHER GPU=$1 COMMIT=$COMMIT \
  nohup setsid bash scripts/pc/teacher_ref.sh >/dev/null 2>&1 < /dev/null & sleep 1; echo teacher reference started on GPU $1 into $TREF"

for r in $ROUTES; do
  gpu=$1; shift
  tag="pc_gen2_${r}_${STAMP}"
  echo "== launch $r on GPU $gpu as $tag"
  ssh "$HOST" "cd $RROOT && mkdir -p results/pc/routes && \
    TAG=$tag ROUTE=$r GPU=$gpu TEACHER=$TEACHER COMMIT=$COMMIT SEED=$SEED VISION_ENVS=$VISION_ENVS \
    DISTILL_ITERS=$DISTILL_ITERS FINETUNE_ITERS=$FINETUNE_ITERS EPISODE_S=36.0 \
    nohup setsid bash scripts/pc/run_route.sh >results/pc/routes/$tag.watcher.log 2>&1 < /dev/null & \
    sleep 2; echo started; tail -n 2 results/pc/routes/$tag.watcher.log"
done
echo "== pull later with: scripts/pull_results.sh pc/routes  (or rsync the two tag dirs)"
