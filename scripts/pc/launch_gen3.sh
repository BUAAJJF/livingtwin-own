#!/usr/bin/env bash
# Third round, under the object-astray termination (commit ad49e04 and after):
#
#   GPU 4  R0  P1BZ, the standard recipe (1500 + 800, 36 s)     -- the baseline of this environment
#          + the v11 teacher under the ruler in this environment (evaluation only)
#          + C0  E0's fine-tune continued for 800 more iterations here (evaluation after)
#   GPU 5  R1  MASK, the depth + target-mask observation, identical recipe -- the pre-registered control
#   GPU 6  R2  P1BZ, budget x 2 (3000 + 1600)
#   GPU 7  R3  P1BZ, 12 s episodes (1500 + 800)
#
#   bash scripts/pc/launch_gen3.sh          # sync named paths, refuse unless GPUs 4-7 are all idle, launch
set -Eeuo pipefail
HOST=${HOST:-shen-teacher}
RROOT=${RROOT:-/home/yunfan/work/piper-push/LivingTwin}
TEACHER=${TEACHER:-logs/rsl_rl/piperx_pick_place_robust_cold/2026-09-05_18-42-17_v11_nosight_36s_teacher/model_9399.pt}
TEACHER_SHA=${TEACHER_SHA:-d685b54821593f724714c4969bda6d7ca8a11e7fe6aa85f4c5a5c4555e19367a}
E0_FINAL=${E0_FINAL:-$(cat results/pc/routes/pc_gen2_P1BZ_20260906T1020/final_checkpoint.txt)}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M)}
SEED=${SEED:-42}
COMMIT=$(git rev-parse --short HEAD)
DIRTY=$(git status --porcelain | { grep -v '^??' || true; } | wc -l)
[ "$DIRTY" = 0 ] || { echo "commit first: $DIRTY tracked files modified" >&2; exit 2; }

echo "== sync code to $HOST ($COMMIT)"
rsync -az --delete src/piper_push/ "$HOST:$RROOT/src/piper_push/"
rsync -az scripts/ "$HOST:$RROOT/scripts/"
rsync -az tests/ "$HOST:$RROOT/tests/"
ssh "$HOST" "cd $RROOT && grep -q ObjectAstray src/piper_push/tasks/pick_place/mdp.py" || { echo "sync did not land" >&2; exit 2; }
ssh "$HOST" "cd $RROOT && sha256sum $TEACHER" | grep -q "^$TEACHER_SHA " || { echo "teacher sha256 mismatch" >&2; exit 2; }

echo "== GPUs"
FREE=$(ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while IFS=", " read -r i m; do
  case "$i" in 4|5|6|7) ;; *) continue ;; esac
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i "$i" | wc -l)
  if [ "$m" -lt 1024 ] && [ "$busy" = 0 ]; then printf "%s " "$i"; fi
done')
echo "free among 4-7: [$FREE]"
set -- $FREE
[ "$#" -ge 4 ] || { echo "need 4 free GPUs among 4-7, have $#: not launching" >&2; exit 3; }
if [ "${DRY:-0}" = 1 ]; then echo "DRY: would launch on $*"; exit 0; fi
G0=$1; G1=$2; G2=$3; G3=$4

launch_route() {   # tag route gpu distill finetune episode
  ssh "$HOST" "cd $RROOT && mkdir -p results/pc/routes && \
    TAG=$1 ROUTE=$2 GPU=$3 TEACHER=$TEACHER COMMIT=$COMMIT SEED=$SEED VISION_ENVS=512 \
    DISTILL_ITERS=$4 FINETUNE_ITERS=$5 EPISODE_S=$6 \
    nohup setsid bash scripts/pc/run_route.sh >results/pc/routes/$1.watcher.log 2>&1 < /dev/null & \
    sleep 2; echo started $1 on GPU $3; tail -n 1 results/pc/routes/$1.watcher.log"
}
echo "== teacher reference (this environment) on GPU $G0"
ssh "$HOST" "cd $RROOT && mkdir -p results/pc/gen3 && OUT=results/pc/gen3/teacher_v11_astray_$STAMP TEACHER=$TEACHER GPU=$G0 COMMIT=$COMMIT \
  nohup setsid bash scripts/pc/teacher_ref.sh >/dev/null 2>&1 < /dev/null & sleep 1; echo started"
echo "== C0: continue E0's fine-tune on GPU $G0"
ssh "$HOST" "cd $RROOT && OUT=results/pc/gen3/c0_E0_continue_$STAMP RESUME=$E0_FINAL TASK=Mjlab-Pick-Place-PiperX-PC-P1BZ-Vision ITERS=1600 GPU=$G0 COMMIT=$COMMIT \
  nohup setsid bash scripts/pc/continue_finetune.sh >/dev/null 2>&1 < /dev/null & sleep 1; echo started"
launch_route "pc_gen3_R0_P1BZ_$STAMP"     P1BZ "$G0" 1500 800  36.0
launch_route "pc_gen3_R1_MASK_$STAMP"     MASK "$G1" 1500 800  36.0
launch_route "pc_gen3_R2_P1BZ_x2_$STAMP"  P1BZ "$G2" 3000 1600 36.0
launch_route "pc_gen3_R3_P1BZ_12s_$STAMP" P1BZ "$G3" 1500 800  12.0
echo "== done; pull with scripts/pull_results.sh pc/routes and scripts/pull_results.sh pc/gen3"
