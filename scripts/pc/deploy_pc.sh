#!/usr/bin/env bash
# Deploy a point-cloud bundle on the rig, in the runbook's order, one step per invocation.
#
#   scripts/pc/deploy_pc.sh scene                    # 0. the table against the known-good scene (touches nothing)
#   scripts/pc/deploy_pc.sh joints                   # 1. jointcheck 1..6, five degrees each (the arm MOVES a little)
#   scripts/pc/deploy_pc.sh shadow                   # 2. live D455 + real joint feedback, drives OFF, 30 s
#   scripts/pc/deploy_pc.sh noarm                    # 3. the guarded loop with the live camera and the DRY arm, 20 s
#   scripts/pc/deploy_pc.sh move                     # 4. first motion, 20 s -- e-stop in hand, one object, asks for `move`
#   scripts/pc/deploy_pc.sh review <recording dir>   #    render a session
#
# Every step writes a fresh recordings/<step>_<bundle>_<stamp>/ and prints where.  Nothing here
# lowers a guard: `move` runs run.py with --home-first, --command-rate-scale 0.5,
# --max-joint-speed-fraction 0.6 and the typed word.  Stop conditions (runbook step 5): any hold
# longer than 1 s, a joint-speed trip, the object leaving the sector, 20 s.
#
# Environment: POLICY (default: the C1 bundle), CUT (default 0.004 -- the cut above the calibrated
# plane; the policy trained at 0.006 and the simulator's stress sweep says lower is free and higher
# is not), SECONDS, RATE (command-rate scale), SPEED (max joint-speed fraction), DEVICE.
set -Eeuo pipefail
STEP=${1:?scene | joints | shadow | noarm | move | review <dir>}
POLICY=${POLICY:-hardware/deploy/policies/pc_P1BZ6_C1_20260907T1245}
CUT=${CUT:-0.004}
DEVICE=${DEVICE:-cuda}
RATE=${RATE:-0.5}
# The software speed guard as a fraction of the hardware shell's trip.  0.6 was
# the runbook's first-motion setting; the first two motions (2026-09-07)
# reached 2.0-2.5 rad/s on the wrist joints -- 52-63 % of their 3.93 rad/s
# shell -- because the real wrist overshoots a rate-limited command by ~1.6x,
# and 0.6 ended the second run on it.  0.9 keeps the guard below the shell.
SPEED=${SPEED:-0.9}
# The log writer: zlib on 848x480 RGB + depth at 30 Hz falls behind the camera
# and a full queue ends the run ("recording queue full", 2026-09-07 on the
# second motion).  Uncompressed frames at ~3 MB each keep the writer ahead of
# the camera (90 MB/s to disk; 32 GB free), and a queue of 600 frames (~1.8 GB
# of RAM) rides out any stall.  A run is still never left unlogged.
RECORD_FLAGS=${RECORD_FLAGS:---no-record-compress --record-queue 600}
cd "$(dirname "$0")/../.."
export MUJOCO_GL=disable
export LD_LIBRARY_PATH="${MAMBA_ROOT:-$HOME/micromamba}/envs/mjlab/lib:${LD_LIBRARY_PATH:-}"
M="micromamba run -n mjlab python"
STAMP=$(date +%Y%m%dT%H%M%S)
NAME=$(basename "$POLICY")
[ -f "$POLICY/manifest.json" ] || { echo "no bundle at $POLICY" >&2; exit 2; }
ROUTE=$(python3 -c "import json; print(json.load(open('$POLICY/manifest.json'))['route'])")
echo "bundle $POLICY  route $ROUTE  cut ${CUT} m  device $DEVICE"
case "$STEP" in
  scene)
    $M -m hardware.deploy.scene --like recordings/v4_stereo_try3 ;;
  joints)
    for j in 1 2 3 4 5 6; do
      echo "== joint $j: the arm should move joint $j by five degrees and back"; $M -m hardware.deploy.jointcheck --joint "$j"
    done ;;
  shadow)
    OUT=recordings/pc_shadow_armread_${NAME}_$STAMP
    $M -m hardware.deploy.pc_run --policy "$POLICY" --camera d455 --arm-read --seconds "${SECONDS_RUN:-30}" --record "$OUT"
    echo "read $OUT/summary.json: control_ms.p95 < 20, perception_ms.p95 well under 33, effective_vision_hz ~30, frame_age_s.p95 < 0.10, holds only at the start, action_api_status ok" ;;
  noarm)
    OUT=recordings/pc_noarm_${NAME}_$STAMP
    $M -m hardware.deploy.run --obs pc --policy "$POLICY" --camera d455 --no-arm --policy-device "$DEVICE" \
      --cloud-height-min "$CUT" --seconds "${SECONDS_RUN:-20}" --record "$OUT" $RECORD_FLAGS
    echo "recorded $OUT -- check the perception line: ~30 Hz, 0 frames with too few workspace points, holds only at the start" ;;
  move)
    OUT=recordings/pc_motion_${NAME}_$STAMP
    echo "FIRST MOTION: e-stop in hand, workspace clear, ONE textured object >= 40 mm tall in the sector, table checked with 'scene'."
    echo "run.py will ask for the word 'move'.  Rate $RATE, speed fraction $SPEED, ${SECONDS_RUN:-20} s, --home-first."
    $M -m hardware.deploy.run --obs pc --policy "$POLICY" --camera d455 --policy-device "$DEVICE" \
      --cloud-height-min "$CUT" --home-first --command-rate-scale "$RATE" --max-joint-speed-fraction "$SPEED" \
      --seconds "${SECONDS_RUN:-20}" --record "$OUT" $RECORD_FLAGS
    echo "recorded $OUT; render with: scripts/pc/deploy_pc.sh review $OUT" ;;
  review)
    $M -m hardware.deploy.review "${2:?recording dir}" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
