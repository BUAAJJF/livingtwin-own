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
# is not), SECONDS_RUN, RATE (command-rate scale), SPEED (max joint-speed fraction), DEVICE,
# DEPTH (sensor | stereo -- where the depth map comes from, see below).
set -Eeuo pipefail
STEP=${1:?scene | joints | shadow | noarm | move | review <dir>}
POLICY=${POLICY:-hardware/deploy/policies/pc_P1BZ6_C1_20260907T1245}
CUT=${CUT:-0.004}
RATE=${RATE:-0.5}
# DEPTH=stereo runs Fast-FoundationStereo (TensorRT, 14 ms/frame) on the raw imagers instead of
# the D455's own depth.  Every point-cloud result so far is on `sensor`; the 09-01 bench on this
# unit gave the model 73% fill against the camera's 89% and 3.7 mm agreement, so it is an A/B,
# not an upgrade.  `noarm` and `move` take it; the pc_run shadow does not (no imager path there).
# A stereo session archives the camera's depth, both imagers and the depth the policy saw
# (`policy_depth`), ~3.6 MB per frame uncompressed -- compare them afterwards with
# scripts/pc/compare_depth_sources.py <recording>.
DEPTH=${DEPTH:-sensor}
# Where the policy runs.  Every sensor-depth session so far ran it on cuda (3.7
# ms a step).  With stereo it CANNOT share the GPU: the engine's kernels
# serialise with onnxruntime's and the control step became 35 ms (25 overruns,
# run ended, 2026-09-07 23:04; reproduced offline: 38 ms concurrent, 18 ms even
# on a separate CUDA stream).  On the CPU the same policy takes 1.9 ms whatever
# the GPU is doing, so stereo runs it there and refuses anything else.
case "$DEPTH" in
  sensor) DEPTH_FLAGS=""; DEVICE=${DEVICE:-cuda} ;;
  stereo)
    DEPTH_FLAGS="--depth-source stereo"; DEVICE=${DEVICE:-cpu}
    [ "$DEVICE" = cpu ] || { echo "DEPTH=stereo needs DEVICE=cpu: the policy on the GPU stalls behind the stereo engine (35 ms control steps)" >&2; exit 2; } ;;
  *) echo "DEPTH must be sensor or stereo, not '$DEPTH'" >&2; exit 2 ;;
esac
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
[ "$DEPTH" = sensor ] || NAME="${NAME}_${DEPTH}"
[ -f "$POLICY/manifest.json" ] || { echo "no bundle at $POLICY" >&2; exit 2; }
ROUTE=$(python3 -c "import json; print(json.load(open('$POLICY/manifest.json'))['route'])")
echo "bundle $POLICY  route $ROUTE  cut ${CUT} m  device $DEVICE  depth $DEPTH"
# Uncompressed frames fill a disk quickly: 2 MB each on sensor depth, 3.6 MB with the imagers
# and the computed depth, at 30 Hz.  A run that dies on a full disk is an unlogged run, so the
# space is checked before the camera opens (a 2x margin over the nominal length).
check_disk() {
  local secs=$1 per_frame_mb=2
  [ "$DEPTH" = sensor ] || per_frame_mb=4
  local need_mb=$(( secs * 30 * per_frame_mb * 2 ))
  local free_mb; free_mb=$(df -Pm recordings | awk 'NR==2 {print $4}')
  if [ "$free_mb" -lt "$need_mb" ]; then
    echo "only ${free_mb} MB free under recordings/, a ${secs} s $DEPTH session wants ~${need_mb} MB (2x margin); free space first (du -sh recordings/*)" >&2
    exit 3
  fi
  echo "disk: ${free_mb} MB free, ~$(( need_mb / 2 )) MB expected for ${secs} s"
}
case "$STEP" in
  scene)
    $M -m hardware.deploy.scene --like recordings/v4_stereo_try3 ;;
  joints)
    for j in 1 2 3 4 5 6; do
      echo "== joint $j: the arm should move joint $j by five degrees and back"; $M -m hardware.deploy.jointcheck --joint "$j"
    done ;;
  shadow)
    [ "$DEPTH" = sensor ] || { echo "the pc_run shadow has no imager path; use DEPTH=$DEPTH with noarm (camera + dry arm, same guards)" >&2; exit 2; }
    OUT=recordings/pc_shadow_armread_${NAME}_$STAMP
    $M -m hardware.deploy.pc_run --policy "$POLICY" --camera d455 --arm-read --seconds "${SECONDS_RUN:-30}" --record "$OUT"
    echo "read $OUT/summary.json: control_ms.p95 < 20, perception_ms.p95 well under 33, effective_vision_hz ~30, frame_age_s.p95 < 0.10, holds only at the start, action_api_status ok" ;;
  noarm)
    OUT=recordings/pc_noarm_${NAME}_$STAMP
    check_disk "${SECONDS_RUN:-20}"
    $M -m hardware.deploy.run --obs pc --policy "$POLICY" --camera d455 --no-arm --policy-device "$DEVICE" \
      --cloud-height-min "$CUT" --seconds "${SECONDS_RUN:-20}" --record "$OUT" $RECORD_FLAGS $DEPTH_FLAGS
    echo "recorded $OUT -- check the perception line: ~30 Hz, 0 frames with too few workspace points, holds only at the start" ;;
  move)
    OUT=recordings/pc_motion_${NAME}_$STAMP
    echo "FIRST MOTION: e-stop in hand, workspace clear, ONE textured object >= 40 mm tall in the sector, table checked with 'scene'."
    echo "run.py will ask for the word 'move'.  Rate $RATE, speed fraction $SPEED, ${SECONDS_RUN:-20} s, --home-first, depth $DEPTH."
    check_disk "${SECONDS_RUN:-20}"
    $M -m hardware.deploy.run --obs pc --policy "$POLICY" --camera d455 --policy-device "$DEVICE" \
      --cloud-height-min "$CUT" --home-first --command-rate-scale "$RATE" --max-joint-speed-fraction "$SPEED" \
      --seconds "${SECONDS_RUN:-20}" --record "$OUT" $RECORD_FLAGS $DEPTH_FLAGS
    echo "recorded $OUT; render with: scripts/pc/deploy_pc.sh review $OUT" ;;
  review)
    $M -m hardware.deploy.review "${2:?recording dir}" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
