#!/usr/bin/env bash
# From a trained checkpoint to a policy that is allowed near the arm.
#
#   scripts/bringup_d455.sh <checkpoint.pt> [name]
#   REMOTE=shen-teacher:/home/yunfan/work/.../model_5999.pt scripts/bringup_d455.sh
#
# Everything here is already in hardware/deploy/README.md as eight numbered
# steps to be typed by hand.  Typing them by hand is how the export bug
# happened: exported on the Distill task, the graph disagreed with the trained
# policy by 4.35 on actions of magnitude 1-5 -- a different network -- and
# check_export.py printed OK, because it had compared that network to itself.
# The only signal was that the actions were near-constant at +-0.09, which
# takes knowing the right magnitude to notice.  So the task is pinned here, the
# comparison that would have caught it is not optional, and no stage may be
# skipped by being forgotten.
#
# Nothing in this script touches the arm or the camera.  It ends by printing
# the two commands that do, in the order they should be run.
set -Eeuo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
CONDA_ENV=${CONDA_ENV:-${MJLAB_ENV:-livingtwin}}
source "$ROOT/scripts/conda_env.sh"

# The deployment task.  NOT the distillation task, and not "whatever the
# checkpoint came from" -- a distillation checkpoint is converted to an actor
# first and then exported on the task it will be deployed on.
TASK=${TASK:-Mjlab-Pick-Place-PiperX-Vision}
# The task whose observation layout the robot rebuilds by hand.  Exported and
# compared rather than assumed: the vision variant drops `grasped` and appends
# `squeeze`, which puts it after `actions` rather than where `grasped` was, and
# a policy fed a correct vector in the wrong order does not crash.
OBS_TASK=${OBS_TASK:-$TASK}
DEVICE=${DEVICE:-cuda:0}
CAMERA=${CAMERA:-d455}
REPLAY=${REPLAY:-}

CKPT=${1:-${REMOTE:-}}
[ -n "$CKPT" ] || { sed -n '2,6p' "$0" >&2; exit 2; }

NAME=${2:-}
if [ -z "$NAME" ]; then
  NAME=$(basename "$CKPT" .pt)
  NAME="d455_v3_${NAME}"
fi
DEST=hardware/deploy/policies/$NAME
STAMP=$(date -Is)

M=(python -u)

say() { printf '\n=== %s\n' "$*"; }
fail() { printf '\nNO-GO: %s\n' "$*" >&2; exit 1; }

mkdir -p "$DEST"
LOG=$DEST/bringup.log
exec > >(tee -a "$LOG") 2>&1
say "bring-up $NAME at $STAMP"

# --- 0. fetch ---------------------------------------------------------------

case "$CKPT" in
  *:*)
    say "0. fetching $CKPT"
    scp -q "$CKPT" "$DEST/checkpoint.pt" || fail "could not fetch $CKPT"
    SRC=$CKPT
    CKPT=$DEST/checkpoint.pt
    ;;
  *)
    [ -f "$CKPT" ] || fail "no checkpoint at $CKPT"
    SRC=$(readlink -f "$CKPT")
    cp -n "$CKPT" "$DEST/checkpoint.pt" 2>/dev/null || true
    ;;
esac
SHA=$(sha256sum "$CKPT" | cut -d' ' -f1)
say "checkpoint $SRC"
echo "    sha256 $SHA"

# --- 1. what the actor expects ---------------------------------------------
#
# Exported to a temporary file and diffed against the one the robot reads.  A
# silent reorder between the task the policy was trained on and the spec on
# disk is the failure this catches, and it is invisible at run time because
# proprio.py only refuses when a term is *missing*, never when it moved.

say "1. observation spec, from $OBS_TASK"
"${M[@]}" scripts/export_obs_spec.py --task "$OBS_TASK" --device "$DEVICE" \
  --out "$DEST/obs_spec.json" || fail "could not export the observation spec"
if ! diff -q hardware/deploy/obs_spec.json "$DEST/obs_spec.json" >/dev/null; then
  diff -u hardware/deploy/obs_spec.json "$DEST/obs_spec.json" || true
  fail "the deployed obs_spec.json is not what $OBS_TASK produces.  Copy the \
new one into hardware/deploy/obs_spec.json only after reading that diff -- a \
reordered vector is not a crash, it is confident nonsense."
fi
echo "    identical to hardware/deploy/obs_spec.json"

# --- 2. actor, then export on the deployment task ---------------------------

say "2. converting to an actor checkpoint"
"${M[@]}" scripts/student_to_actor.py "$CKPT" "$DEST/actor.pt" \
  || fail "student_to_actor failed"
# A distillation checkpoint is rewritten; a fine-tuned one already has
# actor_state_dict and the converter deliberately writes nothing at all.  Both
# are normal, and which happened is decided by what is on disk afterwards.
ACTOR=$DEST/actor.pt
if [ ! -f "$ACTOR" ]; then
  ACTOR=$CKPT
  echo "    already an actor checkpoint; exporting $ACTOR unchanged"
fi

say "3. exporting on $TASK"
"${M[@]}" scripts/check_export.py "$ACTOR" --task "$TASK" \
  --device "$DEVICE" --out "$DEST" || fail "export failed"
[ -f "$DEST/policy.onnx" ] || fail "no policy.onnx in $DEST"

# --- 4. the graph against the trained policy, through the deployment path ----

say "4. selftest against the simulator, with --checkpoint"
"${M[@]}" -m hardware.deploy.selftest --policy "$DEST" \
  --checkpoint "$ACTOR" --device "$DEVICE" \
  || fail "selftest failed -- read the table above before changing anything"

# --- 5. the loop, with no hardware ------------------------------------------

say "5. dry run"
"${M[@]}" -m hardware.deploy.run --policy "$DEST" --camera "$CAMERA" \
  --dry-run --seconds 10 || fail "dry run failed"

if [ -z "$REPLAY" ]; then
  REPLAY=$(find recordings -maxdepth 1 -type d -name "*${CAMERA}*" \
    -exec test -f '{}/meta.json' \; -print 2>/dev/null | sort | tail -n 1)
fi
if [ -n "$REPLAY" ] && [ -f "$REPLAY/meta.json" ]; then
  say "6. replaying $REPLAY -- real frames, and the only measure of what the loop costs"
  "${M[@]}" -m hardware.deploy.run --policy "$DEST" --camera "$CAMERA" \
    --replay "$REPLAY" --allow-nominal --seconds 20 \
    --record "$DEST/replay" || fail "replay failed"
else
  echo "    no recorded ${CAMERA} session found; skipping the replay stage."
  echo "    THIS IS A GAP: the dry run does not touch the segmenter, so"
  echo "    nothing here has measured the perception thread on real frames."
fi

# --- provenance -------------------------------------------------------------

cat >"$DEST/bringup.json" <<JSON
{
  "name": "$NAME",
  "stamp": "$STAMP",
  "source_checkpoint": "$SRC",
  "checkpoint_sha256": "$SHA",
  "exported_actor": "$ACTOR",
  "export_task": "$TASK",
  "obs_spec_task": "$OBS_TASK",
  "camera": "$CAMERA",
  "replay_session": "${REPLAY:-null}",
  "repo_commit": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "repo_dirty": $(git diff --quiet 2>/dev/null && echo false || echo true)
}
JSON

say "GO -- $DEST"
cat <<NEXT

Nothing above has moved anything.  The next two commands are the ones that
matter, in this order.

  1. Is the calibration still the calibration?  Two minutes, no re-solve.
     A fresh solve is NOT the safe default: piper_push.camera is pinned to the
     2026-08-26 D455 hand-eye result and this policy was trained for it.

       python -m hardware.deploy.rig_check \\
         --camera $CAMERA --table-only

  2. The dress rehearsal.  Real camera, real perception, real policy, real
     guard evaluation -- and DryRunArm, so the arm does not move at all.  This
     is where the height floor gets chosen, not on the arm.

       python -m hardware.deploy.run \\
         --policy $DEST --camera $CAMERA --device cuda \\
         --no-arm --seconds 60 --record recordings/${NAME}_noarm_try1 \\
         --min-grasp-height 0.05 --guard-mode hold

     Read guard_holds in its run.json.  Zero means the floor is not being
     tested.  Hundreds means the policy wants to be below it, and putting the
     arm under that is a decision, not a formality.

Only then the arm, with the e-stop in hand and a fresh --record directory.
NEXT
