"""What the calibration measured, expressed as things the simulator gets wrong.

``hardware/deploy/calibrate.py`` produces ``rig.json``: where the camera really
is, and where the table really is.  The deployment reads it and resamples the
real camera into the simulator's one, so the *image* is right whatever the
mount did.  What resampling cannot fix is that a camera somewhere else sees
round corners differently, and that a table at a different height is a
different table.  Those are the residual gaps, and this is what states them.

Three questions, in the order they matter.

**Is the rig inside what the policy was trained against?**  Every camera axis
is compared to ``TRAINED_RANGE`` below, which is the record of
what training randomised.  Inside it, the calibration is done and nothing needs
to change.  Outside it, retraining is not optional -- the policy is being asked
to work at a viewpoint it has never seen, and it will do so confidently.

**If it is not, what changes?**  Two ways to close a gap and they are not
equivalent.  Move the *nominal* (``camera.CAMERA_POS``) and the simulator is
rebuilt around the real rig, which is right when the mount is where it is going
to stay.  Widen the *envelope* (``camera.POS_JITTER_M``) and the policy learns
to tolerate the range, which costs some performance and is right when the mount
is not trusted to stay put.  Both are printed; the choice is not this script's.

**What can the simulator not express at all?**  Three things, and they are
named rather than rounded away: the camera's roll, its lateral offset, and the
table's tilt.  the simulator has no term for any of them, so a rig
with a degree of roll cannot be replayed in simulation -- and a number that
cannot be simulated is the one worth reading, because it is the one no amount
of training has covered.

    python scripts/rig_to_sim.py
    python scripts/rig_to_sim.py --rig hardware/deploy/rig.json --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# mjlab first; see hardware/deploy/config.py for why the order is not free.
import mjlab.tasks  # noqa: F401,E402

from piper_push import camera as sim_camera  # noqa: E402

# What the camera-pose randomisation covered in training, per axis, in the
# axis's own units.  ``None`` means the quantity was not modelled at all.
TRAINED_RANGE: dict[str, tuple[float, float] | None] = {
  "cam_pitch_deg": (-2.0, 2.0),
  "cam_yaw_deg": (-2.0, 2.0),
  "cam_pos_x_m": (-0.02, 0.02),
  "cam_pos_z_m": (-0.02, 0.02),
}

from hardware.deploy import config  # noqa: E402

TABLE_Z_TOLERANCE_M = 0.005
"""How far the real table may sit from the simulator's zero before it is worth
saying so.  Five millimetres because the objects are 24-90 mm tall and the
grasp is planned in the base frame: a table 10 mm low is a gripper closing
10 mm above where the policy learned the object's waist to be, which is a
fifth of the shortest object."""

TABLE_TILT_TOLERANCE_DEG = 0.5
"""A tilt this small moves the far corner of the 0.9 m workspace by 8 mm,
which is inside the depth sensor's own noise at this range.  Beyond it the
table is a plane the simulator does not have."""


def _basis(fwd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The right and up the simulator builds a camera frame from.

  Copied in behaviour, not in code, from ``randomize_camera_pose_offset``:
  right from the forward and the world's up, then up from those two.  It is
  reproduced here because the pitch and yaw reported below are only meaningful
  in the frame they will be applied in.
  """
  right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
  right /= np.linalg.norm(right)
  return right, np.cross(right, fwd)


def decompose(T_base_cam: np.ndarray) -> dict:
  """The measured extrinsic as the simulator's own mismatch axes.

  ``T_base_cam`` is OpenCV: columns right, down, forward.  The simulator's
  camera is MuJoCo's and the conversion lives in ``config``; what is needed
  here is only the optical axis, which is the third column in both.

  Order matters and follows the simulator's.  ``randomize_camera_pose_offset``
  applies the position offset *first* and re-derives the aim direction from the
  offset position, then turns that by pitch and yaw.  So pitch and yaw are
  measured against the direction from the offset position to the fixed aim
  point, not against the nominal optical axis -- measuring them against the
  nominal would double-count the rotation the translation already caused.
  """
  pos = np.asarray(T_base_cam)[:3, 3]
  fwd = np.asarray(T_base_cam)[:3, 2] / np.linalg.norm(T_base_cam[:3, 2])
  nominal = np.asarray(sim_camera.CAMERA_POS, dtype=np.float64)
  aim = np.asarray(sim_camera.CAMERA_AIM, dtype=np.float64)
  d = pos - nominal

  # What the simulator can move: x and z.  There is no y term.
  posed = nominal + np.array([d[0], 0.0, d[2]])
  ref = aim - posed
  ref /= np.linalg.norm(ref)
  right, up = _basis(ref)

  along = float(fwd @ ref)
  pitch = math.degrees(math.atan2(float(fwd @ up), along))
  yaw = math.degrees(math.atan2(float(fwd @ np.cross(up, ref)), along))

  # Roll: the camera's own right axis against the roll-free one the simulator
  # would build for the same optical axis.
  r_free, u_free = _basis(fwd)
  measured_right = np.asarray(T_base_cam)[:3, 0]
  roll = math.degrees(math.atan2(float(measured_right @ u_free),
                                 float(measured_right @ r_free)))
  # The frame is built with +y down in OpenCV and up in the simulator's, so a
  # roll-free mount reads as 180 degrees here; report the departure from that.
  roll = (roll + 180.0) % 360.0 - 180.0
  if abs(roll) > 90.0:
    roll = roll - math.copysign(180.0, roll)

  return {
    "cam_pos_x_m": float(d[0]),
    "cam_pos_y_m": float(d[1]),
    "cam_pos_z_m": float(d[2]),
    "cam_pitch_deg": pitch,
    "cam_yaw_deg": yaw,
    "cam_roll_deg": roll,
    "offset_mm": float(np.linalg.norm(d) * 1000),
    "range_m": float(np.linalg.norm(pos - aim)),
  }


def _verdict(name: str, value: float, unit: str) -> tuple[str, str]:
  """Where one measured axis stands against what training covered."""
  rng = TRAINED_RANGE.get(name)
  if rng is None:
    return "NOT MODELLED", "the simulator has no term for this"
  lo, hi = rng
  if lo <= value <= hi:
    frac = abs(value) / max(hi, -lo, 1e-9)
    return "in distribution", f"{100 * frac:.0f}% of the trained range"
  return "OUT of distribution", f"trained {lo:+g} to {hi:+g} {unit}"


def report(rig: config.Rig, dec: dict) -> dict:
  out = {"decomposition": dec, "flags": []}
  say = out["flags"].append

  print("the camera")
  print(f"  measured        {np.round(rig.T_base_cam[:3, 3], 4).tolist()} m, "
        f"{dec['offset_mm']:.1f} mm from the nominal mount")
  if rig.residual_mm is not None:
    print(f"  hand-eye residual  {rig.residual_mm:.2f} mm")
  print()

  rows = [("cam_pos_x_m", dec["cam_pos_x_m"], "m", 1000, "mm"),
          ("cam_pos_z_m", dec["cam_pos_z_m"], "m", 1000, "mm"),
          ("cam_pitch_deg", dec["cam_pitch_deg"], "deg", 1, "deg"),
          ("cam_yaw_deg", dec["cam_yaw_deg"], "deg", 1, "deg")]
  print(f"  {'axis':16s} {'measured':>12s}   status")
  for name, v, unit, k, shown in rows:
    status, why = _verdict(name, v, unit)
    print(f"  {name:16s} {v * k:+9.2f} {shown:3s}   {status} ({why})")
    if status.startswith("OUT"):
      say(f"{name} is outside what training randomised")

  # The two the parameterisation cannot carry, and the table's tilt below.
  print()
  print(f"  {'cam_pos_y_m':16s} {dec['cam_pos_y_m'] * 1000:+9.2f} mm    "
        "NOT MODELLED (the camera jitter offsets x and z only)")
  print(f"  {'cam_roll_deg':16s} {dec['cam_roll_deg']:+9.2f} deg   "
        "NOT MODELLED (the frame is rebuilt from world up, so roll is zero "
        "by construction)")
  if abs(dec["cam_pos_y_m"]) > sim_camera.POS_JITTER_M:
    say(f"the mount is {dec['cam_pos_y_m'] * 1000:+.0f} mm off in y, which "
        "neither the training jitter nor the mismatch parameterisation covers")
  if abs(dec["cam_roll_deg"]) > math.degrees(sim_camera.ROT_JITTER_RAD):
    say(f"the mount is rolled {dec['cam_roll_deg']:+.1f} deg, which is more "
        "than the free rotation jitter and cannot be replayed in simulation")

  print("\nthe table")
  dz = rig.table_z - config.TABLE_Z_M
  print(f"  height          {rig.table_z * 1000:+.1f} mm in the base frame, "
        f"{dz * 1000:+.1f} mm from the simulator's")
  if abs(dz) > TABLE_Z_TOLERANCE_M:
    say(f"the table is {dz * 1000:+.0f} mm from where the simulator puts it "
        "and the simulator does not randomise this")
  if rig.table_tilt_deg is not None:
    print(f"  tilt            {rig.table_tilt_deg:.2f} deg   "
          "NOT MODELLED (the simulator's table is exactly level)")
    if rig.table_tilt_deg > TABLE_TILT_TOLERANCE_DEG:
      span = 0.9
      say(f"the table is tilted {rig.table_tilt_deg:.1f} deg, which is "
          f"{1000 * span * math.tan(math.radians(rig.table_tilt_deg)):.0f} mm "
          f"across the workspace, and nothing in training varies it")
  if rig.table_flatness_mm is not None:
    print(f"  flatness        {rig.table_flatness_mm:.1f} mm rms about the "
          "fitted plane")
  return out


def prescribe(rig: config.Rig, dec: dict, flags: list[str]) -> None:
  print("\n" + "-" * 74)
  if not flags:
    print("\nEverything measured is inside what the policy was trained "
          "against.\nNo simulator change is needed before deploying; the "
          "resampling in\nhardware/deploy/rectify.py covers the rest.")
  else:
    print("\nOutside the training distribution:\n")
    for f in flags:
      print(f"  - {f}")

  print("\nTo replay this rig in simulation -- evaluate the trained policy at "
        "the\nviewpoint the robot actually has, before touching the robot:\n")
  print("  python scripts/accept_s1.py --checkpoint <ckpt> \\")
  print(f"      --cam-pos-x-m {dec['cam_pos_x_m']:+.4f} "
        f"--cam-pos-z-m {dec['cam_pos_z_m']:+.4f} \\")
  print(f"      --cam-pitch-deg {dec['cam_pitch_deg']:+.3f} "
        f"--cam-yaw-deg {dec['cam_yaw_deg']:+.3f}")
  print("\n  Run it against the nominal too.  The difference between the two "
        "is what\n  this calibration costs, in the only unit that matters, "
        "and it is the number\n  that says whether retraining is worth it.")

  if not flags:
    return

  print("\nTo retrain against it, two options and they are different "
        "decisions:\n")
  pos = np.asarray(rig.T_base_cam)[:3, 3]
  aim = np.asarray(sim_camera.CAMERA_AIM, dtype=np.float64)
  fwd = np.asarray(rig.T_base_cam)[:3, 2]
  fwd = fwd / np.linalg.norm(fwd)
  # Where the measured optical axis meets the measured table: the aim point
  # that reproduces this camera, since the simulator's camera is built by
  # looking from a position at a point.
  denom = fwd[2]
  new_aim = (pos + fwd * ((rig.table_z - pos[2]) / denom)
             if denom < -1e-3 else aim)
  print("  1. Move the nominal, and the simulator is rebuilt around this "
        "mount.\n     In src/piper_push/camera.py:\n")
  print(f"       CAMERA_POS = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
  print(f"       CAMERA_AIM = ({new_aim[0]:.3f}, {new_aim[1]:.3f}, "
        f"{new_aim[2]:.3f})")
  print("\n     Right when the mount is where it is staying.  Everything "
        "downstream --\n     hardware/deploy/config.py, the rig sheet, the "
        "deployment's own nominal --\n     imports these, so there is nothing "
        "else to edit.  It does mean every\n     policy trained before the "
        "edit was trained for a different camera.")
  if abs(dec["cam_roll_deg"]) > 0.5:
    print(f"\n     It does not carry the {dec['cam_roll_deg']:+.1f} deg of "
          "roll.  CAMERA_AIM names a\n     point, and a point cannot say "
          "which way up the camera is; that part stays\n     in the envelope "
          "below however the nominal is set.")
  if abs(dec["cam_roll_deg"]) > 0.5:
    print(f"\n     It does not carry the {dec['cam_roll_deg']:+.1f} deg of "
          "roll: CAMERA_AIM names a point, and\n     a point cannot express "
          "which way up the camera is.  That much stays in the\n     "
          "envelope below whatever else is decided.")

  need_pos = max(abs(dec["cam_pos_x_m"]), abs(dec["cam_pos_y_m"]),
                 abs(dec["cam_pos_z_m"]))
  need_rot = max(abs(dec["cam_pitch_deg"]), abs(dec["cam_yaw_deg"]),
                 abs(dec["cam_roll_deg"]))
  print("\n  2. Widen the envelope, and the policy learns to tolerate the "
        "range rather\n     than this point.  In src/piper_push/camera.py:\n")
  print(f"       POS_JITTER_M = {max(sim_camera.POS_JITTER_M, need_pos * 1.5):.3f}"
        f"        # was {sim_camera.POS_JITTER_M:.3f}")
  print(f"       ROT_JITTER_RAD = math.radians("
        f"{max(math.degrees(sim_camera.ROT_JITTER_RAD), need_rot * 1.5):.1f})"
        f"  # was {math.degrees(sim_camera.ROT_JITTER_RAD):.1f} deg")
  print("\n     1.5x the measured error, so the calibration sits inside the "
        "range rather\n     than on its edge.  This costs performance -- a "
        "policy that must work over\n     a wider envelope is worse at the "
        "middle of it -- and it is the right trade\n     when the mount can "
        "be knocked.")

  if rig.table_tilt_deg and rig.table_tilt_deg > TABLE_TILT_TOLERANCE_DEG:
    print("\n  3. The table's tilt has no option, because nothing in "
          "src/piper_push\n     varies it.  Adding it is an event that "
          "rotates the table body a\n     fraction of a degree per "
          "environment; until then it is an unmodelled\n     "
          f"{rig.table_tilt_deg:.1f} deg, and the honest thing is to shim the "
          "table.")


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--rig", default=str(config.RIG_FILE))
  p.add_argument("--json", default=None, help="write the decomposition")
  a = p.parse_args()

  path = pathlib.Path(a.rig)
  if not path.exists():
    print(f"no calibration at {path}.\n\nRun it first:\n"
          "  python -m hardware.deploy.calibrate --preview   # check the board\n"
          "  python -m hardware.deploy.calibrate --collect   # 8+ poses\n"
          "  python -m hardware.deploy.calibrate --solve")
    return 1
  rig = config.Rig.load(path)
  dec = decompose(rig.T_base_cam)
  out = report(rig, dec)
  prescribe(rig, dec, out["flags"])
  if a.json:
    pathlib.Path(a.json).write_text(json.dumps(out, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
