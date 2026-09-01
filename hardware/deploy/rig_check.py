"""Is the saved calibration still the calibration?  Two minutes, no re-solve.

Re-running ``calibgui`` is an hour and it *moves the answer*: the simulator's
camera pose in ``piper_push.camera`` is the 2026-08-26 D455 hand-eye result,
so every policy trained since is trained for that exact viewpoint.  A fresh
solve that lands 20 mm away does not correct the rig, it invalidates the
policy.  What is actually wanted before a session is the cheaper question --
*has anything moved since* -- and nothing in this directory answered it:
``calibrate.py`` has only ``--collect``/``--solve``/``--preview``, and
``calibgui``'s preflight is a 250 mm "is the board on the gripper at all"
sanity check.

Two stages, and the second one is what makes the first one complete.

**The table plane** costs one depth frame, no board and no motion.  It catches
anything that tilts the camera or changes its height.  It cannot catch a slide
along the table or a rotation about its normal: the plane is unchanged by both.

**One gripped-board pose** closes those two.  The board's pose in the *gripper*
frame is a constant -- that is the whole premise of the hand-eye solve -- so
predicting it through FK and measuring it through the saved extrinsic must
agree.  Any of the six degrees of freedom moving shows up here in millimetres.

A note on the table height, because comparing the wrong pair of numbers is the
easy mistake.  ``rig_d455.json``'s ``table_z`` was fitted from *checkerboard
corners*, whose metric positions are known.  This tool fits *depth*, which
carries the camera's range bias -- measured at -14 mm at 0.7 m on the D405 --
so the two disagree by that bias even when nothing at all has moved.  Height is
therefore compared against a depth-fit baseline this tool writes on its first
run; only the tilt and the normal are compared against the rig, being the parts
a uniform range bias does not move.

(Writing this check found a second reason the two used to disagree, since
fixed: ``calibrate.fit_table`` returned the *centroid* of the visible tabletop
rather than the plane's height at the base origin, which is what ``config.Rig``
documents and what the deployment's table guard subtracts.  On this rig's tilt
and viewing geometry that was 6.1 mm.  ``--solve --table`` and the GUI's live
readout both went through it.)

    # no arm, no board, no motion -- run this one first
    python -m hardware.deploy.rig_check --camera d455 --table-only

    # the full check: grip the board, then three held poses
    python -m hardware.deploy.rig_check --camera d455 --poses 3

Exit status is 0 when every stage passed, 1 when one did not, 2 when the check
could not be run at all.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from . import calibrate, config

HERE = pathlib.Path(__file__).resolve().parent


# --- thresholds -------------------------------------------------------------
#
# Tied to what training randomised, not chosen for looking strict.  The robust
# profile draws the camera over +-30 mm and +-3 deg and the table over +-7 mm
# and +-0.5 deg, so a rig inside the WARN column is inside the distribution the
# policy was trained for, and one past FAIL is a mount that has been knocked.

TABLE_Z_WARN_MM, TABLE_Z_FAIL_MM = 5.0, 10.0
TILT_WARN_DEG, TILT_FAIL_DEG = 0.30, 0.60
BOARD_WARN_MM, BOARD_FAIL_MM = 8.0, 15.0
BOARD_WARN_DEG, BOARD_FAIL_DEG = 1.0, 2.5

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}


def _grade(value: float, warn: float, fail: float) -> str:
  if not np.isfinite(value):
    return FAIL
  return FAIL if value >= fail else (WARN if value >= warn else PASS)


def _worst(*states: str) -> str:
  return max(states, key=lambda s: _RANK[s])


def _line(label: str, text: str, state: str) -> None:
  print(f"  {state:4s}  {label:<26s} {text}")


def _rt(rvec, tvec) -> np.ndarray:
  return calibrate._rt(rvec, tvec)


def _rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
  return calibrate._angle_between(np.asarray(Ra), np.asarray(Rb))


def _fk(kin, joint_pos: np.ndarray) -> np.ndarray:
  """Gripper site pose in the base frame."""
  kin.update(np.asarray(joint_pos, dtype=np.float64))
  T = np.eye(4)
  T[:3, :3] = kin.data.site_xmat[kin.site_id].reshape(3, 3)
  T[:3, 3] = kin.data.site_xpos[kin.site_id]
  return T


def board_in_gripper(records: list[dict], T_base_cam: np.ndarray,
                     gripper_site: str = "grasp_site") -> tuple[np.ndarray, float]:
  """Where the board sits on the gripper, from the recorded session.

  ``rig.json`` does not store it -- only ``solve`` returns it, and only when it
  refined -- so it is recomputed here from the saved poses and the saved
  extrinsic.  The spread is returned with it: it is the same quantity
  ``solve`` reports as the hand-eye residual, and if it does not reproduce the
  residual in the rig file then the pose file and the rig file are not from the
  same session.
  """
  from .proprio import Kinematics

  kin = Kinematics(site_name=gripper_site)
  in_grip = []
  for r in records:
    T_bg = _fk(kin, r["joint_pos"])
    in_grip.append(np.linalg.inv(T_bg) @ T_base_cam @ _rt(r["rvec"], r["tvec"]))
  mean = calibrate._transform_mean(in_grip)
  origins = np.stack([T[:3, 3] for T in in_grip])
  spread_mm = float(np.sqrt(np.mean(
    np.linalg.norm(origins - origins.mean(axis=0), axis=1) ** 2)) * 1000)
  return mean, spread_mm


# --- stage one: the table plane ---------------------------------------------

def check_table(rig, args, baseline: dict | None) -> dict:
  from . import sensor

  print(f"\ntable plane -- {args.frames} depth frames, no board, no motion")
  reader = sensor.Reader(serial=args.serial or rig.serial,
                         backend=args.camera,
                         width=config.D405_WIDTH, height=config.D405_HEIGHT)
  try:
    reader.wait_for_first()
    if rig.serial and reader.serial and str(rig.serial) != str(reader.serial):
      raise RuntimeError(
        f"the rig belongs to camera {rig.serial}, this is {reader.serial}")
    frames, seen = [], set()
    deadline = time.time() + max(5.0, args.frames / 10.0)
    while len(frames) < args.frames and time.time() < deadline:
      f = reader.latest()
      if f is not None and f.index not in seen:
        seen.add(f.index)
        frames.append(np.asarray(f.depth, dtype=np.float64))
      else:
        time.sleep(0.01)
    if len(frames) < 4:
      raise RuntimeError(f"only {len(frames)} frames arrived from the camera")
    stack = np.stack(frames)
    # Median over frames rather than one shot: a third of this sensor's error
    # is temporal, and it is free to remove it before fitting a plane to it.
    depth = np.where((stack > 0).sum(0) >= len(frames) // 2,
                     np.median(np.where(stack > 0, stack, np.nan), axis=0), 0.0)
    depth = np.nan_to_num(depth)
    K = reader.K
  finally:
    reader.close()

  fit = calibrate.fit_table(depth, rig.T_base_cam, K)
  now_z, now_tilt = fit["table_z"], fit["tilt_deg"]
  now_n = np.asarray(fit["normal_base"], dtype=np.float64)

  rig_tilt = rig.table_tilt_deg
  d_tilt = abs(now_tilt - rig_tilt) if rig_tilt is not None else float("nan")
  tilt_state = (_grade(d_tilt, TILT_WARN_DEG, TILT_FAIL_DEG)
                if rig_tilt is not None else WARN)

  if rig.table_normal_base is not None:
    rig_n = np.asarray(rig.table_normal_base, dtype=np.float64)
    d_normal = float(np.degrees(np.arccos(
      np.clip(abs(float(rig_n @ now_n)), -1.0, 1.0))))
    normal_state = _grade(d_normal, TILT_WARN_DEG, TILT_FAIL_DEG)
  else:
    d_normal, normal_state = float("nan"), WARN

  print(f"  fitted from {fit['n_points']} points on the tabletop")
  _line("tilt vs rig",
        f"{now_tilt:.3f} deg now, {rig_tilt:.3f} deg in the rig, "
        f"delta {d_tilt:.3f} deg" if rig_tilt is not None
        else f"{now_tilt:.3f} deg now; the rig records no tilt", tilt_state)
  _line("normal vs rig",
        f"{d_normal:.3f} deg between the plane normals" if np.isfinite(d_normal)
        else "the rig records no normal", normal_state)

  if baseline is None:
    z_state = WARN
    _line("height vs baseline",
          f"{now_z * 1000:+.1f} mm depth-fitted; no baseline yet, recording "
          f"this one (the rig's {rig.table_z * 1000:+.1f} mm is board-fitted "
          "and carries no range bias, so the two are not comparable)", WARN)
    d_z = float("nan")
  else:
    d_z = abs(now_z - baseline["table_z"]) * 1000
    z_state = _grade(d_z, TABLE_Z_WARN_MM, TABLE_Z_FAIL_MM)
    _line("height vs baseline",
          f"{now_z * 1000:+.1f} mm now, {baseline['table_z'] * 1000:+.1f} mm "
          f"at baseline ({baseline.get('stamp', 'unknown')}), "
          f"delta {d_z:.1f} mm", z_state)
  _line("flatness", f"{fit['flatness_mm']:.2f} mm rms about the fitted plane",
        PASS)

  state = _worst(tilt_state, normal_state, z_state)
  print("  the plane is unchanged by a slide along the table or a spin about "
        "its normal; only the board stage sees those.")
  return {"stage": "table", "state": state, "fit": fit,
          "delta_tilt_deg": d_tilt, "delta_normal_deg": d_normal,
          "delta_z_mm": d_z}


# --- stage two: the gripped board -------------------------------------------

def check_board(rig, args) -> dict:
  from . import robot, sensor
  from .proprio import Kinematics

  poses_path = (pathlib.Path(args.poses_file) if args.poses_file
                else HERE / f"calib_poses_{args.camera}.json")
  if not poses_path.exists():
    raise RuntimeError(f"no recorded pose file at {poses_path}")
  board, records = calibrate.load_poses(poses_path)
  T_grip_board, spread_mm = board_in_gripper(records, rig.T_base_cam)
  print(f"\ngripped board -- {len(records)} poses from {poses_path.name}, "
        f"board {board.kind} {board.squares[0]}x{board.squares[1]} "
        f"@ {board.square_m * 1000:.0f} mm")
  against = ("" if rig.residual_mm is None
             else f", against {rig.residual_mm:.2f} mm in the rig")
  print(f"  board-on-gripper reproduced with {spread_mm:.2f} mm spread{against}")

  print(f"\n  Grip the board.  The arm is NOT enabled and will NOT be "
        f"commanded;\n  move it by hand, or leave it where it is.")
  reader = sensor.Reader(serial=args.serial or rig.serial,
                         backend=args.camera, infrared=False,
                         width=args.calib_width, height=args.calib_height,
                         gray_source="left_ir", emitter=args.emitter)
  arm = robot.PiperArm(args.can)
  observations = []
  try:
    reader.wait_for_first()
    arm.connect()
    kin = Kinematics(site_name="grasp_site")
    for i in range(args.poses):
      if input(f"\n  pose {i + 1}/{args.poses}: hold the board in view and "
               "press enter (or 'q' to stop): ").strip().lower() == "q":
        break
      dets, deadline = [], time.time() + 4.0
      seen = set()
      st = arm.read()
      while len(dets) < args.fuse_frames and time.time() < deadline:
        f = reader.latest()
        if f is None or f.index in seen:
          time.sleep(0.01)
          continue
        seen.add(f.index)
        dets.append(calibrate.detect_board(f.gray, reader.K, reader.dist, board))
      fused = calibrate.fuse_detections(dets, reader.K, reader.dist, board,
                                        min_frames=min(8, args.fuse_frames))
      if fused is None:
        found = sum(d is not None for d in dets)
        print(f"        board not found ({found}/{len(dets)} frames detected).  "
              "Not counted; try again with the board fully in view.")
        continue
      T_bg = _fk(kin, np.array([*st.q, st.gripper, -st.gripper]))
      T_cb = _rt(fused["rvec"], fused["tvec"])
      measured = rig.T_base_cam @ T_cb          # through the saved extrinsic
      predicted = T_bg @ T_grip_board           # through forward kinematics
      pos_mm = float(np.linalg.norm(measured[:3, 3] - predicted[:3, 3]) * 1000)
      rot_deg = _rot_deg(measured[:3, :3], predicted[:3, :3])
      print(f"        {fused['n_corners']} corners, "
            f"{fused['reproj_rms_px']:.2f} px rms  ->  "
            f"{pos_mm:.1f} mm, {rot_deg:.2f} deg from where FK says it is")
      observations.append({"T_bg": T_bg, "T_cb": T_cb, "pos_mm": pos_mm,
                           "rot_deg": rot_deg,
                           "n_corners": int(fused["n_corners"]),
                           "reproj_rms_px": float(fused["reproj_rms_px"])})
  finally:
    reader.close()
    # Never disable: DisableArm removes holding torque and the arm can fall.
    try:
      arm.close()
    except Exception:
      pass

  if not observations:
    raise RuntimeError("no pose was measured; nothing to compare")

  pos = np.array([o["pos_mm"] for o in observations])
  rot = np.array([o["rot_deg"] for o in observations])
  pos_state = _grade(float(pos.max()), BOARD_WARN_MM, BOARD_FAIL_MM)
  rot_state = _grade(float(rot.max()), BOARD_WARN_DEG, BOARD_FAIL_DEG)

  print(f"\n  over {len(observations)} pose(s)")
  _line("board position", f"{pos.mean():.1f} mm mean, {pos.max():.1f} mm worst",
        pos_state)
  _line("board orientation",
        f"{rot.mean():.2f} deg mean, {rot.max():.2f} deg worst", rot_state)

  # Three or more poses separate the two ways this can go wrong.  Re-fitting
  # the board-on-gripper transform absorbs a board that was re-clamped in a
  # different place; whatever is left after that is the camera.
  refit_mm = float("nan")
  if len(observations) >= 3:
    in_grip = [np.linalg.inv(o["T_bg"]) @ rig.T_base_cam @ o["T_cb"]
               for o in observations]
    centre = calibrate._transform_mean(in_grip)
    origins = np.stack([T[:3, 3] for T in in_grip])
    refit_mm = float(np.sqrt(np.mean(
      np.linalg.norm(origins - centre[:3, 3], axis=1) ** 2)) * 1000)
    moved_mm = float(np.linalg.norm(centre[:3, 3] - T_grip_board[:3, 3]) * 1000)
    _line("after re-fitting the mount",
          f"{refit_mm:.1f} mm residual, mount offset by {moved_mm:.1f} mm",
          _grade(refit_mm, BOARD_WARN_MM, BOARD_FAIL_MM))
    if pos_state != PASS and refit_mm < BOARD_WARN_MM:
      print("  Read that as the board sitting differently on the gripper, not "
            "the camera having moved:\n  the poses agree with each other, they "
            "just agree about a different mount.")

  return {"stage": "board", "state": _worst(pos_state, rot_state),
          "n_poses": len(observations),
          "position_mm": {"mean": float(pos.mean()), "worst": float(pos.max())},
          "orientation_deg": {"mean": float(rot.mean()),
                              "worst": float(rot.max())},
          "refit_residual_mm": refit_mm,
          "recorded_residual_mm": rig.residual_mm,
          "pose_file_spread_mm": spread_mm}


def main() -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--camera", choices=("d405", "d455"), default="d455")
  p.add_argument("--rig-file", default=None,
                 help="defaults to rig.json for the D405 and rig_<cam>.json "
                      "otherwise, exactly as run.py resolves it")
  p.add_argument("--serial", default=None)
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--poses", type=int, default=3,
                 help="gripped-board poses to measure.  One is enough to see a "
                      "camera that moved; three separate that from a board "
                      "that was re-clamped")
  p.add_argument("--table-only", action="store_true",
                 help="the depth-plane stage alone: no board, no arm, no CAN")
  p.add_argument("--frames", type=int, default=30,
                 help="depth frames to median before fitting the plane")
  p.add_argument("--fuse-frames", type=int, default=12,
                 help="detections fused per board pose, as calibration does")
  p.add_argument("--calib-width", type=int, default=1280)
  p.add_argument("--calib-height", type=int, default=720)
  p.add_argument("--emitter", choices=("on", "off"), default=None)
  p.add_argument("--pose-file", dest="poses_file", default=None)
  p.add_argument("--baseline", default=None,
                 help="where the depth-fit table baseline lives; defaults "
                      "beside the rig")
  p.add_argument("--update-baseline", action="store_true",
                 help="overwrite the table baseline with this run.  Do it "
                      "after a deliberate re-calibration, never to make a "
                      "failing check pass")
  p.add_argument("--json", default=None, help="write the full result here")
  a = p.parse_args()
  a.poses = max(1, int(a.poses))

  suffix = "" if a.camera == "d405" else f"_{a.camera}"
  rig_path = (pathlib.Path(a.rig_file) if a.rig_file
              else pathlib.Path(config.RIG_FILE).with_name(f"rig{suffix}.json"))
  if not rig_path.exists():
    print(f"no calibration at {rig_path}; there is nothing to check", file=sys.stderr)
    return 2
  rig = config.Rig.load(rig_path)

  baseline_path = pathlib.Path(a.baseline) if a.baseline else \
      rig_path.with_name(f"{rig_path.stem}_table_baseline.json")
  baseline = (json.loads(baseline_path.read_text())
              if baseline_path.exists() else None)

  residual = ("" if rig.residual_mm is None
              else f", hand-eye residual {rig.residual_mm:.2f} mm")
  print(f"checking {rig_path.name}: camera {rig.serial}{residual}")

  results, state = [], PASS
  try:
    table = check_table(rig, a, baseline)
    results.append(table)
    state = _worst(state, table["state"])
    if baseline is None or a.update_baseline:
      baseline_path.write_text(json.dumps({
        "table_z": table["fit"]["table_z"],
        "tilt_deg": table["fit"]["tilt_deg"],
        "normal_base": table["fit"]["normal_base"],
        "flatness_mm": table["fit"]["flatness_mm"],
        "rig": rig_path.name,
        "serial": rig.serial,
        "stamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "depth plane fit, hardware.deploy.rig_check",
      }, indent=2) + "\n")
      print(f"  wrote the depth-fit baseline to {baseline_path.name}")
  except Exception as e:
    print(f"  FAIL  table stage could not run: {e}", file=sys.stderr)
    results.append({"stage": "table", "state": FAIL, "error": repr(e)})
    state = FAIL

  if not a.table_only:
    try:
      board = check_board(rig, a)
      results.append(board)
      state = _worst(state, board["state"])
    except Exception as e:
      print(f"  FAIL  board stage could not run: {e}", file=sys.stderr)
      results.append({"stage": "board", "state": FAIL, "error": repr(e)})
      state = FAIL

  print(f"\n{state}: {rig_path.name}")
  if state == PASS:
    print("Use the existing calibration.  Re-solving would move the camera "
          "pose the training run is pinned to.")
  elif state == WARN:
    print("Inside the randomised envelope but drifting.  Deployable; worth a "
          "look at the mount before a long session.")
  else:
    print("Something has moved.  Re-calibrate with calibgui -- and note that "
          "the policy currently training was trained for the OLD pose, so a "
          "new solve means re-checking scripts/rig_to_sim.py before trusting "
          "any hardware result.")

  if a.json:
    pathlib.Path(a.json).write_text(json.dumps({
      "rig": str(rig_path), "serial": rig.serial, "state": state,
      "stamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
      "stages": [{k: v for k, v in r.items() if k != "fit"} | (
        {"fit": r["fit"]} if "fit" in r else {}) for r in results],
    }, indent=2, default=float) + "\n")
    print(f"wrote {a.json}")

  return 0 if state in (PASS, WARN) else 1


if __name__ == "__main__":
  raise SystemExit(main())
