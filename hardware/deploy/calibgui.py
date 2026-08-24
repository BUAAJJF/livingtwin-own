#!/usr/bin/env python3
"""Eye-to-hand calibration, driven from a browser instead of from a prompt.

    python -m hardware.deploy.calibgui      # then open http://127.0.0.1:8771

``calibrate.py --collect`` works and asks the operator to hold four things in
their head at once: whether the board is detected, whether the arm has stopped
moving, whether this pose is different enough from the last one to be worth
recording, and whether the set as a whole has enough *rotation* in it yet.
Only the first is visible, and it is the least important.  The last is the one
that decides: hand-eye is determined by rotation, the solver refuses under 30
degrees of spread, and there is no way to know you are short until you stop
collecting and it says so.

So this shows all four, live, and puts the record button behind three of them.

**Stillness is a gate, not a warning.**  A board photographed while the arm is
settling gives a pose that is confidently wrong, and it is wrong in a way that
survives every downstream check except the residual -- where it turns up as
"one bad pose" with no indication of which of the thirty it was.  The button is
disabled until the view has been still for the whole ring.

**Novelty is a gate too.**  Thirty poses of the same viewpoint constrain
exactly as much as one.  A pose that is within a few degrees and a few
centimetres of one already recorded is refused, with the reason shown, which
turns "collect more poses" into a thing the operator can do deliberately
instead of a number to grind out.

**The rotation span is drawn, not reported.**  Each recorded pose puts a dot on
a disc at the direction the board is facing; the live pose is a ring.  Spreading
the dots is the whole task, and a picture of where they are not is worth more
than a number that says 24 degrees.

Nothing here re-implements the calibration.  Detection, the solve, the residual
and the table fit are all ``calibrate.py``'s, called directly, so the GUI and
the CLI cannot drift apart -- and ``calib_poses.json`` is written in the same
format by both, so a session started in one can be finished in the other.

After five manual seed poses, the same solver is allowed a weaker 15-degree
gate to produce navigation-only guidance.  It predicts nearby board poses in
the D405 grayscale image and lets the operator confirm one slow automatic
move at a time.  That rough result is never saveable; the final solve keeps the
normal 30-degree gate.
"""

from __future__ import annotations

# Importing the deployment model loads MuJoCo, mjlab and Torch and takes about
# 30 seconds on this workstation.  Say so before those imports begin; without
# this line the terminal is completely silent and a healthy first start looks
# hung.
if __name__ == "__main__":
  print("calibgui: loading deployment model (about 30 s on first start) ...",
        flush=True)

import argparse
import dataclasses
import json
import math
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import mujoco
import numpy as np

from . import calibrate, config, proprio
from piper_push import robot as sim_robot

HERE = Path(__file__).resolve().parent

RING = 8
"""Frames the stillness test looks back over.  At ~15 Hz that is half a second
-- long enough that a settling arm has not finished, short enough that holding
a pose deliberately does not feel like waiting."""

STILL_MM = 6.0
STILL_DEG = 1.0
"""How far the board's pose may wander across the ring and still count as still.

Set from what a *stationary* board actually does on this rig, which is not the
bench's 0.71 mm -- that was absolute pose error against a known target, a
different quantity.  Measured here instead: ten consecutive frames of an
untouched board at 0.69 m reported ranges spanning 693-696 mm, so the
frame-to-frame spread along the view axis alone is about 3 mm.  These are twice
that, which leaves the gate closed for anything a hand is doing and open for a
board that has been let go of.

The first pass at this file used 1.5 mm and 0.35 deg, reasoning from the bench
number, and the gate never went green -- the record button would have been
permanently disabled with no indication why.  The live numbers are therefore
shown in the button's own explanation, so a threshold that is wrong for some
other rig is diagnosable by looking at it rather than by reading this comment.
Override with ``--still-mm`` and ``--still-deg``."""

NOVEL_MM = 25.0
NOVEL_DEG = 8.0
"""A new pose must differ from every recorded one by at least one of these.
Rotation is what hand-eye is determined by, so the angular threshold is the
one that matters; the translation one is there so that pure reaches at a fixed
orientation are not rejected outright."""

GUIDANCE_MIN_POSES = 5
GUIDANCE_MIN_ROT_SPAN_DEG = 15.0
"""The deliberately weaker gate for an extrinsic used only to keep the board
in frame while choosing the next pose.  It is never offered for saving: the
real solve still requires ``calibrate.MIN_POSES`` and 30 degrees.  Five poses
and 15 degrees are enough to say which side of the image a small wrist motion
will move towards, which is all the automatic collector asks of it."""

AUTO_SPEED_RAD_S = 0.22
AUTO_RATE_HZ = 30.0
AUTO_TRACKING_ERROR_RAD = 0.45
"""Conservative joint-space motion for calibration.  The page asks before
every move; these constants then keep that confirmed move slow and abort if
feedback falls far behind the streamed target."""


def _ang(Ra: np.ndarray, Rb: np.ndarray) -> float:
  return calibrate._angle_between(Ra, Rb)


def _plausible_camera_transform(T: np.ndarray) -> bool:
  """Basic physical validity, without assuming the simulator's mounting side.

  The real rig may legitimately put the camera across the table and rotated
  90 degrees from the training scene.  A rough solve is guidance for relative
  nearby moves, so consistency and a finite table-scale camera distance are
  the relevant checks; proximity to ``sim_camera_extrinsic`` is not.
  """
  T = np.asarray(T, dtype=np.float64)
  if T.shape != (4, 4) or not np.isfinite(T).all():
    return False
  R, p = T[:3, :3], T[:3, 3]
  return bool(0.15 < np.linalg.norm(p) < 2.0
              and -0.30 < p[2] < 2.0
              and np.allclose(R @ R.T, np.eye(3), atol=1e-3)
              and np.linalg.det(R) > 0.99)


def _joint_trajectory(q0: np.ndarray, q1: np.ndarray,
                      speed_rad_s: float = AUTO_SPEED_RAD_S,
                      rate_hz: float = AUTO_RATE_HZ) -> np.ndarray:
  """A rest-to-rest joint path whose peak speed is bounded.

  ``3 u^2 - 2 u^3`` peaks at 1.5 times its average speed, hence the 1.5 in
  the duration.  The first row is the measured position so enabling the arm
  is immediately followed by a hold at exactly where it already is.
  """
  a = np.asarray(q0, dtype=np.float64).reshape(6)
  b = np.asarray(q1, dtype=np.float64).reshape(6)
  distance = float(np.max(np.abs(b - a)))
  duration = max(0.8, 1.5 * distance / max(float(speed_rad_s), 1e-3))
  n = max(2, int(math.ceil(duration * float(rate_hz))) + 1)
  u = np.linspace(0.0, 1.0, n)
  blend = 3.0 * u ** 2 - 2.0 * u ** 3
  return a[None, :] + blend[:, None] * (b - a)[None, :]


class NextPosePlanner:
  """Choose a nearby joint pose that is novel and keeps the target visible.

  The board-to-gripper transform is recovered from the current detection and
  the rough camera extrinsic.  Candidate joint poses are then run through the
  same MuJoCo kinematics as deployment, projected with the D405 intrinsics,
  and rejected if the board would leave the grayscale image, cross a joint
  safety margin, or introduce a new self-collision.
  """

  def __init__(self, board: calibrate.Board, K: np.ndarray,
               image_size: tuple[int, int]):
    self.board = board
    self.K = np.asarray(K, dtype=np.float64)
    self.width, self.height = (int(image_size[0]), int(image_size[1]))
    self.kin = proprio.Kinematics()
    self.lo = np.array([sim_robot.SAFE_TARGET_CLIP[f"joint{i}"][0]
                        for i in range(1, 7)], dtype=np.float64) + 0.04
    self.hi = np.array([sim_robot.SAFE_TARGET_CLIP[f"joint{i}"][1]
                        for i in range(1, 7)], dtype=np.float64) - 0.04

  def _fk(self, q7: np.ndarray, forward: bool = False) -> np.ndarray:
    q7 = np.asarray(q7, dtype=np.float64).reshape(7)
    self.kin.update(np.array([*q7[:6], q7[6], -q7[6]]))
    if forward:
      mujoco.mj_forward(self.kin.model, self.kin.data)
    T = np.eye(4)
    T[:3, :3] = self.kin.data.site_xmat[self.kin.site_id].reshape(3, 3)
    T[:3, 3] = self.kin.data.site_xpos[self.kin.site_id]
    return T

  def _contacts(self, q7: np.ndarray) -> set[tuple[int, int]]:
    self._fk(q7, forward=True)
    return {tuple(sorted((int(self.kin.data.contact[i].geom1),
                          int(self.kin.data.contact[i].geom2))))
            for i in range(self.kin.data.ncon)}

  def _collision_free(self, q0: np.ndarray, q1: np.ndarray,
                      allowed: set[tuple[int, int]]) -> bool:
    for u in np.linspace(0.0, 1.0, 9)[1:]:
      q = np.asarray(q0) + u * (np.asarray(q1) - np.asarray(q0))
      if not self._contacts(q).issubset(allowed):
        return False
    return True

  def _outline(self) -> np.ndarray:
    nx, ny = self.board.squares
    if self.board.kind == "charuco":
      w, h = nx * self.board.square_m, ny * self.board.square_m
    else:
      w, h = ((nx - 1) * self.board.square_m,
              (ny - 1) * self.board.square_m)
    return np.array([[0, 0, 0, 1], [w, 0, 0, 1],
                     [w, h, 0, 1], [0, h, 0, 1]], dtype=np.float64)

  def _project(self, T_cb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    xyz = (T_cb @ self._outline().T).T[:, :3]
    if np.any(xyz[:, 2] <= 0.15):
      return None
    uvw = (self.K @ xyz.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    centre = uv.mean(axis=0)
    return uv, centre

  def _candidates(self, q: np.ndarray, seed: int) -> list[np.ndarray]:
    out = []
    # Deliberate single- and two-axis wrist tilts make early coverage
    # predictable; random combinations fill the view-constrained gaps.
    for amount in np.radians((10.0, 16.0, 22.0)):
      for j in (3, 4, 5):
        for sign in (-1.0, 1.0):
          d = np.zeros(6)
          d[j] = sign * amount
          out.append(q[:6] + d)
      for s4 in (-1.0, 1.0):
        for s5 in (-1.0, 1.0):
          d = np.zeros(6)
          d[3], d[4] = s4 * amount, s5 * amount
          out.append(q[:6] + d)
    rng = np.random.default_rng(seed)
    scale = np.array([0.12, 0.10, 0.10, 0.25, 0.25, 0.30])
    cap = np.array([0.22, 0.18, 0.18, 0.38, 0.38, 0.42])
    random = np.clip(rng.normal(size=(700, 6)) * scale, -cap, cap)
    out.extend(q[:6] + d for d in random)
    return [np.clip(x, self.lo, self.hi) for x in out]

  def plan(self, T_base_cam: np.ndarray, T_cam_board: np.ndarray,
           q7: np.ndarray, records: list[dict]) -> dict:
    q7 = np.asarray(q7, dtype=np.float64).reshape(7)
    T_bg_now = self._fk(q7)
    T_grip_board = (np.linalg.inv(T_bg_now) @ np.asarray(T_base_cam)
                    @ np.asarray(T_cam_board))
    T_cam_base = np.linalg.inv(T_base_cam)

    recorded_R, recorded_p, recorded_n = [], [], []
    for r in records:
      rq = np.asarray(r["joint_pos"], dtype=np.float64)
      T_bg = self._fk(np.array([*rq[:6], rq[6]]))
      recorded_R.append(T_bg[:3, :3])
      recorded_p.append(T_bg[:3, 3])
      T_cb = calibrate._rt(r["rvec"], r["tvec"])
      n = T_cb[:3, :3] @ np.array([0.0, 0.0, 1.0])
      recorded_n.append(n / np.linalg.norm(n))

    current_contacts = self._contacts(q7)
    options = []
    margin = 34.0
    image_centre = np.array([self.width / 2.0, self.height / 2.0])
    for qc in self._candidates(q7, seed=7919 + len(records) * 104729):
      motion = float(np.linalg.norm(qc - q7[:6]))
      if motion < np.radians(7.0):
        continue
      c7 = np.array([*qc, q7[6]])
      T_bg = self._fk(c7)
      T_cb = T_cam_base @ T_bg @ T_grip_board
      projected = self._project(T_cb)
      if projected is None:
        continue
      uv, centre = projected
      if (uv[:, 0].min() < margin or uv[:, 0].max() > self.width - margin
          or uv[:, 1].min() < margin or uv[:, 1].max() > self.height - margin):
        continue
      area = abs(float(cv2.contourArea(uv.astype(np.float32))))
      if area < 500.0:
        continue
      # Do not turn the printed face away from the camera in one move.
      n_now = T_cam_board[:3, :3] @ np.array([0.0, 0.0, 1.0])
      n_new = T_cb[:3, :3] @ np.array([0.0, 0.0, 1.0])
      if float(np.dot(n_now, n_new)) < math.cos(math.radians(42.0)):
        continue

      nearest_a = min((_ang(T_bg[:3, :3], R) for R in recorded_R),
                      default=180.0)
      nearest_mm = min((float(np.linalg.norm(T_bg[:3, 3] - p)) * 1000
                        for p in recorded_p), default=1000.0)
      nn = n_new / np.linalg.norm(n_new)
      normal_a = min((math.degrees(math.acos(float(np.clip(np.dot(nn, n),
                                                               -1.0, 1.0))))
                      for n in recorded_n), default=90.0)
      if records and nearest_a < NOVEL_DEG and nearest_mm < NOVEL_MM:
        continue
      centre_cost = float(np.linalg.norm(
        (centre - image_centre) / np.array([self.width, self.height])))
      boundary = float(np.min(np.minimum(qc - self.lo, self.hi - qc)))
      score = (1.8 * normal_a + nearest_a + 0.018 * nearest_mm
               + 4.0 * min(boundary, 0.3) - 12.0 * centre_cost
               - 4.0 * motion)
      options.append((score, c7, T_bg, T_cb, uv, centre, nearest_a,
                      nearest_mm, normal_a, current_contacts))

    if not options:
      return {"available": False,
              "why": "no nearby pose is both novel and fully inside the D405 gray image"}
    best = None
    for option in sorted(options, key=lambda x: x[0], reverse=True):
      if self._collision_free(q7, option[1], option[-1]):
        best = option
        break
    if best is None:
      return {"available": False,
              "why": "all visible novel poses have a model collision on their path"}
    (_, c7, T_bg, T_cb, uv, centre, nearest_a, nearest_mm, normal_a,
     _allowed) = best
    return {
      "available": True,
      "q": c7.tolist(),
      "from_q": q7.tolist(),
      "q_deg": [round(float(x), 1) for x in np.degrees(c7[:6])],
      "ee_position_m": [round(float(x), 4) for x in T_bg[:3, 3]],
      "polygon_px": np.round(uv, 1).tolist(),
      "center_px": [round(float(x), 1) for x in centre],
      "range_mm": round(float(np.linalg.norm(T_cb[:3, 3])) * 1000, 0),
      "motion_deg": round(math.degrees(float(np.max(np.abs(c7[:6] - q7[:6])))), 1),
      "nearest_rotation_deg": round(nearest_a, 1),
      "nearest_translation_mm": round(nearest_mm, 0),
      "normal_novelty_deg": round(normal_a, 1),
      "status": "ready",
    }


# ---------------------------------------------------------------------------


class Session:
  """The camera, the arm, and the poses recorded so far.

  One worker thread owns the hardware and publishes a snapshot; the HTTP
  handlers only ever read that snapshot or take the lock to mutate the pose
  list.  The alternative -- letting request threads touch the camera -- means
  two browser tabs are two readers of one D405.
  """

  def __init__(self, board: calibrate.Board, serial: str | None,
               can: str, no_arm: bool, poses_file: Path,
               still_mm: float = STILL_MM, still_deg: float = STILL_DEG):
    self.board = board
    self.still_mm, self.still_deg = float(still_mm), float(still_deg)
    self.poses_file = poses_file
    self.records: list[dict] = []
    self.solution: dict | None = None
    self.table: dict | None = None
    self.saved = False
    self.note = ""
    self.guidance_T: np.ndarray | None = None
    self.guidance: dict | None = None
    self.guidance_failure = ""
    self.next_target: dict | None = None
    self.motion: dict = {"status": "idle", "progress": 0.0}
    self._motion_thread: threading.Thread | None = None
    self._motion_cancel = threading.Event()
    self._last_plan_attempt = 0.0
    self._plan_epoch = 0

    self.lock = threading.Lock()
    self.arm_lock = threading.Lock()
    self.kin_lock = threading.Lock()
    self._stop = threading.Event()
    self._jpeg = b""
    self._live: dict = {"detected": False}
    self._ring: deque = deque(maxlen=RING)
    self.error = ""

    if poses_file.exists():
      stored, recs = calibrate.load_poses(poses_file)
      if stored == board:
        self.records = recs
        self.note = f"resumed {len(recs)} pose(s) from {poses_file.name}"
      else:
        self.note = (f"{poses_file.name} holds poses from a different board "
                     f"({stored.describe()}); they are not loaded")

    from . import sensor
    # The deployed perception path is the D405's grayscale stream.  Calibration
    # uses that exact image too; using the optional raw IR stream here would
    # make the GUI validate a different optical path from the one deployed.
    print("calibgui: opening D405 grayscale stream ...", flush=True)
    try:
      self.reader = sensor.Reader(serial=serial, infrared=False)
    except RuntimeError as e:
      if "VIDIOC_S_FMT" in str(e) or "Input/output error" in str(e):
        raise RuntimeError(
          "D405 refused to start its video stream. Another process usually "
          "has /dev/video* open; close RealSense Viewer and any older "
          "calibgui, then check `fuser /dev/video*`. Unplug/replug the D405 "
          "if no owner is reported. Original error: " + str(e)) from e
      raise
    self.reader.wait_for_first()

    self.arm = None
    if not no_arm:
      from . import robot
      print(f"calibgui: connecting arm on {can} ...", flush=True)
      try:
        self.arm = robot.PiperArm(can)
        self.arm.connect()
      except Exception:
        # A failed CAN bring-up must not leave the camera thread alive and make
        # the next launch fail with the unrelated-looking VIDIOC_S_FMT error.
        self.reader.close()
        raise
    self.kin = proprio.Kinematics()
    self.planner = NextPosePlanner(
      board, self.reader.K, (config.D405_WIDTH, config.D405_HEIGHT))

    if config.RIG_FILE.exists():
      try:
        rig = config.Rig.load()
        if (rig.serial and self.reader.serial
            and str(rig.serial) != str(self.reader.serial)):
          raise ValueError(
            f"rig belongs to D405 {rig.serial}, connected camera is "
            f"{self.reader.serial}")
        self.guidance_T = rig.T_base_cam.copy()
        self.guidance = {
          "source": f"existing {config.RIG_FILE.name}",
          "rough": False,
          "session": False,
          "position_m": [round(float(x), 4) for x in rig.T_base_cam[:3, 3]],
          "residual_mm": rig.residual_mm,
        }
        suffix = f"using {config.RIG_FILE.name} as initial guidance"
        self.note = f"{self.note}; {suffix}" if self.note else suffix
      except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        suffix = f"could not load {config.RIG_FILE.name} for guidance: {e}"
        self.note = f"{self.note}; {suffix}" if self.note else suffix
    self._update_guidance_from_records()

    self._thread = threading.Thread(target=self._run, daemon=True,
                                    name="calibgui")
    self._thread.start()

  # -- the worker ------------------------------------------------------------

  def _run(self) -> None:
    while not self._stop.is_set():
      try:
        self._tick()
      except Exception:                       # a worker that dies goes silent
        self.error = traceback.format_exc(limit=3)
        time.sleep(0.5)
      time.sleep(1 / 15)

  def _tick(self) -> None:
    frame = self.reader.latest()
    if frame is None:
      return
    # Explicitly detect the ChArUco/ArUco pattern in the same D405 grayscale
    # image the deployed stack consumes.
    gray = np.asarray(frame.gray, dtype=np.uint8)
    # Detected once and reused by the overlay: the detection is the expensive
    # part of the tick and doing it twice halves the preview rate.
    found = self.board.detect(gray)
    pose = None
    if found is not None:
      obj, img = found
      ok, rvec, tvec = cv2.solvePnP(obj, img, self.reader.K, np.zeros(5),
                                    flags=cv2.SOLVEPNP_ITERATIVE)
      if ok:
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.reader.K,
                                    np.zeros(5))
        pose = {"rvec": rvec, "tvec": tvec, "n_corners": int(len(img)),
                "reproj_rms_px": float(np.sqrt(
                  ((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2)
                  .sum(1).mean()))}

    live: dict = {
      "detected": pose is not None,
      "frame_age_s": round(self.reader.age, 3),
    }
    T_cb = None
    if pose is not None:
      T_cb = calibrate._rt(pose["rvec"], pose["tvec"])
      live.update(n_corners=pose["n_corners"],
                  reproj_px=round(pose["reproj_rms_px"], 3),
                  range_mm=round(float(np.linalg.norm(T_cb[:3, 3])) * 1000, 1),
                  normal=self._normal(T_cb))
      self._ring.append((time.time(), T_cb))
    else:
      self._ring.clear()

    live["still"], live["still_detail"] = self._still()
    live["novel"], live["novel_detail"] = self._novel(T_cb)
    live["arm_novel"], live["arm_novel_detail"] = False, "no arm feedback"

    st = None
    if self.arm is not None:
      with self.arm_lock:
        st = self.arm.read()
      live["arm"] = {
        "q_deg": [round(float(x), 2) for x in np.degrees(st.q)],
        "gripper_mm": round(st.gripper * 1000, 1),
      }
      live["arm_novel"], live["arm_novel_detail"] = self._arm_novel(st)
      if T_cb is not None:
        live["on_gripper"] = self._on_gripper(st, T_cb)

    self._maybe_plan(T_cb, st)
    with self.lock:
      target = dict(self.next_target) if self.next_target is not None else None

    with self.lock:
      self._live = live
      self._jpeg = self._render(gray, found, live, target)
      self._last = (T_cb, st)

  def _normal(self, T_cb: np.ndarray) -> list[float]:
    """Where the board faces, in the camera frame, as a point on a disc.

    The board's own +z, which for a target held up to a camera points back
    towards it.  Two numbers are enough to draw: the third is determined and
    always negative.
    """
    n = T_cb[:3, :3] @ np.array([0.0, 0.0, 1.0])
    if n[2] > 0:
      n = -n
    return [round(float(n[0]), 4), round(float(n[1]), 4)]

  def _still(self) -> tuple[bool, str]:
    if len(self._ring) < RING:
      return False, f"settling ({len(self._ring)}/{RING} frames)"
    Ts = [T for _, T in self._ring]
    ref = Ts[-1]
    dt = max(float(np.linalg.norm(T[:3, 3] - ref[:3, 3])) for T in Ts) * 1000
    dr = max(_ang(T[:3, :3], ref[:3, :3]) for T in Ts)
    if dt > self.still_mm or dr > self.still_deg:
      return False, f"moving ({dt:.1f} mm, {dr:.2f} deg over {RING} frames)"
    return True, f"still ({dt:.1f} mm, {dr:.2f} deg)"

  def _novel(self, T_cb: np.ndarray | None) -> tuple[bool, str]:
    if T_cb is None:
      return False, "no board"
    best_d, best_a = 1e9, 1e9
    for r in self.records:
      T = calibrate._rt(r["rvec"], r["tvec"])
      best_d = min(best_d, float(np.linalg.norm(T[:3, 3] - T_cb[:3, 3])) * 1000)
      best_a = min(best_a, _ang(T[:3, :3], T_cb[:3, :3]))
    if not self.records:
      return True, "first pose"
    if best_a >= NOVEL_DEG or best_d >= NOVEL_MM:
      return True, f"new ({best_a:.0f} deg, {best_d:.0f} mm from the nearest)"
    return False, (f"too close to a recorded pose ({best_a:.0f} deg, "
                   f"{best_d:.0f} mm) -- turn the board, do not just move it")

  def _arm_novel(self, st) -> tuple[bool, str]:
    """Require the arm, not merely the detected board, to have moved.

    Hand-eye assumes one rigid board-to-gripper transform.  If the operator
    holds the board and changes its image pose while the arm stays still, the
    old image-only novelty gate accepts mutually impossible measurements.
    """
    with self.lock:
      records = list(self.records)
    if not records:
      return True, "first arm pose"
    q = np.array([*st.q, st.gripper, -st.gripper], dtype=np.float64)
    with self.kin_lock:
      self.kin.update(q)
      p = self.kin.data.site_xpos[self.kin.site_id].copy()
      R = self.kin.data.site_xmat[self.kin.site_id].reshape(3, 3).copy()
      nearest = None
      for i, r in enumerate(records):
        self.kin.update(np.asarray(r["joint_pos"], dtype=np.float64))
        rp = self.kin.data.site_xpos[self.kin.site_id].copy()
        rR = self.kin.data.site_xmat[self.kin.site_id].reshape(3, 3).copy()
        d = float(np.linalg.norm(p - rp)) * 1000.0
        a = _ang(R, rR)
        score = max(d / 15.0, a / 5.0)
        if nearest is None or score < nearest[0]:
          nearest = (score, i, d, a)
    _, i, d, a = nearest
    if d < 15.0 and a < 5.0:
      return False, (f"arm matches pose #{i} ({d:.1f} mm, {a:.1f} deg); "
                     "move the arm with the board rigidly clamped")
    return True, f"arm moved ({d:.0f} mm, {a:.0f} deg from nearest #{i})"

  @staticmethod
  def _conflicting_stationary_poses(records: list[dict]) -> list[list[int]]:
    """Groups where one arm pose reports incompatible board poses."""
    bad: list[list[int]] = []
    used: set[int] = set()
    for i, a in enumerate(records):
      if i in used:
        continue
      qa = np.asarray(a["joint_pos"][:6], dtype=np.float64)
      Ta = calibrate._rt(a["rvec"], a["tvec"])
      group = [i]
      conflict = False
      for j in range(i + 1, len(records)):
        qb = np.asarray(records[j]["joint_pos"][:6], dtype=np.float64)
        if np.max(np.abs(qa - qb)) > math.radians(0.5):
          continue
        group.append(j)
        Tb = calibrate._rt(records[j]["rvec"], records[j]["tvec"])
        if (np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) > 0.005
            or _ang(Ta[:3, :3], Tb[:3, :3]) > 2.0):
          conflict = True
      if conflict:
        bad.append(group)
        used.update(group)
    return bad

  def _update_guidance_from_records(self) -> None:
    """Promote the current manual seed poses to a navigation-only extrinsic."""
    with self.lock:
      records = list(self.records)
    if len(records) < GUIDANCE_MIN_POSES:
      self.guidance_failure = (
        f"need {GUIDANCE_MIN_POSES - len(records)} more distinct arm pose(s)")
      return
    conflicts = self._conflicting_stationary_poses(records)
    if conflicts:
      labels = ", ".join("/".join(f"#{i}" for i in g) for g in conflicts)
      self.guidance_failure = (
        f"same arm pose has different board detections at {labels}; "
        "drop those records and keep the board rigidly clamped")
      return
    try:
      out = calibrate.solve(
        records, board=self.board,
        min_rotation_span_deg=GUIDANCE_MIN_ROT_SPAN_DEG)
    except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
      self.guidance_failure = f"rough solve failed: {e}"
      return
    T = out.get("T_base_cam")
    if T is None or not np.isfinite(T).all():
      self.guidance_failure = (
        f"arm rotation spread {out.get('rot_span_deg', 0.0):.1f} deg; "
        f"need {GUIDANCE_MIN_ROT_SPAN_DEG:.0f} deg")
      return
    # A rough solution is allowed to be visibly worse than the final 4 mm
    # calibration, but not so incoherent that it cannot guide a small move.
    if not np.isfinite(out["residual_mm"]) or out["residual_mm"] > 35.0:
      self.guidance_failure = (
        f"rough residual {out['residual_mm']:.1f} mm exceeds 35 mm; "
        "drop inconsistent poses")
      return
    guidance_T = np.asarray(T, dtype=np.float64)
    guidance = {
      "source": f"rough solve from {len(records)} current poses",
      "rough": True,
      "session": True,
      "position_m": [round(float(x), 4) for x in guidance_T[:3, 3]],
      "residual_mm": round(float(out["residual_mm"]), 2),
      "rotation_span_deg": round(float(out["rot_span_deg"]), 1),
    }
    if not _plausible_camera_transform(guidance_T):
      self.guidance_failure = (
        "rough camera transform is not a finite table-scale rigid pose; "
        "collect more varied rigid arm poses")
      return
    with self.lock:
      # Do not install a result computed across a pose that was dropped while
      # the solver was running.
      if len(self.records) == len(records):
        self.guidance_T = guidance_T
        self.guidance = guidance
        self.guidance_failure = ""

  def _maybe_plan(self, T_cb: np.ndarray | None, st) -> None:
    if T_cb is None or st is None:
      return
    with self.lock:
      if self.guidance_T is None:
        return
      if self.motion.get("status") == "moving":
        return
      if (self.next_target is not None
          and self.next_target.get("status") == "reached"):
        return
      if (self.next_target is not None and self.next_target.get("available")
          and self.next_target.get("status") == "ready"):
        start_q = np.asarray(self.next_target.get("from_q", [])[:6])
        if start_q.size == 6 and np.max(np.abs(start_q - st.q)) < math.radians(3):
          return
        # The operator moved the arm by hand after this preview was made, so
        # its predicted polygon is stale and must not remain clickable.
        self.next_target = None
      elif (self.next_target is not None
            and time.monotonic() - self._last_plan_attempt < 1.0):
        return
      else:
        self.next_target = None
      records = list(self.records)
      T = self.guidance_T.copy()
      guidance_source = self.guidance["source"] if self.guidance else ""
      epoch = self._plan_epoch
      self._last_plan_attempt = time.monotonic()
    q7 = np.array([*st.q, st.gripper], dtype=np.float64)
    try:
      target = self.planner.plan(T, T_cb, q7, records)
    except Exception as e:
      target = {"available": False, "why": f"planner failed: {e}"}
    target["guidance_source"] = guidance_source
    with self.lock:
      # A record/solve request may have invalidated the pending plan while the
      # kinematic search was running.
      if self.next_target is None and self._plan_epoch == epoch:
        self.next_target = target

  def _invalidate_target(self) -> None:
    self._plan_epoch += 1
    self.next_target = None
    if self.motion.get("status") != "moving":
      self.motion = {"status": "idle", "progress": 0.0}

  def _on_gripper(self, st, T_cb: np.ndarray) -> dict:
    """Is the board actually held by the arm?

    Eye-to-hand needs the board carried by the gripper: one lying on the table
    has an unknown pose in both frames and constrains nothing, and the symptom
    is a calibration that refuses at the residual gate after every pose has
    been collected.  This catches it before the first one.

    Before guidance exists this uses the nominal extrinsic; afterwards it uses
    the current rough or solved one.  The tolerance is wide enough to survive
    a badly aimed mount.  It answers
    "within an arm's wrist" or "somewhere else entirely", which is the only
    resolution needed.
    """
    with self.kin_lock:
      self.kin.update(np.array([*st.q, st.gripper, -st.gripper]))
      grip = self.kin.data.site_xpos[self.kin.site_id].copy()
    with self.lock:
      T_base_cam = (self.guidance_T.copy() if self.guidance_T is not None
                    else config.sim_camera_extrinsic())
    board = (T_base_cam @ np.append(T_cb[:3, 3], 1.0))[:3]
    d = float(np.linalg.norm(board - grip))
    return {"separation_mm": round(d * 1000, 0), "ok": bool(d < 0.25)}

  # -- the preview -----------------------------------------------------------

  def _render(self, gray: np.ndarray, found, live: dict,
              target: dict | None = None) -> bytes:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if target and target.get("available") and target.get("polygon_px"):
      poly = np.rint(target["polygon_px"]).astype(np.int32).reshape(-1, 1, 2)
      cv2.polylines(img, [poly], True, (255, 210, 40), 2, cv2.LINE_AA)
      c = tuple(np.rint(target["center_px"]).astype(int))
      cv2.drawMarker(img, c, (255, 210, 40), cv2.MARKER_CROSS, 18, 2,
                     cv2.LINE_AA)
      cv2.putText(img, "NEXT", (c[0] + 10, c[1] - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 40), 2,
                  cv2.LINE_AA)
    if found is not None:
      _, pts = found
      ok = live.get("still") and live.get("novel")
      colour = (80, 220, 100) if ok else (60, 160, 240)
      for p in pts.reshape(-1, 2):
        cv2.circle(img, (int(p[0]), int(p[1])), 4, colour, -1, cv2.LINE_AA)
      xy = pts.reshape(-1, 2)
      lo, hi = xy.min(0).astype(int), xy.max(0).astype(int)
      cv2.rectangle(img, tuple(lo - 8), tuple(hi + 8), colour, 1, cv2.LINE_AA)
    else:
      cv2.putText(img, "board not detected", (16, 34),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 80, 240), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return buf.tobytes() if ok else b""

  def preview(self) -> bytes:
    with self.lock:
      return self._jpeg

  # -- what the page reads ---------------------------------------------------

  def rotation_span(self, records: list[dict] | None = None) -> float:
    """The gate the solver applies, computed the way the solver computes it.

    From the *gripper* rotations, not the board's: ``solve`` measures the span
    of ``R_bg`` and refuses under ``MIN_ROT_SPAN_DEG``.  Reporting the board's
    spread instead would be a different number that agrees most of the time.
    """
    records = list(self.records if records is None else records)
    Rs = []
    with self.kin_lock:
      for r in records:
        self.kin.update(np.asarray(r["joint_pos"], dtype=np.float64))
        Rs.append(self.kin.data.site_xmat[self.kin.site_id].reshape(3, 3).T.copy())
    return max((_ang(Rs[i], Rs[j]) for i in range(len(Rs))
                for j in range(i + 1, len(Rs))), default=0.0)

  def state(self) -> dict:
    with self.lock:
      live = dict(self._live)
      records = list(self.records)
      target = dict(self.next_target) if self.next_target is not None else None
      motion = dict(self.motion)
    return {
      "board": {"describe": self.board.describe(),
                "n_corners": self.board.n_corners(),
                "suggested_poses": self.board.suggested_poses()},
      "live": live,
      "n_poses": len(records),
      "min_poses": calibrate.MIN_POSES,
      "rotation_span_deg": round(self.rotation_span(records), 1),
      "min_rotation_span_deg": calibrate.MIN_ROT_SPAN_DEG,
      "poses": [{"i": i, "n_corners": r.get("n_corners"),
                 "reproj_px": round(r.get("reproj_rms_px", 0.0), 2),
                 "normal": self._normal(calibrate._rt(r["rvec"], r["tvec"]))}
                for i, r in enumerate(records)],
      "solution": self.solution,
      "guidance": self.guidance,
      "guidance_failure": self.guidance_failure,
      "next_target": target,
      "motion": motion,
      "image_source": "D405 gray",
      "guidance_min_poses": GUIDANCE_MIN_POSES,
      "guidance_min_rotation_span_deg": GUIDANCE_MIN_ROT_SPAN_DEG,
      "table": self.table,
      "saved": self.saved,
      "has_arm": self.arm is not None,
      "note": self.note,
      "error": self.error,
      "poses_file": str(self.poses_file),
      "rig_file": str(config.RIG_FILE),
    }

  # -- what the buttons do ---------------------------------------------------

  def record(self) -> dict:
    with self.lock:
      live = dict(self._live)
      T_cb, st = getattr(self, "_last", (None, None))
      moving = self.motion.get("status") == "moving"
    if moving:
      return {"ok": False, "why": "arm is moving"}
    if T_cb is None:
      return {"ok": False, "why": "no board in view"}
    if not live.get("still"):
      return {"ok": False, "why": live.get("still_detail", "moving")}
    if not live.get("novel"):
      return {"ok": False, "why": live.get("novel_detail", "not a new pose")}
    if not live.get("arm_novel"):
      return {"ok": False, "why": live.get(
        "arm_novel_detail", "arm pose has not changed")}
    if st is None:
      return {"ok": False, "why": "no arm; poses cannot be recorded"}
    rec = {
      "joint_pos": [*st.q.tolist(), st.gripper, -st.gripper],
      "rvec": cv2.Rodrigues(T_cb[:3, :3])[0].ravel().tolist(),
      "tvec": T_cb[:3, 3].tolist(),
      "n_corners": live.get("n_corners"),
      "reproj_rms_px": live.get("reproj_px"),
    }
    with self.lock:
      self.records.append(rec)
      self.solution = None
      self.saved = False
      calibrate.save_poses(self.board, self.records, self.poses_file)
      self._invalidate_target()
      if self.guidance and self.guidance.get("session"):
        self.guidance_T = None
        self.guidance = None
    self._update_guidance_from_records()
    return {"ok": True}

  def drop(self, i: int) -> dict:
    with self.lock:
      if not (0 <= i < len(self.records)):
        return {"ok": False, "why": "no such pose"}
      self.records.pop(i)
      self.solution = None
      self.saved = False
      calibrate.save_poses(self.board, self.records, self.poses_file)
      self._invalidate_target()
      if self.guidance and self.guidance.get("session"):
        self.guidance_T = None
        self.guidance = None
    self._update_guidance_from_records()
    return {"ok": True}

  def solve(self) -> dict:
    with self.lock:
      records = list(self.records)
    if len(records) < calibrate.MIN_POSES:
      return {"ok": False, "why": f"{len(records)} poses; "
                                  f"{calibrate.MIN_POSES} is the minimum"}
    out = calibrate.solve(records, board=self.board)
    if out["T_base_cam"] is None:
      self.solution = {"ok": False,
                       "rot_span_deg": round(out["rot_span_deg"], 1),
                       "why": "not enough rotation; the equation is satisfied "
                              "by any answer"}
      return self.solution
    T = out["T_base_cam"]
    nominal = config.sim_camera_extrinsic()
    self.solution = {
      "ok": True,
      "residual_mm": round(out["residual_mm"], 2),
      "worst_mm": round(out["worst_mm"], 2),
      "worst_pose": out["worst_pose"],
      "per_pose_mm": [round(x, 2) for x in out["per_pose_mm"]],
      "rot_span_deg": round(out["rot_span_deg"], 1),
      "n_flipped": out.get("n_flipped", 0),
      "position_m": [round(float(x), 4) for x in T[:3, 3]],
      "vs_sim_mm": round(float(np.linalg.norm(T[:3, 3] - nominal[:3, 3]))
                         * 1000, 1),
      "vs_sim_deg": round(_ang(T[:3, :3], nominal[:3, :3]), 2),
      "T_base_cam": T.tolist(),
    }
    guide_ok = (out["residual_mm"] <= 35.0
                and _plausible_camera_transform(T))
    with self.lock:
      self._invalidate_target()
      if guide_ok:
        self.guidance_T = T.copy()
        self.guidance = {
          "source": f"accepted solve from {len(records)} current poses",
          "rough": False,
          "session": True,
          "position_m": self.solution["position_m"],
          "residual_mm": self.solution["residual_mm"],
          "rotation_span_deg": self.solution["rot_span_deg"],
        }
        self.guidance_failure = ""
      elif self.guidance and self.guidance.get("session"):
        self.guidance_T = None
        self.guidance = None
    return self.solution

  def move_next(self) -> dict:
    with self.lock:
      if self.arm is None:
        return {"ok": False, "why": "no arm connected"}
      if self.motion.get("status") == "moving":
        return {"ok": False, "why": "arm is already moving"}
      target = dict(self.next_target) if self.next_target is not None else None
      if not (target and target.get("available")
              and target.get("status") == "ready"):
        return {"ok": False, "why": (target or {}).get(
          "why", "no planned target")}
      target["status"] = "moving"
      self.next_target = target
      self.motion = {"status": "moving", "progress": 0.0,
                     "why": "enabling arm"}
      self._motion_cancel.clear()
    self._motion_thread = threading.Thread(
      target=self._move_worker, args=(target,), daemon=True,
      name="calibgui-motion")
    self._motion_thread.start()
    return {"ok": True}

  def _move_worker(self, target: dict) -> None:
    """Execute one confirmed target, with feedback and a tracking watchdog."""
    try:
      with self.arm_lock:
        start = self.arm.read()
        self.arm.enable()
        # Hold the measured pose immediately after enabling, before the first
        # interpolated command.  This prevents a stale drive setpoint from
        # becoming the arm's first enabled target.
        self.arm.command(np.array([*start.q, start.gripper]))
      path = _joint_trajectory(start.q, np.asarray(target["q"][:6]))
      period = 1.0 / AUTO_RATE_HZ
      deadline = time.monotonic()
      for i, q in enumerate(path):
        if self._stop.is_set():
          raise RuntimeError("session is closing")
        if self._motion_cancel.is_set():
          raise RuntimeError("motion cancelled by operator")
        with self.arm_lock:
          st = self.arm.read()
          command = np.array([*q, target["q"][6]], dtype=np.float64)
          self.arm.command(command, period)
        tracking = float(np.max(np.abs(st.q - q)))
        # Allow the servo to establish motion for the first half second; after
        # that, a large error means a disabled/stalled joint or lost CAN path.
        if i > AUTO_RATE_HZ * 0.5 and tracking > AUTO_TRACKING_ERROR_RAD:
          raise RuntimeError(
            f"tracking error {math.degrees(tracking):.1f} deg exceeds "
            f"{math.degrees(AUTO_TRACKING_ERROR_RAD):.1f} deg")
        with self.lock:
          self.motion = {
            "status": "moving",
            "progress": round(i / max(len(path) - 1, 1), 3),
            "tracking_error_deg": round(math.degrees(tracking), 1),
            "why": "streaming limited joint trajectory",
          }
        deadline += period
        time.sleep(max(0.0, deadline - time.monotonic()))

      # Give the firmware a short settling window while continuing to hold the
      # exact endpoint.  The normal stillness gate remains the authority on
      # when a sample may actually be recorded.
      goal = np.asarray(target["q"][:6])
      final = float("inf")
      for _ in range(40):
        if self._motion_cancel.is_set():
          raise RuntimeError("motion cancelled by operator")
        with self.arm_lock:
          st = self.arm.read()
          self.arm.command(np.array([*goal, target["q"][6]]))
        final = float(np.max(np.abs(st.q - goal)))
        if final < math.radians(2.0):
          break
        time.sleep(0.05)
      if final >= math.radians(3.0):
        raise RuntimeError(f"arm stopped {math.degrees(final):.1f} deg from target")
      with self.lock:
        if self.next_target is not None:
          self.next_target["status"] = "reached"
        self.motion = {"status": "reached", "progress": 1.0,
                       "tracking_error_deg": round(math.degrees(final), 1),
                       "why": "target reached; wait for the stillness gate"}
    except Exception as e:
      try:
        with self.arm_lock:
          self.arm.hold()
      except Exception:
        pass
      with self.lock:
        if self.next_target is not None:
          self.next_target["status"] = "failed"
        self.motion = {"status": "failed", "progress": 0.0,
                       "why": str(e)}

  def stop_motion(self) -> dict:
    with self.lock:
      if self.motion.get("status") != "moving":
        return {"ok": False, "why": "arm is not moving"}
      self._motion_cancel.set()
      self.motion["why"] = "stop requested; holding at feedback pose"
    return {"ok": True}

  def save(self) -> dict:
    if not (self.solution and self.solution.get("ok")):
      return {"ok": False, "why": "solve first"}
    T = np.asarray(self.solution["T_base_cam"], dtype=np.float64)
    rig = config.Rig(T_base_cam=T, K=self.reader.K,
                     serial=self.reader.serial,
                     residual_mm=self.solution["residual_mm"])
    frame = self.reader.latest()
    if frame is not None:
      try:
        t = calibrate.fit_table(frame.depth, T, self.reader.K)
        rig.table_z = t["table_z"]
        rig.table_tilt_deg = t["tilt_deg"]
        rig.table_flatness_mm = t["flatness_mm"]
        self.table = {k: round(float(v), 4) if isinstance(v, float) else v
                      for k, v in t.items()}
      except RuntimeError as e:
        self.table = {"error": str(e)}
    rig.save()
    self.saved = True
    return {"ok": True, "rig_file": str(config.RIG_FILE), "table": self.table}

  def close(self) -> None:
    self._stop.set()
    self._motion_cancel.set()
    if self._motion_thread is not None:
      self._motion_thread.join(timeout=3.0)
    self._thread.join(timeout=2.0)
    self.reader.close()
    if self.arm is not None:
      with self.arm_lock:
        self.arm.close()


# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
  server_version = "calibgui/1.0"

  def log_message(self, *a):
    pass

  def _json(self, obj, code=200):
    body = json.dumps(obj, default=str).encode()
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):
    path = self.path.split("?")[0]
    sess = self.server.session
    if path in ("/", "/index.html"):
      body = (HERE / "calibgui.html").read_bytes()
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
    elif path == "/state":
      self._json(sess.state())
    elif path == "/stream":
      self.send_response(200)
      self.send_header("Age", "0")
      self.send_header("Cache-Control", "no-cache, private")
      self.send_header("Content-Type",
                       "multipart/x-mixed-replace; boundary=frame")
      self.end_headers()
      try:
        while True:
          buf = sess.preview()
          if buf:
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                             b"Content-Length: " + str(len(buf)).encode() +
                             b"\r\n\r\n" + buf + b"\r\n")
          time.sleep(1 / 15)
      except (BrokenPipeError, ConnectionResetError):
        pass
    else:
      self.send_error(404)

  def do_POST(self):
    path = self.path.split("?")[0]
    sess = self.server.session
    n = int(self.headers.get("Content-Length") or 0)
    body = json.loads(self.rfile.read(n) or b"{}") if n else {}
    if path == "/record":
      self._json(sess.record())
    elif path == "/drop":
      self._json(sess.drop(int(body.get("i", -1))))
    elif path == "/solve":
      self._json(sess.solve())
    elif path == "/save":
      self._json(sess.save())
    elif path == "/move-next":
      self._json(sess.move_next())
    elif path == "/stop-motion":
      self._json(sess.stop_motion())
    else:
      self.send_error(404)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--port", type=int, default=8771)
  p.add_argument("--serial", default=None)
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--no-arm", action="store_true",
                 help="camera only; poses cannot be recorded, but the board "
                      "and the framing can be checked without CAN")
  p.add_argument("--poses", default=str(calibrate.POSES_FILE))
  p.add_argument("--still-mm", type=float, default=STILL_MM)
  p.add_argument("--still-deg", type=float, default=STILL_DEG)
  calibrate_group = p.add_argument_group("the board")
  calibrate_group.add_argument("--board", default=None)
  calibrate_group.add_argument("--board-kind",
                               choices=("charuco", "checker"), default=None)
  calibrate_group.add_argument("--squares", default=None)
  calibrate_group.add_argument("--square-mm", type=float, default=None)
  calibrate_group.add_argument("--marker-mm", type=float, default=None)
  calibrate_group.add_argument("--dict", default=None)
  calibrate_group.add_argument("--legacy", action="store_true")
  a = p.parse_args()

  board = calibrate.board_from_args(a)
  print(f"board: {board.describe()}", flush=True)
  sess = Session(board, a.serial, a.can, a.no_arm, Path(a.poses),
                 a.still_mm, a.still_deg)
  httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
  httpd.session = sess
  print(f"\n  open http://127.0.0.1:{a.port}\n", flush=True)
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    sess.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
