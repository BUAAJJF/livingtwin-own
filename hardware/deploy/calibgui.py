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
RECOVERY_MAX_MOTION_RAD = math.radians(30.0)
"""Conservative joint-space motion for calibration.  The page asks before
every move; these constants then keep that confirmed move slow and abort if
feedback falls far behind the streamed target.  A recovery pose is also
refused if any joint is more than 30 degrees away: recovery is only meant to
reverse the immediately preceding small next-pose move, never to navigate
from an unknown arm configuration."""

BOARD_BORDER_M = 0.006
BOARD_COLLISION_PAD_M = 0.010
BOARD_HALF_THICKNESS_M = 0.012
BOARD_TABLE_CLEARANCE_M = 0.060
BOARD_PATH_SAMPLES = 33
"""Collision envelope for the calibration board carried by the gripper.

The compact white print has a 168 x 140 mm pattern on a 180 x 152 mm cut
panel, hence the 6 mm physical border.  The mount puts the panel roughly
100 mm beyond the robot's normal tip, so checking the robot model alone is
not a payload collision check.  Inflate every panel edge by another 10 mm,
treat it as a 24 mm slab, and keep that inflated volume at least 60 mm above
the measured table throughout an automatic move.  The clearance deliberately
also covers error in the rough five-pose hand-eye solution used for guidance.
"""

CALIB_WIDTH, CALIB_HEIGHT = 1280, 720
GUI_RATE_HZ = 15.0
PREVIEW_WIDTH = 960
FUSION_WINDOW = 24
FUSION_MIN_FRAMES = 8
MIN_PROJECTED_MARKER_PX = 22.0
TABLE_BOARD = calibrate.Board(
  kind="checker", squares=(11, 8), square_m=0.025, min_corners=88)
TABLE_MIN_SAMPLES = 3
TABLE_RECOMMENDED_SAMPLES = 6
TABLE_NOVEL_MM = 50.0
TABLE_SEARCH_INTERVAL_S = 0.50
TABLE_TRACK_INTERVAL_S = 0.13
TABLE_POSES_FILE = HERE / "calib_table_poses.json"
"""Calibration-only acquisition settings.  Deployment remains 848x480.

At roughly 15 GUI ticks/s the fusion window covers 1.6 seconds.  Eight valid
frames are enough to reject a one-frame decode while keeping the record button
responsive.  Twenty-two pixels per marker keeps a 4x4 code above about 3.5
pixels per module including its black border.
"""


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
  """See ``robot.joint_trajectory``; kept as a name so this file reads the
  same, but there is one implementation and the deployment loop uses it too."""
  from . import robot as _robot
  return _robot.joint_trajectory(q0, q1, speed_rad_s, rate_hz)


class NextPosePlanner:
  """Choose a nearby joint pose that is novel and keeps the target visible.

  The board-to-gripper transform is recovered from the current detection and
  the rough camera extrinsic.  Candidate joint poses are then run through the
  same MuJoCo kinematics as deployment, projected with the D405 intrinsics,
  and rejected if the board would leave the grayscale image, cross a joint
  safety margin, introduce a new self-collision, or bring the complete board
  payload close to the table.
  """

  def __init__(self, board: calibrate.Board, K: np.ndarray,
               image_size: tuple[int, int], table_z: float = config.TABLE_Z_M):
    self.board = board
    self.K = np.asarray(K, dtype=np.float64)
    self.width, self.height = (int(image_size[0]), int(image_size[1]))
    self.table_z = float(table_z)
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

  def _board_payload(self) -> np.ndarray:
    """Eight corners of the inflated physical board in board coordinates."""
    nx, ny = self.board.squares
    if self.board.kind == "charuco":
      w, h = nx * self.board.square_m, ny * self.board.square_m
    else:
      w, h = ((nx - 1) * self.board.square_m,
              (ny - 1) * self.board.square_m)
    pad = BOARD_BORDER_M + BOARD_COLLISION_PAD_M
    return np.array([
      [x, y, z, 1.0]
      for z in (-BOARD_HALF_THICKNESS_M, BOARD_HALF_THICKNESS_M)
      for y in (-pad, h + pad)
      for x in (-pad, w + pad)
    ], dtype=np.float64)

  def board_clearance(self, q7: np.ndarray,
                      T_grip_board: np.ndarray) -> float:
    """Minimum inflated-board height above the calibrated table, in metres."""
    T_base_board = self._fk(q7) @ np.asarray(T_grip_board, dtype=np.float64)
    xyz = (T_base_board @ self._board_payload().T).T[:, :3]
    return float(np.min(xyz[:, 2]) - self.table_z)

  def validate_path(
    self, q0: np.ndarray, q1: np.ndarray, T_grip_board: np.ndarray,
    allowed: set[tuple[int, int]] | None = None,
  ) -> tuple[bool, float, str]:
    """Check arm contacts and the board swept volume along a whole move."""
    q0 = np.asarray(q0, dtype=np.float64).reshape(7)
    q1 = np.asarray(q1, dtype=np.float64).reshape(7)
    if allowed is None:
      allowed = self._contacts(q0)
    minimum = float("inf")
    for u in np.linspace(0.0, 1.0, BOARD_PATH_SAMPLES):
      q = q0 + u * (q1 - q0)
      if not self._contacts(q).issubset(allowed):
        return False, minimum, "robot model collision on path"
      clearance = self.board_clearance(q, T_grip_board)
      minimum = min(minimum, clearance)
      if clearance < BOARD_TABLE_CLEARANCE_M:
        return False, minimum, (
          f"calibration board would pass {clearance * 1000:.0f} mm above "
          f"the table; require {BOARD_TABLE_CLEARANCE_M * 1000:.0f} mm")
    return True, minimum, ""

  def _collision_free(self, q0: np.ndarray, q1: np.ndarray,
                      allowed: set[tuple[int, int]],
                      T_grip_board: np.ndarray) -> bool:
    return self.validate_path(q0, q1, T_grip_board, allowed)[0]

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
    for amount in np.radians((7.0, 10.0, 12.0)):
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
    scale = np.array([0.07, 0.06, 0.06, 0.12, 0.12, 0.14])
    cap = np.array([0.12, 0.10, 0.10, 0.19, 0.19, 0.21])
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
    mount_info = {
      "T_grip_board": T_grip_board.tolist(),
      "table_z_m": self.table_z,
      "required_board_clearance_mm": BOARD_TABLE_CLEARANCE_M * 1000.0,
    }
    current_clearance = self.board_clearance(q7, T_grip_board)
    if current_clearance < BOARD_TABLE_CLEARANCE_M:
      return {
        "available": False,
        "why": (f"current calibration board is only "
                f"{current_clearance * 1000:.0f} mm above the table; move it "
                f"manually above {BOARD_TABLE_CLEARANCE_M * 1000:.0f} mm "
                "before enabling automatic views"),
        "board_clearance_mm": round(current_clearance * 1000.0, 1),
        **mount_info,
      }

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
    margin = 42.0
    image_centre = np.array([self.width / 2.0, self.height / 2.0])
    current_projected = self._project(T_cam_board)
    if current_projected is None:
      return {"available": False,
              "why": "current detected board cannot be projected",
              **mount_info}
    current_uv, _ = current_projected
    current_area = abs(float(cv2.contourArea(current_uv.astype(np.float32))))
    current_max_edge = max(float(np.linalg.norm(
      current_uv[(i + 1) % 4] - current_uv[i])) for i in range(4))
    current_shape = current_area / max(current_max_edge ** 2, 1.0)
    n_now = T_cam_board[:3, :3] @ np.array([0.0, 0.0, 1.0])
    n_now /= np.linalg.norm(n_now)
    current_facing = abs(float(n_now[2]))
    for qc in self._candidates(q7, seed=7919 + len(records) * 104729):
      motion = float(np.linalg.norm(qc - q7[:6]))
      if motion < np.radians(7.0):
        continue
      c7 = np.array([*qc, q7[6]])
      T_bg = self._fk(c7)
      endpoint_clearance = self.board_clearance(c7, T_grip_board)
      if endpoint_clearance < BOARD_TABLE_CLEARANCE_M:
        continue
      T_cb = T_cam_base @ T_bg @ T_grip_board
      projected = self._project(T_cb)
      if projected is None:
        continue
      uv, centre = projected
      if (uv[:, 0].min() < margin or uv[:, 0].max() > self.width - margin
          or uv[:, 1].min() < margin or uv[:, 1].max() > self.height - margin):
        continue
      area = abs(float(cv2.contourArea(uv.astype(np.float32))))
      max_edge = max(float(np.linalg.norm(uv[(i + 1) % 4] - uv[i]))
                     for i in range(4))
      shape = area / max(max_edge ** 2, 1.0)
      nx, ny = self.board.squares
      horizontal = min(float(np.linalg.norm(uv[1] - uv[0])),
                       float(np.linalg.norm(uv[2] - uv[3]))) / max(nx, 1)
      vertical = min(float(np.linalg.norm(uv[2] - uv[1])),
                     float(np.linalg.norm(uv[3] - uv[0]))) / max(ny, 1)
      marker_px = (min(horizontal, vertical)
                   * self.board.marker_m / self.board.square_m
                   if self.board.kind == "charuco"
                   else min(horizontal, vertical))
      # A polygon can be completely inside the image and still be an
      # undetectable sliver.  Preserve a substantial fraction of the currently
      # detected board's pixel support and projected thickness.
      if (area < max(2500.0, 0.50 * current_area)
          or shape < max(0.12, 0.55 * current_shape)
          or marker_px < (MIN_PROJECTED_MARKER_PX
                          * self.width / CALIB_WIDTH)):
        continue
      # Do not turn the printed face away from the camera in one move.
      n_new = T_cb[:3, :3] @ np.array([0.0, 0.0, 1.0])
      n_new /= np.linalg.norm(n_new)
      facing = abs(float(n_new[2]))
      if (float(np.dot(n_now, n_new)) < math.cos(math.radians(20.0))
          or facing < max(0.28, 0.65 * current_facing)):
        continue

      nearest_a = min((_ang(T_bg[:3, :3], R) for R in recorded_R),
                      default=180.0)
      nearest_mm = min((float(np.linalg.norm(T_bg[:3, 3] - p)) * 1000
                        for p in recorded_p), default=1000.0)
      normal_a = min((math.degrees(math.acos(float(np.clip(np.dot(n_new, n),
                                                               -1.0, 1.0))))
                      for n in recorded_n), default=90.0)
      if records and nearest_a < NOVEL_DEG and nearest_mm < NOVEL_MM:
        continue
      centre_cost = float(np.linalg.norm(
        (centre - image_centre) / np.array([self.width, self.height])))
      boundary = float(np.min(np.minimum(qc - self.lo, self.hi - qc)))
      score = (0.7 * normal_a + 0.7 * nearest_a + 0.018 * nearest_mm
               + 4.0 * min(boundary, 0.3) - 12.0 * centre_cost
               - 4.0 * motion + 30.0 * facing + 20.0 * shape)
      options.append((score, c7, T_bg, T_cb, uv, centre, nearest_a,
                      nearest_mm, normal_a, area, shape, facing, marker_px,
                      current_contacts, endpoint_clearance))

    if not options:
      return {"available": False,
              "why": ("no nearby pose is visible, novel, and keeps the full "
                      "calibration board clear of the table"),
              **mount_info}
    best = None
    rejected_reason = ""
    path_clearance = float("nan")
    for option in sorted(options, key=lambda x: x[0], reverse=True):
      safe, clearance, reason = self.validate_path(
        q7, option[1], T_grip_board, option[-2])
      if safe:
        best = option
        path_clearance = clearance
        break
      rejected_reason = reason
    if best is None:
      return {"available": False,
              "why": ("all visible novel poses are unsafe on their path: "
                      + (rejected_reason or "collision")),
              **mount_info}
    (_, c7, T_bg, T_cb, uv, centre, nearest_a, nearest_mm, normal_a,
     area, shape, facing, marker_px, _allowed, endpoint_clearance) = best
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
      "projected_area_px2": round(area, 0),
      "projected_shape": round(shape, 3),
      "facing_cos": round(facing, 3),
      "projected_marker_px": round(marker_px, 1),
      "board_clearance_mm": round(endpoint_clearance * 1000.0, 1),
      "path_board_clearance_mm": round(path_clearance * 1000.0, 1),
      **mount_info,
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
               still_mm: float = STILL_MM, still_deg: float = STILL_DEG,
               recovery_q: np.ndarray | None = None,
               calib_width: int = CALIB_WIDTH,
               calib_height: int = CALIB_HEIGHT,
               raw_gray: bool = True,
               table_poses_file: Path = TABLE_POSES_FILE,
               camera_backend: str = "d455",
               rig_file: Path = config.RIG_FILE,
               emitter: str | None = None):
    self.board = board
    self.still_mm, self.still_deg = float(still_mm), float(still_deg)
    self.poses_file = poses_file
    self.records: list[dict] = []
    self.table_board = TABLE_BOARD
    self.table_poses_file = Path(table_poses_file)
    self.camera_backend = str(camera_backend)
    self.emitter = emitter
    self.rig_file = Path(rig_file)
    self.table_records: list[dict] = []
    self.solution: dict | None = None
    self.table: dict | None = None
    self.saved = False
    self.note = ""
    self.guidance_T: np.ndarray | None = None
    self.guidance: dict | None = None
    self.guidance_failure = ""
    self.next_target: dict | None = None
    self.board_mount_T: np.ndarray | None = None
    self.motion: dict = {"status": "idle", "progress": 0.0}
    self.recovery_q = (None if recovery_q is None else
                       np.asarray(recovery_q, dtype=np.float64).reshape(6))
    self._motion_thread: threading.Thread | None = None
    self._motion_cancel = threading.Event()
    self._last_plan_attempt = 0.0
    self._plan_epoch = 0

    self.lock = threading.Lock()
    self.arm_lock = threading.Lock()
    self.kin_lock = threading.Lock()
    self._stop = threading.Event()
    self._jpeg = b""
    self._overlay_found = None
    self._overlay_table_found = None
    self._live: dict = {"detected": False}
    self._ring: deque = deque(maxlen=FUSION_WINDOW)
    self._table_live: dict = {"detected": False}
    self._table_ring: deque = deque(maxlen=FUSION_WINDOW)
    self._last_table_scan_s = float("-inf")
    self.error = ""

    if poses_file.exists():
      stored, recs = calibrate.load_poses(poses_file)
      if stored == board:
        self.records = recs
        self.note = f"resumed {len(recs)} pose(s) from {poses_file.name}"
        # The most recent sample is, by definition, a pose where the board was
        # detected.  It is useful after a GUI restart, but the 30-degree gate
        # in ``return_visible`` still prevents stale files from causing a
        # large move.
        if self.recovery_q is None and recs:
          self.recovery_q = np.asarray(
            recs[-1]["joint_pos"][:6], dtype=np.float64).copy()
      else:
        self.note = (f"{poses_file.name} holds poses from a different board "
                     f"({stored.describe()}); they are not loaded")

    if self.table_poses_file.exists():
      try:
        stored, recs = calibrate.load_poses(self.table_poses_file)
        if stored == self.table_board:
          self.table_records = recs
          suffix = (f"resumed {len(recs)} table sample(s) from "
                    f"{self.table_poses_file.name}")
          self.note = f"{self.note}; {suffix}" if self.note else suffix
        else:
          suffix = (f"ignored table samples for {stored.describe()}; expected "
                    f"{self.table_board.describe()}")
          self.note = f"{self.note}; {suffix}" if self.note else suffix
      except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        suffix = f"could not load {self.table_poses_file.name}: {e}"
        self.note = f"{self.note}; {suffix}" if self.note else suffix

    from . import sensor
    # The deployed perception path is the D405's grayscale stream.  Calibration
    # uses that exact image too; using the optional raw IR stream here would
    # make the GUI validate a different optical path from the one deployed.
    source = "raw left grayscale" if raw_gray else "depth-aligned grayscale"
    print(f"calibgui: opening {self.camera_backend.upper()} {source} at "
          f"{calib_width}x{calib_height} ...", flush=True)
    try:
      self.reader = sensor.Reader(
        serial=serial, infrared=False, backend=self.camera_backend,
        width=int(calib_width), height=int(calib_height),
        gray_source="left_ir" if raw_gray else "aligned_color",
        emitter=self.emitter)
    except RuntimeError as e:
      if "VIDIOC_S_FMT" in str(e) or "Input/output error" in str(e):
        raise RuntimeError(
          f"{self.camera_backend.upper()} refused to start its video stream. "
          "Another process usually "
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
    table_z = config.TABLE_Z_M
    # A new camera has no rig yet, but the robot and table have not moved.
    # Reuse the existing rig's measured table height only for collision
    # clearance; never reuse its camera extrinsic for the new calibration.
    for table_rig in (self.rig_file, config.RIG_FILE):
      try:
        if table_rig.exists():
          table_z = config.Rig.load(table_rig).table_z
          break
      except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    self.planner = NextPosePlanner(
      board, self.reader.K,
      tuple(int(x) for x in self.reader.meta["resolution"]), table_z=table_z)

    if self.rig_file.exists():
      try:
        rig = config.Rig.load(self.rig_file)
        if (rig.serial and self.reader.serial
            and str(rig.serial) != str(self.reader.serial)):
          raise ValueError(
            f"rig belongs to camera {rig.serial}, connected camera is "
            f"{self.reader.serial}")
        self.guidance_T = rig.T_base_cam.copy()
        self.guidance = {
          "source": f"existing {self.rig_file.name}",
          "rough": False,
          "session": False,
          "position_m": [round(float(x), 4) for x in rig.T_base_cam[:3, 3]],
          "residual_mm": rig.residual_mm,
        }
        suffix = f"using {self.rig_file.name} as initial guidance"
        self.note = f"{self.note}; {suffix}" if self.note else suffix
      except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        suffix = f"could not load {self.rig_file.name} for guidance: {e}"
        self.note = f"{self.note}; {suffix}" if self.note else suffix
    self._update_guidance_from_records()

    self._thread = threading.Thread(target=self._run, daemon=True,
                                    name="calibgui")
    self._thread.start()
    self._preview_thread = threading.Thread(
      target=self._preview_run, daemon=True, name="calibgui-preview")
    self._preview_thread.start()

  # -- the worker ------------------------------------------------------------

  def _run(self) -> None:
    deadline = time.monotonic()
    while not self._stop.is_set():
      try:
        self._tick()
      except Exception:                       # a worker that dies goes silent
        self.error = traceback.format_exc(limit=3)
        time.sleep(0.5)
      # Fixed-rate scheduling: sleeping a full period *after* detection made
      # the old preview rate equal to processing time plus 1/15 s (about 9 Hz
      # on the D455), even though both camera and HTTP stream were healthy.
      deadline += 1.0 / GUI_RATE_HZ
      delay = deadline - time.monotonic()
      if delay > 0:
        time.sleep(delay)
      else:
        # Do not accumulate lag after an occasional expensive full-frame
        # checkerboard search.
        deadline = time.monotonic()

  def _tick(self) -> None:
    frame = self.reader.latest()
    if frame is None:
      return
    # Explicitly detect the ChArUco/ArUco pattern in the same D405 grayscale
    # image the deployed stack consumes.
    gray = np.asarray(frame.gray, dtype=np.uint8)
    # Detected once and reused by the overlay: the detection is the expensive
    # part of the tick and doing it twice halves the preview rate.
    pose = calibrate.detect_board(
      gray, self.reader.K, self.reader.dist, self.board)
    found = (pose["object_points"], pose["image_points"]) \
      if pose is not None else None
    # A failed full-frame 11x8 checkerboard search is substantially more
    # expensive than the compact ChArUco detection.  During hand-eye capture
    # the loose table board is normally absent, so searching for it on every
    # 1280x720 preview frame only makes the GUI lag.  Poll slowly while absent
    # and promptly increase the rate after acquisition.  Hand-eye detection
    # remains full-rate, and every table sample still uses the original image.
    now = time.monotonic()
    with self.lock:
      table_was_detected = bool(self._table_live.get("detected"))
    table_interval = (TABLE_TRACK_INTERVAL_S if table_was_detected
                      else TABLE_SEARCH_INTERVAL_S)
    table_scan = now - self._last_table_scan_s >= table_interval
    table_pose = None
    if table_scan:
      self._last_table_scan_s = now
      table_pose = calibrate.detect_board(
        gray, self.reader.K, self.reader.dist, self.table_board)
    table_found = (table_pose["object_points"], table_pose["image_points"]) \
      if table_pose is not None else None

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
    self._ring.append((time.time(), pose))

    table_live: dict = {
      "detected": table_pose is not None,
      "pattern": self.table_board.describe(),
    }
    T_ct = None
    if table_pose is not None:
      T_ct = calibrate._rt(table_pose["rvec"], table_pose["tvec"])
      centre_obj = self.table_board.object_points().mean(0)
      centre_cam = T_ct[:3, :3] @ centre_obj + T_ct[:3, 3]
      table_live.update(
        n_corners=table_pose["n_corners"],
        reproj_px=round(table_pose["reproj_rms_px"], 3),
        range_mm=round(float(np.linalg.norm(centre_cam)) * 1000, 1),
        center_cam=centre_cam.tolist())
      with self.lock:
        T_bc = (np.asarray(self.solution["T_base_cam"], dtype=np.float64)
                if self.solution and self.solution.get("ok")
                else (self.guidance_T.copy()
                      if self.guidance_T is not None else None))
      if T_bc is not None:
        n = T_bc[:3, :3] @ T_ct[:3, 2]
        if n[2] < 0:
          n = -n
        table_live["tilt_deg"] = round(float(np.degrees(
          np.arccos(np.clip(n[2], -1.0, 1.0)))), 3)
    if table_scan:
      if table_pose is None:
        # Do not let old detections keep the table gate green after the board
        # has left the image; reacquisition must collect a fresh fusion set.
        self._table_ring.clear()
      else:
        self._table_ring.append((time.time(), table_pose))
    else:
      # Preserve the most recent table status between the deliberately sparse
      # scans.  The overlay is omitted because its pixels belong to an older
      # frame, but the status panel should not flicker to "not detected".
      with self.lock:
        table_live = dict(self._table_live)
    table_live["still"], table_live["still_detail"] = self._still_ring(
      self._table_ring)
    table_valid = sum(p is not None for _, p in self._table_ring)
    table_live["fusion_frames"] = table_valid
    table_live["fusion_required"] = FUSION_MIN_FRAMES
    table_live["novel"], table_live["novel_detail"] = \
      self._table_novel(table_live.get("center_cam"))

    # A confirmed recovery move has done its job as soon as the board is
    # visible again.  Clear it so the ordinary planner may use this recovered
    # view without requiring a duplicate sample at the old pose.
    with self.lock:
      recovered = bool(T_cb is not None and self.next_target is not None
                       and self.next_target.get("recovery")
                       and self.next_target.get("status") == "reached")
      if recovered:
        self.next_target = None
        self.recovery_q = None
        self.motion = {"status": "idle", "progress": 0.0,
                       "why": "board visible again"}

    live["still"], live["still_detail"] = self._still()
    valid_fusion = sum(p is not None for _, p in self._ring)
    live["fusion_frames"] = valid_fusion
    live["fusion_required"] = FUSION_MIN_FRAMES
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
        # Keep the last *settled* visible configuration.  During a move the
        # board may flash through a barely detectable edge view; using that as
        # the retreat point would defeat the recovery button's purpose.
        with self.lock:
          if (live["still"]
              and self.motion.get("status") != "moving"):
            self.recovery_q = st.q.copy()

    self._maybe_plan(T_cb, st)
    with self.lock:
      target = dict(self.next_target) if self.next_target is not None else None

    with self.lock:
      self._live = live
      self._table_live = table_live
      self._overlay_found = found
      if table_scan:
        # Keep the last successful table overlay between the intentionally
        # sparse scans.  Clearing it on every skipped scan made a 100%-healthy
        # detector look as if it were flashing at half the preview rate.
        self._overlay_table_found = table_found
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

  def _still_ring(self, ring: deque) -> tuple[bool, str]:
    poses = [p for _, p in ring if p is not None]
    if len(poses) < FUSION_MIN_FRAMES:
      return False, (f"collecting stable detections ({len(poses)}/"
                     f"{FUSION_MIN_FRAMES} frames)")
    Ts = [calibrate._rt(p["rvec"], p["tvec"]) for p in poses]
    ref = Ts[-1]
    dt = max(float(np.linalg.norm(T[:3, 3] - ref[:3, 3])) for T in Ts) * 1000
    dr = max(_ang(T[:3, :3], ref[:3, :3]) for T in Ts)
    if dt > self.still_mm or dr > self.still_deg:
      return False, (f"moving ({dt:.1f} mm, {dr:.2f} deg over "
                     f"{len(Ts)} detections)")
    return True, f"still ({dt:.1f} mm, {dr:.2f} deg; {len(Ts)} fused frames)"

  def _still(self) -> tuple[bool, str]:
    return self._still_ring(self._ring)

  def _fused_detection(self) -> dict | None:
    detections = [p for _, p in list(self._ring) if p is not None]
    return calibrate.fuse_detections(
      detections, self.reader.K, self.reader.dist, self.board,
      min_frames=FUSION_MIN_FRAMES)

  def _fused_table_detection(self) -> dict | None:
    detections = [p for _, p in list(self._table_ring) if p is not None]
    return calibrate.fuse_detections(
      detections, self.reader.K, self.reader.dist, self.table_board,
      min_frames=FUSION_MIN_FRAMES)

  def _table_novel(self, center_cam) -> tuple[bool, str]:
    if center_cam is None:
      return False, "11x8 checkerboard not detected"
    centre = np.asarray(center_cam, dtype=np.float64)
    with self.lock:
      records = list(self.table_records)
    if not records:
      return True, "first table sample"
    distances = [float(np.linalg.norm(
      centre - np.asarray(r["center_cam"], dtype=np.float64))) * 1000
      for r in records]
    nearest = min(distances)
    if nearest < TABLE_NOVEL_MM:
      return False, (f"only {nearest:.0f} mm from the nearest table sample; "
                     f"move the board at least {TABLE_NOVEL_MM:.0f} mm")
    return True, f"new table area ({nearest:.0f} mm from nearest sample)"

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
        min_rotation_span_deg=GUIDANCE_MIN_ROT_SPAN_DEG,
        K=self.reader.K, dist=self.reader.dist)
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
        mount = target.get("T_grip_board")
        if mount is not None:
          self.board_mount_T = np.asarray(mount, dtype=np.float64)

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

  def _preview_run(self) -> None:
    """Publish smooth camera frames independently of expensive detection.

    D455 corner detection remains on the untouched 1280x720 frames in
    ``_tick``.  Rendering it in that same loop made the browser inherit the
    detector's variable cadence.  This loop samples the reader at a steady
    rate and overlays the most recent detection, which is normally less than
    one tenth of a second old.
    """
    deadline = time.monotonic()
    while not self._stop.is_set():
      frame = self.reader.latest()
      if frame is not None:
        with self.lock:
          found = self._overlay_found
          table_found = self._overlay_table_found
          live = dict(self._live)
          table_live = dict(self._table_live)
          target = (dict(self.next_target)
                    if self.next_target is not None else None)
        buf = self._render(np.asarray(frame.gray, dtype=np.uint8), found, live,
                           target, table_found, table_live)
        with self.lock:
          self._jpeg = buf
      deadline += 1.0 / GUI_RATE_HZ
      delay = deadline - time.monotonic()
      if delay > 0:
        time.sleep(delay)
      else:
        deadline = time.monotonic()

  def _render(self, gray: np.ndarray, found, live: dict,
              target: dict | None = None, table_found=None,
              table_live: dict | None = None) -> bytes:
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
    elif table_found is None:
      cv2.putText(img, "board not detected", (16, 34),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 80, 240), 2, cv2.LINE_AA)
    if table_found is not None:
      _, pts = table_found
      ready = bool(table_live and table_live.get("still")
                   and table_live.get("novel"))
      colour = (220, 110, 255) if ready else (180, 90, 210)
      xy = pts.reshape(-1, 2)
      for p in xy:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, colour, -1, cv2.LINE_AA)
      hull = cv2.convexHull(np.rint(xy).astype(np.int32).reshape(-1, 1, 2))
      cv2.polylines(img, [hull], True, colour, 2, cv2.LINE_AA)
      lo = xy.min(0).astype(int)
      cv2.putText(img, "TABLE 11x8 / 25mm", (int(lo[0]), max(24, int(lo[1]) - 10)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    # Detection, pose estimation and fusion above always use the untouched
    # 1280x720 grayscale.  The browser does not need those extra transport
    # pixels, so downscale only after all full-resolution overlays are drawn.
    if img.shape[1] > PREVIEW_WIDTH:
      h = int(round(img.shape[0] * PREVIEW_WIDTH / img.shape[1]))
      img = cv2.resize(img, (PREVIEW_WIDTH, h), interpolation=cv2.INTER_AREA)
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
      table_live = dict(self._table_live)
      records = list(self.records)
      table_records = list(self.table_records)
      target = dict(self.next_target) if self.next_target is not None else None
      motion = dict(self.motion)
      recovery_q = (None if self.recovery_q is None
                    else self.recovery_q.copy())
      _, last_st = getattr(self, "_last", (None, None))
    recovery = None
    if recovery_q is not None:
      delta = (float("inf") if last_st is None else
               float(np.max(np.abs(recovery_q - last_st.q))))
      available = bool(np.isfinite(delta)
                       and delta <= RECOVERY_MAX_MOTION_RAD)
      recovery = {
        "available": available,
        "q_deg": [round(float(x), 1) for x in np.degrees(recovery_q)],
        "motion_deg": (None if not np.isfinite(delta)
                       else round(math.degrees(delta), 1)),
        "why": ("" if available else
                ("no current arm feedback" if not np.isfinite(delta) else
                 f"last visible pose is {math.degrees(delta):.1f} deg away; "
                 "30 deg automatic-recovery limit exceeded")),
      }
    return {
      "board": {"describe": self.board.describe(),
                "n_corners": self.board.n_corners(),
                "suggested_poses": self.board.suggested_poses()},
      "live": live,
      "table_live": table_live,
      "table_board": {
        "describe": self.table_board.describe(),
        "min_samples": TABLE_MIN_SAMPLES,
        "recommended_samples": TABLE_RECOMMENDED_SAMPLES,
      },
      "n_table_samples": len(table_records),
      "table_samples": [
        {"i": i,
         "reproj_px": round(float(r.get("reproj_rms_px", 0.0)), 3),
         "fusion_frames": r.get("fusion_frames"),
         "corner_spread_px": (round(float(r["corner_spread_px"]), 3)
                              if r.get("corner_spread_px") is not None
                              else None),
         "center_cam_mm": [round(float(x) * 1000, 1)
                           for x in r.get("center_cam", [0, 0, 0])]}
        for i, r in enumerate(table_records)],
      "n_poses": len(records),
      "min_poses": calibrate.MIN_POSES,
      "rotation_span_deg": round(self.rotation_span(records), 1),
      "min_rotation_span_deg": calibrate.MIN_ROT_SPAN_DEG,
      "poses": [{"i": i, "n_corners": r.get("n_corners"),
                 "reproj_px": round(r.get("reproj_rms_px", 0.0), 2),
                 "fusion_frames": r.get("fusion_frames"),
                 "corner_spread_px": (round(r["corner_spread_px"], 2)
                                      if r.get("corner_spread_px") is not None
                                      else None),
                 "normal": self._normal(calibrate._rt(r["rvec"], r["tvec"]))}
                for i, r in enumerate(records)],
      "solution": self.solution,
      "guidance": self.guidance,
      "guidance_failure": self.guidance_failure,
      "next_target": target,
      "motion": motion,
      "recovery": recovery,
      "image_source": (f"{self.reader.meta.get('model', self.camera_backend)} "
                       f"raw left grayscale "
                       f"{self.reader.meta['resolution'][0]}x"
                       f"{self.reader.meta['resolution'][1]}"),
      "guidance_min_poses": GUIDANCE_MIN_POSES,
      "guidance_min_rotation_span_deg": GUIDANCE_MIN_ROT_SPAN_DEG,
      "table": self.table,
      "saved": self.saved,
      "has_arm": self.arm is not None,
      "note": self.note,
      "error": self.error,
      "poses_file": str(self.poses_file),
      "table_poses_file": str(self.table_poses_file),
      "rig_file": str(self.rig_file),
    }

  # -- what the buttons do ---------------------------------------------------

  def record(self) -> dict:
    with self.lock:
      live = dict(self._live)
      _, st = getattr(self, "_last", (None, None))
      moving = self.motion.get("status") == "moving"
    if moving:
      return {"ok": False, "why": "arm is moving"}
    if not live.get("still"):
      return {"ok": False, "why": live.get("still_detail", "moving")}
    fused = self._fused_detection()
    if fused is None:
      return {"ok": False,
              "why": "not enough consistent board corners to fuse"}
    T_cb = calibrate._rt(fused["rvec"], fused["tvec"])
    if not live.get("novel"):
      return {"ok": False, "why": live.get("novel_detail", "not a new pose")}
    if not live.get("arm_novel"):
      return {"ok": False, "why": live.get(
        "arm_novel_detail", "arm pose has not changed")}
    if st is None:
      return {"ok": False, "why": "no arm; poses cannot be recorded"}
    rec = {
      "joint_pos": [*st.q.tolist(), st.gripper, -st.gripper],
      "rvec": np.asarray(fused["rvec"]).ravel().tolist(),
      "tvec": np.asarray(fused["tvec"]).ravel().tolist(),
      "n_corners": fused["n_corners"],
      "reproj_rms_px": fused["reproj_rms_px"],
      "fusion_frames": fused["fusion_frames"],
      "corner_spread_px": fused["corner_spread_px"],
      "object_points": np.asarray(fused["object_points"]).tolist(),
      "image_points": np.asarray(fused["image_points"]).tolist(),
      "camera_K": self.reader.K.tolist(),
      "camera_dist": self.reader.dist.tolist(),
      "image_size": list(self.reader.meta["resolution"]),
      "image_source": self.reader.meta.get("gray_source"),
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

  def record_table(self) -> dict:
    """Record one settled placement of the loose tabletop checkerboard."""
    with self.lock:
      live = dict(self._table_live)
      moving = self.motion.get("status") == "moving"
    if moving:
      return {"ok": False, "why": "arm is moving"}
    if not live.get("still"):
      return {"ok": False, "why": live.get(
        "still_detail", "table board is moving")}
    if not live.get("novel"):
      return {"ok": False, "why": live.get(
        "novel_detail", "move the table board to a new area")}
    fused = self._fused_table_detection()
    if fused is None:
      return {"ok": False,
              "why": "not enough consistent checkerboard corners to fuse"}
    T_ct = calibrate._rt(fused["rvec"], fused["tvec"])
    centre_obj = self.table_board.object_points().mean(0)
    centre_cam = T_ct[:3, :3] @ centre_obj + T_ct[:3, 3]
    rec = {
      "rvec": np.asarray(fused["rvec"]).ravel().tolist(),
      "tvec": np.asarray(fused["tvec"]).ravel().tolist(),
      "center_cam": centre_cam.tolist(),
      "n_corners": fused["n_corners"],
      "reproj_rms_px": fused["reproj_rms_px"],
      "fusion_frames": fused["fusion_frames"],
      "corner_spread_px": fused["corner_spread_px"],
      "object_points": np.asarray(fused["object_points"]).tolist(),
      "image_points": np.asarray(fused["image_points"]).tolist(),
      "camera_K": self.reader.K.tolist(),
      "camera_dist": self.reader.dist.tolist(),
      "image_size": list(self.reader.meta["resolution"]),
      "image_source": self.reader.meta.get("gray_source"),
    }
    with self.lock:
      self.table_records.append(rec)
      self.table = None
      self.saved = False
      calibrate.save_poses(
        self.table_board, self.table_records, self.table_poses_file)
    return {"ok": True, "n_table_samples": len(self.table_records)}

  def drop_table(self, i: int) -> dict:
    with self.lock:
      if not (0 <= i < len(self.table_records)):
        return {"ok": False, "why": "no such table sample"}
      self.table_records.pop(i)
      self.table = None
      self.saved = False
      calibrate.save_poses(
        self.table_board, self.table_records, self.table_poses_file)
    return {"ok": True}

  def reset_table(self) -> dict:
    with self.lock:
      self.table_records.clear()
      self._table_ring.clear()
      self.table = None
      self.saved = False
      calibrate.save_poses(
        self.table_board, self.table_records, self.table_poses_file)
    return {"ok": True}

  def solve(self) -> dict:
    with self.lock:
      records = list(self.records)
    if len(records) < calibrate.MIN_POSES:
      return {"ok": False, "why": f"{len(records)} poses; "
                                  f"{calibrate.MIN_POSES} is the minimum"}
    out = calibrate.solve(records, board=self.board,
                          K=self.reader.K, dist=self.reader.dist)
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
      "residual_all_mm": round(out.get("residual_all_mm", out["residual_mm"]), 2),
      "n_inliers": out.get("n_inliers", len(records)),
      "outlier_poses": out.get("outlier_poses", []),
      "rot_span_deg": round(out["rot_span_deg"], 1),
      "n_flipped": out.get("n_flipped", 0),
      "position_m": [round(float(x), 4) for x in T[:3, 3]],
      "vs_sim_mm": round(float(np.linalg.norm(T[:3, 3] - nominal[:3, 3]))
                         * 1000, 1),
      "vs_sim_deg": round(_ang(T[:3, :3], nominal[:3, :3]), 2),
      "T_base_cam": T.tolist(),
      "joint_refined": bool(out.get("joint_refined")),
      "reprojection_rms_px": (round(out["reprojection_rms_px"], 3)
                              if out.get("joint_refined") else None),
      "reprojection_before_px": (round(out["reprojection_before_px"], 3)
                                 if out.get("joint_refined") else None),
      "n_corner_observations": out.get("n_corner_observations"),
    }
    guide_ok = (out["residual_mm"] <= 35.0
                and _plausible_camera_transform(T))
    with self.lock:
      self._invalidate_target()
      if guide_ok:
        self.guidance_T = T.copy()
        self.guidance = {
          "source": (f"accepted robust solve from "
                     f"{out.get('n_inliers', len(records))}/{len(records)} poses"),
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

  def return_visible(self) -> dict:
    """Offer a click-confirmed retreat to the last board-visible joint pose."""
    with self.lock:
      if self.arm is None:
        return {"ok": False, "why": "no arm connected"}
      if self.motion.get("status") == "moving":
        return {"ok": False, "why": "arm is already moving"}
      if self.recovery_q is None:
        return {"ok": False, "why": "no recovery pose is available"}
      _, st = getattr(self, "_last", (None, None))
      if st is None:
        return {"ok": False, "why": "no current arm feedback"}
      goal = self.recovery_q.copy()
      delta = float(np.max(np.abs(goal - st.q)))
      if not np.isfinite(delta) or delta > RECOVERY_MAX_MOTION_RAD:
        return {
          "ok": False,
          "why": (f"last board-visible pose is {math.degrees(delta):.1f} deg "
                  "away; refusing automatic recovery above 30 deg. "
                  "Reposition manually until the board is visible."),
        }
      mount = getattr(self, "board_mount_T", None)
      if mount is None:
        return {"ok": False,
                "why": "board payload pose is unknown; refusing automatic recovery"}
      self.next_target = {
        "available": True,
        "recovery": True,
        "q": [*goal.tolist(), float(st.gripper)],
        "from_q": [*st.q.tolist(), float(st.gripper)],
        "q_deg": [round(float(x), 1) for x in np.degrees(goal)],
        "motion_deg": round(math.degrees(delta), 1),
        "T_grip_board": np.asarray(mount).tolist(),
        "required_board_clearance_mm": BOARD_TABLE_CLEARANCE_M * 1000.0,
        "status": "ready",
      }
      self.motion = {"status": "idle", "progress": 0.0}
    return self.move_next()

  def _move_worker(self, target: dict) -> None:
    """Execute one confirmed target, with feedback and a tracking watchdog."""
    try:
      with self.arm_lock:
        start = self.arm.read()
      mount_raw = target.get("T_grip_board")
      if mount_raw is None:
        raise RuntimeError(
          "target has no calibration-board payload transform; refusing motion")
      mount = np.asarray(mount_raw, dtype=np.float64)
      start7 = np.array([*start.q, start.gripper], dtype=np.float64)
      goal7 = np.asarray(target["q"], dtype=np.float64)
      safe, preflight_clearance, reason = self.planner.validate_path(
        start7, goal7, mount)
      if not safe:
        raise RuntimeError("execution preflight rejected target: " + reason)
      with self.arm_lock:
        # Preload the measured pose while drives are still disabled.  PiPER
        # may retain an old position target across enable cycles; enabling
        # before replacing it can produce an immediate jump towards that stale
        # target.
        self.arm.command(np.array([*start.q, start.gripper]))
        self.arm.enable()
        # Repeat after enable, before the first interpolated command.
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
          actual7 = np.array([*st.q, st.gripper], dtype=np.float64)
          clearance = self.planner.board_clearance(actual7, mount)
          if clearance < BOARD_TABLE_CLEARANCE_M:
            raise RuntimeError(
              f"live calibration-board clearance {clearance * 1000:.0f} mm "
              f"is below {BOARD_TABLE_CLEARANCE_M * 1000:.0f} mm")
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
            "board_clearance_mm": round(clearance * 1000.0, 1),
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
    deployment_K = self.reader.intrinsics(
      config.D405_WIDTH, config.D405_HEIGHT)
    rig = config.Rig(T_base_cam=T, K=deployment_K,
                     serial=self.reader.serial,
                     residual_mm=self.solution["residual_mm"])
    frame = self.reader.latest()
    with self.lock:
      table_records = list(self.table_records)
    if len(table_records) >= TABLE_MIN_SAMPLES:
      try:
        t = calibrate.fit_table_board_samples(
          table_records, T, self.table_board)
        rig.table_z = t["table_z"]
        rig.table_tilt_deg = t["tilt_deg"]
        rig.table_flatness_mm = t["flatness_mm"]
        rig.table_normal_base = np.asarray(t["normal_base"], dtype=np.float64)
        self.table = self._display_table(t)
      except RuntimeError as e:
        self.table = {"error": str(e)}
    elif frame is not None:
      try:
        t = calibrate.fit_table(frame.depth, T, self.reader.K)
        rig.table_z = t["table_z"]
        rig.table_tilt_deg = t["tilt_deg"]
        rig.table_flatness_mm = t["flatness_mm"]
        rig.table_normal_base = np.asarray(t["normal_base"], dtype=np.float64)
        self.table = self._display_table(t)
      except RuntimeError as e:
        self.table = {"error": str(e)}
    rig.save(self.rig_file)
    self.saved = True
    return {"ok": True, "rig_file": str(self.rig_file), "table": self.table}

  @staticmethod
  def _display_table(t: dict) -> dict:
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating))
                else [round(float(x), 4) for x in v]
                if isinstance(v, list) else v)
            for k, v in t.items()}

  def save_table(self) -> dict:
    """Fit the moved checkerboard samples and update the saved rig."""
    if not (self.solution and self.solution.get("ok")):
      return {"ok": False, "why": "solve hand-eye first"}
    with self.lock:
      records = list(self.table_records)
    if len(records) < TABLE_MIN_SAMPLES:
      return {"ok": False,
              "why": (f"need {TABLE_MIN_SAMPLES} table samples; "
                      f"have {len(records)}")}
    T = np.asarray(self.solution["T_base_cam"], dtype=np.float64)
    try:
      t = calibrate.fit_table_board_samples(records, T, self.table_board)
    except RuntimeError as e:
      return {"ok": False, "why": str(e)}
    if max(t["coverage_x_mm"], t["coverage_y_mm"]) < 100.0:
      return {"ok": False, "why": (
        "table samples cover less than 100 mm; move the board farther across "
        "the tabletop before saving"), "table": self._display_table(t)}
    if t["flatness_mm"] > 10.0:
      return {"ok": False, "why": (
        f"placements disagree by {t['flatness_mm']:.1f} mm; keep the board "
        "flat and drop/recollect the worst sample"),
        "table": self._display_table(t)}
    deployment_K = self.reader.intrinsics(
      config.D405_WIDTH, config.D405_HEIGHT)
    rig = config.Rig(
      T_base_cam=T, K=deployment_K, serial=self.reader.serial,
      residual_mm=self.solution["residual_mm"], table_z=t["table_z"],
      table_normal_base=np.asarray(t["normal_base"], dtype=np.float64),
      table_tilt_deg=t["tilt_deg"],
      table_flatness_mm=t["flatness_mm"])
    rig.save(self.rig_file)
    with self.lock:
      self.table = self._display_table(t)
      self.saved = True
    return {"ok": True, "rig_file": str(self.rig_file),
            "table": self.table}

  def close(self) -> None:
    self._stop.set()
    self._motion_cancel.set()
    if self._motion_thread is not None:
      self._motion_thread.join(timeout=3.0)
    self._thread.join(timeout=2.0)
    self._preview_thread.join(timeout=2.0)
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
    elif path == "/record-table":
      self._json(sess.record_table())
    elif path == "/drop":
      self._json(sess.drop(int(body.get("i", -1))))
    elif path == "/drop-table":
      self._json(sess.drop_table(int(body.get("i", -1))))
    elif path == "/reset-table":
      self._json(sess.reset_table())
    elif path == "/solve":
      self._json(sess.solve())
    elif path == "/save":
      self._json(sess.save())
    elif path == "/save-table":
      self._json(sess.save_table())
    elif path == "/move-next":
      self._json(sess.move_next())
    elif path == "/stop-motion":
      self._json(sess.stop_motion())
    elif path == "/return-visible":
      self._json(sess.return_visible())
    else:
      self.send_error(404)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--port", type=int, default=8771)
  p.add_argument("--serial", default=None)
  p.add_argument("--camera", choices=("d405", "d455"), default="d455",
                 help="RealSense model to calibrate")
  p.add_argument("--emitter", choices=("on", "off"), default=None,
                 help="D455 infrared projector during calibration; defaults "
                      "to off so its dot pattern cannot perturb corners")
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--no-arm", action="store_true",
                 help="camera only; poses cannot be recorded, but the board "
                      "and the framing can be checked without CAN")
  p.add_argument("--poses", default=None,
                 help="hand-eye samples; defaults to a camera-specific file")
  p.add_argument("--table-poses", default=None,
                 help="table samples; defaults to a camera-specific file")
  p.add_argument("--rig-file", default=None,
                 help="calibration output; defaults to rig.json for D405 and "
                      "rig_<camera>.json otherwise")
  p.add_argument("--calib-width", type=int, default=CALIB_WIDTH)
  p.add_argument("--calib-height", type=int, default=CALIB_HEIGHT)
  p.add_argument("--aligned-gray", action="store_true",
                 help="fallback to the old depth-aligned colour grayscale; "
                      "raw unwarped left grayscale is the default")
  p.add_argument("--still-mm", type=float, default=STILL_MM)
  p.add_argument("--still-deg", type=float, default=STILL_DEG)
  p.add_argument("--recovery-q-deg", default=None,
                 help=argparse.SUPPRESS)
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

  # The D455 projector helps passive-stereo depth in textureless scenes, but
  # its dots contaminate the printed black/white edges used for sub-pixel pose
  # estimation.  Keep deployment's D455 default on; only calibration defaults
  # to a clean passive left-IR image.
  emitter = a.emitter
  if emitter is None and a.camera == "d455":
    emitter = "off"

  suffix = "" if a.camera == "d405" else f"_{a.camera}"
  poses_file = Path(a.poses) if a.poses else Path(
    calibrate.POSES_FILE).with_name(f"calib_poses{suffix}.json")
  table_poses_file = Path(a.table_poses) if a.table_poses else Path(
    TABLE_POSES_FILE).with_name(f"calib_table_poses{suffix}.json")
  rig_file = Path(a.rig_file) if a.rig_file else Path(
    config.RIG_FILE).with_name(f"rig{suffix}.json")

  board = calibrate.board_from_args(a)
  recovery_q = None
  if a.recovery_q_deg:
    values = [float(x) for x in a.recovery_q_deg.split(",")]
    if len(values) != 6:
      p.error("--recovery-q-deg needs six comma-separated joint angles")
    recovery_q = np.radians(values)
  print(f"board: {board.describe()}", flush=True)
  sess = Session(board, a.serial, a.can, a.no_arm, poses_file,
                 a.still_mm, a.still_deg, recovery_q,
                 a.calib_width, a.calib_height, not a.aligned_gray,
                 table_poses_file, a.camera, rig_file, emitter)
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
