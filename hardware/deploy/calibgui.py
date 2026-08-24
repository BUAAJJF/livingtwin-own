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
"""

from __future__ import annotations

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
import numpy as np

from . import calibrate, config, proprio

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


def _ang(Ra: np.ndarray, Rb: np.ndarray) -> float:
  return calibrate._angle_between(Ra, Rb)


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

    self.lock = threading.Lock()
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
    self.reader = sensor.Reader(serial=serial)
    self.reader.wait_for_first()

    self.arm = None
    if not no_arm:
      from . import robot
      self.arm = robot.PiperArm(can)
      self.arm.connect()
    self.kin = proprio.Kinematics()

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
    gray = frame.gray
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

    st = None
    if self.arm is not None:
      st = self.arm.read()
      live["arm"] = {
        "q_deg": [round(float(x), 2) for x in np.degrees(st.q)],
        "gripper_mm": round(st.gripper * 1000, 1),
      }
      if T_cb is not None:
        live["on_gripper"] = self._on_gripper(st, T_cb)

    with self.lock:
      self._live = live
      self._jpeg = self._render(gray, found, live)
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

  def _on_gripper(self, st, T_cb: np.ndarray) -> dict:
    """Is the board actually held by the arm?

    Eye-to-hand needs the board carried by the gripper: one lying on the table
    has an unknown pose in both frames and constrains nothing, and the symptom
    is a calibration that refuses at the residual gate after every pose has
    been collected.  This catches it before the first one.

    The extrinsic is the thing being measured, so this uses the *nominal* one
    and a tolerance wide enough to survive a badly aimed mount.  It answers
    "within an arm's wrist" or "somewhere else entirely", which is the only
    resolution needed.
    """
    self.kin.update(np.array([*st.q, st.gripper, -st.gripper]))
    grip = self.kin.data.site_xpos[self.kin.site_id].copy()
    board = (config.sim_camera_extrinsic() @ np.append(T_cb[:3, 3], 1.0))[:3]
    d = float(np.linalg.norm(board - grip))
    return {"separation_mm": round(d * 1000, 0), "ok": bool(d < 0.25)}

  # -- the preview -----------------------------------------------------------

  def _render(self, gray: np.ndarray, found, live: dict) -> bytes:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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

  def rotation_span(self) -> float:
    """The gate the solver applies, computed the way the solver computes it.

    From the *gripper* rotations, not the board's: ``solve`` measures the span
    of ``R_bg`` and refuses under ``MIN_ROT_SPAN_DEG``.  Reporting the board's
    spread instead would be a different number that agrees most of the time.
    """
    Rs = []
    for r in self.records:
      self.kin.update(np.asarray(r["joint_pos"], dtype=np.float64))
      Rs.append(self.kin.data.site_xmat[self.kin.site_id].reshape(3, 3).T)
    return max((_ang(Rs[i], Rs[j]) for i in range(len(Rs))
                for j in range(i + 1, len(Rs))), default=0.0)

  def state(self) -> dict:
    with self.lock:
      live = dict(self._live)
      records = list(self.records)
    return {
      "board": {"describe": self.board.describe(),
                "n_corners": self.board.n_corners(),
                "suggested_poses": self.board.suggested_poses()},
      "live": live,
      "n_poses": len(records),
      "min_poses": calibrate.MIN_POSES,
      "rotation_span_deg": round(self.rotation_span(), 1),
      "min_rotation_span_deg": calibrate.MIN_ROT_SPAN_DEG,
      "poses": [{"i": i, "n_corners": r.get("n_corners"),
                 "reproj_px": round(r.get("reproj_rms_px", 0.0), 2),
                 "normal": self._normal(calibrate._rt(r["rvec"], r["tvec"]))}
                for i, r in enumerate(records)],
      "solution": self.solution,
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
    if T_cb is None:
      return {"ok": False, "why": "no board in view"}
    if not live.get("still"):
      return {"ok": False, "why": live.get("still_detail", "moving")}
    if not live.get("novel"):
      return {"ok": False, "why": live.get("novel_detail", "not a new pose")}
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
    return {"ok": True}

  def drop(self, i: int) -> dict:
    with self.lock:
      if not (0 <= i < len(self.records)):
        return {"ok": False, "why": "no such pose"}
      self.records.pop(i)
      self.solution = None
      self.saved = False
      calibrate.save_poses(self.board, self.records, self.poses_file)
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
    return self.solution

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
    self._thread.join(timeout=2.0)
    self.reader.close()
    if self.arm is not None:
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
  print(f"board: {board.describe()}")
  sess = Session(board, a.serial, a.can, a.no_arm, Path(a.poses),
                 a.still_mm, a.still_deg)
  httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
  httpd.session = sess
  print(f"\n  open http://127.0.0.1:{a.port}\n")
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    sess.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
