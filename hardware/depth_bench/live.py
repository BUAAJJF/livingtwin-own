#!/usr/bin/env python3
"""Live side-by-side comparison of every depth camera on the bench.

    python live.py            # then open http://127.0.0.1:8770

Press start, carry the cameras around the printed target together, and stop.
While the session runs, a coordinator watches the pose of the target in each
camera and takes a synchronised snapshot whenever the viewpoint has become
*new* -- far enough from every snapshot already taken -- and everything is
momentarily still.  Stop produces the comparison.

Three decisions are worth stating, because each of them is what makes the
resulting comparison mean anything.

**The cameras are snapshotted at the same instant, on the same scene.**  The
alternative -- characterise one camera, unplug it, characterise the next -- lets
the lamp, the table and the operator drift between them, and those differences
land on the camera's score.  Here a shot is a pair, and the pair shares
everything except the sensor.

**The trigger is novelty, not a timer.**  A timer rewards standing still and
fills the session with the same viewpoint measured fifty times, which looks
like a lot of data and constrains nothing.  A shot is taken only when the pose
is far from every shot already held, so the captures spread themselves over
distance and angle without anyone having to plan them.

**Nothing is captured while anything is moving.**  A rolling-shutter smear or a
motion-blurred stereo match is not sensor noise, but it is indistinguishable
from it once it is in the numbers.  The worker tracks how long the view has
been still, and a shot needs the whole ring buffer to have been still -- so the
frames that get measured all predate the trigger and none of them contain the
approach.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics as M
import view
from capture import Capture, open_backend, rays_from_K

VIS_NEAR, VIS_FAR = 0.10, 1.50
"""Depth colour scale, shared by every camera so the panels are comparable by
eye.  1.5 m is CUTOFF_M from src/piper_push/camera.py -- the far plane the
policy's depth normalisation uses."""

RING = 24
"""Frames kept per camera.  The same stack length measure.py uses for its
temporal statistics, so live shots and one-shot captures are comparable."""


# --------------------------------------------------------------------------
# one camera


class CameraWorker(threading.Thread):
  """Owns a Stream, and is the only thread that touches it."""

  def __init__(self, name: str, backend, args, serial: str | None, board, spec):
    super().__init__(daemon=True)
    self.name = name
    self.backend = backend
    self.args = args
    self.serial = serial
    self.board, self.spec = board, spec

    self.lock = threading.Lock()
    self.ring: deque = deque(maxlen=RING)
    self.gray: np.ndarray | None = None
    self.depth: np.ndarray | None = None
    self.pose: dict | None = None
    self.pose_age = 0.0
    self.motion = float("inf")
    self.still_frames = 0
    self.fps = 0.0
    self.error: str | None = None
    self.meta: dict = {}
    self.K = self.dist = self.rays = None
    self.T_dg = np.eye(4)
    self._stop = threading.Event()
    self._prev_small: np.ndarray | None = None
    self._jpeg: bytes | None = None
    self._jpeg_at = 0.0

  # -- the loop ------------------------------------------------------------

  def run(self) -> None:
    try:
      stream = self.backend.Stream(self.args, self.serial)
    except Exception as e:
      self.error = f"{type(e).__name__}: {e}"
      return
    self.K, self.dist, self.meta = stream.K, stream.dist, stream.meta
    # Taken from the stream when it has one -- a lidar's geometry is measured,
    # not derived -- and otherwise built from K once the first frame has said
    # what the depth grid actually is, rather than trusting a requested size.
    self.rays = getattr(stream, "rays", None)
    self.T_dg = getattr(stream, "T_dg", np.eye(4))

    n, t0, pose_every, i, fails = 0, time.time(), 2, 0, 0
    try:
      while not self._stop.is_set():
        try:
          depth, gray = stream.read()
          fails = 0
        except Exception as e:
          self.error = f"{type(e).__name__}: {e}"
          fails += 1
          # A camera that has stopped delivering frames usually will not start
          # again by itself -- the D405 wedges when the Odin 1 streams on the
          # same USB controller -- so recover it rather than showing a dead
          # panel for the rest of the session.
          if fails >= 25:
            self.error = "recovering the camera..."
            try:
              stream.close()
            except Exception:
              pass
            if hasattr(self.backend, "reset"):
              self.backend.reset(self.serial)
            try:
              stream = self.backend.Stream(self.args, self.serial)
              self.K, self.dist = stream.K, stream.dist
              fails = 0
            except Exception as e2:
              self.error = f"reopen failed: {type(e2).__name__}: {e2}"
              time.sleep(3.0)
          time.sleep(0.2)
          continue
        i += 1

        # Motion on a 1/4-scale image: enough to see a hand or a camera move,
        # cheap enough to run on every frame of every camera.
        small = cv2.resize(gray, None, fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        motion = (float(np.mean(np.abs(small - self._prev_small)))
                  if self._prev_small is not None else float("inf"))
        self._prev_small = small

        pose = None
        if i % pose_every == 0:
          try:
            pose = M.detect_pose(gray, self.K, self.dist, self.board)
          except RuntimeError:
            pose = None

        if self.rays is None:
          self.rays = rays_from_K(self.K, depth.shape)

        with self.lock:
          self.depth, self.gray = depth, gray
          self.ring.append(depth)
          self.motion = motion
          self.still_frames = (self.still_frames + 1
                               if motion < self.args.still else 0)
          if i % pose_every == 0:
            self.pose, self.pose_age = pose, time.time()
          self.error = None

        n += 1
        if n >= 15:
          self.fps, n, t0 = n / (time.time() - t0), 0, time.time()
    finally:
      stream.close()

  def stop(self) -> None:
    self._stop.set()

  # -- what the page asks for ---------------------------------------------

  def snapshot(self) -> Capture | None:
    """The ring buffer as a Capture, or None if it is not full yet."""
    with self.lock:
      if len(self.ring) < RING or self.gray is None:
        return None
      return Capture(depth=np.stack(self.ring), gray=self.gray.copy(),
                     K=self.K, dist=self.dist,
                     meta=dict(self.meta, n_frames=RING),
                     rays=self.rays, T_dg=self.T_dg)

  def state(self) -> dict:
    with self.lock:
      p = self.pose
      return {
        "name": self.name,
        "error": self.error,
        "fps": round(self.fps, 1),
        "motion": None if self.motion == float("inf") else round(self.motion, 2),
        "still_frames": self.still_frames,
        "ring": len(self.ring),
        "model": self.meta.get("model", "-"),
        "serial": self.meta.get("serial", "-"),
        "resolution": self.meta.get("resolution"),
        "emitter": self.meta.get("emitter"),
        "sees_target": p is not None,
        "distance_mm": None if p is None else round(p["plane_distance_m"] * 1000),
        "tilt_deg": None if p is None else round(p["tilt_deg"], 1),
        "corners": None if p is None else p["n_corners"],
        "reproj_px": None if p is None else round(p["reproj_rms_px"], 2),
      }

  def preview(self, width: int = 480) -> bytes:
    """Greyscale over depth, as one JPEG.  Cached: several browser tabs asking
    for the same stream must not each re-encode the same frame."""
    now = time.time()
    if self._jpeg is not None and now - self._jpeg_at < 0.05:
      return self._jpeg
    with self.lock:
      gray, depth, pose = self.gray, self.depth, self.pose
    if gray is None or depth is None:
      img = np.zeros((360, 640, 3), np.uint8)
      cv2.putText(img, self.error or "starting...", (20, 180),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 220), 2)
    else:
      img = view.compose(gray, depth, self.K, self.dist, self.rays, self.T_dg,
                         self.spec, pose_img=pose, width=width,
                         banner=None if pose is not None else "target not visible")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
    self._jpeg, self._jpeg_at = buf.tobytes(), now
    return self._jpeg


# --------------------------------------------------------------------------
# the session


def pose_gap(a: dict, b: dict) -> tuple[float, float]:
  """Translation (m) and rotation (deg) between two camera-to-target poses."""
  dt = float(np.linalg.norm(a["tvec"].ravel() - b["tvec"].ravel()))
  Rrel = a["R"].T @ b["R"]
  ang = float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
  return dt, ang


class Session:
  def __init__(self, workers, board, spec, args, outdir: Path):
    self.workers = workers
    self.board, self.spec, self.args = board, spec, args
    self.outdir = outdir
    self.lock = threading.Lock()
    self.running = False
    self.shots: list[dict] = []
    self.reason = "idle"
    self.novelty = 0.0
    self.report: dict | None = None
    self.busy = False
    self._t0 = 0.0
    self._last_fire = 0.0

  # -- the rule ------------------------------------------------------------

  def novelty_of(self, poses: dict) -> float:
    """How new this viewpoint is: 1.0 is exactly at threshold.

    Measured against *every* shot already taken, not just the last one.  Against
    only the last, walking back and forth between two places keeps scoring as
    new and fills the session with two viewpoints.
    """
    if not self.shots:
      return float("inf")
    worst = float("inf")
    for shot in self.shots:
      for name, p in poses.items():
        q = shot["poses"].get(name)
        if q is None:
          continue
        dt, ang = pose_gap(p, q)
        worst = min(worst, max(dt / self.args.d_dist, ang / self.args.d_angle))
    return worst

  def tick(self) -> None:
    with self.lock:
      if not self.running or self.busy:
        return
    poses, blocked = {}, None
    for w in self.workers:
      st = w.state()
      if st["error"]:
        blocked = f"{w.name}: {st['error']}"
        break
      with w.lock:
        p = w.pose
      if p is None:
        blocked = f"{w.name} cannot see the target"
        break
      if w.still_frames < RING:
        blocked = f"{w.name} still settling ({w.still_frames}/{RING})"
        break
      poses[w.name] = p
    if blocked:
      with self.lock:
        self.reason, self.novelty = blocked, 0.0
      return

    nov = self.novelty_of(poses)
    with self.lock:
      self.novelty = 99.0 if nov == float("inf") else nov
      if nov < 1.0:
        self.reason = "viewpoint too close to one already captured — move further"
        return
      if time.time() - self._last_fire < self.args.min_interval:
        self.reason = "waiting out the minimum interval"
        return
      self.busy = True
      self.reason = "capturing"
      self._last_fire = time.time()
    threading.Thread(target=self._fire, args=(poses,), daemon=True).start()

  def _fire(self, poses: dict, manual: bool = False) -> None:
    try:
      idx = len(self.shots)
      caps = {w.name: w.snapshot() for w in self.workers}
      if any(c is None for c in caps.values()):
        with self.lock:
          self.reason = "ring buffer not full yet"
        return
      shot = {"index": idx, "t": time.time() - self._t0, "manual": manual,
              "poses": poses, "cameras": {}}
      sd = self.outdir / f"shot{idx:02d}"
      sd.mkdir(parents=True, exist_ok=True)
      for name, cap in caps.items():
        try:
          res = M.evaluate(cap, self.board, self.spec)
        except Exception as e:
          shot["cameras"][name] = {"error": f"{type(e).__name__}: {e}"}
          continue
        res["capture"] = cap.meta
        shot["cameras"][name] = res
        worker = next(w for w in self.workers if w.name == name)
        (sd / f"{name}.jpg").write_bytes(worker.preview(400))
        if self.args.save_raw:
          cap.save(sd / f"{name}.npz")
      with self.lock:
        self.shots.append(shot)
        self.reason = f"captured shot {idx}"
      (sd / "metrics.json").write_text(json.dumps(
        {k: v for k, v in shot.items() if k != "poses"}, indent=2, default=str))
    except Exception:
      traceback.print_exc()
      with self.lock:
        self.reason = "capture failed, see console"
    finally:
      with self.lock:
        self.busy = False

  # -- controls ------------------------------------------------------------

  def start(self) -> None:
    with self.lock:
      self.running, self.shots, self.report = True, [], None
      self._t0 = self._last_fire = time.time()
      self.reason = "running"
    self.outdir.mkdir(parents=True, exist_ok=True)

  def manual(self) -> None:
    poses = {}
    for w in self.workers:
      with w.lock:
        if w.pose is not None:
          poses[w.name] = w.pose
    if not poses:
      with self.lock:
        self.reason = "no camera can see the target"
      return
    with self.lock:
      if self.busy:
        return
      self.busy = True
    threading.Thread(target=self._fire, args=(poses, True), daemon=True).start()

  def stop(self) -> dict:
    with self.lock:
      self.running = False
      self.reason = "building the comparison"
    rep = build_report(self.shots, [w.name for w in self.workers], self.outdir)
    with self.lock:
      self.report = rep
      self.reason = f"done — {len(self.shots)} shots"
    return rep

  def state(self) -> dict:
    with self.lock:
      return {
        "running": self.running, "busy": self.busy, "reason": self.reason,
        "novelty": round(self.novelty, 2), "n_shots": len(self.shots),
        "elapsed": round(time.time() - self._t0, 1) if self._t0 else 0.0,
        "has_report": self.report is not None,
        "shots": [{"index": s["index"], "t": round(s["t"], 1),
                   "manual": s["manual"],
                   "distance_mm": {k: (None if "error" in v else
                                       round(v["pose"]["distance_m"] * 1000))
                                   for k, v in s["cameras"].items()}}
                  for s in self.shots],
        "thresholds": {"d_dist": self.args.d_dist, "d_angle": self.args.d_angle,
                       "still": self.args.still,
                       "min_interval": self.args.min_interval},
      }


# --------------------------------------------------------------------------
# the comparison


REGIONS = ["charuco", "white", "black"]
LEGEND = {"charuco": "textured", "white": "blank white", "black": "solid black"}


def fit_quadratic(z, s) -> float:
  ok = np.isfinite(z) & np.isfinite(s)
  z, s = np.asarray(z)[ok], np.asarray(s)[ok]
  if z.size < 2:
    return float("nan")
  return float((z**2 @ s) / (z**2 @ z**2))


def build_report(shots, cam_names, outdir: Path) -> dict:
  rows = {c: {r: {"z": [], "rms": [], "fill": [], "bias": [], "temp": []}
              for r in REGIONS} for c in cam_names}
  for s in shots:
    for cam, res in s["cameras"].items():
      if "error" in res:
        continue
      for r in REGIONS:
        m = res["regions"].get(r, {})
        if not m.get("n_pixels"):
          continue
        rows[cam][r]["z"].append(m["range_m"])
        rows[cam][r]["rms"].append(m["spatial_rms_m"])
        rows[cam][r]["fill"].append(m["fill"])
        rows[cam][r]["bias"].append(m["bias_m"])
        rows[cam][r]["temp"].append(m["temporal_std_m"])

  summary = {}
  for cam in cam_names:
    summary[cam] = {}
    for r in REGIONS:
      d = rows[cam][r]
      a = fit_quadratic(d["z"], d["rms"])
      summary[cam][r] = {
        "n": len(d["z"]),
        "a_per_m": a,
        "sigma_at_0.70m": a * 0.70**2 if np.isfinite(a) else float("nan"),
        "fill_mean": float(np.mean(d["fill"])) if d["fill"] else float("nan"),
        "fill_min": float(np.min(d["fill"])) if d["fill"] else float("nan"),
        "bias_median_m": float(np.median(d["bias"])) if d["bias"] else float("nan"),
        "temporal_median_m": (float(np.nanmedian(d["temp"])) if d["temp"]
                              else float("nan")),
        "range_m": ([float(np.min(d["z"])), float(np.max(d["z"]))] if d["z"]
                    else None),
      }
  rep = {"cameras": cam_names, "n_shots": len(shots), "summary": summary,
         "raw": {c: {r: rows[c][r] for r in REGIONS} for c in cam_names}}
  fig = plot_report(rows, cam_names)
  if fig:
    (outdir / "comparison.png").write_bytes(fig)
    rep["figure"] = "comparison.png"
  outdir.mkdir(parents=True, exist_ok=True)
  (outdir / "report.json").write_text(json.dumps(rep, indent=2))
  return rep


def plot_report(rows, cam_names) -> bytes | None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  have = any(rows[c][r]["z"] for c in cam_names for r in REGIONS)
  if not have:
    return None
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
  colours = plt.cm.tab10.colors
  marks = {"charuco": "o", "white": "s", "black": "^"}

  for ci, cam in enumerate(cam_names):
    c = colours[ci % len(colours)]
    for r in REGIONS:
      d = rows[cam][r]
      if not d["z"]:
        continue
      z = np.array(d["z"])
      axes[0].scatter(z, np.array(d["rms"]) * 1000, color=c, marker=marks[r],
                      alpha=0.8, label=f"{cam} · {LEGEND[r]}")
      axes[1].scatter(z, np.array(d["fill"]) * 100, color=c, marker=marks[r],
                      alpha=0.8, label=f"{cam} · {LEGEND[r]}")
      axes[2].scatter(z, np.array(d["bias"]) * 1000, color=c, marker=marks[r],
                      alpha=0.8, label=f"{cam} · {LEGEND[r]}")
    d = rows[cam]["charuco"]
    if len(d["z"]) >= 2:
      a = fit_quadratic(d["z"], d["rms"])
      zz = np.linspace(min(d["z"]) * 0.9, max(max(d["z"]) * 1.1, 0.75), 50)
      axes[0].plot(zz, a * zz**2 * 1000, color=c, lw=1.4,
                   label=f"{cam} fit a={a:.4f}")

  axes[0].axvline(0.70, color="0.5", ls="--", lw=1)
  axes[0].text(0.70, axes[0].get_ylim()[1] * 0.95, " sim camera 0.70 m",
               color="0.4", fontsize=8, va="top")
  axes[0].set_xlabel("distance (m)")
  axes[0].set_ylabel("spatial RMS vs plane (mm)")
  axes[0].set_title("noise")
  axes[1].set_xlabel("distance (m)")
  axes[1].set_ylabel("fill (%)")
  axes[1].set_ylim(-2, 102)
  axes[1].set_title("dropout")
  axes[2].set_xlabel("distance (m)")
  axes[2].set_ylabel("bias vs plane (mm)")
  axes[2].axhline(0, color="0.7", lw=0.8)
  axes[2].set_title("bias")
  axes[1].legend(fontsize=7, loc="lower left")
  for ax in axes:
    ax.grid(alpha=0.25)
  fig.tight_layout()
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=130)
  plt.close(fig)
  return buf.getvalue()


# --------------------------------------------------------------------------
# server


class Handler(BaseHTTPRequestHandler):
  server_version = "depthbench/1.0"

  def log_message(self, *a):  # the console belongs to the capture threads
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
    app = self.server.app
    if path in ("/", "/index.html"):
      body = (HERE / "live.html").read_bytes()
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
    elif path == "/state":
      self._json({"session": app["session"].state(),
                  "cameras": [w.state() for w in app["workers"]]})
    elif path == "/report":
      self._json(app["session"].report or {})
    elif path.startswith("/stream/"):
      self._stream(path.rsplit("/", 1)[-1])
    elif path.startswith("/file/"):
      self._file(path[len("/file/"):])
    else:
      self.send_error(404)

  def _file(self, rel: str):
    app = self.server.app
    p = (app["outdir"] / rel).resolve()
    if not str(p).startswith(str(app["outdir"].resolve())) or not p.is_file():
      self.send_error(404)
      return
    body = p.read_bytes()
    self.send_response(200)
    self.send_header("Content-Type",
                     "image/png" if p.suffix == ".png" else "image/jpeg")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()
    self.wfile.write(body)

  def _stream(self, name: str):
    w = next((x for x in self.server.app["workers"] if x.name == name), None)
    if w is None:
      self.send_error(404)
      return
    self.send_response(200)
    self.send_header("Age", "0")
    self.send_header("Cache-Control", "no-cache, private")
    self.send_header("Content-Type",
                     "multipart/x-mixed-replace; boundary=frame")
    self.end_headers()
    try:
      while True:
        buf = w.preview()
        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                         b"Content-Length: " + str(len(buf)).encode() +
                         b"\r\n\r\n" + buf + b"\r\n")
        time.sleep(1 / 15)
    except (BrokenPipeError, ConnectionResetError):
      pass

  def do_POST(self):
    app = self.server.app
    sess = app["session"]
    n = int(self.headers.get("Content-Length") or 0)
    payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
    path = self.path.split("?")[0]
    if path == "/start":
      sess.start()
      self._json(sess.state())
    elif path == "/stop":
      self._json(sess.stop())
    elif path == "/shot":
      sess.manual()
      self._json(sess.state())
    elif path == "/config":
      for k in ("d_dist", "d_angle", "still", "min_interval"):
        if k in payload:
          setattr(sess.args, k, float(payload[k]))
      self._json(sess.state())
    else:
      self.send_error(404)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--port", type=int, default=8770)
  ap.add_argument("--target", type=Path, default=HERE / "targets" / "target_a4.json")
  ap.add_argument("--out", type=Path, default=HERE / "results" / "live")
  ap.add_argument("--backends", default="auto",
                  help="comma separated, or 'auto' to use whatever is plugged in")
  ap.add_argument("--d-dist", dest="d_dist", type=float, default=0.06,
                  help="metres of viewpoint change that counts as a new shot")
  ap.add_argument("--d-angle", dest="d_angle", type=float, default=12.0,
                  help="degrees of viewpoint change that counts as a new shot")
  ap.add_argument("--still", type=float, default=3.0,
                  help="grey levels of inter-frame motion still counted as still")
  ap.add_argument("--min-interval", dest="min_interval", type=float, default=1.0)
  ap.add_argument("--save-raw", action="store_true",
                  help="also write the frame stacks (~40 MB per camera per shot)")
  ap.add_argument("--width", type=int, default=848)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--preset", default="default")
  ap.add_argument("--depth-units", dest="depth_units", type=float, default=1e-4)
  ap.add_argument("--filters", action="store_true")
  ap.add_argument("--serial", default=None)
  # Odin 1 knobs.  Not pulled in via its add_args because that would collide
  # with --width/--height/--fps, which mean the D405's stream here.
  ap.add_argument("--odin-rate", type=int, default=2, choices=(0, 1, 2))
  ap.add_argument("--conf-min", type=int, default=30)
  ap.add_argument("--undistort-f", type=float, default=620.0)
  ap.add_argument("--undistort-size", default="1280x1024")
  ap.add_argument("--rebuild-rays", action="store_true")
  args = ap.parse_args()

  board, spec = M.load_target(args.target)

  names = (["d405", "zedx", "odin1"] if args.backends == "auto"
           else args.backends.split(","))
  found = []
  for bname in names:
    try:
      backend = open_backend(bname)
    except SystemExit:
      continue
    try:
      devs = backend.available()
    except Exception as e:
      print(f"  {bname}: cannot enumerate ({e})")
      continue
    for d in devs:
      found.append((backend, d))
      print(f"  found {d.get('model', bname)} "
            f"serial {d.get('serial', '?')} via {bname}")
  if not found:
    raise SystemExit("no depth camera found on any backend")

  workers = []
  for backend, dev in found:
    label = dev.get("model", backend.NAME).replace("RealSense ", "").replace(" ", "")
    name = label if not any(w.name == label for w in workers) \
      else f"{label}-{dev.get('serial', len(workers))}"
    w = CameraWorker(name, backend, args, dev.get("serial"), board, spec)
    w.start()
    workers.append(w)

  stamp = time.strftime("%Y%m%d-%H%M%S")
  outdir = args.out / stamp
  session = Session(workers, board, spec, args, outdir)

  def coordinator():
    while True:
      try:
        session.tick()
      except Exception:
        traceback.print_exc()
      time.sleep(0.1)

  threading.Thread(target=coordinator, daemon=True).start()

  httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
  httpd.daemon_threads = True
  httpd.app = {"workers": workers, "session": session, "outdir": outdir}
  print(f"\n  {len(workers)} camera(s): {', '.join(w.name for w in workers)}")
  print(f"  session -> {outdir}")
  print(f"\n  open  http://127.0.0.1:{args.port}\n")
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    for w in workers:
      w.stop()


if __name__ == "__main__":
  main()
