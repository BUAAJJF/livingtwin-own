#!/usr/bin/env python3
"""Browser view of the exact camera tensor consumed by the vision policy.

This is a perception diagnostic, not a control program. It opens the camera,
runs the same fused segmenter, tracker, reprojector and camera_obs functions as
run.py, and never imports or opens the robot CAN backend. An optional exported
policy may be evaluated against a clearly-labelled frozen nominal arm state;
its action is displayed but never mapped or sent anywhere.

  python -m hardware.deploy.policygui --camera d455 \
    --serial 262822300638 --port 8773
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from . import config, mask, obs, overlay, proprio, rectify, sensor

HERE = pathlib.Path(__file__).resolve().parent
JPEG_QUALITY = 88


def _jpeg(image: np.ndarray) -> bytes:
  ok, encoded = cv2.imencode(
    ".jpg", np.asarray(image), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
  return encoded.tobytes() if ok else b""


def _depth_preview(depth: np.ndarray, valid: np.ndarray,
                   cutoff: float = config.CUTOFF_M) -> np.ndarray:
  d = np.asarray(depth, dtype=np.float32)
  v = np.asarray(valid, dtype=bool)
  x = np.clip(d / max(float(cutoff), 1e-6), 0.0, 1.0)
  # Near is warm and far is cool; invalid remains unambiguously black.
  image = cv2.applyColorMap(
    np.round((1.0 - x) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
  image[~v] = 0
  return image


class Session:
  def __init__(self, camera: str, serial: str | None,
               rig_path: pathlib.Path, yolo_weights: pathlib.Path,
               yolo_conf: float, policy_path: pathlib.Path | None):
    if not rig_path.exists():
      raise RuntimeError(f"no calibration at {rig_path}")
    if not yolo_weights.exists():
      raise RuntimeError(f"no YOLO weights at {yolo_weights}")
    self.camera_name = camera
    self.rig_path = rig_path
    self.yolo_weights = yolo_weights
    self.yolo_conf = float(yolo_conf)
    self.rig = config.Rig.load(rig_path)
    self.reader = sensor.Reader(serial=serial, backend=camera)
    self.reader.wait_for_first()
    if (self.rig.serial and self.reader.serial
        and str(self.rig.serial) != str(self.reader.serial)):
      self.reader.close()
      raise RuntimeError(
        f"{rig_path} belongs to camera {self.rig.serial}, connected camera "
        f"is {self.reader.serial}")
    self.rig.K = self.reader.K
    self.rig.serial = self.reader.serial

    self.reproj = rectify.Reprojector(self.rig)
    self.depth_segmenter = mask.DepthSegmenter(self.rig, self.reproj)
    yolo_cfg = dataclasses.replace(mask.YoloCfg(), conf=self.yolo_conf)
    self.yolo_segmenter = mask.YoloSegmenter(
      str(yolo_weights), self.rig, self.reproj, device="cuda:0",
      yolo_cfg=yolo_cfg)
    self.segmenter = mask.FusedSegmenter(
      self.depth_segmenter, self.yolo_segmenter)
    self.tracker = mask.TargetTracker()

    # No CAN. The nominal kinematics are used only by the target-selection
    # distance rule. Arm subtraction is disabled rather than pretending this
    # nominal pose is the pose of the physical arm.
    self.builder = proprio.ProprioBuilder()
    self.nominal_q = self.builder.default_q.astype(np.float64)
    self.builder.kin.update(self.nominal_q)
    self.nominal_hand = self.builder.kin.site_pos.copy()
    self.nominal_feedback = proprio.JointFeedback(
      position=self.nominal_q.copy(), velocity=np.zeros(8),
      target=self.nominal_q.copy(), gripper_effort=0.0)
    self.last_action = np.zeros(7, dtype=np.float32)

    self.policy = None
    self.policy_error = None
    if policy_path is not None:
      try:
        from .policy import Policy
        self.policy = Policy(policy_path, providers=["CPUExecutionProvider"])
      except Exception as e:
        self.policy_error = str(e)

    self.lock = threading.Lock()
    self._images: dict[str, bytes] = {}
    self._state = {
      "ready": False, "why": "waiting for first perception result",
      "frame_age_ms": None,
    }
    self._stop = threading.Event()
    self._reset = threading.Event()
    self._periods: list[float] = []
    self._thread = threading.Thread(target=self._loop, daemon=True,
                                    name="policy-input-visualizer")
    self._thread.start()

  def _loop(self):
    last_index = -1
    while not self._stop.is_set():
      frame = self.reader.latest()
      if frame is None or frame.index == last_index:
        time.sleep(0.002)
        continue
      last_index = frame.index
      if self._reset.is_set():
        self.tracker.clear()
        if self.policy is not None:
          self.policy.reset()
        self.last_action[:] = 0
        self._reset.clear()
      t0 = time.perf_counter()
      try:
        seg = self.segmenter(frame.depth, rgb=frame.gray, arm=None)
        label = self.tracker.update(seg, self.nominal_hand)
        payload = (mask.full_mask(seg, label, self.segmenter.decimate)
                   if label else None)
        depth, valid, target = self.reproj(frame.depth, payload=payload)
        if target is None:
          target = np.zeros_like(valid, dtype=np.int32)
        camera = obs.camera_obs(depth, valid, target > 0)

        action = None
        if self.policy is not None:
          flat = self.builder(self.nominal_feedback, self.last_action)
          action = self.policy(flat, camera)
          self.last_action = action.astype(np.float32)

        raw = cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
        for inst in seg.instances:
          binary = mask.full_mask(
            seg, inst.label, self.segmenter.decimate).astype(np.uint8)
          contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          colour = ((0, 235, 255) if inst.label == label
                    else (70, 220, 90))
          cv2.drawContours(raw, contours, -1, colour, 2)
          x, y, _w, _h = inst.bbox
          cv2.putText(raw, str(inst.label), (x, max(18, y)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2,
                      cv2.LINE_AA)
        # The sector and the bin, projected through the calibration.  An
        # object outside the sector is invisible to the policy however well it
        # is segmented, and nothing in a depth image says where the boundary
        # is -- so it is drawn, and the operator can place objects against it.
        try:
          overlay.draw(raw, self.rig)
        except Exception:
          pass
        caption = (f"fused {len(seg.instances)} | target {label or '-'} | "
                   f"YOLO +{self.segmenter.n_from_yolo}")
        cv2.putText(raw, caption, (15, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.70, (0, 235, 255), 2, cv2.LINE_AA)

        ch0 = np.round(camera[0] * 255).astype(np.uint8)
        ch1 = np.round(camera[1] * 255).astype(np.uint8)
        ch2 = np.round(camera[2] * 255).astype(np.uint8)
        composite = np.stack([ch0, np.maximum(ch0, ch1), ch2], axis=-1)
        process_ms = (time.perf_counter() - t0) * 1000
        self._periods.append(process_ms)
        del self._periods[:-120]
        target_inst = next(
          (x for x in seg.instances if x.label == label), None)
        state = {
          "ready": True, "why": "",
          "frame_index": int(frame.index),
          "frame_age_ms": max(0.0, (time.time() - frame.stamp) * 1000),
          "process_ms": process_ms,
          "process_p50_ms": float(np.median(self._periods)),
          "sensor_shape": list(frame.depth.shape),
          "policy_shape": list(camera.shape),
          "sensor_depth_fill": float((frame.depth > 0).mean()),
          "policy_depth_fill": float(valid.mean()),
          "instances": len(seg.instances),
          "target_label": int(label),
          "target_pixels_policy": int(camera[1].sum()),
          "from_yolo": int(self.segmenter.n_from_yolo),
          "yolo_unplaced": int(self.yolo_segmenter.n_unplaced),
          "target_center_base_m": (
            None if target_inst is None else
            [float(x) for x in target_inst.centroid_base]),
          "action": None if action is None else [float(x) for x in action],
        }
        images = {
          "raw": _jpeg(raw),
          "sensor_depth": _jpeg(_depth_preview(
            frame.depth, frame.depth > 0)),
          "policy_depth": _jpeg(ch0),
          "policy_mask": _jpeg(ch1),
          "policy_masked_depth": _jpeg(ch2),
          "policy_composite": _jpeg(composite),
        }
        with self.lock:
          self._state = state
          self._images = images
      except Exception as e:
        with self.lock:
          self._state = {
            "ready": False, "why": f"perception failed: {e}",
            "frame_age_ms": self.reader.age * 1000,
          }
        time.sleep(0.05)

  def state(self) -> dict:
    with self.lock:
      live = dict(self._state)
    live["frame_age_ms"] = self.reader.age * 1000
    return {
      "camera": self.camera_name.upper(),
      "serial": self.reader.serial,
      "rig": str(self.rig_path.resolve()),
      "yolo_weights": str(self.yolo_weights.resolve()),
      "yolo_conf": self.yolo_conf,
      "arm_state": "nominal (CAN disconnected)",
      "can_connected": False,
      "arm_commands": False,
      "policy_loaded": self.policy is not None,
      "policy_path": None if self.policy is None else self.policy.path,
      "policy_error": self.policy_error,
      "live": live,
    }

  def image(self, name: str) -> bytes:
    with self.lock:
      return self._images.get(name, b"")

  def reset_target(self) -> dict:
    self._reset.set()
    return {"ok": True}

  def close(self):
    self._stop.set()
    self._thread.join(timeout=3.0)
    self.reader.close()


class Handler(BaseHTTPRequestHandler):
  server_version = "policygui/1.0"

  def log_message(self, *_):
    pass

  def _json(self, value):
    body = json.dumps(value).encode()
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):
    path = self.path.split("?")[0]
    if path in ("/", "/index.html"):
      body = (HERE / "policygui.html").read_bytes()
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
    elif path == "/state":
      self._json(self.server.session.state())
    elif path.startswith("/image/"):
      name = path.removeprefix("/image/")
      body = self.server.session.image(name)
      if not body:
        self.send_error(503, "image not ready")
        return
      self.send_response(200)
      self.send_header("Content-Type", "image/jpeg")
      self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
    else:
      self.send_error(404)

  def do_POST(self):
    if self.path.split("?")[0] == "/reset-target":
      self._json(self.server.session.reset_target())
    else:
      self.send_error(404)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--camera", choices=("d405", "d455"), default="d455")
  p.add_argument("--serial", default=None)
  p.add_argument("--rig-file", default=None)
  p.add_argument("--yolo-weights", default=None)
  p.add_argument("--yolo-conf", type=float, default=0.25)
  p.add_argument("--policy", default=None,
                 help="optional recurrent policy.onnx; actions are displayed "
                      "against a frozen nominal arm and never sent")
  p.add_argument("--port", type=int, default=8773)
  a = p.parse_args()
  suffix = "" if a.camera == "d405" else f"_{a.camera}"
  rig = (pathlib.Path(a.rig_file) if a.rig_file else
         pathlib.Path(config.RIG_FILE).with_name(f"rig{suffix}.json"))
  default_yolo_dir = "yolo" if a.camera == "d405" else f"yolo_{a.camera}"
  yolo = (pathlib.Path(a.yolo_weights) if a.yolo_weights else
          HERE / default_yolo_dir / "best.pt")
  policy = pathlib.Path(a.policy) if a.policy else None
  sess = Session(a.camera, a.serial, rig, yolo, a.yolo_conf, policy)
  httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
  httpd.session = sess
  print("camera-only diagnostic: CAN is not opened; no arm command exists")
  print(f"policy tensor: 3 x {config.HEIGHT} x {config.WIDTH}")
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
