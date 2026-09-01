#!/usr/bin/env python3
"""Manual browser capture of sparse D405/D455 YOLO teacher frames.

Only the camera is opened. CAN is never opened and no arm command can be sent.
Put the arm outside the workspace, arrange the objects, wait for the green
overlay to contain the expected number of objects, and click once.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from . import config, mask, rectify, sensor

HERE = pathlib.Path(__file__).resolve().parent


class Session:
  def __init__(self, camera: str, serial: str | None, rig_path: pathlib.Path,
               out: pathlib.Path, expected: int):
    if not rig_path.exists():
      raise RuntimeError(f"no calibration at {rig_path}; save it first")
    if out.exists() and any(out.iterdir()):
      raise RuntimeError(f"refusing to overwrite non-empty session {out}")
    self.camera = camera
    self.out = out
    self.expected = max(0, int(expected))
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

    out.mkdir(parents=True, exist_ok=True)
    self.rig.save(out / "rig.json")
    (out / "capture.json").write_text(json.dumps({
      "camera": camera, "serial": self.reader.serial,
      "model": self.reader.meta.get("model"),
      "gray_source": self.reader.meta.get("gray_source"),
      "emitter": self.reader.meta.get("emitter"),
      "resolution": self.reader.meta.get("resolution"),
      "fps": self.reader.meta.get("fps"),
      "rig_source": str(rig_path.resolve()),
      "manual_capture": True, "expected_objects": self.expected,
      "arm_feedback": False, "arm_commands": False,
    }, indent=2) + "\n")

    self.segmenter = mask.DepthSegmenter(
      self.rig, rectify.Reprojector(self.rig))
    self.lock = threading.Lock()
    self.records: list[dict] = []
    self._preview = b""
    self._candidate = None
    self._live = {
      "ready": False, "detected": 0, "frame_age_s": None,
      "depth_fill": 0.0, "why": "waiting for camera",
    }
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._loop, daemon=True,
                                    name="yolo-capture-preview")
    self._thread.start()

  def _loop(self):
    last_index = -1
    while not self._stop.is_set():
      frame = self.reader.latest()
      if frame is None or frame.index == last_index:
        time.sleep(0.01)
        continue
      last_index = frame.index
      try:
        seg = self.segmenter(frame.depth, arm=None)
        detected = len(seg.instances)
        fill = float((frame.depth > 0).mean())
        ready = detected == self.expected if self.expected else detected > 0
        why = (f"detected {detected}/{self.expected} objects"
               if self.expected else f"detected {detected} objects")
        view = cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
        for inst in seg.instances:
          binary = mask.full_mask(seg, inst.label,
                                  self.segmenter.decimate).astype(np.uint8)
          contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
          cv2.drawContours(view, contours, -1, (60, 220, 80), 2)
          x, y, _w, _h = inst.bbox
          scale = self.segmenter.decimate
          cv2.putText(view, str(inst.label), (x * scale, max(18, y * scale)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 220, 80), 2,
                      cv2.LINE_AA)
        colour = (60, 220, 80) if ready else (40, 180, 255)
        cv2.putText(view, why, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    colour, 2, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", view,
                                   [cv2.IMWRITE_JPEG_QUALITY, 88])
        with self.lock:
          self._candidate = (frame.index, frame.stamp, frame.depth.copy(),
                             frame.gray.copy(), detected, fill)
          self._live = {
            "ready": ready, "detected": detected,
            "frame_age_s": max(0.0, time.time() - frame.stamp),
            "depth_fill": fill, "why": why,
          }
          if ok:
            self._preview = encoded.tobytes()
      except Exception as e:
        with self.lock:
          self._live = {"ready": False, "detected": 0,
                        "frame_age_s": None, "depth_fill": 0.0,
                        "why": f"segmentation failed: {e}"}
        time.sleep(0.1)

  def state(self) -> dict:
    with self.lock:
      live = dict(self._live)
      n = len(self.records)
    live["frame_age_s"] = self.reader.age
    return {
      "camera": self.camera.upper(), "serial": self.reader.serial,
      "out": str(self.out.resolve()), "expected": self.expected,
      "saved": n, "live": live,
      "can_connected": False, "arm_commands": False,
    }

  def preview(self) -> bytes:
    with self.lock:
      return self._preview

  def save(self) -> dict:
    with self.lock:
      candidate = self._candidate
      live = dict(self._live)
      if candidate is None:
        return {"ok": False, "why": "no camera frame yet"}
      if time.time() - candidate[1] > 0.5:
        return {"ok": False, "why": "camera frame is stale"}
      if not live.get("ready"):
        return {"ok": False, "why": live.get("why", "object gate failed")}
      _frame_i, stamp, depth, gray, detected, fill = candidate
      i = len(self.records)
      np.savez_compressed(
        self.out / f"{i:06d}.npz",
        depth=(np.asarray(depth) * 10000).astype(np.uint16),
        gray=np.asarray(gray, dtype=np.uint8),
      )
      self.records.append({
        "i": i, "t": float(stamp), "joint_pos": None,
        "manual_capture": True, "detected_instances": int(detected),
        "depth_fill": float(fill),
      })
      (self.out / "meta.json").write_text(
        json.dumps(self.records, indent=2) + "\n")
      return {"ok": True, "saved": len(self.records),
              "detected": int(detected), "depth_fill": float(fill)}

  def close(self):
    self._stop.set()
    self._thread.join(timeout=2.0)
    self.reader.close()
    with self.lock:
      (self.out / "meta.json").write_text(
        json.dumps(self.records, indent=2) + "\n")
    print(f"saved {len(self.records)} manually confirmed frame(s) to {self.out}")


class Handler(BaseHTTPRequestHandler):
  server_version = "collect-yolo-gui/1.0"

  def log_message(self, *_):
    pass

  def _json(self, obj):
    body = json.dumps(obj).encode()
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):
    path = self.path.split("?")[0]
    sess = self.server.session
    if path in ("/", "/index.html"):
      body = (HERE / "collect_yolo_gui.html").read_bytes()
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
                             b"Content-Length: " + str(len(buf)).encode()
                             + b"\r\n\r\n" + buf + b"\r\n")
          time.sleep(1 / 12)
      except (BrokenPipeError, ConnectionResetError):
        pass
    else:
      self.send_error(404)

  def do_POST(self):
    if self.path.split("?")[0] == "/save":
      self._json(self.server.session.save())
    else:
      self.send_error(404)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--camera", choices=("d405", "d455"), default="d455")
  p.add_argument("--serial", default=None)
  p.add_argument("--rig-file", default=None)
  p.add_argument("--out", default=None)
  p.add_argument("--expected", type=int, default=4,
                 help="enable save only with this many detected objects; "
                      "use 0 to accept any non-empty scene")
  p.add_argument("--port", type=int, default=8772)
  a = p.parse_args()
  suffix = "" if a.camera == "d405" else f"_{a.camera}"
  rig_path = (pathlib.Path(a.rig_file) if a.rig_file else
              pathlib.Path(config.RIG_FILE).with_name(f"rig{suffix}.json"))
  out = pathlib.Path(a.out) if a.out else pathlib.Path(
    "recordings") / f"{a.camera}_yolo" / time.strftime("manual_%Y%m%d_%H%M%S")
  sess = Session(a.camera, a.serial, rig_path, out, a.expected)
  httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
  httpd.session = sess
  print("camera only: CAN is not opened and the arm cannot be commanded")
  print(f"session: {out.resolve()}")
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
