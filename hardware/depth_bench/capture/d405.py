"""Intel RealSense D405 backend.

Two properties of this camera shape everything the bench does with it.

It has **no infrared projector** -- ``rs-enumerate-devices -o`` lists no Laser
Power and no Emitter Enabled, unlike every other D400.  It is passive stereo,
so it matches on whatever texture the scene already has.  That is why the
target sheet carries a blank white patch: on a projector-equipped camera that
patch is unremarkable, and on this one it is the failure mode.

Its useful range is short -- Intel quote roughly 7 cm to 50 cm.  The simulator
camera in ``src/piper_push/camera.py`` sits 0.70 m from what it aims at, which
is outside that.  The bench therefore measures across whatever distances the
sheet is actually placed at and fits noise against distance, rather than
assuming a single operating point.

Depth units default to 0.1 mm rather than the factory 1 mm.  At 0.2 m the
quantisation step of a 1 mm unit is a large fraction of the sensor's own noise,
and we would end up measuring the integer grid instead of the camera.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pyrealsense2 as rs

from . import Capture

NAME = "d405"

PRESETS = {
  "custom": 0, "default": 1, "hand": 2,
  "high_accuracy": 3, "high_density": 4, "medium_density": 5,
}


def add_args(ap) -> None:
  ap.add_argument("--width", type=int, default=848)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--preset", choices=sorted(PRESETS), default="default")
  ap.add_argument("--depth-units", type=float, default=1e-4,
                  help="metres per depth integer; 1e-4 avoids quantising the noise")
  ap.add_argument("--filters", action="store_true",
                  help="apply the stock spatial+temporal filter chain "
                       "(default is raw, which is what the noise model wants)")
  ap.add_argument("--serial", default=None)


def available() -> list[dict]:
  """Every RealSense on the bus, so the live viewer can find them itself."""
  out = []
  for dev in rs.context().query_devices():
    try:
      out.append({
        "backend": NAME,
        "serial": dev.get_info(rs.camera_info.serial_number),
        "model": dev.get_info(rs.camera_info.name),
        "firmware": dev.get_info(rs.camera_info.firmware_version),
        "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
      })
    except Exception:
      continue
  return out


def reset(serial: str | None = None) -> bool:
  """Power-cycle the camera over USB.

  Needed because this D405 shares its xHCI controller with the Odin 1, and once
  the lidar starts streaming the RealSense stops delivering frames and does not
  come back on its own -- ``wait_for_frames`` times out for ever, including
  after the offending process exits.  A hardware reset recovers it, so the
  bench does that rather than asking someone to find the cable.  The real fix
  is a different USB controller; this is what makes the session survive until
  someone moves it.
  """
  done = False
  for dev in rs.context().query_devices():
    if serial and dev.get_info(rs.camera_info.serial_number) != serial:
      continue
    try:
      dev.hardware_reset()
      done = True
    except Exception:
      pass
  if done:
    time.sleep(6.0)  # it disappears from the bus and re-enumerates
  return done


def _post(args):
  """The stock filter chain, in the order librealsense documents.

  Off by default: the simulator's noise model should be fitted to the depth the
  deployed pipeline actually consumes, and until that pipeline is decided the
  honest baseline is the unfiltered sensor.
  """
  if not getattr(args, "filters", False):
    return []
  return [rs.disparity_transform(True), rs.spatial_filter(),
          rs.temporal_filter(), rs.disparity_transform(False)]


class Stream:
  """A running camera.

  Both the one-shot ``grab`` and the live viewer go through this, so there is
  exactly one place that decides resolution, preset, depth units and alignment.
  Two code paths configuring the same camera differently is how a live preview
  ends up flattering a sensor that the measurement then contradicts.
  """

  def __init__(self, args, serial: str | None = None):
    serial = serial or getattr(args, "serial", None)
    if not rs.context().query_devices().size():
      raise RuntimeError("no RealSense device found; check the USB 3 cable")

    cfg = rs.config()
    if serial:
      cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16,
                      args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8,
                      args.fps)

    self._pipe = rs.pipeline()
    profile = self._pipe.start(cfg)
    dev = profile.get_device()
    sensor = dev.first_depth_sensor()
    if sensor.supports(rs.option.visual_preset):
      sensor.set_option(rs.option.visual_preset, float(PRESETS[args.preset]))
    if sensor.supports(rs.option.depth_units):
      sensor.set_option(rs.option.depth_units, args.depth_units)
    self._scale = sensor.get_depth_scale()

    # Colour is resampled into the depth grid, never the other way round: the
    # depth samples are the measurement and must not be interpolated.
    self._align = rs.align(rs.stream.depth)
    self._filters = _post(args)

    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile() \
      .get_intrinsics()
    self.K = np.array([[intr.fx, 0.0, intr.ppx],
                       [0.0, intr.fy, intr.ppy],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    self.dist = np.array(intr.coeffs, dtype=np.float64)

    di = dev.get_info
    self.meta = {
      "backend": NAME,
      "model": di(rs.camera_info.name),
      "serial": di(rs.camera_info.serial_number),
      "firmware": di(rs.camera_info.firmware_version),
      "usb": di(rs.camera_info.usb_type_descriptor),
      "librealsense": rs.__version__ if hasattr(rs, "__version__") else "unknown",
      "resolution": [args.width, args.height],
      "fps": args.fps,
      "preset": args.preset,
      "depth_units_m": self._scale,
      "filters": bool(getattr(args, "filters", False)),
      "emitter": "none (D405 is passive stereo)",
      "stereo_baseline_m": (sensor.get_option(rs.option.stereo_baseline) / 1000.0
                            if sensor.supports(rs.option.stereo_baseline) else None),
      "fx_px": float(intr.fx),
    }

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    """One aligned pair: depth in metres (0 = invalid) and greyscale."""
    frames = self._align.process(self._pipe.wait_for_frames())
    d = frames.get_depth_frame()
    for f in self._filters:
      d = f.process(d)
    depth = (np.asanyarray(d.as_depth_frame().get_data()).astype(np.float32)
             * self._scale)
    c = frames.get_color_frame()
    if not c:
      raise RuntimeError("no colour frame; cannot locate the target")
    gray = cv2.cvtColor(np.asanyarray(c.get_data()), cv2.COLOR_BGR2GRAY)
    return depth, gray

  def close(self) -> None:
    try:
      self._pipe.stop()
    except Exception:
      pass


def grab(args, n_frames: int, warmup: int = 30) -> Capture:
  s = Stream(args)
  try:
    for _ in range(warmup):  # auto-exposure needs time or the first frames lie
      s.read()
    depths, gray = [], None
    for _ in range(n_frames):
      d, gray = s.read()
      depths.append(d)
    meta = dict(s.meta, n_frames=n_frames)
    return Capture(depth=np.stack(depths), gray=gray, K=s.K, dist=s.dist, meta=meta)
  finally:
    s.close()
