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


def available(model_filter: str = "D405") -> list[dict]:
  """Matching RealSense cameras on the bus."""
  out = []
  for dev in rs.context().query_devices():
    try:
      item = {
        "backend": NAME,
        "serial": dev.get_info(rs.camera_info.serial_number),
        "model": dev.get_info(rs.camera_info.name),
        "firmware": dev.get_info(rs.camera_info.firmware_version),
        "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
      }
      if model_filter.lower() in item["model"].lower():
        out.append(item)
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

  def __init__(self, args, serial: str | None = None, *,
               backend_name: str = NAME, expected_model: str = "D405",
               emitter_description: str = "none (D405 is passive stereo)"):
    serial = serial or getattr(args, "serial", None)
    devices = rs.context().query_devices()
    if not devices.size():
      raise RuntimeError("no RealSense device found; check the USB 3 cable")

    if serial is None:
      matches = [dev.get_info(rs.camera_info.serial_number) for dev in devices
                 if expected_model.lower() in
                 dev.get_info(rs.camera_info.name).lower()]
      if len(matches) == 1:
        serial = matches[0]
      elif not matches:
        raise RuntimeError(f"no RealSense {expected_model} found")
      else:
        raise RuntimeError(
          f"multiple RealSense {expected_model} cameras found; pass --serial")

    cfg = rs.config()
    if serial:
      cfg.enable_device(serial)
    self.gray_source = str(getattr(args, "gray_source", "aligned_color"))
    if self.gray_source not in ("aligned_color", "left_ir"):
      raise ValueError(
        f"unknown D405 gray source {self.gray_source!r}; expected "
        "'aligned_color' or 'left_ir'")
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16,
                      args.fps)
    if self.gray_source == "aligned_color":
      cfg.enable_stream(rs.stream.color, args.width, args.height,
                        rs.format.bgr8, args.fps)
    # The left infrared imager, optionally, and NOT aligned to anything.
    #
    # The colour stream below is warped into the depth grid, which is right for
    # the bench -- it puts colour and depth on one grid so a region measured in
    # one is the same region in the other.  It is wrong for anything that has
    # to *locate* something in the image, because the warp is driven by the
    # depth map: where depth is missing the output is black, and where depth is
    # noisy the colour pixel lands somewhere else.  A calibration target is
    # mostly edges, edges are where this camera drops out, and the target ends
    # up holed exactly along the features being measured.
    #
    # The left imager needs no warp at all: RealSense defines the depth frame
    # as that imager's frame, so its intrinsics are the depth intrinsics --
    # checked on this camera, identical to five decimal places and zero
    # distortion, where the colour stream carries -0.052 of radial.
    self.infrared = (self.gray_source == "left_ir"
                     or bool(getattr(args, "infrared", False)))
    # The RIGHT imager as well, for anything that wants to compute its own
    # disparity instead of reading the ASIC's.  It is off by default: it is a
    # third USB stream this camera does not need, and the bench measured this
    # unit wedging permanently when its controller is oversubscribed.
    self.stereo = bool(getattr(args, "stereo", False))
    if self.stereo:
      self.infrared = True
    if self.infrared:
      cfg.enable_stream(rs.stream.infrared, 1, args.width, args.height,
                        rs.format.y8, args.fps)
    if self.stereo:
      cfg.enable_stream(rs.stream.infrared, 2, args.width, args.height,
                        rs.format.y8, args.fps)

    self._pipe = rs.pipeline()
    profile = self._pipe.start(cfg)
    dev = profile.get_device()
    model = dev.get_info(rs.camera_info.name)
    if expected_model.lower() not in model.lower():
      self._pipe.stop()
      raise RuntimeError(
        f"{backend_name} backend selected {model!r}; pass that camera's serial "
        f"or connect a RealSense {expected_model}")
    sensor = dev.first_depth_sensor()
    self._depth_sensor = sensor
    if sensor.supports(rs.option.visual_preset):
      sensor.set_option(rs.option.visual_preset, float(PRESETS[args.preset]))
    if sensor.supports(rs.option.depth_units):
      sensor.set_option(rs.option.depth_units, args.depth_units)
    self._scale = sensor.get_depth_scale()
    emitter = getattr(args, "emitter", None)
    if emitter is not None and sensor.supports(rs.option.emitter_enabled):
      sensor.set_option(rs.option.emitter_enabled,
                        1.0 if emitter == "on" else 0.0)
    laser_power = getattr(args, "laser_power", None)
    if laser_power is not None and sensor.supports(rs.option.laser_power):
      sensor.set_option(rs.option.laser_power, float(laser_power))

    # Colour is resampled into the depth grid, never the other way round: the
    # depth samples are the measurement and must not be interpolated.
    self._align = (rs.align(rs.stream.depth)
                   if self.gray_source == "aligned_color" else None)
    self._filters = _post(args)

    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile() \
      .get_intrinsics()
    self.K = np.array([[intr.fx, 0.0, intr.ppx],
                       [0.0, intr.fy, intr.ppy],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    self.dist = np.array(intr.coeffs, dtype=np.float64)

    di = dev.get_info
    self.meta = {
      "backend": backend_name,
      "model": model,
      "serial": di(rs.camera_info.serial_number),
      "firmware": di(rs.camera_info.firmware_version),
      "usb": di(rs.camera_info.usb_type_descriptor),
      "librealsense": rs.__version__ if hasattr(rs, "__version__") else "unknown",
      "resolution": [args.width, args.height],
      "fps": args.fps,
      "gray_source": self.gray_source,
      "preset": args.preset,
      "depth_units_m": self._scale,
      "filters": bool(getattr(args, "filters", False)),
      "emitter": (
        "on" if (sensor.supports(rs.option.emitter_enabled)
                 and sensor.get_option(rs.option.emitter_enabled) > 0.5)
        else "off" if sensor.supports(rs.option.emitter_enabled)
        else emitter_description
      ),
      "laser_power": (sensor.get_option(rs.option.laser_power)
                      if sensor.supports(rs.option.laser_power) else None),
      "stereo_baseline_m": (sensor.get_option(rs.option.stereo_baseline) / 1000.0
                            if sensor.supports(rs.option.stereo_baseline) else None),
      "fx_px": float(intr.fx),
      # What turns disparity into metres.  Read from the device rather than the
      # rig file: the rig's intrinsics describe the depth grid, and a learned
      # stereo model is producing disparity in the LEFT IMAGER, which is the
      # same frame but need not stay so if the streams are ever reconfigured.
      "stereo_focal_baseline": (
        float(intr.fx) * float(sensor.get_option(rs.option.stereo_baseline)) / 1000.0
        if sensor.supports(rs.option.stereo_baseline) else None),
    }

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    """One aligned pair: depth in metres (0 = invalid) and greyscale."""
    return self.read3()[:2]

  def read4(self):
    """``read3`` plus the raw right infrared frame, or None if not enabled."""
    d, g, ir = self.read3()
    return d, g, ir, self._right

  def read3(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The aligned pair, plus the raw left infrared frame if it was enabled."""
    raw = self._pipe.wait_for_frames()
    ir = None
    self._right = None
    if self.infrared:
      f = raw.get_infrared_frame(1)
      if f:
        ir = np.asanyarray(f.get_data()).copy()
    if self.stereo:
      f = raw.get_infrared_frame(2)
      if f:
        self._right = np.asanyarray(f.get_data()).copy()
    if self.gray_source == "left_ir":
      # The depth frame is defined in this left imager's optical frame.  No
      # colour-to-depth warp, no black alignment holes at marker edges, and no
      # factory extrinsic is needed before hand-eye calibration.
      d = raw.get_depth_frame()
      if not d or ir is None:
        raise RuntimeError("no raw left grayscale/depth frame")
      for f in self._filters:
        d = f.process(d)
      depth = (np.asanyarray(d.as_depth_frame().get_data()).astype(np.float32)
               * self._scale)
      return depth, ir, ir

    frames = self._align.process(raw)
    d = frames.get_depth_frame()
    for f in self._filters:
      d = f.process(d)
    depth = (np.asanyarray(d.as_depth_frame().get_data()).astype(np.float32)
             * self._scale)
    c = frames.get_color_frame()
    if not c:
      raise RuntimeError("no colour frame; cannot locate the target")
    gray = cv2.cvtColor(np.asanyarray(c.get_data()), cv2.COLOR_BGR2GRAY)
    return depth, gray, ir

  def intrinsics(self, width: int, height: int) -> np.ndarray:
    """Factory depth intrinsics for another supported resolution.

    Calibration runs the raw imager at 1280x720, while deployment deliberately
    remains at the characterised 848x480 mode.  Extrinsics are resolution
    independent; the rig file still needs the latter mode's exact per-device
    intrinsics rather than a scaled approximation.
    """
    for p in self._depth_sensor.get_stream_profiles():
      try:
        if p.stream_type() != rs.stream.depth or p.format() != rs.format.z16:
          continue
        v = p.as_video_stream_profile()
        if v.width() != int(width) or v.height() != int(height):
          continue
        intr = v.get_intrinsics()
        return np.array([[intr.fx, 0.0, intr.ppx],
                         [0.0, intr.fy, intr.ppy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
      except RuntimeError:
        continue
    raise RuntimeError(
      f"{self.meta['model']} does not report a {width}x{height} depth profile")

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
