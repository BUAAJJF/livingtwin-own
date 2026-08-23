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


def _post(args):
  """The stock filter chain, in the order librealsense documents.

  Off by default: the simulator's noise model should be fitted to the depth the
  deployed pipeline actually consumes, and until that pipeline is decided the
  honest baseline is the unfiltered sensor.
  """
  if not args.filters:
    return []
  spatial = rs.spatial_filter()
  temporal = rs.temporal_filter()
  return [rs.disparity_transform(True), spatial, temporal,
          rs.disparity_transform(False)]


def grab(args, n_frames: int, warmup: int = 30) -> Capture:
  ctx = rs.context()
  devices = list(ctx.query_devices())
  if not devices:
    raise SystemExit("no RealSense device found; check the USB 3 cable")

  cfg = rs.config()
  if args.serial:
    cfg.enable_device(args.serial)
  cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
  cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8,
                    args.fps)

  pipe = rs.pipeline()
  profile = pipe.start(cfg)
  try:
    dev = profile.get_device()
    sensor = dev.first_depth_sensor()
    if sensor.supports(rs.option.visual_preset):
      sensor.set_option(rs.option.visual_preset, float(PRESETS[args.preset]))
    if sensor.supports(rs.option.depth_units):
      sensor.set_option(rs.option.depth_units, args.depth_units)
    depth_scale = sensor.get_depth_scale()

    # Colour is resampled into the depth grid, never the other way round: the
    # depth samples are the measurement and must not be interpolated.
    align = rs.align(rs.stream.depth)
    filters = _post(args)

    baseline = (sensor.get_option(rs.option.stereo_baseline) / 1000.0
                if sensor.supports(rs.option.stereo_baseline) else None)

    dprof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = dprof.get_intrinsics()
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0, intr.fy, intr.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)

    for _ in range(warmup):  # auto-exposure needs time or the first frames lie
      pipe.wait_for_frames()

    depths, gray = [], None
    for _ in range(n_frames):
      frames = align.process(pipe.wait_for_frames())
      d = frames.get_depth_frame()
      for f in filters:
        d = f.process(d)
      depths.append(np.asanyarray(d.as_depth_frame().get_data()).astype(np.float32)
                    * depth_scale)
      c = frames.get_color_frame()
      if c:
        import cv2

        gray = cv2.cvtColor(np.asanyarray(c.get_data()), cv2.COLOR_BGR2GRAY)

    if gray is None:
      raise SystemExit("no colour frame arrived; cannot locate the target")

    di = dev.get_info
    meta = {
      "backend": NAME,
      "model": di(rs.camera_info.name),
      "serial": di(rs.camera_info.serial_number),
      "firmware": di(rs.camera_info.firmware_version),
      "usb": di(rs.camera_info.usb_type_descriptor),
      "librealsense": rs.__version__ if hasattr(rs, "__version__") else "unknown",
      "resolution": [args.width, args.height],
      "fps": args.fps,
      "preset": args.preset,
      "depth_units_m": depth_scale,
      "filters": bool(args.filters),
      "emitter": "none (D405 is passive stereo)",
      "stereo_baseline_m": baseline,
      "fx_px": float(intr.fx),
      "n_frames": n_frames,
    }
    return Capture(depth=np.stack(depths), gray=gray, K=K, dist=dist, meta=meta)
  finally:
    pipe.stop()
