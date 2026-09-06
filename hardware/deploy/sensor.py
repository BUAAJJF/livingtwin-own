"""The D405, read on its own thread.

The camera runs at 30 Hz and the policy at 50, so the control loop must never
wait for a frame -- and ``wait_for_frames`` is a blocking call that will happily
sit for 5 seconds when the USB link hiccups.  A reader thread publishes the
latest frame and the loop takes whatever is there, which is also what the
simulator does: the camera sensor is sampled at the control rate and the
renderer does not produce a new image on every physics step either.

Every setting comes from ``hardware/depth_bench/capture/d405.py``, which is
where the sensor was characterised.  Resolution, preset and depth units in
particular: the bench ran at 848x480 with 0.1 mm depth units rather than the
factory 1 mm, and the noise model in ``piper_push.depth_noise`` was fitted to
that.  A deployment on different settings would be a deployment against a
different sensor from the one the policy was trained for.

Stale-frame detection is not optional.  A camera that stops delivering leaves
the last good frame in the buffer, the policy keeps acting on it, and the arm
keeps moving through a scene it can no longer see.  ``Reader.age`` is how old
the current frame is and ``run.py`` stops the arm when it exceeds a threshold.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import threading
import time

import numpy as np

from . import config

_BENCH = pathlib.Path(__file__).resolve().parents[1] / "depth_bench"


def _bench_backend(name: str = "d455"):
  """The bench's own RealSense backend, imported from where it lives.

  Imported rather than reimplemented: two files configuring the same camera
  differently is how a deployment ends up running a sensor the measurements do
  not describe.
  """
  if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))
  from capture import open_backend   # noqa: E402  -- path set above
  return open_backend(name)


@dataclasses.dataclass
class Frame:
  depth: np.ndarray
  """``(480, 848)`` metres, 0 where the sensor returned nothing."""
  gray: np.ndarray
  """uint8 depth-grid grayscale.  In calibration ``gray_source='left_ir'``
  makes this the unwarped raw left imager; deployment keeps the historical
  depth-aligned colour stream."""
  stamp: float
  index: int
  rgb: np.ndarray | None = None
  """Synchronized raw BGR frame from the independent colour imager.

  It has never been aligned through depth. YOLO and SAM use this image and
  their masks are projected back to the left/depth grid afterwards.
  """
  policy_depth: np.ndarray | None = None
  """The depth the POLICY was given, when that is not ``depth``.

  Set by the perception thread when ``--depth-source stereo`` replaces the
  camera's own map with a computed one.  The recorder writes this in
  preference to ``depth``, because a recording whose depth is not the run's
  depth cannot be replayed or reviewed as that run."""
  ir_right: np.ndarray | None = None
  """``(480, 848)`` uint8 from the RIGHT infrared imager, or None unless the
  reader was opened with ``stereo=True``.

  Only one thing needs this: computing disparity outside the camera.  The
  ASIC's own answer already arrives as ``depth``, and on this unit the two
  agree to 3.7 mm where both are defined, so a second opinion is worth having
  only where the ASIC has none."""
  ir: np.ndarray | None = None
  """``(480, 848)`` uint8 from the left infrared imager, unwarped, or None
  unless the reader was opened with ``infrared=True``.

  This is the image to *locate* things in.  ``gray`` is the colour stream
  warped into the depth grid, so it is black wherever the depth is missing and
  displaced wherever the depth is noisy -- and both of those happen at edges,
  which is what a calibration target is made of.  ``ir`` has neither problem
  and its intrinsics are the depth intrinsics, so a pose measured in it needs
  no further transform to mean something in the depth frame."""
  detection_labels: np.ndarray | None = None
  """Detector instance labels on its native grid, attached by perception.

  This and the fields below are optional recording telemetry.  They are never
  read by the control path; keeping them on the camera frame lets the
  asynchronous recorder preserve exactly what perception produced without
  running the detector a second time on the control thread.
  """
  detection_rgb_labels: np.ndarray | None = None
  """Accepted YOLO instances on the original RGB grid, when YOLO is active."""
  source_mask: np.ndarray | None = None
  """Final sensor-grid target mask after SAM/depth arbitration."""
  sam_raw_mask: np.ndarray | None = None
  """Unfiltered SAM output before watchdog and depth fallback."""
  sam_rgb_mask: np.ndarray | None = None
  """Unfiltered SAM output on the original colour-imager grid."""
  policy_mask: np.ndarray | None = None
  """Final target mask on the policy's 168x224 camera grid."""
  detections: list[dict] | None = None
  """JSON-safe instance boxes and 3-D measurements for log overlays."""
  mask_state: str | None = None
  """Per-frame mask source/state, e.g. depth or SAM watchdog state."""
  sensor_meta: dict | None = None
  """RealSense device timestamps and frame numbers for each raw stream."""


class Reader:
  """Latest-frame-wins reader for the D405."""

  def __init__(self, serial: str | None = None, infrared: bool = False,
               backend: str = "d455", stereo: bool = False, **overrides):
    overrides.setdefault("infrared", infrared or stereo)
    overrides.setdefault("stereo", stereo)
    self.backend = str(backend)
    if self.backend not in ("d405", "d455"):
      raise ValueError(f"unsupported deployment camera backend {self.backend!r}")
    if self.backend == "d455":
      overrides.setdefault("emitter", "on")
    self._camera = _bench_backend(self.backend)
    self._infrared = bool(overrides.get("infrared", False))
    self._stereo = bool(overrides.get("stereo", False))
    self._gray_source = str(overrides.get("gray_source", "aligned_color"))
    args = _Args(serial=serial, **overrides)
    self._stream = self._camera.Stream(args)
    self.K = self._stream.K
    self.dist = self._stream.dist
    self.meta = self._stream.meta
    self.serial = self.meta.get("serial")

    self._lock = threading.Lock()
    self._frame: Frame | None = None
    self._stop = threading.Event()
    self._errors = 0
    self._thread = threading.Thread(target=self._run, daemon=True,
                                    name=f"{self.backend}-reader")
    self._thread.start()

  def _run(self) -> None:
    index = 0
    while not self._stop.is_set():
      try:
        if self._stereo:
          depth, gray, ir, ir_right = self._stream.read4()
        else:
          depth, gray, ir = self._stream.read3()
          ir_right = None
        rgb = getattr(self._stream, "raw_color", None)
        sensor_meta = getattr(self._stream, "frame_meta", None)
        self._errors = 0
      except Exception:
        self._errors += 1
        if self._errors > 25:
          # The bench found this camera wedges permanently when it shares an
          # xHCI controller with another depth sensor -- a hardware reset is
          # the only thing that recovers it, and it is cheaper than asking
          # someone to find the cable mid-run.
          try:
            self._stream.close()
            self._camera.reset(self.serial)
            self._stream = self._camera.Stream(
              _Args(serial=self.serial, infrared=self._infrared,
                    stereo=self._stereo, gray_source=self._gray_source,
                    width=int(self.meta["resolution"][0]),
                    height=int(self.meta["resolution"][1])))
            self._errors = 0
          except Exception:
            time.sleep(0.5)
        continue
      index += 1
      with self._lock:
        self._frame = Frame(depth=depth, gray=gray, rgb=rgb, ir=ir,
                            ir_right=ir_right,
                            stamp=time.time(), index=index,
                            sensor_meta=(None if sensor_meta is None else
                                         dict(sensor_meta)))

  def latest(self) -> Frame | None:
    with self._lock:
      return self._frame

  @property
  def age(self) -> float:
    """Seconds since the current frame arrived; ``inf`` before the first."""
    f = self.latest()
    return float("inf") if f is None else time.time() - f.stamp

  SETTLE_S = 1.5
  """How long to keep taking frames after the first one arrives.

  The D405's auto-exposure has not converged when the first frame lands, and
  the frames before it does are dark enough to be useless without being
  obviously broken.  What that cost, once: ``calibrate --preview`` reported
  "board not found" on twelve consecutive frames with the board squarely in
  view and perfectly detectable a second later, which sends you looking at the
  dictionary, the square size and the mounting -- everything except the
  exposure.  1.5 s is about 45 frames at 30 Hz; Intel's own guidance is to
  discard the first 30.
  """

  def wait_for_first(self, timeout: float = 10.0,
                     settle_s: float | None = None) -> Frame:
    settle = self.SETTLE_S if settle_s is None else float(settle_s)
    deadline = time.time() + timeout
    first = None
    while time.time() < deadline:
      f = self.latest()
      if f is not None:
        if first is None:
          first = time.time()
        if time.time() - first >= settle:
          return f
      time.sleep(0.02)
    raise RuntimeError(
      f"no frame from the {self.meta.get('model', self.backend)} within "
      f"{timeout} s. Check `rs-enumerate-devices` "
      "and that it is on a USB 3 port -- the bench found this camera stops "
      "delivering entirely when it shares a controller with another sensor."
    )

  def close(self) -> None:
    self._stop.set()
    self._thread.join(timeout=2.0)
    self._stream.close()

  def intrinsics(self, width: int, height: int) -> np.ndarray:
    return self._stream.intrinsics(width, height)


@dataclasses.dataclass
class _Args:
  """The bench backend's argument object, at the settings it was measured on."""

  width: int = config.D405_WIDTH
  height: int = config.D405_HEIGHT
  fps: int = config.D405_FPS
  preset: str = "default"
  depth_units: float = 1e-4
  infrared: bool = False
  """Also stream the raw left infrared imager; see ``Frame.ir``."""
  stereo: bool = False
  """Also stream the RIGHT imager, so a learned stereo model can compute its
  own disparity.  Implies ``infrared``; see ``Frame.ir_right``."""
  gray_source: str = "aligned_color"
  """``left_ir`` is the direct unwarped D405 grayscale used for calibration."""
  filters: bool = False
  """Raw, because the noise model was fitted to raw.  Turning the stock spatial
  and temporal filters on here would make the robot's depth quieter than the
  simulator's, in a way whose shape no one has measured."""
  serial: str | None = None
  emitter: str | None = None
  laser_power: float | None = None
