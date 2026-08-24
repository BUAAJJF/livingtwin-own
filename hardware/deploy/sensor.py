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


def _bench_backend():
  """The bench's own D405 backend, imported from where it lives.

  Imported rather than reimplemented: two files configuring the same camera
  differently is how a deployment ends up running a sensor the measurements do
  not describe.
  """
  if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))
  from capture import d405           # noqa: E402  -- path set above
  return d405


@dataclasses.dataclass
class Frame:
  depth: np.ndarray
  """``(480, 848)`` metres, 0 where the sensor returned nothing."""
  gray: np.ndarray
  """``(480, 848)`` uint8, resampled into the depth frame by the driver."""
  stamp: float
  index: int


class Reader:
  """Latest-frame-wins reader for the D405."""

  def __init__(self, serial: str | None = None, **overrides):
    self._d405 = _bench_backend()
    args = _Args(serial=serial, **overrides)
    self._stream = self._d405.Stream(args)
    self.K = self._stream.K
    self.meta = self._stream.meta
    self.serial = self.meta.get("serial")

    self._lock = threading.Lock()
    self._frame: Frame | None = None
    self._stop = threading.Event()
    self._errors = 0
    self._thread = threading.Thread(target=self._run, daemon=True,
                                    name="d405-reader")
    self._thread.start()

  def _run(self) -> None:
    index = 0
    while not self._stop.is_set():
      try:
        depth, gray = self._stream.read()
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
            self._d405.reset(self.serial)
            self._stream = self._d405.Stream(_Args(serial=self.serial))
            self._errors = 0
          except Exception:
            time.sleep(0.5)
        continue
      index += 1
      with self._lock:
        self._frame = Frame(depth=depth, gray=gray, stamp=time.time(),
                            index=index)

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
      f"no frame from the D405 within {timeout} s.  Check `rs-enumerate-devices` "
      "and that it is on a USB 3 port -- the bench found this camera stops "
      "delivering entirely when it shares a controller with another sensor."
    )

  def close(self) -> None:
    self._stop.set()
    self._thread.join(timeout=2.0)
    self._stream.close()


@dataclasses.dataclass
class _Args:
  """The bench backend's argument object, at the settings it was measured on."""

  width: int = config.D405_WIDTH
  height: int = config.D405_HEIGHT
  fps: int = config.D405_FPS
  preset: str = "default"
  depth_units: float = 1e-4
  filters: bool = False
  """Raw, because the noise model was fitted to raw.  Turning the stock spatial
  and temporal filters on here would make the robot's depth quieter than the
  simulator's, in a way whose shape no one has measured."""
  serial: str | None = None
