"""Intel RealSense D455 backend.

The D455 shares librealsense plumbing with the D405 but not its sensor model:
it has a wider stereo baseline and an active infrared projector.  This wrapper
keeps captures in ``results/d455`` and records the emitter state so that an
active D455 measurement can never be mistaken for the passive D405 data.
"""

from __future__ import annotations

import numpy as np

from . import Capture
from . import d405 as _rs

NAME = "d455"


def add_args(ap) -> None:
  _rs.add_args(ap)
  ap.add_argument("--emitter", choices=("on", "off"), default="on",
                  help="D455 infrared projector state")
  ap.add_argument("--laser-power", type=float, default=None,
                  help="projector power in the device's librealsense units")


def available() -> list[dict]:
  return [d | {"backend": NAME} for d in _rs.available("D455")]


reset = _rs.reset


class Stream(_rs.Stream):
  def __init__(self, args, serial: str | None = None):
    super().__init__(args, serial, backend_name=NAME, expected_model="D455",
                     emitter_description="configurable D455 projector")


def grab(args, n_frames: int, warmup: int = 30) -> Capture:
  s = Stream(args)
  try:
    for _ in range(warmup):
      s.read()
    depths, gray = [], None
    for _ in range(n_frames):
      d, gray = s.read()
      depths.append(d)
    meta = dict(s.meta, n_frames=n_frames)
    return Capture(depth=np.stack(depths), gray=gray, K=s.K, dist=s.dist,
                   meta=meta)
  finally:
    s.close()
