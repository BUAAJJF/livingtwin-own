"""Camera backends for the depth bench.

Three cameras are being compared and only one of them exists on the desk at a
time, so the thing that has to stay fixed is the *interface*, not the code
behind it.  A backend's only job is to hand back a stack of depth frames on a
known pixel grid, one grayscale image registered to that same grid, and the
intrinsics of the grid.  Every number the bench reports is computed from those
three things by ``metrics.py``, which has never heard of RealSense.

Depth is metres, float32, with 0 meaning "no measurement".  The grayscale image
is what the ChArUco is detected in; it must be on the depth grid, because the
whole point is that the ground-truth plane and the depth samples share a
coordinate system with no resampling of the depth in between.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Capture:
  """One recording session: N frames of the same static scene."""

  depth: np.ndarray  # (N, H, W) float32 metres, 0 = invalid
  gray: np.ndarray  # (H, W) uint8, registered to the depth grid
  K: np.ndarray  # (3, 3) float64, intrinsics of the depth grid
  dist: np.ndarray  # (5,) float64 distortion of the depth grid
  meta: dict  # everything needed to say what produced this

  def __post_init__(self) -> None:
    if self.depth.ndim != 3:
      raise ValueError(f"depth must be (N, H, W), got {self.depth.shape}")
    if self.gray.shape != self.depth.shape[1:]:
      raise ValueError(
        f"gray {self.gray.shape} is not on the depth grid {self.depth.shape[1:]}"
      )

  def save(self, path) -> None:
    import json

    np.savez_compressed(
      path,
      depth=self.depth,
      gray=self.gray,
      K=self.K,
      dist=self.dist,
      meta=json.dumps(self.meta),
    )

  @staticmethod
  def load(path) -> "Capture":
    import json

    z = np.load(path, allow_pickle=False)
    return Capture(
      depth=z["depth"], gray=z["gray"], K=z["K"], dist=z["dist"],
      meta=json.loads(str(z["meta"])),
    )


def open_backend(name: str):
  """Import a backend lazily -- each one needs an SDK that may not be installed."""
  if name == "d405":
    from . import d405

    return d405
  if name == "zedx":
    from . import zedx

    return zedx
  if name == "odin1":
    from . import odin1

    return odin1
  raise SystemExit(f"unknown backend {name!r}; have: d405, zedx, odin1")
