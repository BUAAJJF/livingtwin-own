"""Camera backends for the depth bench.

Three cameras are being compared and only one of them exists on the desk at a
time, so the thing that has to stay fixed is the *interface*, not the code
behind it.  A backend's only job is to hand back a stack of depth frames on a
known pixel grid, one grayscale image registered to that same grid, and the
intrinsics of the grid.  Every number the bench reports is computed from those
three things by ``metrics.py``, which has never heard of RealSense.

Depth is metres, float32, with 0 meaning "no measurement".

The geometry is carried as a **ray table** rather than an intrinsics matrix.
A pinhole K was the obvious choice and it is wrong: the Odin 1 is a lidar, and
fitting a pinhole to its measured ray directions leaves a 17-pixel residual on
a 256x192 grid -- its grid is not a rectilinear projection and no (fx, fy, cx,
cy) describes it.  A ray table describes both sensors exactly, and for a
pinhole camera it is computed from K in one line, so nothing is lost.

The greyscale image is what the ChArUco is detected in.  It does *not* have to
be on the depth grid -- on the Odin 1 the board is far too small in the 256x192
lidar image to resolve a marker, so the board is found in the 1600x1296 colour
image instead and the resulting pose is carried into the depth frame by
``T_dg``.  What must never happen is the reverse: resampling the depth to meet
the image, because the depth samples are the measurement.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def rays_from_K(K, shape) -> np.ndarray:
  """Z-normalised ray directions for a pinhole grid: the (H, W, 3) with z == 1."""
  h, w = shape
  u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
  return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                   np.ones_like(u)], axis=-1)


@dataclasses.dataclass
class Capture:
  """One recording session: N frames of the same static scene."""

  depth: np.ndarray  # (N, H, W) float32 metres, 0 = invalid
  gray: np.ndarray  # (h, w) uint8, the image the ChArUco is found in
  K: np.ndarray  # (3, 3) float64, intrinsics of *gray* (undistorted pinhole)
  dist: np.ndarray  # (5,) float64 distortion of gray
  meta: dict  # everything needed to say what produced this
  rays: np.ndarray | None = None
  """(H, W, 3) z-normalised ray direction per depth pixel, in the depth frame.
  ``None`` means the depth grid is the same pinhole grid as ``gray``, and the
  rays are derived from ``K``."""
  T_dg: np.ndarray | None = None
  """(4, 4) taking a pose from the ``gray`` camera frame into the depth frame.
  ``None`` means they are the same frame."""

  def __post_init__(self) -> None:
    if self.depth.ndim != 3:
      raise ValueError(f"depth must be (N, H, W), got {self.depth.shape}")
    grid = self.depth.shape[1:]
    if self.rays is None:
      if self.gray.shape != grid:
        raise ValueError(
          f"gray {self.gray.shape} is not on the depth grid {grid}, so the rays "
          "cannot come from K -- pass an explicit ray table"
        )
      self.rays = rays_from_K(self.K, grid)
    if self.rays.shape != (*grid, 3):
      raise ValueError(f"rays {self.rays.shape} does not match the depth grid {grid}")
    if self.T_dg is None:
      self.T_dg = np.eye(4)

  def save(self, path) -> None:
    import json

    np.savez_compressed(
      path,
      depth=self.depth,
      gray=self.gray,
      K=self.K,
      dist=self.dist,
      rays=self.rays,
      T_dg=self.T_dg,
      meta=json.dumps(self.meta),
    )

  @staticmethod
  def load(path) -> "Capture":
    import json

    z = np.load(path, allow_pickle=False)
    return Capture(
      depth=z["depth"], gray=z["gray"], K=z["K"], dist=z["dist"],
      meta=json.loads(str(z["meta"])),
      rays=z["rays"] if "rays" in z else None,
      T_dg=z["T_dg"] if "T_dg" in z else None,
    )


def open_backend(name: str):
  """Import a backend lazily -- each one needs an SDK that may not be installed."""
  if name == "d405":
    from . import d405

    return d405
  if name == "d455":
    from . import d455

    return d455
  raise SystemExit(f"unknown backend {name!r}; have: d405, d455")
