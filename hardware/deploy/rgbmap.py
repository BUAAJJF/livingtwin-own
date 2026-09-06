"""Project masks between the D455 colour imager and its left/depth imager.

Visual models see the synchronized, unaligned RGB frame.  Geometry and the
policy stay on the left-infrared/depth grid.  A depth pixel is deprojected,
transformed with the factory depth-to-colour extrinsic, and projected into the
RGB imager; sampling the RGB mask there yields a mask tied to the same surface
as that depth value.  The RGB image itself is never warped through depth.
"""

from __future__ import annotations

import collections
import time

import cv2
import numpy as np


def _intrinsics(meta: dict, name: str) -> dict:
  value = meta.get(name)
  if not isinstance(value, dict):
    raise ValueError(f"camera metadata has no {name}")
  required = ("width", "height", "fx", "fy", "ppx", "ppy")
  missing = [key for key in required if key not in value]
  if missing:
    raise ValueError(f"{name} is missing {missing}")
  return value


class RgbDepthMapper:
  """Vectorized mask projection using one RealSense frameset's calibration."""

  def __init__(self, camera_meta: dict, seed_dilate_px: int = 1):
    d = _intrinsics(camera_meta, "depth_intrinsics")
    c = _intrinsics(camera_meta, "color_intrinsics")
    extr = camera_meta.get("depth_to_color")
    if not isinstance(extr, dict):
      raise ValueError("camera metadata has no depth_to_color")
    self.depth_shape = (int(d["height"]), int(d["width"]))
    self.color_shape = (int(c["height"]), int(c["width"]))
    self.dfx, self.dfy = np.float32(d["fx"]), np.float32(d["fy"])
    self.dcx, self.dcy = np.float32(d["ppx"]), np.float32(d["ppy"])
    self.cfx, self.cfy = np.float32(c["fx"]), np.float32(c["fy"])
    self.ccx, self.ccy = np.float32(c["ppx"]), np.float32(c["ppy"])
    self.color_model = str(c.get("model", "distortion.none")).split(".")[-1]
    self.color_coeffs = np.asarray(c.get("coeffs", [0] * 5),
                                   dtype=np.float32)
    depth_coeffs = np.asarray(d.get("coeffs", [0] * 5), dtype=np.float64)
    # The D455 left/depth stream on the measured unit is distortion-free.
    # Deprojecting a distorted depth grid as pinhole would silently shift the
    # target, so refuse a different device profile instead of approximating.
    if np.max(np.abs(depth_coeffs), initial=0.0) > 1e-5:
      raise ValueError(
        "depth/left imager is distorted; vectorized deprojection needs its "
        "inverse model before RGB deployment")
    # rs2_extrinsics.rotation is a flat column-major matrix. Reshaping it in
    # NumPy's default row-major order silently applies the inverse rotation.
    self.R = np.asarray(extr.get("rotation"), dtype=np.float32).reshape(
      3, 3, order="F")
    self.t = np.asarray(extr.get("translation"),
                        dtype=np.float32).reshape(3)
    if not np.isfinite(self.R).all() or not np.isfinite(self.t).all():
      raise ValueError("non-finite depth_to_color extrinsic")
    if abs(np.linalg.det(self.R) - 1.0) > 1e-3:
      raise ValueError("depth_to_color rotation is not a rotation")
    self.seed_dilate_px = max(0, int(seed_dilate_px))
    yy, xx = np.indices(self.depth_shape, dtype=np.float32)
    self._rx = (xx - self.dcx) / self.dfx
    self._ry = (yy - self.dcy) / self.dfy
    # Rigid projection is affine in depth. Precompute the rotated ray once so
    # each frame needs only three multiply-adds, rather than materializing X/Y
    # and evaluating a 3x3 transform for every pixel.
    self._ray_x = (self.R[0, 0] * self._rx + self.R[0, 1] * self._ry
                   + self.R[0, 2])
    self._ray_y = (self.R[1, 0] * self._rx + self.R[1, 1] * self._ry
                   + self.R[1, 2])
    self._ray_z = (self.R[2, 0] * self._rx + self.R[2, 1] * self._ry
                   + self.R[2, 2])
    self._cached_depth = None
    self._cached_map = None
    self._cached_color_index = None
    self._stats = collections.Counter()
    self._timing_ms = collections.defaultdict(list)

  def _distort(self, x, y):
    model = self.color_model
    if model in ("none", "0"):
      return x, y
    if model not in ("brown_conrady", "modified_brown_conrady",
                     "inverse_brown_conrady", "4", "2", "1"):
      raise ValueError(f"unsupported RGB distortion model {model!r}")
    k1, k2, p1, p2, k3 = self.color_coeffs[:5]
    r2 = x * x + y * y
    f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    if model in ("modified_brown_conrady", "1"):
      xf, yf = x * f, y * f
      # RealSense's modified model applies tangential displacement after the
      # radial coordinates have been modified.
      r2f = xf * xf + yf * yf
      return (xf + 2 * p1 * xf * yf + p2 * (r2f + 2 * xf * xf),
              yf + 2 * p2 * xf * yf + p1 * (r2f + 2 * yf * yf))
    return (x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * f + 2 * p2 * x * y + p1 * (r2 + 2 * y * y))

  def map(self, depth: np.ndarray):
    """Return nearest RGB pixel per depth pixel and a valid projection mask."""
    z = np.asarray(depth, dtype=np.float32)
    if z.shape != self.depth_shape:
      raise ValueError(f"depth is {z.shape}, expected {self.depth_shape}")
    if z is self._cached_depth:
      self._stats["map_cache_hits"] += 1
      return self._cached_map
    started = time.perf_counter()
    pcx = self._ray_x * z + self.t[0]
    pcy = self._ray_y * z + self.t[1]
    pcz = self._ray_z * z + self.t[2]
    good_z = (z > 0.0) & np.isfinite(z) & (pcz > 1e-6)
    nx = np.divide(pcx, pcz, out=np.zeros_like(pcx), where=good_z)
    ny = np.divide(pcy, pcz, out=np.zeros_like(pcy), where=good_z)
    nx, ny = self._distort(nx, ny)
    u = np.rint(nx * self.cfx + self.ccx).astype(np.int32)
    v = np.rint(ny * self.cfy + self.ccy).astype(np.int32)
    h, w = self.color_shape
    valid = good_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    result = (u, v, valid)
    # One flat gather maps a whole RGB mask. Clipping is harmless because the
    # validity mask clears out-of-frame projections immediately afterwards.
    self._cached_color_index = (
      np.clip(v, 0, h - 1) * w + np.clip(u, 0, w - 1)).reshape(-1)
    # Keep a strong reference to make identity caching safe even if callers
    # release their frame immediately. YOLO and SAM share this map.
    self._cached_depth = z
    self._cached_map = result
    self._stats["map_compute_calls"] += 1
    self._timing_ms["map_compute"].append(
      (time.perf_counter() - started) * 1000.0)
    return result

  def rgb_masks_to_depth(self, masks: np.ndarray,
                         depth: np.ndarray) -> np.ndarray:
    """Sample one or more RGB masks at every valid depth surface pixel."""
    started = time.perf_counter()
    value = np.asarray(masks, dtype=bool)
    one = value.ndim == 2
    if one:
      value = value[None]
    if value.ndim != 3 or value.shape[1:] != self.color_shape:
      raise ValueError(
        f"RGB masks are {value.shape}, expected (N, {self.color_shape[0]}, "
        f"{self.color_shape[1]})")
    u, v, valid = self.map(depth)
    del u, v
    out = value.reshape(value.shape[0], -1)[:, self._cached_color_index]
    out = out.reshape(value.shape[0], *self.depth_shape)
    out[:, ~valid] = False
    self._stats["rgb_to_depth_calls"] += 1
    self._stats["rgb_mask_pixels"] += int(value.sum())
    self._stats["depth_mask_pixels"] += int(out.sum())
    self._timing_ms["rgb_to_depth_total"].append(
      (time.perf_counter() - started) * 1000.0)
    return out[0] if one else out

  def depth_mask_to_rgb(self, mask: np.ndarray,
                        depth: np.ndarray) -> np.ndarray:
    """Splat a depth-grid seed into RGB for YOLO/SAM prompting."""
    started = time.perf_counter()
    value = np.asarray(mask, dtype=bool)
    if value.shape != self.depth_shape:
      raise ValueError(f"depth mask is {value.shape}, expected {self.depth_shape}")
    u, v, valid = self.map(depth)
    take = valid & value
    out = np.zeros(self.color_shape, dtype=np.uint8)
    out[v[take], u[take]] = 1
    if self.seed_dilate_px and out.any():
      n = 2 * self.seed_dilate_px + 1
      kernel = np.ones((n, n), np.uint8)
      out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
      out = cv2.dilate(out, kernel, iterations=1)
    self._stats["depth_to_rgb_calls"] += 1
    self._stats["depth_seed_pixels"] += int(value.sum())
    self._stats["rgb_seed_pixels"] += int(out.sum())
    self._timing_ms["depth_to_rgb_total"].append(
      (time.perf_counter() - started) * 1000.0)
    return out.astype(bool)

  def report(self) -> dict:
    def summary(values):
      a = np.asarray(values, dtype=np.float64)
      return {"n": int(a.size), "p50": float(np.percentile(a, 50)),
              "p95": float(np.percentile(a, 95)), "max": float(a.max())}
    return {
      "counts": {key: int(value) for key, value in self._stats.items()},
      "timing_ms": {key: summary(value)
                    for key, value in self._timing_ms.items() if value},
    }


class MappedSamPredictor:
  """Keep SamTargetTracker on depth masks while SAM itself sees raw RGB."""

  def __init__(self, predictor, mapper: RgbDepthMapper):
    self.predictor = predictor
    self.mapper = mapper
    self.depth: np.ndarray | None = None
    self.last_rgb_mask: np.ndarray | None = None

  def set_depth(self, depth: np.ndarray) -> None:
    self.depth = np.asarray(depth)
    # Compute once here (or hit YOLO's map) and reuse it for SAM prompting and
    # propagation on this same synchronized frame.
    self.mapper.map(self.depth)

  def _seed(self, mask: np.ndarray) -> np.ndarray:
    if self.depth is None:
      raise RuntimeError("set_depth must be called before SAM")
    return self.mapper.depth_mask_to_rgb(mask, self.depth)

  def anchor(self, image: np.ndarray, mask: np.ndarray) -> None:
    self.predictor.anchor(image, self._seed(mask))

  def reanchor(self, mask: np.ndarray) -> None:
    self.predictor.reanchor(self._seed(mask))

  def propagate(self, image: np.ndarray) -> np.ndarray | None:
    raw = self.predictor.propagate(image)
    self.last_rgb_mask = None if raw is None else np.asarray(raw, dtype=bool)
    if raw is None:
      return None
    if self.depth is None:
      raise RuntimeError("set_depth must be called before SAM")
    return self.mapper.rgb_masks_to_depth(raw, self.depth)

  def reset(self) -> None:
    self.last_rgb_mask = None
    self.predictor.reset()

  def report(self) -> dict:
    base = getattr(self.predictor, "report", None)
    return {"model": base() if callable(base) else None,
            "projection": self.mapper.report()}
