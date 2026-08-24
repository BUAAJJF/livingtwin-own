"""Turn what the D405 measured into the image the policy was trained on.

The policy sees a 224x168 depth image from a 52-degree camera at a particular
place in the robot's base frame.  The D405 delivers 848x480 from an 87-degree
camera at whatever place the mount actually put it.  Those are two different
cameras, and the gap between them is not a resize.

So the point cloud is re-rendered: every measured sample is put back in space
using the D405's own intrinsics, moved into the base frame with the calibrated
extrinsic, and photographed again by the simulator's camera.  That handles the
field of view, the resolution, the residual rotation of the mount and the
residual translation, all in one operation, and it is the only version that is
right when the mount is a centimetre off -- an inverse warp through a rotation
alone cannot be, because a translated camera sees round corners differently.

Two details are worth more than they look.

**Nearest, then average.**  The D405 supplies about seven samples per policy
pixel, so something has to combine them.  Taking the nearest is what a
z-buffer does and it is right at an occlusion boundary, where the samples come
from two different surfaces -- but on a flat surface it takes the minimum of
seven noisy measurements, which is biased low by about 1.3 standard deviations,
9 mm at 0.7 m.  So: nearest to pick the surface, then the mean of everything
within a window of it to measure that surface.  The bias goes away and the
occlusion behaviour stays.

**Averaging buys almost nothing, and that is expected.**  The seven samples
sit inside 2.7 sensor pixels, and the sensor's noise is correlated over 8.5
(``piper_push.depth_noise``), so they are nearly the same sample.  The
reduction is about 9%, not the 2.7x independence would predict.  This is why
``depth_noise`` states its sigma on the policy's grid: if resampling really did
average the noise down, the simulator would be training against a quieter
camera than the robot has.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from . import config


@dataclasses.dataclass(frozen=True)
class VirtualCamera:
  """The pinhole the policy looks through.

  Defined by the same three numbers the simulator's ``CameraSensorCfg`` is,
  and derived from them rather than restated, so there is one definition of
  what the policy sees.
  """

  width: int = config.WIDTH
  height: int = config.HEIGHT
  fovy_deg: float = config.FOVY_DEG

  @property
  def K(self) -> np.ndarray:
    f = 0.5 * self.height / math.tan(math.radians(self.fovy_deg) / 2)
    return np.array([[f, 0.0, (self.width - 1) / 2.0],
                     [0.0, f, (self.height - 1) / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def ray_grid(K: np.ndarray, width: int, height: int) -> np.ndarray:
  """``(H, W, 3)`` directions with z == 1, in the OpenCV camera frame."""
  u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                     np.arange(height, dtype=np.float64))
  x = (u - K[0, 2]) / K[0, 0]
  y = (v - K[1, 2]) / K[1, 1]
  return np.stack([x, y, np.ones_like(x)], axis=-1)


class Reprojector:
  """Re-render a D405 depth image as the policy's camera would have seen it.

  Holds the fixed parts -- the source ray table and the transform -- so the
  per-frame work is a matrix multiply, a projection and two scatters.  On the
  GPU that is well under a millisecond; on the CPU it is a few.

  ``payload`` lets anything else measured on the sensor's grid come along for
  the ride.  The segmenter runs at 848x480, where a 40 mm object is 25 pixels
  across instead of 9, and its labels have to end up on the policy's grid
  aligned with the depth to the pixel.  Carrying the index of the sample that
  won each cell is the only way to guarantee that: the label and the depth then
  come from the same measurement, by construction, rather than from two
  resamplings that agree until they do not.
  """

  def __init__(
    self,
    rig: "config.Rig",
    virtual: VirtualCamera | None = None,
    device: str | torch.device = "cpu",
    average_window_m: float = 0.015,
    refine: int = 2,
    T_base_virtual: np.ndarray | None = None,
    margin_translation_m: float = 0.02,
    margin_near_m: float = 0.25,
    decimate: int = 1,
  ) -> None:
    self.virtual = virtual or VirtualCamera()
    self.device = torch.device(device)
    self.average_window_m = float(average_window_m)
    self.refine = int(refine)

    # ``decimate`` takes every n-th row and column of the source.  Exact, not
    # approximate: pixel (i, j) of ``depth[::n, ::n]`` is pixel (ni, nj) of the
    # original, so the intrinsics divide by n and the rays are the same rays.
    #
    # It is what makes the segmenter affordable.  At full resolution a 40 mm
    # object is 25 pixels across and the segmenter costs 39 ms; at half it is
    # 12, which is still a third more than the policy's own image will have,
    # and it costs a quarter of that.  The *depth* is never decimated -- that
    # is the measurement, and it is resampled at full resolution.
    self.decimate = max(1, int(decimate))
    K_src = np.asarray(rig.K if rig.K is not None else _default_d405_K(),
                       dtype=np.float64).copy()
    K_src[:2] /= self.decimate
    self.K_src = K_src
    self.src_width = config.D405_WIDTH // self.decimate
    self.src_height = config.D405_HEIGHT // self.decimate
    rays = ray_grid(self.K_src, self.src_width, self.src_height)
    self._rays = torch.as_tensor(rays.reshape(-1, 3), dtype=torch.float32,
                                 device=self.device)

    T_bv = config.sim_camera_extrinsic() if T_base_virtual is None \
      else np.asarray(T_base_virtual, dtype=np.float64)
    T = np.linalg.inv(T_bv) @ rig.T_base_cam       # source -> virtual
    self.T_virtual_src = T
    self._R = torch.as_tensor(T[:3, :3], dtype=torch.float32, device=self.device)
    self._t = torch.as_tensor(T[:3, 3], dtype=torch.float32, device=self.device)

    K = self.virtual.K
    self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
    self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
    self._n = self.virtual.width * self.virtual.height

    # Which source pixels can possibly land in the policy's image.  The D405
    # sees 87 x 58 degrees and the policy 66 x 52, so a third of every frame is
    # of the room and is carried through the whole pipeline for nothing.
    #
    # Decided by direction alone, which ignores the translation between the two
    # cameras -- so the margin has to cover the parallax.  At the 20 mm the
    # mount is allowed to be out and the 0.25 m of the nearest thing the camera
    # will see, that is under 5 degrees; the margin is set from those two
    # numbers rather than picked.
    margin = float(np.degrees(np.arctan2(margin_translation_m,
                                         margin_near_m)))
    dirs = self._rays @ self._R.T          # rotation only, src -> virtual
    zz = dirs[:, 2].clamp_min(1e-6)
    uu = dirs[:, 0] / zz * self.fx + self.cx
    vv = dirs[:, 1] / zz * self.fy + self.cy
    pad_u = self.fx * math.tan(math.radians(margin))
    pad_v = self.fy * math.tan(math.radians(margin))
    self._in_fov = torch.nonzero(
      (dirs[:, 2] > 0) & (uu > -pad_u) & (uu < self.virtual.width + pad_u)
      & (vv > -pad_v) & (vv < self.virtual.height + pad_v),
      as_tuple=False).reshape(-1)
    self.fov_fraction = float(self._in_fov.numel()) / self._rays.shape[0]

  # -----------------------------------------------------------------------

  def points_base(self, depth: np.ndarray, rig: "config.Rig") -> np.ndarray:
    """The measured cloud in the robot base frame, ``(N, 3)``, valid only.

    Segmentation wants this and not the image: "above the table and inside the
    workspace" is a statement about the room, and it is a two-line test there
    and a mess in image coordinates.
    """
    d = torch.as_tensor(np.ascontiguousarray(depth, dtype=np.float32),
                        device=self.device).reshape(-1)
    src = self._in_fov[d[self._in_fov] > 0]
    pts = self._rays[src] * d[src, None]
    R = torch.as_tensor(rig.T_base_cam[:3, :3], dtype=torch.float32,
                        device=self.device)
    t = torch.as_tensor(rig.T_base_cam[:3, 3], dtype=torch.float32,
                        device=self.device)
    self.last_src = src.cpu().numpy()
    """Which source pixels the returned points came from.  The segmenter needs
    it to write its labels back onto the sensor's grid, and it is no longer
    simply "every valid pixel" now that the ones outside the policy's field of
    view are dropped before anything looks at them."""
    return (pts @ R.T + t).cpu().numpy()

  def rays_base(self, idx: np.ndarray, rig: "config.Rig") -> np.ndarray:
    """Unit rays for the given source pixels, ``(N, 3)`` in the base frame.

    For placing something the depth could not measure.  A detection whose
    pixels are all holes has no cloud to average, and dropping it would defeat
    the backend that exists for exactly that case -- so the ray through it is
    intersected with the fitted table plane instead.  The answer is the point
    on the table under the object rather than the object, which is half its
    height low and well inside the tracker's 60 mm gate.
    """
    r = self._rays[torch.as_tensor(np.asarray(idx, dtype=np.int64),
                                   device=self.device)]
    R = torch.as_tensor(rig.T_base_cam[:3, :3], dtype=torch.float32,
                        device=self.device)
    out = (r @ R.T).cpu().numpy()
    return out / np.linalg.norm(out, axis=-1, keepdims=True)

  @staticmethod
  def camera_origin_base(rig: "config.Rig") -> np.ndarray:
    """Where the camera is, in the base frame -- the origin those rays start
    from."""
    return np.asarray(rig.T_base_cam[:3, 3], dtype=np.float64)

  def source(self, depth: np.ndarray) -> np.ndarray:
    """The frame at this reprojector's own resolution."""
    if self.decimate == 1:
      return depth
    return np.ascontiguousarray(depth[::self.decimate, ::self.decimate])

  def __call__(
    self, depth: np.ndarray, payload: np.ndarray | None = None
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resample one frame.

    Args:
      depth: metres at this reprojector's source resolution, 0 where the
        sensor returned nothing.
      payload: optional ``(480, 848)`` of anything, carried to the winning
        cell.  Integer labels are the intended use.

    Returns:
      ``(depth, valid, payload)`` on the virtual grid.  ``depth`` is metres and
      is 0 where nothing landed; ``valid`` is bool; ``payload`` is None if none
      was given.
    """
    H, W = self.virtual.height, self.virtual.width
    d = torch.as_tensor(np.ascontiguousarray(depth, dtype=np.float32),
                        device=self.device).reshape(-1)
    src = self._in_fov[d[self._in_fov] > 0]
    if src.numel() == 0:
      return (np.zeros((H, W), np.float32), np.zeros((H, W), bool),
              None if payload is None else np.zeros((H, W), payload.dtype))

    p = (self._rays[src] * d[src, None]) @ self._R.T + self._t
    z = p[:, 2]
    ok = z > 1e-3
    p, z, src = p[ok], z[ok], src[ok]

    u = torch.round(p[:, 0] / z * self.fx + self.cx).long()
    v = torch.round(p[:, 1] / z * self.fy + self.cy).long()
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, src = u[inside], v[inside], z[inside], src[inside]
    if z.numel() == 0:
      return (np.zeros((H, W), np.float32), np.zeros((H, W), bool),
              None if payload is None else np.zeros((H, W), payload.dtype))
    cell = v * W + u

    # Pass one: the nearest surface in each cell.
    near = torch.full((self._n,), float("inf"), device=self.device)
    near.scatter_reduce_(0, cell, z, reduce="amin", include_self=True)

    # Pass two: the mean of the samples that belong to that surface.
    def _mean(keep):
      tot = torch.zeros(self._n, device=self.device)
      cnt = torch.zeros(self._n, device=self.device)
      tot.scatter_add_(0, cell[keep], z[keep])
      cnt.scatter_add_(0, cell[keep],
                       torch.ones(int(keep.sum()), device=self.device))
      return tot, cnt

    tot, cnt = _mean(z <= near[cell] + self.average_window_m)

    # And then recentre, twice, which is not decoration.  The window above is
    # one-sided -- it runs from the minimum upwards -- so on a flat surface it
    # keeps the lower tail of the noise and throws away the upper, and the mean
    # of what is left sits below the truth.  With independent samples at the
    # sigma this sensor has, that bias is 3 mm; one recentring leaves 1.3 mm
    # and two leave less than a third of a millimetre.  The surface is still
    # chosen by the minimum, so the occlusion behaviour is unchanged: this only
    # moves which samples of the *already chosen* surface are averaged.
    for _ in range(self.refine):
      centre = tot / cnt.clamp_min(1.0)
      tot, cnt = _mean((z - centre[cell]).abs() <= self.average_window_m)
    valid = cnt > 0
    out = torch.where(valid, tot / cnt.clamp_min(1.0),
                      torch.zeros((), device=self.device))
    on_surface = z <= near[cell] + self.average_window_m

    out_payload = None
    if payload is not None:
      # Any sample on the winning surface will do, and taking the lowest index
      # makes it deterministic.  Not the argmin of z: that would pick whichever
      # sample the noise happened to push nearest, which changes frame to frame
      # on a flat surface and would make the label flicker for no reason.
      pick = torch.full((self._n,), self._rays.shape[0], dtype=torch.long,
                        device=self.device)
      pick.scatter_reduce_(0, cell[on_surface], src[on_surface],
                           reduce="amin", include_self=True)
      flat = torch.as_tensor(np.ascontiguousarray(payload).reshape(-1),
                             device=self.device)
      taken = torch.zeros(self._n, dtype=flat.dtype, device=self.device)
      got = pick < self._rays.shape[0]
      taken[got] = flat[pick[got]]
      out_payload = taken.reshape(H, W).cpu().numpy()

    return (out.reshape(H, W).cpu().numpy(),
            valid.reshape(H, W).cpu().numpy(),
            out_payload)


def mujoco_K(width: int, height: int, fovy_deg: float) -> np.ndarray:
  """The intrinsics of a MuJoCo camera, which are not quite anyone else's.

  MuJoCo renders with square pixels and the principal point at the exact centre
  of the sensor, so ``cx = (W-1)/2``: pixel *centres* sit at ``u + 0.5``, and
  the half-pixel is the difference between this and ``W/2``.

  It matters when the simulator is standing in for the camera.  A real D405's
  principal point is 1.2 px off centre horizontally and 3.5 px vertically, and
  feeding the real camera's intrinsics to a rendered image throws every ray by
  up to 8 mrad -- 6 mm of lateral error at 0.7 m, which arrives as several
  millimetres of depth error on a table seen at an angle.  That was the first
  thing ``selftest.py`` caught.
  """
  f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2)
  return np.array([[f, 0.0, (width - 1) / 2.0],
                   [0.0, f, (height - 1) / 2.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)


def _default_d405_K() -> np.ndarray:
  """The intrinsics of the D405 the bench measured, at 848x480.

  A fallback for replaying a recording that predates the rig file, and for
  tests.  The live pipeline reads the intrinsics off the camera -- every unit
  is slightly different and the difference is larger than the resampling error
  this module is careful about.
  """
  return np.array([[430.75195312, 0.0, 422.27249146],
                   [0.0, 430.75195312, 243.01411438],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
