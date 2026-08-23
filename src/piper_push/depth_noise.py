"""What a RealSense D405 does to a depth image, measured and reproduced.

This replaces two numbers -- ``DEPTH_NOISE_M = 0.004`` and
``DEPTH_DROPOUT = 0.02`` -- that were placeholders, and said so.  The camera
was then put on a bench against a printed ChArUco target with a blank white
patch and a solid black patch (``hardware/depth_bench``), and the fitted model
is in ``hardware/depth_bench/model/d405_noise.json``.  Everything below carries
the number it came from.

Four things were measured, and keeping them apart is the whole point.  A single
Gaussian gets the magnitude roughly right and the *character* completely wrong,
and a convolutional policy keys on the character.

**Error grows as the square of distance.**  Passive stereo recovers disparity,
and ``z = fB/d`` turns a fixed disparity error into ``sigma_z = z^2/(fB) *
sigma_d``.  Measured across 0.29-0.80 m on the textured target: ``sigma = 0.0203
z^2``, residual 0.9 mm.  The old constant 4 mm was right at 0.44 m and wrong
everywhere else; at the 0.70 m this camera actually sits at it understates the
error by 2.5x.

**A third of it does not change between frames.**  Stereo matches on whatever
texture is there, and if the texture does not move the disparity error does not
move either.  Split on 40 static frames: ``sigma_static = 0.0078 z^2`` and
``sigma_temporal = 0.0106 z^2``, so 35% of the variance is a fixed pattern.
That part is indistinguishable from real geometry -- no amount of temporal
filtering removes it -- and iid-per-frame noise models none of it.

**It is correlated across about eight sensor pixels.**  Measured by
autocorrelating the plane residual: 1/e at 8 px static, 9 px temporal, on the
848-wide native grid.  This is why resampling to the policy's 224x168 buys
almost nothing: the 2.7x downsample averages ~7 pixels that are already the
same sample.  Assuming independence there would predict a 2.7x noise reduction
that does not happen.

**It fails at depth discontinuities, which is where the object is.**  Fill rate
against the local depth step, per radian so it transfers across resolutions:

    gradient /rad      0-0.9   0.9-2.2   2.2-4.3  4.3-8.6  8.6-21  21-43   43+
    fill                0.97      0.96      0.87     0.75    0.65   0.46  0.31
    flickering          0.20      0.19      0.48     0.62    0.78   0.98  1.00

fitted as ``p = 0.97 / (1 + (g/29.4)^0.99)``, RMS 0.024.  A 25-45 mm object at
0.7 m is almost all silhouette, so this is not an edge case, it is the object.
And at the steep end the pixels do not merely vanish -- they *flicker*, present
one frame and gone the next, which is a far harder thing for a recurrent policy
to sit through than a steady hole.

**Untextured surfaces are twice as noisy and full of holes.**  Same sheet, same
frame, three surfaces: textured ``a = 0.0203`` and 99.6% fill; solid black
``a = 0.0345`` and 98.9%; blank white ``a = 0.0405`` and 88.1% fill, dropping to
41.9% in the worst shot.  The camera has no projector -- it is passive stereo --
so a plain white object is the failure mode.  The simulator has no textures, so
this is drawn per object at reset instead, which is the honest mapping: some of
the things on the table will be featureless and the policy cannot tell which
until it tries.

Everything is randomised around the measured value rather than pinned to it.
The bench measured one camera on one day, and the point of the exercise is a
policy that survives the next one.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Measured constants.  hardware/depth_bench/model/d405_noise.json
# ---------------------------------------------------------------------------

SIGMA_STATIC_PER_M = 0.0078
"""``sigma_static = a z^2``, metres per metre squared.  The frozen part."""

SIGMA_TEMPORAL_PER_M = 0.0106
"""``sigma_temporal = a z^2``.  Redrawn every frame: lag-1 autocorrelation was
-0.007, i.e. nothing to carry over."""

CORR_LEN_NATIVE_PX = 8.5
"""1/e correlation length on the 848x480 native grid."""

NATIVE_F_PX_PER_RAD = 430.8
"""D405 focal length at 848x480.  Correlation and gradient thresholds are
converted through this so they transfer to any other grid."""

EDGE_P_FLAT = 0.97
EDGE_G50_PER_RAD = 29.4
EDGE_EXP = 0.99
"""``p_valid = p_flat / (1 + (g/g50)^n)`` against relative depth gradient per
radian.  Fit RMS 0.024 over seven decades-wide bins."""

TEXTURE_PENALTY = (1.0, 2.1)
"""Noise multiplier drawn per object-surface.  1.0 is the textured target,
2.0 the blank white patch (0.0405/0.0203), 1.7 the black one."""

SURFACE_FILL = (0.88, 1.0)
"""Steady fill rate drawn per surface.  Measured means: textured 0.996, black
0.989, blank white 0.881 -- and 0.419 in the worst single shot, which is what
the low end of the noise scale is for rather than this range."""

BIAS_SCALE = -0.013
BIAS_OFFSET_M = -0.005
"""Range bias fitted as ``bias = s*z + b``: the camera reads about 14 mm short
at 0.70 m.  Randomised rather than applied, because it is a calibration error
and the next camera's will be different -- the policy should not be able to
tell."""

BIAS_SCALE_JITTER = 0.02
BIAS_OFFSET_JITTER_M = 0.008


@dataclasses.dataclass
class DepthNoiseCfg:
  """How hard to push each measured effect.

  Every field defaults to the measurement.  ``strength`` scales the whole thing
  at once, which is what a curriculum wants, and what an ablation wants when it
  needs to ask whether the noise model is doing anything at all.
  """

  strength: float = 1.0
  sigma_static_per_m: float = SIGMA_STATIC_PER_M
  sigma_temporal_per_m: float = SIGMA_TEMPORAL_PER_M
  corr_len_native_px: float = CORR_LEN_NATIVE_PX
  edge_p_flat: float = EDGE_P_FLAT
  edge_g50_per_rad: float = EDGE_G50_PER_RAD
  edge_exp: float = EDGE_EXP
  texture_penalty: tuple[float, float] = TEXTURE_PENALTY
  surface_fill: tuple[float, float] = SURFACE_FILL
  bias_scale: float = BIAS_SCALE
  bias_offset_m: float = BIAS_OFFSET_M
  bias_scale_jitter: float = BIAS_SCALE_JITTER
  bias_offset_jitter_m: float = BIAS_OFFSET_JITTER_M
  hole_dilate: bool = True
  """Grow each hole by one pixel.  A stereo hole is an occlusion shadow: the
  region the second camera cannot see is wider than the discontinuity that
  casts it, and the measured flicker fraction next to an edge (0.48 in the
  2.2-4.3 bin, where fill is still 0.87) is that shadow breathing."""


class DepthCorruption:
  """Per-environment sensor state, redrawn at reset.

  Holds two things that cannot be redrawn every step without misrepresenting
  the sensor: the frozen noise field, and the per-surface texture quality and
  range bias.  Everything else is stateless.

  Sized in the *policy's* grid, and every measured constant is converted into
  it through the focal lengths, so changing ``camera.WIDTH`` or ``FOVY_DEG``
  does not silently change what the sensor does.
  """

  def __init__(
    self,
    num_envs: int,
    height: int,
    width: int,
    f_px_per_rad: float,
    device: torch.device | str,
    cfg: DepthNoiseCfg | None = None,
  ) -> None:
    self.cfg = cfg or DepthNoiseCfg()
    self.num_envs = num_envs
    self.height, self.width = height, width
    self.device = torch.device(device)
    self.f_px_per_rad = float(f_px_per_rad)

    # The correlation length is a property of the stereo matching window, so it
    # is fixed in angle, not in pixels.  On the policy grid a pixel is 2.7x
    # wider, so the same window covers 2.7x fewer of them.
    self.corr_px = max(
      1.0, self.cfg.corr_len_native_px * self.f_px_per_rad / NATIVE_F_PX_PER_RAD
    )
    self._lo_h = max(2, int(round(height / self.corr_px)))
    self._lo_w = max(2, int(round(width / self.corr_px)))

    # Bilinear upsampling of white noise is a low-pass filter, so the result
    # does not have unit variance.  Measure the loss once rather than deriving
    # it: the exact factor depends on align_corners and on how the requested
    # size rounds, and a wrong constant here silently rescales every noise
    # number in this file.
    with torch.no_grad():
      probe = self._smooth(torch.randn(8, 1, self._lo_h, self._lo_w,
                                       device=self.device))
      self._gain = float(1.0 / probe.std().clamp_min(1e-6))

    self.static_field = torch.zeros(num_envs, 1, self._lo_h, self._lo_w,
                                    device=self.device)
    self.texture = torch.ones(num_envs, 1, 1, 1, device=self.device)
    self.fill = torch.ones(num_envs, 1, 1, 1, device=self.device)
    self.bias_scale = torch.zeros(num_envs, 1, 1, 1, device=self.device)
    self.bias_offset = torch.zeros(num_envs, 1, 1, 1, device=self.device)
    self.reset(None)

  # -- state ----------------------------------------------------------------

  def _smooth(self, lo: torch.Tensor) -> torch.Tensor:
    return F.interpolate(lo, size=(self.height, self.width), mode="bilinear",
                         align_corners=False)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    """Draw a new sensor and a new set of surfaces for these environments."""
    if env_ids is None or isinstance(env_ids, slice):
      idx = torch.arange(self.num_envs, device=self.device)
    else:
      idx = env_ids.to(self.device)
    n = int(idx.numel())
    if n == 0:
      return
    c = self.cfg

    self.static_field[idx] = torch.randn(n, 1, self._lo_h, self._lo_w,
                                         device=self.device)

    def _u(lo: float, hi: float) -> torch.Tensor:
      return lo + (hi - lo) * torch.rand(n, 1, 1, 1, device=self.device)

    self.texture[idx] = _u(*c.texture_penalty)
    self.fill[idx] = _u(*c.surface_fill)
    self.bias_scale[idx] = c.bias_scale + _u(-1.0, 1.0) * c.bias_scale_jitter
    self.bias_offset[idx] = (
      c.bias_offset_m + _u(-1.0, 1.0) * c.bias_offset_jitter_m
    )

  # -- the sensor -----------------------------------------------------------

  def relative_gradient(self, depth: torch.Tensor) -> torch.Tensor:
    """Local depth step as a fraction of range, per radian.

    Per radian rather than per pixel so the threshold measured on the sensor's
    own 848x480 grid means the same thing here.  Forward and backward
    differences are taken separately and the larger kept: at a silhouette the
    step is on one side only, and averaging the two halves it and puts the
    object's outline in the wrong bin.
    """
    d = depth
    sx = (d[..., :, 1:] - d[..., :, :-1]).abs()
    sy = (d[..., 1:, :] - d[..., :-1, :]).abs()
    dx = torch.maximum(F.pad(sx, (1, 0)), F.pad(sx, (0, 1)))
    dy = torch.maximum(F.pad(sy, (0, 0, 1, 0)), F.pad(sy, (0, 0, 0, 1)))
    return torch.sqrt(dx * dx + dy * dy) / d.clamp_min(1e-3) * self.f_px_per_rad

  def __call__(
    self, depth: torch.Tensor, generator: torch.Generator | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt a clean depth image.

    Args:
      depth: ``(B, 1, H, W)`` metres, already clamped to the far plane.

    Returns:
      ``(depth, valid)`` -- metres and a boolean of the same shape.  The caller
      decides what an invalid pixel reads as, because that convention has to
      match whatever the deployed pipeline does with a zero from the driver,
      and this module is not the right place to know that.
    """
    c = self.cfg
    if c.strength <= 0.0:
      return depth, torch.ones_like(depth, dtype=torch.bool)

    b = depth.shape[0]
    scale = c.strength * self.texture[:b] * depth * depth

    static = self._smooth(self.static_field[:b]) * self._gain
    temporal = self._smooth(
      torch.randn(b, 1, self._lo_h, self._lo_w, device=depth.device,
                  dtype=depth.dtype, generator=generator)
    ) * self._gain

    out = depth + scale * (
      static * c.sigma_static_per_m + temporal * c.sigma_temporal_per_m
    )
    # Range bias last, on the corrupted depth: it is what the camera reports,
    # not a property of the surface.
    out = out * (1.0 + self.bias_scale[:b]) + self.bias_offset[:b]

    # Validity is decided from the *clean* depth.  The geometry is what casts
    # the occlusion shadow; letting the noise decide where the edges are would
    # put holes in the middle of flat surfaces.
    g = self.relative_gradient(depth)
    p = c.edge_p_flat / (1.0 + (g / c.edge_g50_per_rad).pow(c.edge_exp))
    p = p * (self.fill[:b] / c.edge_p_flat).clamp(max=1.0)
    valid = torch.rand(depth.shape, device=depth.device, dtype=depth.dtype,
                       generator=generator) < p
    if c.hole_dilate:
      valid = ~(F.max_pool2d((~valid).to(depth.dtype), 3, 1, 1) > 0.5)
    return out, valid
