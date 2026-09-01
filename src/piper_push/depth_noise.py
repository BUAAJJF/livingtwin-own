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
radian.  Fit RMS 0.024 over seven decades-wide bins.  ``p_flat`` is not used at
runtime: it is the fill rate of the flat parts of that particular desk, and the
runtime draws that per surface from ``SURFACE_FILL`` instead.  It is kept here
because the other two constants are meaningless without the normalisation they
were fitted alongside."""

SHADOW_MRAD = 5.8
"""How far a hole spreads beyond the discontinuity that causes it, in
milliradians -- an angle, like the correlation length, and for the same reason:
it is a property of the stereo matching window and not of whatever grid the
model happens to be evaluated on.  It was a pixel of the policy's image, which
is 5.8 mrad; stated that way it was 2.7 times wider in angle when the same
model ran on the sensor's own grid, and the two paths disagreed on the hole
rate by a factor of 2.3 as a result.  A stereo hole is an occlusion shadow -- the strip the second
camera cannot see is wider than the step that casts it -- and the measurement
shows it directly: in the 2.2-4.3 per-radian bin fill is still 0.87 but 48% of
those pixels flicker, against 19% on flat ground.  Implemented by widening the
*gradient* rather than by dilating the holes, which is both what physically
happens and the only version that does not inflate the hole rate: dilating
independently drawn holes by 3x3 turns 6% into 43%."""

TEXTURE_PENALTY = (1.0, 2.1)
"""Noise multiplier.  1.0 is the textured target, 2.0 the blank white patch
(0.0405/0.0203), 1.7 the black one."""

SURFACE_FILL = (0.88, 0.995)
"""Steady fill rate on a flat surface.  Measured means: textured 0.996, black
0.989, blank white 0.881 -- and 0.419 in the worst single shot, which is what
the low end of the range is reaching towards rather than matching.

Both of these are drawn **once per environment**, not per object.  The
simulator renders untextured geometry, so there is nothing to key a per-object
quality off, and one draw for the whole scene says "everything in this
environment is as featureless as a blank sheet" a fraction of the time.  That
is the conservative reading of the measurement rather than the faithful one --
a real table has a textured mat and one white object on it -- and it is the
obvious thing to refine if the policy turns out to be over-cautious."""

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
  native_f_px_per_rad: float = NATIVE_F_PX_PER_RAD
  edge_g50_per_rad: float = EDGE_G50_PER_RAD
  edge_exp: float = EDGE_EXP
  texture_penalty: tuple[float, float] = TEXTURE_PENALTY
  surface_fill: tuple[float, float] = SURFACE_FILL
  bias_scale: float = BIAS_SCALE
  bias_offset_m: float = BIAS_OFFSET_M
  bias_scale_jitter: float = BIAS_SCALE_JITTER
  bias_offset_jitter_m: float = BIAS_OFFSET_JITTER_M
  shadow_mrad: float = SHADOW_MRAD
  stereo_baseline_m: float = 0.0
  stereo_focal_px: float = 0.0
  disparity_subpixel_levels: int = 0
  """Optional active/passive stereo quantisation.

  RealSense D400 depth is produced in disparity and quantised at a fixed
  sub-pixel resolution before it becomes metres.  Zero keeps the historical
  D405 model byte-identical; the fitted D455 config supplies its 95 mm
  baseline, native focal length and 1/32-pixel disparity grid.
  """


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
      1.0,
      self.cfg.corr_len_native_px * self.f_px_per_rad
      / self.cfg.native_f_px_per_rad,
    )
    self._lo_h = max(2, int(round(height / self.corr_px)))
    self._lo_w = max(2, int(round(width / self.corr_px)))
    self.shadow_px = int(round(self.cfg.shadow_mrad * 1e-3 * self.f_px_per_rad))

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
    own 848x480 grid means the same thing here.

    A central difference, matching ``np.gradient``, because that is what the
    threshold was fitted with.  A one-sided difference is a defensible choice
    on its own -- at a silhouette the step really is on one side -- but it
    returns twice as much at a step edge as the central difference does, so
    used against a threshold fitted the other way it would double the dropout
    on every object outline in the scene.

    Pad once and slice, rather than convolve with a difference kernel.  The
    convolution is the obvious way to write it and it is fifteen times slower
    here -- 4.6 ms against 0.3 -- because cuDNN has no good plan for one input
    channel and two output channels, and picks a general one.  At 512
    environments that difference is 4 ms of every control step.
    """
    d = F.pad(depth, (1, 1, 1, 1), mode="replicate")
    gx = d[..., 1:-1, 2:] - d[..., 1:-1, :-2]
    gy = d[..., 2:, 1:-1] - d[..., :-2, 1:-1]
    return torch.sqrt(gx * gx + gy * gy) * (0.5 * self.f_px_per_rad) \
      / depth.clamp_min(1e-3)

  def __call__(
    self, depth: torch.Tensor, generator: torch.Generator | None = None,
    featureless: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt a clean depth image.

    Args:
      depth: ``(B, 1, H, W)`` metres, already clamped to the far plane.
      featureless: optional ``(B, 1, H, W)`` in [0, 1] saying which pixels the
        camera has nothing to match on.  1 gets this environment's drawn
        surface quality, 0 gets a textured one.  Omitted, the whole frame gets
        the drawn quality, which is the pessimistic reading -- it says every
        surface in the scene is as bad as the worst one.

        The distinction is worth making because the deployment can arrange half
        of it.  The table gets a textured mat, so it is a 0; the objects are
        whatever they are, so they are 1s.  With the whole frame at the drawn
        quality the simulator spends a good fraction of its episodes claiming
        the *table* is a blank white sheet, and the measured consequence is not
        small: it is most of the reason a purely geometric segmenter reports
        phantoms across the tablecloth.

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

    # All three correlated fields in one upsample.  They are the frozen error,
    # this frame's error, and the variate that decides where the holes go; they
    # are independent but they share a correlation length, so they share the
    # filter that gives them one.
    fresh = torch.randn(b, 2, self._lo_h, self._lo_w, device=depth.device,
                        dtype=depth.dtype, generator=generator)
    fields = self._smooth(
      torch.cat([self.static_field[:b].to(depth.dtype), fresh], dim=1)
    ) * self._gain
    static, temporal, w = fields[:, 0:1], fields[:, 1:2], fields[:, 2:3]

    if featureless is None:
      texture, fill = self.texture[:b], self.fill[:b]
    else:
      q = featureless.to(depth.dtype).clamp(0.0, 1.0)
      texture = 1.0 + (self.texture[:b] - 1.0) * q
      fill = c.surface_fill[1] + (self.fill[:b] - c.surface_fill[1]) * q

    scale = (c.strength * texture) * depth * depth
    out = depth + scale * (
      static * c.sigma_static_per_m + temporal * c.sigma_temporal_per_m
    )
    # Range bias last, on the corrupted depth: it is what the camera reports,
    # not a property of the surface.
    out = out * (1.0 + self.bias_scale[:b]) + self.bias_offset[:b]

    # The D400 ASIC estimates disparity, not depth.  Quantising in metres
    # would make the error constant with range; quantising disparity preserves
    # the z^2 growth dictated by stereo geometry.  Clamp the denominator only
    # against numerical zero -- the clean renderer has already imposed the
    # sensor's near/far range.
    if (c.disparity_subpixel_levels > 0 and c.stereo_baseline_m > 0.0
        and c.stereo_focal_px > 0.0):
      fb = c.stereo_focal_px * c.stereo_baseline_m
      levels = float(c.disparity_subpixel_levels)
      disparity = fb / out.clamp_min(1.0e-4)
      disparity = torch.round(disparity * levels) / levels
      out = fb / disparity.clamp_min(1.0 / levels)

    # Validity is decided from the *clean* depth.  The geometry is what casts
    # the occlusion shadow; letting the noise decide where the edges are would
    # put holes in the middle of flat surfaces.
    g = self.relative_gradient(depth)
    if self.shadow_px > 0:
      k = 2 * self.shadow_px + 1
      g = F.max_pool2d(g, k, 1, self.shadow_px)
    p = fill / (1.0 + (g / c.edge_g50_per_rad).pow(c.edge_exp))

    # The draw is correlated over the same distance as the noise, because holes
    # come in blobs.  Passive stereo fails over a region -- a patch with no
    # texture, a strip in occlusion shadow -- and never one isolated pixel at a
    # time.  Drawn as a smoothed normal put through its own CDF, which is a
    # correlated uniform, so the marginal probability is still exactly ``p``.
    u = 0.5 * (1.0 + torch.erf(w * 0.70710678118654752))
    return out, u < p
