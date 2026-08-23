"""The sensor model has to reproduce what the bench measured, not just be noisy.

Every number checked here is read back out of a simulated image and compared to
``hardware/depth_bench/model/d405_noise.json``.  That is the point: a noise
model whose output does not have the measured statistics is a decorative one,
and the failure is invisible -- training still runs, the images still look
plausible, and the policy is still being prepared for a camera that does not
exist.

Four properties are worth more than the rest, because each one is a thing the
old two-constant placeholder got wrong and each one changes what a
convolutional policy can do about it:

* the error grows as z^2, so a constant fitted at one distance is wrong at
  every other;
* a third of it does not change between frames, so it cannot be filtered away;
* it is correlated across several pixels, so downsampling does not average it
  out;
* the dropout sits on depth discontinuities, so it lands on exactly the
  silhouette that says how big the object is.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from piper_push import camera as cam
from piper_push import depth_noise

H, W = 168, 224
F_PX_PER_RAD = cam.f_px_per_rad(height=H)


def make(num_envs: int = 64, **kw) -> depth_noise.DepthCorruption:
  cfg = dataclasses.replace(depth_noise.DepthNoiseCfg(), **kw)
  torch.manual_seed(0)
  return depth_noise.DepthCorruption(num_envs, H, W, F_PX_PER_RAD, "cpu", cfg)


def flat(z: float, num_envs: int = 64) -> torch.Tensor:
  return torch.full((num_envs, 1, H, W), z)


# ---------------------------------------------------------------------------
# Magnitude and its distance dependence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("z", [0.3, 0.5, 0.7, 0.9])
def test_sigma_grows_as_z_squared(z):
  """sigma = a z^2, with a the quadrature sum of the two measured components.

  Checked against the model's own constants rather than a hard-coded number so
  that re-fitting the bench data moves the test with it -- what is being
  asserted is that the implementation delivers what it claims, which is a
  different question from whether the claim is right.
  """
  c = make(texture_penalty=(1.0, 1.0))     # pin the surface so only z varies
  d = flat(z)
  out, _ = c(d)
  # Undo the range bias, which is a systematic and not part of sigma.
  out = (out - c.bias_offset) / (1.0 + c.bias_scale)
  a = math.hypot(c.cfg.sigma_static_per_m, c.cfg.sigma_temporal_per_m)
  assert (out - d).std().item() == pytest.approx(a * z * z, rel=0.08)


def test_texture_penalty_scales_the_noise():
  """A featureless surface is noisier, by the measured factor.

  The bench put a printed pattern, a blank white patch and a solid black patch
  in one frame: a = 0.0203, 0.0405, 0.0345.  The simulator has no textures, so
  the ratio is drawn per environment instead, and this checks it reaches the
  image at full strength.
  """
  # Bias pinned: its per-environment spread is larger than the noise itself at
  # this range, and it does not scale with the surface, so leaving it in would
  # dilute the ratio towards one.
  pin = dict(bias_scale=0.0, bias_offset_m=0.0, bias_scale_jitter=0.0,
             bias_offset_jitter_m=0.0)
  lo = make(texture_penalty=(1.0, 1.0), **pin)
  hi = make(texture_penalty=(2.0, 2.0), **pin)
  d = flat(0.7)
  assert (hi(d)[0] - d).std().item() / (lo(d)[0] - d).std().item() == \
    pytest.approx(2.0, rel=0.08)


# ---------------------------------------------------------------------------
# The split that the placeholder had no way to express
# ---------------------------------------------------------------------------


def test_a_third_of_the_variance_survives_averaging():
  """Averaging many frames removes the twinkle and leaves the fixed pattern.

  This is the property that decides whether a temporal filter on the robot
  would help.  Measured share of variance that is static: 0.35.
  """
  c = make(texture_penalty=(1.0, 1.0), bias_scale=0.0, bias_offset_m=0.0,
           bias_scale_jitter=0.0, bias_offset_jitter_m=0.0)
  d = flat(0.7, num_envs=16)
  n = 200
  acc = torch.zeros_like(d)
  for _ in range(n):
    acc += c(d)[0]
  residual_of_mean = (acc / n - d).std().item()

  total = math.hypot(c.cfg.sigma_static_per_m, c.cfg.sigma_temporal_per_m)
  expected_static = c.cfg.sigma_static_per_m * 0.7**2
  # sigma_t/sqrt(n) is still in there; it is 7% of the static term at n=200.
  assert residual_of_mean == pytest.approx(expected_static, rel=0.12)
  share = c.cfg.sigma_static_per_m**2 / total**2
  assert share == pytest.approx(0.35, abs=0.03)


def test_reset_redraws_the_fixed_pattern():
  c = make(texture_penalty=(1.0, 1.0))
  d = flat(0.7)
  before = c.static_field.clone()
  c.reset(torch.tensor([0, 1, 2]))
  assert not torch.allclose(before[:3], c.static_field[:3])
  assert torch.allclose(before[3:], c.static_field[3:])
  del d


def test_frozen_between_resets():
  """Two calls with no reset in between share the static half of the error."""
  c = make(texture_penalty=(1.0, 1.0), sigma_temporal_per_m=0.0)
  d = flat(0.7)
  a, _ = c(d)
  b, _ = c(d)
  assert torch.allclose(a, b, atol=1e-7)


# ---------------------------------------------------------------------------
# Correlation length
# ---------------------------------------------------------------------------


def test_noise_is_correlated_over_the_measured_distance():
  """1/e at 8.5 sensor pixels, which is ~3.4 pixels of the policy's image.

  Assumed independent, the 2.7x downsample from the sensor's grid to this one
  would cut the noise by 2.7x.  It does not, and this is why.
  """
  c = make(texture_penalty=(1.0, 1.0), sigma_static_per_m=0.0,
           bias_scale=0.0, bias_offset_m=0.0,
           bias_scale_jitter=0.0, bias_offset_jitter_m=0.0)
  d = flat(0.7)
  e = c(d)[0] - d
  e = e - e.mean()
  v = (e * e).mean()
  prof = [((e[..., :, :W - k] * e[..., :, k:]).mean() / v).item()
          for k in range(12)]
  below = [i for i, p in enumerate(prof) if p < math.exp(-1.0)]
  expected = depth_noise.CORR_LEN_NATIVE_PX * F_PX_PER_RAD / \
    depth_noise.NATIVE_F_PX_PER_RAD
  assert below, f"noise decorrelates beyond 12 px, profile={prof}"
  assert below[0] == pytest.approx(expected, abs=1.6)


# ---------------------------------------------------------------------------
# Where the holes are
# ---------------------------------------------------------------------------


def test_flat_surfaces_are_nearly_full():
  c = make(surface_fill=(1.0, 1.0), hole_dilate=False)
  _, valid = c(flat(0.7))
  assert valid.float().mean().item() == pytest.approx(
    depth_noise.EDGE_P_FLAT, abs=0.02
  )


def test_holes_concentrate_on_depth_edges():
  """A silhouette is where the sensor fails, and the object is all silhouette.

  Built as a step from 0.65 m to 0.75 m, which is roughly what the top of a
  40 mm object against the table behind it looks like from this camera.
  """
  c = make(surface_fill=(1.0, 1.0), hole_dilate=False)
  d = flat(0.7, num_envs=32)
  d[..., W // 2:] = 0.62
  _, valid = c(d)
  edge = valid[..., W // 2 - 1:W // 2 + 1].float().mean().item()
  interior = valid[..., 8:W // 2 - 8].float().mean().item()
  assert interior == pytest.approx(depth_noise.EDGE_P_FLAT, abs=0.02)

  g = 0.08 / 0.7 * F_PX_PER_RAD
  predicted = depth_noise.EDGE_P_FLAT / (
    1.0 + (g / depth_noise.EDGE_G50_PER_RAD) ** depth_noise.EDGE_EXP
  )
  assert edge == pytest.approx(predicted, abs=0.05)
  assert edge < 0.75 * interior


def test_dilation_widens_the_occlusion_shadow():
  c = make(surface_fill=(1.0, 1.0), hole_dilate=True)
  d = flat(0.7)
  assert c(d)[1].float().mean().item() < depth_noise.EDGE_P_FLAT


def test_surface_fill_reaches_flat_regions():
  c = make(surface_fill=(0.5, 0.5), hole_dilate=False)
  assert c(flat(0.7))[1].float().mean().item() == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# Range bias
# ---------------------------------------------------------------------------


def test_bias_is_a_scale_and_an_offset_and_differs_per_environment():
  """Measured -1.3% and -5 mm, so about -14 mm at 0.70 m.

  Randomised rather than applied, because it is a calibration error: the next
  camera's will be different and the policy must not be able to depend on it.
  """
  c = make(texture_penalty=(1.0, 1.0), sigma_static_per_m=0.0,
           sigma_temporal_per_m=0.0)
  per_env = (c(flat(0.7))[0] - 0.7).flatten(1).mean(1)
  assert per_env.mean().item() == pytest.approx(-0.0141, abs=0.002)
  assert per_env.std().item() > 0.004
  far = (c(flat(1.0))[0] - 1.0).flatten(1).mean(1)
  assert (far - per_env).mean().item() == pytest.approx(
    depth_noise.BIAS_SCALE * 0.3, abs=0.004
  )


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


def test_strength_zero_is_a_pass_through():
  """``play`` has to return the clean image untouched, bit for bit.

  Recorded rollouts are compared to each other across runs, and a sensor that
  is only nearly off would make two identical policies look different.
  """
  c = make(strength=0.0)
  d = flat(0.7)
  out, valid = c(d)
  assert torch.equal(out, d)
  assert bool(valid.all())
