"""The tanh-squashed Gaussian head: bounded output, exact density, sane entropy.

The reference density is torch's own ``TransformedDistribution(Normal,
TanhTransform)``; the head has to agree with it to float precision, because
the PPO ratio is a difference of two of these and a bias there is a bias in
every update.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from piper_push.squashed import SquashedGaussianDistribution, log1m_tanh2


def _head(std=0.6, dim=7):
  d = SquashedGaussianDistribution(dim, init_std=std)
  mu = torch.linspace(-3.0, 3.0, dim).unsqueeze(0).repeat(5, 1)
  mu = mu + 0.1 * torch.arange(5.0).unsqueeze(1)
  d.update(mu)
  return d, mu


def test_samples_and_deterministic_output_stay_inside_the_unit_box():
  d, mu = _head()
  torch.manual_seed(0)
  s = torch.stack([d.sample() for _ in range(200)])
  assert s.abs().max() < 1.0
  assert torch.allclose(d.deterministic_output(mu), torch.tanh(mu))
  assert torch.allclose(d.mean, torch.tanh(mu))


def test_log_prob_matches_torch_transformed_distribution():
  d, mu = _head()
  ref = TransformedDistribution(Normal(mu, d.std), [TanhTransform(cache_size=1)])
  torch.manual_seed(1)
  a = torch.tanh(Normal(mu, d.std).sample())
  got = d.log_prob(a)
  want = ref.log_prob(a).sum(-1)
  assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), (got - want).abs().max()


def test_log1m_tanh2_is_the_stable_form():
  u = torch.tensor([-20.0, -5.0, -1.0, 0.0, 0.5, 3.0, 20.0])
  naive = torch.log1p(-torch.tanh(u) ** 2)
  ok = u.abs() < 8  # the naive form underflows to -inf beyond that
  assert torch.allclose(log1m_tanh2(u)[ok], naive[ok], atol=1e-3)
  assert torch.isfinite(log1m_tanh2(u)).all()


def test_log_prob_is_finite_at_the_clamp_and_has_a_gradient():
  d = SquashedGaussianDistribution(3, init_std=0.6)
  mu = torch.zeros(2, 3, requires_grad=True)
  d.update(mu)
  a = torch.tensor([[0.0, 0.999999, -1.0], [1.0, -0.5, 0.3]])
  lp = d.log_prob(a)
  assert torch.isfinite(lp).all()
  lp.sum().backward()
  assert torch.isfinite(mu.grad).all()


def test_entropy_falls_as_the_mean_saturates():
  """The change-of-variables term is what pulls a saturating mean back: the
  entropy bonus must get worse, not stay flat, as |mu| grows."""
  torch.manual_seed(0)
  d = SquashedGaussianDistribution(1, init_std=0.6, entropy_samples=256)
  ents = []
  for m in (0.0, 1.0, 2.0, 4.0):
    d.update(torch.full((1, 1), m))
    ents.append(d.entropy.detach().item())
  assert all(a > b for a, b in zip(ents, ents[1:])), ents
  d.update(torch.full((1, 1), 4.0, requires_grad=True))
  (g,) = torch.autograd.grad(d.entropy.sum(), d._normal.mean)
  assert g.item() < 0.0


def test_entropy_is_bounded_by_the_box_and_stops_paying_for_noise():
  """A bounded variable cannot hold more than log 2 nats per dimension.  The
  mean-evaluated proxy reported 12 nats for seven dimensions at sigma ~ 2 and
  PPO's entropy bonus inflated sigma to get there; the sampled estimate must
  stay under the bound and stop rising once the samples are bang-bang."""
  import math
  torch.manual_seed(0)
  ents = {}
  for std in (0.1, 0.6, 2.0, 5.0):
    d = SquashedGaussianDistribution(7, init_std=std, std_range=(1e-3, 10.0), entropy_samples=512)
    d.update(torch.zeros(1, 7))
    ents[std] = d.entropy.detach().item()
  assert all(e < 7 * math.log(2.0) + 0.1 for e in ents.values()), ents
  assert ents[0.6] > ents[0.1]
  assert ents[5.0] < ents[2.0], ents  # more noise in u is LESS entropy in a
  # And sigma gets a gradient from it, so the bonus can push sigma down.
  d = SquashedGaussianDistribution(7, init_std=3.0, std_range=(1e-3, 10.0), entropy_samples=64)
  d.update(torch.zeros(1, 7))
  (g,) = torch.autograd.grad(d.entropy.sum(), d.std_param)
  assert (g < 0).all(), g


def test_kl_is_taken_in_u_space_and_is_zero_for_equal_params():
  d, mu = _head()
  p = d.params
  assert torch.allclose(d.kl_divergence(p, p), torch.zeros(5))
  q = (mu + 1.0, d.std)
  assert (d.kl_divergence(p, q) > 0).all()


def test_export_module_is_a_tanh():
  d, mu = _head()
  m = d.as_deterministic_output_module()
  assert isinstance(m, nn.Module)
  assert torch.allclose(m(mu), torch.tanh(mu))
  scripted = torch.jit.script(m)
  assert torch.allclose(scripted(mu), torch.tanh(mu))


def test_rsl_rl_model_uses_the_head_through_its_class_name():
  from rsl_rl.models import MLPModel
  from tensordict import TensorDict
  obs = TensorDict({"x": torch.randn(4, 6)}, batch_size=[4])
  m = MLPModel(obs, {"actor": ["x"]}, "actor", 7, hidden_dims=(16,),
               distribution_cfg={"class_name": "piper_push.squashed:SquashedGaussianDistribution",
                                 "init_std": 0.6, "std_type": "scalar"})
  a = m(obs, stochastic_output=True)
  assert a.shape == (4, 7) and a.abs().max() < 1.0
  assert torch.isfinite(m.get_output_log_prob(a)).all()
  assert torch.isfinite(m.output_entropy).all()
  assert m(obs).abs().max() < 1.0


def test_small_sigma_near_saturation_keeps_the_ratio_finite():
  """The v10 teacher crash: sigma 0.03, mean near 7, actions stored in
  float32.  Recovering u from the stored a was off by more than sigma, the
  PPO ratio overflowed and sigma went NaN.  Old and new log-probs of the
  same stored action must now differ by a finite, moderate amount."""
  torch.manual_seed(0)
  d = SquashedGaussianDistribution(7, init_std=0.03)
  mu_old = torch.full((256, 7), 7.0)
  d.update(mu_old)
  a = d.sample().float()                    # what rsl_rl stores
  assert torch.isfinite(a).all() and a.abs().max() < 1.0
  lp_old = d.log_prob(a)
  d.update(mu_old + 0.01)                   # one KL-limited step later
  lp_new = d.log_prob(a)
  ratio = torch.exp(lp_new - lp_old)
  assert torch.isfinite(lp_old).all() and torch.isfinite(lp_new).all()
  assert torch.isfinite(ratio).all() and ratio.max() < 10.0, ratio.max()
  # and the gradient that reaches sigma is finite
  d2 = SquashedGaussianDistribution(7, init_std=0.03)
  d2.update(mu_old + 0.01)
  (g,) = torch.autograd.grad(d2.log_prob(a).sum(), d2.std_param)
  assert torch.isfinite(g).all()


def test_sigma_has_a_floor_below_which_the_density_would_lie():
  d = SquashedGaussianDistribution(3, init_std=1e-4)
  d.update(torch.zeros(1, 3))
  assert (d.std >= 0.02).all()
