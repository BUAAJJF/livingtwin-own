"""The bounded action head: a Gaussian on u, tanh applied downstream, entropy of tanh(u).

The density is only ever evaluated on the stored sample.  The first version
recovered u = atanh(a) from a float32 a and both v10 teachers went NaN on it;
these tests pin the design that replaced it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from piper_push.squashed import PreSquashGaussianDistribution, log1m_tanh2


def _head(std=0.6, dim=7):
  d = PreSquashGaussianDistribution(dim, init_std=std)
  mu = torch.linspace(-3.0, 3.0, dim).unsqueeze(0).repeat(5, 1) + 0.1 * torch.arange(5.0).unsqueeze(1)
  d.update(mu)
  return d, mu


def test_the_head_emits_u_and_its_density_is_the_gaussian_of_the_stored_sample():
  d, mu = _head()
  torch.manual_seed(0)
  u = d.sample()
  assert u.dtype == torch.float32 and u.abs().max() > 1.0  # not squashed here
  want = Normal(mu, d.std).log_prob(u).sum(-1)
  assert torch.allclose(d.log_prob(u), want)
  assert torch.allclose(d.deterministic_output(mu), mu) and torch.allclose(d.mean, mu)


def test_no_atanh_anywhere_in_the_density_path():
  import inspect
  import piper_push.squashed as m
  src = inspect.getsource(m.PreSquashGaussianDistribution)
  assert "atanh" not in src and ".double()" not in src


def test_small_sigma_far_out_keeps_the_ratio_finite():
  """The crash configuration: sigma 0.03, mean 7, float32 samples.  With the
  density on the stored u, old and new log-probs of the same sample differ
  by a moderate, finite amount."""
  torch.manual_seed(0)
  d = PreSquashGaussianDistribution(7, init_std=0.03)
  mu = torch.full((256, 7), 7.0)
  d.update(mu)
  u = d.sample()
  lp_old = d.log_prob(u)
  d.update(mu + 0.01)
  lp_new = d.log_prob(u)
  ratio = torch.exp(lp_new - lp_old)
  # 0.01 is a third of a sigma; the ratio of a 5-sigma sample moves by e^1.7
  assert torch.isfinite(ratio).all() and ratio.max() < 20.0, ratio.max()
  (g,) = torch.autograd.grad(d.log_prob(u).sum(), d.std_param)
  assert torch.isfinite(g).all()


def test_log1m_tanh2_is_the_stable_form():
  u = torch.tensor([-20.0, -5.0, -1.0, 0.0, 0.5, 3.0, 20.0])
  naive = torch.log1p(-torch.tanh(u) ** 2)
  ok = u.abs() < 8
  assert torch.allclose(log1m_tanh2(u)[ok], naive[ok], atol=1e-3)
  assert torch.isfinite(log1m_tanh2(u)).all()


def test_entropy_is_of_tanh_u_bounded_by_the_box_and_falls_when_the_mean_saturates():
  torch.manual_seed(0)
  ents = {}
  for std in (0.1, 0.6, 2.0, 5.0):
    d = PreSquashGaussianDistribution(7, init_std=std, std_range=(1e-3, 10.0), entropy_samples=512)
    d.update(torch.zeros(1, 7))
    ents[std] = d.entropy.detach().item()
  assert all(e < 7 * math.log(2.0) + 0.1 for e in ents.values()), ents
  assert ents[0.6] > ents[0.1] and ents[5.0] < ents[2.0], ents
  d = PreSquashGaussianDistribution(1, init_std=0.6, entropy_samples=256)
  es = []
  for m in (0.0, 1.0, 2.0, 4.0):
    d.update(torch.full((1, 1), m))
    es.append(d.entropy.detach().item())
  assert all(a > b for a, b in zip(es, es[1:])), es
  d.update(torch.full((1, 1), 4.0, requires_grad=True))
  (g,) = torch.autograd.grad(d.entropy.sum(), d._normal.mean)
  assert g.item() < 0.0
  d = PreSquashGaussianDistribution(7, init_std=3.0, std_range=(1e-3, 10.0), entropy_samples=64)
  d.update(torch.zeros(1, 7))
  (g,) = torch.autograd.grad(d.entropy.sum(), d.std_param)
  assert (g < 0).all()


def test_kl_is_the_gaussian_kl_and_sigma_has_a_floor():
  d, mu = _head()
  p = d.params
  assert torch.allclose(d.kl_divergence(p, p), torch.zeros(5))
  assert (d.kl_divergence(p, (mu + 1.0, d.std)) > 0).all()
  d = PreSquashGaussianDistribution(3, init_std=1e-4)
  d.update(torch.zeros(1, 3))
  assert (d.std >= 0.02).all()


def test_export_module_is_the_identity_so_the_graph_emits_u():
  d, mu = _head()
  m = d.as_deterministic_output_module()
  assert isinstance(m, nn.Identity)
  assert torch.equal(torch.jit.script(m)(mu), mu)


def test_per_dimension_init_std():
  d = PreSquashGaussianDistribution(3, init_std=[0.1, 0.2, 0.3])
  assert torch.allclose(d.std_param, torch.tensor([0.1, 0.2, 0.3]))
  try:
    PreSquashGaussianDistribution(3, init_std=[0.1, 0.2])
  except ValueError:
    pass
  else:
    raise AssertionError("wrong-length init_std accepted")


def test_rsl_rl_model_uses_the_head_through_its_class_name():
  from rsl_rl.models import MLPModel
  from tensordict import TensorDict
  obs = TensorDict({"x": torch.randn(4, 6)}, batch_size=[4])
  m = MLPModel(obs, {"actor": ["x"]}, "actor", 7, hidden_dims=(16,),
               distribution_cfg={"class_name": "piper_push.squashed:PreSquashGaussianDistribution",
                                 "init_std": 0.6, "std_type": "scalar"})
  u = m(obs, stochastic_output=True)
  assert u.shape == (4, 7)
  assert torch.isfinite(m.get_output_log_prob(u)).all() and torch.isfinite(m.output_entropy).all()
