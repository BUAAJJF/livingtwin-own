"""A Gaussian squashed through tanh: the action head is bounded, and it has a gradient everywhere.

Why the raw Gaussian head had to go.  The action term maps ``a`` onto a joint
target with ``offset + scale * a`` and clips the *target*; nothing bounded
``a``.  Every value of the gripper action below -1 is the same closed jaw, so
a teacher drifted to -14 (results/audit_20260904/gripper_action_*.json: 95%
of its steps outside [-1, 1]) at no cost -- ``action_rate`` is a difference,
a constant has none -- while the same number was fed back as the ``actions``
observation and became the label the student regressed on.  The arm did the
same with a different excuse: its scales spanned a quarter of the safe range,
so reaching the workspace needed |a| of 3.

A tanh on the mean alone would not fix it: with |mu| large the gradient of
tanh vanishes and the latch comes back as a dead unit.  The head here is the
SAC construction -- ``u ~ N(mu, sigma)``, ``a = tanh(u)``, with the Jacobian
``sum log(1 - a^2)`` in the log-probability -- so the policy gradient is
exact in ``a`` and the entropy bonus, which carries ``log(1 - tanh^2(mu))``,
pulls a saturating mean back toward the range where the action still does
something.

Everything is expressed in ``u`` except the number that leaves the model.
``sample`` and ``deterministic_output`` return ``a``; the environment, the
``actions`` observation, the distillation loss and the exported graph all
see ``a`` in (-1, 1).  ``params`` and ``kl_divergence`` are in ``u`` space,
where the KL between two squashed Gaussians equals the KL between the
Gaussians (tanh is a bijection).  ``entropy`` is ``H(u) + E_u[log(1 -
tanh^2(u))]`` with the expectation taken over a few reparameterised samples.
The cheaper proxy that evaluates the Jacobian term at the mean is unbounded
in sigma -- H(u) grows with log(sigma) and the term at the mean does not
know -- and PPO's entropy bonus drove sigma up until the reported entropy
was 12 nats for seven dimensions that cannot hold more than 7 log 2 = 4.85
(the first 200-iteration run on the state task: reward peaked at iteration
125 and then fell as the samples went bang-bang).  Sampled, the term falls
like -2 E|u| and the bonus stops paying for noise the tanh throws away.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules.distribution import Distribution
from torch.distributions import Normal


def log1m_tanh2(u: torch.Tensor) -> torch.Tensor:
  """``log(1 - tanh(u)^2)``, evaluated in a form that does not cancel."""
  return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))


class SquashedGaussianDistribution(Distribution):
  """``a = tanh(u)``, ``u ~ N(mu, sigma)``, state-independent ``sigma``.

  Constructor arguments mirror ``rsl_rl.modules.GaussianDistribution`` so the
  runner configs differ by ``class_name`` only.  ``init_std`` is in ``u``
  space; 0.6 about ``mu = 0`` is a spread of about 0.5 in ``a``.
  """

  def __init__(
    self,
    output_dim: int,
    init_std: float | list[float] | tuple[float, ...] = 0.6,
    std_range: tuple[float, float] = (1e-6, 1e6),
    std_type: str = "scalar",
    learn_std: bool = True,
    atanh_eps: float = 1e-6,
    entropy_samples: int = 4,
  ) -> None:
    super().__init__(output_dim)
    self.std_type = std_type
    # ``init_std`` may be one number or one per dimension.  Per dimension is
    # how the head matches the old convention's exploration in JOINT space:
    # sigma 0.6 on a scale of 0.9 rad was 0.54 rad of noise on joint 1; the
    # same sigma on the bounded scale of 2.62 rad would be 1.3 rad, and the
    # first 200-iteration comparison learned at half the old speed for it.
    init = torch.as_tensor(init_std, dtype=torch.float32).reshape(-1)
    if init.numel() == 1:
      init = init.expand(output_dim).clone()
    if init.numel() != output_dim:
      raise ValueError(f"init_std has {init.numel()} entries for {output_dim} outputs")
    if std_type == "scalar":
      self.std_param = nn.Parameter(init, requires_grad=learn_std)
    elif std_type == "log":
      self.log_std_param = nn.Parameter(torch.log(init), requires_grad=learn_std)
    else:
      raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")
    self.std_range = [max(float(std_range[0]), 1e-6), float(std_range[1])]
    self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]
    self.atanh_eps = float(atanh_eps)
    self.entropy_samples = int(entropy_samples)
    self._normal: Normal | None = None
    Normal.set_default_validate_args(False)

  # -- parameters -----------------------------------------------------------

  def _std(self) -> torch.Tensor:
    if self.std_type == "scalar":
      return self.std_param.clamp(self.std_range[0], self.std_range[1])
    return torch.exp(self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1]))

  def update(self, mlp_output: torch.Tensor) -> None:
    self._normal = Normal(mlp_output, self._std())

  # -- outputs, in a ---------------------------------------------------------

  def sample(self) -> torch.Tensor:
    return torch.tanh(self._normal.sample())  # type: ignore[union-attr]

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    return torch.tanh(mlp_output)

  def as_deterministic_output_module(self) -> nn.Module:
    return nn.Tanh()

  @property
  def input_dim(self) -> int:
    return self.output_dim

  @property
  def mean(self) -> torch.Tensor:
    """The deterministic action, ``tanh(mu)``."""
    return torch.tanh(self._normal.mean)  # type: ignore[union-attr]

  @property
  def std(self) -> torch.Tensor:
    """``sigma`` in ``u`` space; what the logs call the action std."""
    return self._normal.stddev  # type: ignore[union-attr]

  # -- densities, in u ---------------------------------------------------------

  def _u(self, outputs: torch.Tensor) -> torch.Tensor:
    return torch.atanh(outputs.clamp(-1.0 + self.atanh_eps, 1.0 - self.atanh_eps))

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    u = self._u(outputs)
    return (self._normal.log_prob(u) - log1m_tanh2(u)).sum(dim=-1)  # type: ignore[union-attr]

  @property
  def entropy(self) -> torch.Tensor:
    """``H(u) + E[log(1 - tanh^2 u)]``, the expectation over ``entropy_samples``
    reparameterised draws so the gradient reaches both ``mu`` and ``sigma``."""
    n = self._normal
    u = n.rsample((self.entropy_samples,))  # type: ignore[union-attr]
    return n.entropy().sum(dim=-1) + log1m_tanh2(u).mean(dim=0).sum(dim=-1)  # type: ignore[union-attr]

  @property
  def params(self) -> tuple[torch.Tensor, ...]:
    return (self._normal.mean, self._normal.stddev)  # type: ignore[union-attr]

  def kl_divergence(self, old_params, new_params) -> torch.Tensor:
    old_mean, old_std = old_params
    new_mean, new_std = new_params
    return torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(dim=-1)
