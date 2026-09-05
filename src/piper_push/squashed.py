"""The bounded action head: a Gaussian on u, and tanh applied by the environment.

What the policy emits is ``u``, a plain Gaussian sample.  What the arm gets is
``a = tanh(u)``, applied inside the action term (``actions.py``, ``bounded``),
fed back to the policy as the ``actions`` observation, penalised by the
smoothness terms, regressed on by distillation and applied again by the deploy
mapper on the robot.  So every consumer of the action sees a bounded number,
and PPO never has to invert the tanh:

* rsl_rl stores what ``sample`` returns and hands it back to ``log_prob``.
  That is ``u`` itself, in the float32 it was drawn in, so the density is the
  Gaussian density of the stored sample -- no ``atanh``, no rounding.  The
  first version of this head returned ``a`` and recovered ``u = atanh(a)``
  from float32; next to 1.0 that is off by 0.15 at |u| = 8, five sigmas when
  sigma is 0.03, and the PPO ratio overflowed (surrogate loss 3.7e7) and took
  sigma to NaN in both v10 teachers.  The rule since: the density is always
  evaluated on the stored sample, never on a value reconstructed from it.

* The importance ratio in ``a`` equals the ratio in ``u``: the Jacobian of
  the tanh is a property of the sample, not of the parameters, and cancels
  between old and new.  ``kl_divergence`` likewise is the Gaussian KL.

* The entropy is the entropy OF ``a``, ``H(u) + E[log(1 - tanh^2 u)]``, over
  a few reparameterised draws.  ``H(u)`` alone is unbounded in sigma and the
  entropy bonus inflated sigma until the samples were bang-bang; the
  Jacobian term falls like -2 E|u| and stops paying for noise the tanh throws
  away, and its gradient pulls a saturating mean back toward the range where
  the action still does something.

Why a bounded head at all: the old term mapped ``a`` onto the target with
``offset + scale * a`` and clipped the target; nothing bounded ``a``, and the
arm scales spanned a quarter of the safe range.  The teachers ran the gripper
between -28 and +14 with 95% of steps past +-1, joint 4 of the deployed v4 sat
at a = -3.5 (results/audit_20260904/gripper_action_*.json), a constant
saturated value cost nothing, was fed back verbatim as the ``actions``
observation and was the label the student regressed on.
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


class PreSquashGaussianDistribution(Distribution):
  """``u ~ N(mu, sigma)``; the environment applies ``tanh``.

  Constructor arguments mirror ``rsl_rl.modules.GaussianDistribution`` so the
  runner configs differ by ``class_name`` only.  ``init_std`` may be one number
  or one per output; the bounded task uses one per joint so that the initial
  exploration in joint space matches the old convention's.
  """

  def __init__(
    self,
    output_dim: int,
    init_std: float | list[float] | tuple[float, ...] = 0.6,
    std_range: tuple[float, float] = (0.02, 2.0),
    std_type: str = "scalar",
    learn_std: bool = True,
    entropy_samples: int = 4,
  ) -> None:
    super().__init__(output_dim)
    self.std_type = std_type
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
    self.entropy_samples = int(entropy_samples)
    self._normal: Normal | None = None
    Normal.set_default_validate_args(False)

  def _std(self) -> torch.Tensor:
    if self.std_type == "scalar":
      return self.std_param.clamp(self.std_range[0], self.std_range[1])
    return torch.exp(self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1]))

  def update(self, mlp_output: torch.Tensor) -> None:
    self._normal = Normal(mlp_output, self._std())

  # -- outputs: u, always ------------------------------------------------------

  def sample(self) -> torch.Tensor:
    return self._normal.sample()  # type: ignore[union-attr]

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    return mlp_output

  def as_deterministic_output_module(self) -> nn.Module:
    # The exported graph emits u; the deploy mapper applies the tanh, exactly
    # as the action term does in simulation (action_spec["squashed"]).
    return nn.Identity()

  @property
  def input_dim(self) -> int:
    return self.output_dim

  @property
  def mean(self) -> torch.Tensor:
    return self._normal.mean  # type: ignore[union-attr]

  @property
  def std(self) -> torch.Tensor:
    return self._normal.stddev  # type: ignore[union-attr]

  # -- densities: on the stored u ---------------------------------------------

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self._normal.log_prob(outputs).sum(dim=-1)  # type: ignore[union-attr]

  @property
  def entropy(self) -> torch.Tensor:
    """Entropy of ``a = tanh(u)``: ``H(u) + E[log(1 - tanh^2 u)]``, the
    expectation over ``entropy_samples`` reparameterised draws."""
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
