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

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules.distribution import Distribution
from torch.distributions import Normal


def log1m_tanh2(u: torch.Tensor) -> torch.Tensor:
  """``log(1 - tanh(u)^2)``, evaluated in a form that does not cancel."""
  return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))


class UTelemetry:
  """Per-dimension statistics of the pre-squash action, reported every ``every`` samples.

  Answers, from the run itself, where a large |u| comes from: the mean having
  drifted (``frac_mu_gt6``, ``mean_abs_mu``), sigma having grown (``sigma``),
  or the sampling tail (the ``worst`` record carries mu, sigma and u of the
  largest |u| in the window, so ``|u| - |mu|`` is the noise's share).  Also
  the saturation of what the arm receives, ``|tanh(u)| > 0.99`` and ``> 0.999``.

  Written as one line to stdout and, if ``PIPER_U_TELEMETRY`` names a file, as
  one JSON object per window appended to it (``PIPER_U_TELEMETRY_TAG`` labels
  the stage).  All accumulation stays on the device; the only sync is at
  report time.
  """

  def __init__(self, dim: int, every: int = 320, rollout_only: bool = True) -> None:
    self.dim, self.every = dim, int(every)
    # rsl_rl samples once per rollout step under torch.inference_mode(), and
    # once more per PPO minibatch (outside it, result unused) to refresh the
    # distribution before log_prob.  Only the rollout samples are actions the
    # environment saw, so only those are counted.
    self.rollout_only = bool(rollout_only)
    self.calls = 0
    self.windows = 0
    self._t = None

  def _init(self, like: torch.Tensor) -> None:
    # Ordinary tensors, not inference tensors: an inference tensor cannot be
    # updated in place outside inference mode, and the first call may come
    # from either side.
    with torch.inference_mode(False):
      z = lambda: torch.zeros(self.dim, dtype=torch.float64, device=like.device)
      self._t = {k: z() for k in ("n", "gt6", "gt10", "sat99", "sat999", "abs_mu", "mu_gt6", "abs_u", "sigma")}
      self._worst_val = torch.zeros((), dtype=torch.float32, device=like.device)
      self._worst = torch.zeros(5, dtype=torch.float32, device=like.device)  # dim, env, mu, sigma, u
    self._sigma_n = 0

  def observe(self, mu: torch.Tensor, std: torch.Tensor, u: torch.Tensor) -> None:
    if self.rollout_only and not torch.is_inference_mode_enabled():
      return
    if self._t is None:
      self._init(u)
    mu, u = mu.reshape(-1, u.shape[-1]), u.reshape(-1, u.shape[-1])
    with torch.no_grad():
      t = self._t
      au = u.abs()
      at = torch.tanh(au)
      b = float(u.shape[0]) if u.dim() > 1 else 1.0
      t["n"] += b
      t["gt6"] += (au > 6.0).sum(0)
      t["gt10"] += (au > 10.0).sum(0)
      t["sat99"] += (at > 0.99).sum(0)
      t["sat999"] += (at > 0.999).sum(0)
      t["abs_mu"] += mu.abs().sum(0)
      t["mu_gt6"] += (mu.abs() > 6.0).sum(0)
      t["abs_u"] += au.sum(0)
      t["sigma"] += (std.expand_as(u) if std.dim() < u.dim() else std).mean(0)
      self._sigma_n += 1
      flat = au.reshape(-1)
      idx = flat.argmax()
      val = flat[idx]
      if bool(val > self._worst_val):  # one sync per call; cheap next to the rollout
        self._worst_val.copy_(val)
        d = int(idx % u.shape[-1]); e = int(idx // u.shape[-1])
        mu_e = mu.reshape(-1)[idx] if mu.numel() == u.numel() else mu.reshape(-1, u.shape[-1])[0, d]
        sd_e = std.reshape(-1)[d] if std.numel() == u.shape[-1] else std.reshape(-1, u.shape[-1])[e, d]
        self._worst.copy_(torch.stack([torch.tensor(float(d), device=u.device), torch.tensor(float(e), device=u.device),
                                       mu_e.float(), sd_e.float(), u.reshape(-1)[idx].float()]))
    self.calls += 1
    if self.calls % self.every == 0:
      self.report()

  def report(self) -> dict:
    t = {k: v.cpu() for k, v in self._t.items()}
    n = t["n"].clamp_min(1.0)
    rec = {
      "tag": os.environ.get("PIPER_U_TELEMETRY_TAG", ""),
      "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
      "window": self.windows, "calls": self.calls, "samples_per_dim": float(n[0]),
      "frac_u_gt6": (t["gt6"] / n).tolist(), "frac_u_gt10": (t["gt10"] / n).tolist(),
      "frac_sat99": (t["sat99"] / n).tolist(), "frac_sat999": (t["sat999"] / n).tolist(),
      "mean_abs_mu": (t["abs_mu"] / n).tolist(), "frac_mu_gt6": (t["mu_gt6"] / n).tolist(),
      "mean_abs_u": (t["abs_u"] / n).tolist(), "sigma": (t["sigma"] / max(self._sigma_n, 1)).tolist(),
    }
    w = self._worst.cpu().tolist()
    rec["worst"] = {"dim": int(w[0]), "env": int(w[1]), "mu": w[2], "sigma": w[3], "u": w[4],
                    "noise_share": (abs(w[4]) - abs(w[2])) / max(abs(w[4]), 1e-9)}
    f = lambda xs: " ".join(f"{x:.3f}" for x in xs)
    print(f"[u-telemetry {rec['tag']} w{self.windows}] |u|>6 {f(rec['frac_u_gt6'])} | |u|>10 {f(rec['frac_u_gt10'])} | "
          f"sat>0.99 {f(rec['frac_sat99'])} | sat>0.999 {f(rec['frac_sat999'])} | mean|mu| {f(rec['mean_abs_mu'])} | "
          f"sigma {f(rec['sigma'])} | worst dim {rec['worst']['dim']} env {rec['worst']['env']} "
          f"u {w[4]:+.2f} = mu {w[2]:+.2f} + noise (sigma {w[3]:.3f})", flush=True)
    path = os.environ.get("PIPER_U_TELEMETRY")
    if path:
      try:
        with open(path, "a") as fh:
          fh.write(json.dumps(rec) + "\n")
      except OSError as e:  # telemetry must never take the run down
        print(f"[u-telemetry] could not write {path}: {e}", flush=True)
    self.windows += 1
    self._t = None
    return rec


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
    telemetry_every: int = 320,
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
    # The floor may be per dimension (v10d: 0.1 x the old convention's joint-space
    # noise per joint, so exploration never collapses to a bang-bang mean).
    lo = torch.as_tensor(std_range[0], dtype=torch.float32).reshape(-1).clamp_min(1e-6)
    if lo.numel() == 1:
      lo = lo.expand(output_dim).clone()
    if lo.numel() != output_dim:
      raise ValueError(f"std_range[0] has {lo.numel()} entries for {output_dim} outputs")
    self.register_buffer("std_min", lo)
    self.std_min_filled_on_load = False
    self.std_range = [float(lo.min()), float(std_range[1])]
    self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]
    self.entropy_samples = int(entropy_samples)
    self.telemetry = UTelemetry(output_dim, every=telemetry_every) if telemetry_every > 0 else None
    self._normal: Normal | None = None
    Normal.set_default_validate_args(False)

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    # ``std_min`` is a configuration constant (the sigma floor, v10d), not a
    # trained tensor.  Checkpoints written before it existed (v10b, v10c)
    # carry the same head without it; loading one keeps the floor the task
    # config built and records that it did, instead of failing a strict load
    # on a buffer nothing ever learned.  The action convention and its spec
    # hash are not involved: that check runs before any weight is loaded.
    key = prefix + "std_min"
    if key not in state_dict:
      state_dict[key] = self.std_min.detach().clone()
      self.std_min_filled_on_load = True
    super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs)

  def _std(self) -> torch.Tensor:
    if self.std_type == "scalar":
      return torch.maximum(self.std_param, self.std_min).clamp(max=self.std_range[1])
    return torch.maximum(torch.exp(self.log_std_param.clamp(max=self.log_std_range[1])), self.std_min)

  def update(self, mlp_output: torch.Tensor) -> None:
    self._normal = Normal(mlp_output, self._std())

  # -- outputs: u, always ------------------------------------------------------

  def sample(self) -> torch.Tensor:
    u = self._normal.sample()  # type: ignore[union-attr]
    if self.telemetry is not None:
      self.telemetry.observe(self._normal.mean, self._normal.stddev, u)  # type: ignore[union-attr]
    return u

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
