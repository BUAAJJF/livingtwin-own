"""A small parameter-conditioned model of what the plant does next.

Not a world model in the pixel sense: nothing here reconstructs a 224x168
depth image.  The policy's own encoder already turns the image into a 100-wide
latent, and that latent is what the policy's decision is a function of, so it
is what has to be predicted.

    F_psi(z_t, p_t, a_t, theta, h_t) -> N(z_{t+1}), N(p_{t+1} - p_t), N(e_{t+1})

with ``theta`` the observation delay in control steps.  Three heads because
the three feed three different scores: proprioception and servo error give
``S_state``, which any trajectory-matching method could compute; the latent
gives ``S_latent``; and pushing the predicted latent through the frozen actor
gives ``S_action``, which is the only one that asks what the *decision* would
have been.

Design choices worth stating, because each of them would be a bug if made the
other way:

* the heads predict a *residual* from the current value, so an untrained model
  is the identity rather than zero, and the loss starts from "nothing moves"
  rather than "everything is at the origin";
* log-variances are learned and clamped, because a homoscedastic model scores
  every candidate domain by squared error in the units that happen to be
  largest, which for this data is the latent and not the plant;
* normalisation statistics come from the training split alone.  Fitting them
  on the target session would be a channel through which the target's
  identity reaches the estimator without anybody deciding that it should;
* the ensemble members differ in initialisation *and* in which windows they
  see, so their disagreement is epistemic rather than a seed artefact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

LOGVAR_MIN, LOGVAR_MAX = -10.0, 4.0


# ---------------------------------------------------------------------------
# The frozen actor, from the checkpoint alone
# ---------------------------------------------------------------------------


class ActorHead(nn.Module):
  """The deployed policy from the encoder latent onward.

  Rebuilt from ``actor_state_dict`` rather than by constructing the task's
  model, so that posterior inference needs no simulator, no environment and no
  GPU -- which is the point: this is the part that would run on the robot.

  ``tests/test_wm_model.py`` checks it against
  :func:`piper_push.wm_data.actor_head` on the real model class.
  """

  def __init__(self, gru: nn.GRU, mlp: nn.Sequential,
               obs_dim_1d: int = 0) -> None:
    super().__init__()
    self.gru = gru
    self.mlp = mlp
    self.obs_dim_1d = obs_dim_1d
    """Where the encoder latent stops being the one-dimensional observation and
    starts being the image.  ``_encode`` concatenates the *normalised* 1-D
    observation group first and the convolutional encoding after it, so
    ``enc[..., obs_dim_1d:]`` is what the camera contributed and
    ``enc[..., :obs_dim_1d]`` is proprioception the robot already had.  The
    split matters for B1b: a fit from the whole latent to joint positions is
    trivially perfect at every candidate lag, because the answer is in the
    first 36 columns."""

  @classmethod
  def from_checkpoint(cls, path, map_location="cpu") -> "ActorHead":
    sd = torch.load(path, map_location=map_location,
                    weights_only=False)["actor_state_dict"]
    layers = sum(1 for k in sd if k.startswith("rnn.rnn.weight_ih_l"))
    if layers == 0:
      raise ValueError(f"{path} has no recurrent layer; not a vision policy")
    in_dim = sd["rnn.rnn.weight_ih_l0"].shape[1]
    hidden = sd["rnn.rnn.weight_hh_l0"].shape[1]
    gru = nn.GRU(in_dim, hidden, layers)
    gru.load_state_dict({k[len("rnn.rnn."):]: v for k, v in sd.items()
                         if k.startswith("rnn.rnn.")})

    # mlp.0, mlp.2, mlp.4 ... are the linear layers; the odd indices are the
    # activations, which carry no parameters and so are not in the state dict.
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
    mods: list[nn.Module] = []
    for j, i in enumerate(idx):
      w = sd[f"mlp.{i}.weight"]
      lin = nn.Linear(w.shape[1], w.shape[0])
      lin.weight.data.copy_(w)
      lin.bias.data.copy_(sd[f"mlp.{i}.bias"])
      mods.append(lin)
      if j < len(idx) - 1:
        mods.append(nn.ELU())
    mean = sd.get("obs_normalizer._mean")
    head = cls(gru, nn.Sequential(*mods),
               obs_dim_1d=int(mean.shape[-1]) if mean is not None else 0)
    # GaussianDistribution's deterministic output is the identity on the MLP
    # output -- the mean.  std_param is not used at inference.
    return head.eval()

  @torch.no_grad()
  def forward(self, enc: torch.Tensor, hidden: torch.Tensor | None = None):
    """``(action, next_hidden)`` for a batch of latents."""
    out, h = self.gru(enc.unsqueeze(0), hidden)
    return self.mlp(out.squeeze(0)), h


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


@dataclass
class Norm:
  mean: torch.Tensor
  std: torch.Tensor

  @classmethod
  def fit(cls, x: torch.Tensor, eps: float = 1e-4) -> "Norm":
    flat = x.reshape(-1, x.shape[-1]).float()
    return cls(flat.mean(0), flat.std(0).clamp(min=eps))

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / self.std

  def to(self, device) -> "Norm":
    return Norm(self.mean.to(device), self.std.to(device))

  def state(self) -> dict:
    return {"mean": self.mean, "std": self.std}

  @classmethod
  def load(cls, d: dict) -> "Norm":
    return cls(d["mean"], d["std"])


# ---------------------------------------------------------------------------
# The dynamics model
# ---------------------------------------------------------------------------


def gaussian_nll(x: torch.Tensor, mu: torch.Tensor,
                 logvar: torch.Tensor) -> torch.Tensor:
    """Per-element negative log likelihood, without the 2*pi constant.

    The constant is dropped because every use of this is a *comparison*
    between candidate domains over the same number of elements, and carrying
    it would only make the printed numbers less readable.  It is added back in
    :func:`nll_constant` wherever an absolute NLL is reported.
    """
    return 0.5 * ((x - mu) ** 2 * torch.exp(-logvar) + logvar)


def nll_constant(dim: int) -> float:
  return 0.5 * dim * math.log(2.0 * math.pi)


class LatentDynamics(nn.Module):
  """One ensemble member."""

  def __init__(self, z_dim: int, p_dim: int, a_dim: int, n_theta: int,
               hidden: int = 192, theta_dim: int = 8) -> None:
    super().__init__()
    self.z_dim, self.p_dim, self.a_dim, self.n_theta = z_dim, p_dim, a_dim, n_theta
    self.theta_emb = nn.Embedding(n_theta, theta_dim)
    self.inp = nn.Sequential(
      nn.Linear(z_dim + p_dim + a_dim + theta_dim, hidden), nn.ELU(),
      nn.Linear(hidden, hidden), nn.ELU())
    self.gru = nn.GRU(hidden, hidden)
    self.z_head = nn.Linear(hidden, 2 * z_dim)
    self.p_head = nn.Linear(hidden, 2 * p_dim)
    self.e_head = nn.Linear(hidden, 2)

  def _split(self, raw: torch.Tensor, dim: int):
    mu, logvar = raw[..., :dim], raw[..., dim:]
    return mu, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

  def step(self, z: torch.Tensor, p: torch.Tensor, a: torch.Tensor,
           theta: torch.Tensor, h: torch.Tensor | None):
    """One step.  All inputs already normalised.  Shapes ``(B, ...)``."""
    x = torch.cat([z, p, a, self.theta_emb(theta)], dim=-1)
    out, h = self.gru(self.inp(x).unsqueeze(0), h)
    f = out.squeeze(0)
    dz, z_logvar = self._split(self.z_head(f), self.z_dim)
    dp, p_logvar = self._split(self.p_head(f), self.p_dim)
    e_mu, e_logvar = self._split(self.e_head(f), 1)
    # Residual: the identity is the prior, not the origin.
    return (z + dz, z_logvar, dp, p_logvar, e_mu, e_logvar), h

  def teacher_forced(self, z: torch.Tensor, p: torch.Tensor, a: torch.Tensor,
                     theta: torch.Tensor, h: torch.Tensor | None = None):
    """Run over a whole window with the real inputs.  Shapes ``(T, B, ...)``."""
    outs = []
    for t in range(z.shape[0]):
      out, h = self.step(z[t], p[t], a[t], theta, h)
      outs.append(out)
    return [torch.stack([o[i] for o in outs]) for i in range(6)], h

  def open_loop(self, z0: torch.Tensor, p0: torch.Tensor, a: torch.Tensor,
                theta: torch.Tensor, h: torch.Tensor | None = None):
    """Feed the model's own predictions back for ``a.shape[0]`` steps.

    The actions stay the real ones: they are in the log, so a method that
    needs them needs nothing a robot does not have.  What is *not* given back
    is the state, which is the whole point -- a model that is only ever asked
    for one step can be right by copying its input.
    """
    z, p = z0, p0
    outs = []
    for t in range(a.shape[0]):
      out, h = self.step(z, p, a[t], theta, h)
      outs.append(out)
      z = out[0]
      p = p + out[2]
    return [torch.stack([o[i] for o in outs]) for i in range(6)], h


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class Ensemble(nn.Module):
  """Members disagree where the data did not say; that disagreement is the
  epistemic part of the uncertainty and is reported rather than folded in."""

  def __init__(self, members: list[LatentDynamics], norms: dict[str, Norm]) -> None:
    super().__init__()
    self.members = nn.ModuleList(members)
    self.norms = norms

  def to_device(self, device) -> "Ensemble":
    self.to(device)
    self.norms = {k: v.to(device) for k, v in self.norms.items()}
    return self

  def save(self, path) -> None:
    m0 = self.members[0]
    torch.save({
      "state_dicts": [m.state_dict() for m in self.members],
      "dims": {"z": m0.z_dim, "p": m0.p_dim, "a": m0.a_dim,
               "n_theta": m0.n_theta,
               "hidden": m0.gru.hidden_size,
               "theta_dim": m0.theta_emb.embedding_dim},
      "norms": {k: v.state() for k, v in self.norms.items()},
    }, path)

  @classmethod
  def load(cls, path, map_location="cpu") -> "Ensemble":
    d = torch.load(path, map_location=map_location, weights_only=False)
    dims = d["dims"]
    members = []
    for sd in d["state_dicts"]:
      m = LatentDynamics(dims["z"], dims["p"], dims["a"], dims["n_theta"],
                         hidden=dims["hidden"], theta_dim=dims["theta_dim"])
      m.load_state_dict(sd)
      members.append(m.eval())
    return cls(members, {k: Norm.load(v) for k, v in d["norms"].items()})
