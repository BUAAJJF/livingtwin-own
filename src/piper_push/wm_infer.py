"""Infer the simulator's observation delay from one reward-free session.

Five estimators, all reading the same channels and all given the same data
budget.  They are listed in increasing order of machinery, and the point of
running the cheap ones is that the expensive one has to beat them:

``B0``  prior.  No target data at all: the answer is the source domain.  The
        floor every other method is measured against, and the policy that ships
        if calibration is not worth it.

``B1a`` analytic cross-correlation between the commanded action and the joint
        response.  This is the textbook latency estimator and it is included
        *because* it should fail here: an observation delay does not move the
        actuator's response to a command, so the command-to-response lag is
        zero in every candidate domain.  Reporting it flat is a result about
        which mismatches this classical tool covers, not a bug.

``B1b`` analytic cross-correlation between the *image* encoding and
        proprioception.  The one that does apply: the depth image contains the
        arm, so a delayed image is a picture of where the arm was ``theta``
        steps ago, and a ridge fit from the image encoding to past joint
        positions is cheapest at the true lag.  No simulator, no training, no
        world model -- this is the baseline the rest of the pipeline has to
        justify itself against.

``B2``  a small recurrent classifier from the raw history to ``q(theta)``.
        Amortised system identification: all the simulator knowledge is spent
        offline, and inference is one forward pass.

``B3/B4/DA`` scores from the parameter-conditioned dynamics model.  ``B3`` uses
        proprioception and servo error only -- classical trajectory matching
        with a learned model.  ``B4`` uses the actor latent and the action the
        frozen policy would have taken given the predicted latent.  ``DA`` is
        the weighted combination, with weights and temperature fitted on
        simulation validation domains and never on the target.

Every function here takes only :data:`piper_push.wm_data.DEPLOYABLE` channels.
The label is not an argument to any of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from piper_push import latency, wm_data, wm_model

CONTROL_HZ = 50.0


# ---------------------------------------------------------------------------
# A session, cut to a budget
# ---------------------------------------------------------------------------


@dataclass
class SessionView:
  """One environment's contiguous log, truncated to ``arm_seconds``.

  The label is deliberately not a field.  Scoring compares a returned
  posterior against the truth held by the caller, which keeps the estimator
  and the answer in different scopes.
  """

  enc: torch.Tensor        # (T, D)
  hidden: torch.Tensor     # (T, H)
  proprio: torch.Tensor    # (T, P)
  action: torch.Tensor     # (T, A)
  servo: torch.Tensor      # (T, 1)
  done: torch.Tensor       # (T,)
  arm_seconds: float

  @classmethod
  def from_session(cls, s: wm_data.SessionSet, env: int, arm_seconds: float,
                   device=None) -> "SessionView":
    n = min(int(round(arm_seconds * CONTROL_HZ)), s.steps)

    def take(x):
      v = x[:n, env]
      if v.dtype == torch.half:
        v = v.float()
      return v.to(device) if device is not None else v

    return cls(take(s.enc), take(s.hidden), take(s.proprio), take(s.action),
               take(s.servo), s.done[:n, env], arm_seconds)

  @property
  def steps(self) -> int:
    return int(self.enc.shape[0])

  def windows(self, length: int) -> list[int]:
    """Non-overlapping start indices that do not cross an episode boundary.

    Non-overlapping because the scores are summed as log-likelihoods, and
    overlapping windows would count the same control step several times and
    sharpen the posterior without adding evidence.
    """
    out, t0 = [], 0
    while t0 + length <= self.steps:
      if not bool(self.done[t0:t0 + length - 1].any()):
        out.append(t0)
      t0 += length
    return out

  def stack(self, starts: list[int], length: int) -> dict[str, torch.Tensor]:
    ts = torch.tensor(starts, device=self.enc.device)
    ts = ts.unsqueeze(0) + torch.arange(length, device=ts.device).unsqueeze(1)
    return {k: getattr(self, k)[ts] for k in
            ("enc", "hidden", "proprio", "action", "servo")}


# ---------------------------------------------------------------------------
# B1a: command -> joint response
# ---------------------------------------------------------------------------


def xcorr_action_joint(v: SessionView, max_lag: int = 6) -> list[float]:
  """Correlation between the commanded step and the joint's response, by shift.

  Returned as a cost (negative correlation) over shifts ``0..max_lag`` so that
  it can be read the same way as every other score in this module.
  """
  a = v.action[:, :6]
  dq = v.proprio[1:, :6] - v.proprio[:-1, :6]
  out = []
  for lag in range(max_lag + 1):
    if lag >= len(dq):
      out.append(0.0)
      continue
    x = a[: len(dq) - lag]
    y = dq[lag:]
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    num = (x * y).sum(0)
    den = x.norm(dim=0) * y.norm(dim=0)
    out.append(float(-(num / den.clamp(min=1e-9)).mean()))
  return out


# ---------------------------------------------------------------------------
# B1b: image encoding -> past proprioception
# ---------------------------------------------------------------------------


def _ridge(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
  x = torch.cat([x, x.new_ones(len(x), 1)], 1)
  a = x.T @ x + lam * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
  return torch.linalg.solve(a, x.T @ y)


def xcorr_latent_proprio(v: SessionView, enc_split: int,
                         candidates=latency.LAGS, lam: float = 10.0,
                         fit_frac: float = 0.6) -> list[float]:
  """Held-out residual of a ridge fit from the image encoding to past joints.

  ``enc_split`` is where the encoder latent stops being the one-dimensional
  observation and starts being the image: the first part already *contains*
  proprioception, so including it would let every candidate predict the target
  perfectly and the comparison would measure nothing.

  The fit and the score use disjoint halves of the same session, and every
  candidate is fitted on the same time range, so a longer shift cannot win by
  having fewer or easier samples.
  """
  img = v.enc[:, enc_split:].double()
  q = v.proprio[:, :6].double()
  hi = max(candidates)
  n = len(img)
  if n <= hi + 20:
    return [0.0] * len(candidates)
  # One index range for every candidate.
  t = torch.arange(hi, n)
  cut = hi + int(fit_frac * (n - hi))
  fit = t[t < cut]
  test = t[t >= cut]
  if len(fit) < 10 or len(test) < 10:
    return [0.0] * len(candidates)
  out = []
  for c in candidates:
    w = _ridge(img[fit], q[fit - c], lam)
    pred = torch.cat([img[test], img.new_ones(len(test), 1)], 1) @ w
    out.append(float(((pred - q[test - c]) ** 2).mean()))
  return out


# ---------------------------------------------------------------------------
# B2: a recurrent classifier over the raw history
# ---------------------------------------------------------------------------


class HistoryClassifier(nn.Module):
  """``q(theta | history)`` in one forward pass.

  Deliberately small.  The question is whether the signature is there, not
  whether a large network can be made to find it, and a model with more
  parameters than a session has control steps would answer a different
  question.
  """

  CHANNELS = ("enc", "proprio", "action", "servo")

  def __init__(self, dims: dict[str, int], n_theta: int, hidden: int = 96):
    super().__init__()
    wm_data.assert_deployable(self.CHANNELS)
    self.dims = dims
    in_dim = sum(dims[k] for k in self.CHANNELS)
    self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU())
    self.gru = nn.GRU(hidden, hidden)
    self.head = nn.Linear(hidden, n_theta)

  def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
    """``b[k]`` is ``(T, B, C)``.  Returns per-window logits ``(B, n_theta)``."""
    x = torch.cat([b[k] for k in self.CHANNELS], dim=-1)
    out, _ = self.gru(self.inp(x))
    return self.head(out[-1])

  def state(self) -> dict:
    return {"state_dict": self.state_dict(), "dims": self.dims,
            "n_theta": self.head.out_features,
            "hidden": self.gru.hidden_size}

  @classmethod
  def load(cls, d: dict, map_location="cpu") -> "HistoryClassifier":
    m = cls(d["dims"], d["n_theta"], hidden=d["hidden"])
    m.load_state_dict(d["state_dict"])
    return m.eval().to(map_location)


class DoneOnlyClassifier(nn.Module):
  """The leakage control: episode boundaries and nothing else.

  Reset cadence is a real consequence of the domain -- a policy that fails more
  often restarts more often -- and it is visible to a robot.  It is still not
  evidence about the *plant*, so it is excluded from every other feature set
  and measured here instead, so that the report can say how much of the
  identification a method could have got without looking at the arm at all.
  """

  def __init__(self, n_theta: int, hidden: int = 32):
    super().__init__()
    self.gru = nn.GRU(1, hidden)
    self.head = nn.Linear(hidden, n_theta)

  def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
    out, _ = self.gru(b["done"].float().unsqueeze(-1))
    return self.head(out[-1])


# ---------------------------------------------------------------------------
# B3 / B4 / DA: scores from the dynamics model
# ---------------------------------------------------------------------------


@torch.no_grad()
def model_scores(ens: wm_model.Ensemble, head: wm_model.ActorHead | None,
                 b: dict[str, torch.Tensor], burn_in: int, horizon: int,
                 candidates=latency.LAGS) -> dict[str, torch.Tensor]:
  """Per-candidate ``S_state``, ``S_latent``, ``S_action`` and the ensemble's
  own disagreement, averaged over the windows in ``b``.

  ``b[k]`` is ``(burn_in + horizon + 1, B, C)``.  Everything is in the
  normalisation the model was fitted with, so the three are on comparable
  scales before any weighting.
  """
  norms = ens.norms
  z = norms["z"](b["enc"])
  p = norms["p"](b["proprio"])
  a = norms["a"](b["action"])
  e = norms["e"](b["servo"])
  lo, hi = burn_in, burn_in + horizon
  nb = z.shape[1]

  out = {k: torch.zeros(len(candidates), device=z.device)
         for k in ("state", "latent", "action", "epistemic")}
  for ci, c in enumerate(candidates):
    theta = torch.full((nb,), c, dtype=torch.long, device=z.device)
    z_mus = []
    for m in ens.members:
      _, h = m.teacher_forced(z[:lo], p[:lo], a[:lo], theta, None)
      (z_mu, z_lv, dp, p_lv, e_mu, e_lv), _ = m.teacher_forced(
        z[lo:hi], p[lo:hi], a[lo:hi], theta, h)
      out["state"][ci] += (
        wm_model.gaussian_nll(p[lo + 1:hi + 1] - p[lo:hi], dp, p_lv).sum(-1).mean()
        + wm_model.gaussian_nll(e[lo + 1:hi + 1], e_mu, e_lv).sum(-1).mean())
      out["latent"][ci] += wm_model.gaussian_nll(
        z[lo + 1:hi + 1], z_mu, z_lv).sum(-1).mean()
      z_mus.append(z_mu)
      if head is not None:
        # The action the frozen policy would have taken had the next
        # observation been the predicted one, against the action it actually
        # took.  The real branch needs no forward pass: it is in the log.
        pred_enc = (z_mu * norms["z"].std + norms["z"].mean)
        t_sel = torch.arange(lo + 1, hi + 1, device=z.device)
        hid = b["hidden"][t_sel].reshape(1, -1, b["hidden"].shape[-1])
        act, _ = head(pred_enc.reshape(-1, pred_enc.shape[-1]), hid)
        real = b["action"][t_sel].reshape(-1, b["action"].shape[-1])
        out["action"][ci] += ((act - real) ** 2).sum(-1).mean()
    # Disagreement between members about the same prediction.
    out["epistemic"][ci] = torch.stack(z_mus).var(0).mean() if len(z_mus) > 1 \
      else torch.zeros((), device=z.device)
  n = len(ens.members)
  for k in ("state", "latent", "action"):
    out[k] /= n
  return out


# ---------------------------------------------------------------------------
# Posteriors and metrics
# ---------------------------------------------------------------------------


def posterior(scores, temperature: float,
              prior: latency.LatencyPrior | None = None) -> latency.LatencyPrior:
  """``q(theta) ~ p(theta) exp(-S(theta) / T)``, with the scores centred first.

  Centring changes nothing about the result and everything about whether it
  can be computed: session scores are sums over hundreds of windows, so the
  raw exponent overflows long before the ratio does.
  """
  s = torch.as_tensor(scores, dtype=torch.float64).flatten()
  return latency.LatencyPrior.from_scores(s - s.min(), temperature, prior)


def brier(q: latency.LatencyPrior, truth: int) -> float:
  onehot = [1.0 if v == truth else 0.0 for v in latency.LAGS]
  return float(sum((a - b) ** 2 for a, b in zip(q.probs, onehot)))


def nll(q: latency.LatencyPrior, truth: int, floor: float = 1e-12) -> float:
  import math
  return float(-math.log(max(q.mass(truth), floor)))


def ece(qs: list[latency.LatencyPrior], truths: list[int],
        bins: int = 10) -> float:
  """Expected calibration error of the top-1 confidence."""
  conf = [max(q.probs) for q in qs]
  correct = [1.0 if q.argmax == t else 0.0 for q, t in zip(qs, truths)]
  total, n = 0.0, len(qs)
  if n == 0:
    return float("nan")
  for i in range(bins):
    lo, hi = i / bins, (i + 1) / bins
    sel = [j for j in range(n) if (conf[j] > lo or i == 0) and conf[j] <= hi]
    if not sel:
      continue
    acc = sum(correct[j] for j in sel) / len(sel)
    avg = sum(conf[j] for j in sel) / len(sel)
    total += len(sel) / n * abs(acc - avg)
  return float(total)


def balanced_accuracy(pred: list[int], truth: list[int]) -> float:
  """Per-class recall, averaged.  With unequal session counts per domain a
  plain accuracy rewards guessing whichever domain has the most sessions."""
  recalls = []
  for c in latency.LAGS:
    sel = [i for i, t in enumerate(truth) if t == c]
    if sel:
      recalls.append(sum(1 for i in sel if pred[i] == c) / len(sel))
  return float(sum(recalls) / max(len(recalls), 1))


def confusion(pred: list[int], truth: list[int]) -> list[list[int]]:
  m = [[0] * len(latency.LAGS) for _ in latency.LAGS]
  for p, t in zip(pred, truth):
    m[latency.LAGS.index(t)][latency.LAGS.index(p)] += 1
  return m


def fit_temperature(score_rows: list[list[float]], truths: list[int],
                    grid=None, prior=None) -> float:
  """The temperature that minimises posterior NLL on the given (non-target)
  sessions.  A grid rather than a gradient because the objective is cheap, one
  dimensional, and not convex in general."""
  if grid is None:
    grid = [10.0 ** k for k in torch.arange(-3, 5.01, 0.25).tolist()]
  best, best_t = float("inf"), grid[0]
  for t in grid:
    total = sum(nll(posterior(s, t, prior), y)
                for s, y in zip(score_rows, truths))
    if total < best:
      best, best_t = total, t
  return float(best_t)
