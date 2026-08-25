"""A differentiable stand-in for one control step of the NOMINAL simulator.

MJWarp is not differentiable, so a residual that corrects the command cannot
be trained by backpropagating through the thing it corrects.  Phase RA-Sim-0
allows one way around that: fit a small frozen surrogate to the **nominal**
simulator -- the one we own, whose every internal quantity is ours to read --
and use it only as the bridge that carries a gradient from a transition error
back to the residual's parameters.

Three rules the surrogate lives under, all of which the phase's specification
sets and none of which is negotiable:

1. It is fitted on nominal-simulator data only.  Not on target-domain data,
   and never on the target's hidden effective command.
2. It is frozen while the residual trains.  A surrogate that could move would
   let the pair explain a mismatch by mis-modelling the simulator instead of
   by correcting it.
3. Nothing it predicts is ever reported as a result.  Every number in
   ``docs/ra_sim0_results.md`` comes from a real MJWarp rollout; the surrogate
   appears in the training loop and nowhere else.

What it maps: ``(q, qdot, y_prev, y) -> (q', qdot')`` for the six arm joints,
where ``y`` is the position target handed to the servo for this control step
and ``y_prev`` the one before it -- the action term ramps between the two
across the decimated substeps, so the pair is what the servo actually sees.
Object contact is *not* an input: the surrogate is a model of the arm's
command path, which is what the residual acts on, and the phase's accuracy
gate is settled in the real simulator where contact is present.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

N_JOINTS = 6


@dataclass(frozen=True)
class SurrogateNorm:
  """Feature and label scaling, stored with the weights."""

  x_mean: torch.Tensor
  x_std: torch.Tensor
  y_mean: torch.Tensor
  y_std: torch.Tensor

  def to(self, device) -> "SurrogateNorm":
    return SurrogateNorm(self.x_mean.to(device), self.x_std.to(device),
                         self.y_mean.to(device), self.y_std.to(device))


class TransitionSurrogate(nn.Module):
  """MLP on 24 inputs predicting the *change* in the 12 arm states.

  Predicting the delta rather than the next state is what makes a randomly
  initialised surrogate merely wrong instead of catastrophic: the identity
  map is already most of a control step at 50 Hz.
  """

  def __init__(self, width: int = 256, depth: int = 2) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    d = 4 * N_JOINTS
    for _ in range(depth):
      layers += [nn.Linear(d, width), nn.SiLU()]
      d = width
    layers += [nn.Linear(d, 2 * N_JOINTS)]
    self.net = nn.Sequential(*layers)
    self.register_buffer("x_mean", torch.zeros(4 * N_JOINTS))
    self.register_buffer("x_std", torch.ones(4 * N_JOINTS))
    self.register_buffer("y_mean", torch.zeros(2 * N_JOINTS))
    self.register_buffer("y_std", torch.ones(2 * N_JOINTS))

  def set_norm(self, norm: SurrogateNorm) -> None:
    self.x_mean.copy_(norm.x_mean)
    self.x_std.copy_(norm.x_std.clamp_min(1e-8))
    self.y_mean.copy_(norm.y_mean)
    self.y_std.copy_(norm.y_std.clamp_min(1e-8))

  @staticmethod
  def features(q: torch.Tensor, qd: torch.Tensor, y_prev: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
    """``[q, qdot, y - q, y_prev - q]``, which is the same information.

    Reparameterised rather than fed raw because the servo answers the
    *error*, and at this arm's scale the error is a twentieth of the
    position: handing the network ``y`` and ``q`` separately makes the only
    quantity that matters a small difference of two large normalised inputs,
    and the fit spends its capacity recovering a subtraction.
    """
    return torch.cat([q, qd, y - q, y_prev - q], dim=-1)

  def forward(self, q: torch.Tensor, qd: torch.Tensor,
              y_prev: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = self.features(q, qd, y_prev, y)
    out = self.net((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean
    dq, dqd = out[..., :N_JOINTS], out[..., N_JOINTS:]
    return q + dq, qd + dqd


def fit_surrogate(feats: torch.Tensor, labels: torch.Tensor, *,
                  device: str = "cpu", epochs: int = 60, batch: int = 8192,
                  lr: float = 1e-3, width: int = 256,
                  val_frac: float = 0.1, seed: int = 0,
                  log=print) -> tuple[TransitionSurrogate, dict]:
  """Least squares on ``(q, qdot, y_prev, y) -> (dq, dqdot)``.

  ``feats`` is ``[N, 24]`` and ``labels`` ``[N, 12]``; the split is by row,
  which is right here and only here -- the surrogate is a function fit, not a
  claim about generalisation across sessions, and its held-out number exists
  to say the fit converged rather than to support a conclusion.
  """
  g = torch.Generator().manual_seed(seed)
  n = feats.shape[0]
  perm = torch.randperm(n, generator=g)
  n_val = max(int(n * val_frac), 1)
  vi, ti = perm[:n_val], perm[n_val:]
  model = TransitionSurrogate(width=width).to(device)
  fq, fqd, fyp, fy = feats[ti].split(N_JOINTS, dim=-1)
  xs = TransitionSurrogate.features(fq, fqd, fyp, fy)
  model.set_norm(SurrogateNorm(
    xs.mean(0), xs.std(0),
    labels[ti].mean(0), labels[ti].std(0)).to(device))
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  xt, yt = feats[ti].to(device), labels[ti].to(device)
  xv, yv = feats[vi].to(device), labels[vi].to(device)
  hist = []
  for ep in range(epochs):
    idx = torch.randperm(xt.shape[0], device=device)
    tot = 0.0
    for s in range(0, xt.shape[0], batch):
      b = idx[s:s + batch]
      x, lab = xt[b], yt[b]
      q, qd, yp, ycmd = x.split(N_JOINTS, dim=-1)
      pq, pqd = model(q, qd, yp, ycmd)
      pred = torch.cat([pq - q, pqd - qd], dim=-1)
      loss = torch.nn.functional.mse_loss(
        (pred - model.y_mean) / model.y_std, (lab - model.y_mean) / model.y_std)
      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()
      tot += float(loss) * b.numel()
    with torch.no_grad():
      q, qd, yp, ycmd = xv.split(N_JOINTS, dim=-1)
      pq, pqd = model(q, qd, yp, ycmd)
      pred = torch.cat([pq - q, pqd - qd], dim=-1)
      vl = float(torch.nn.functional.mse_loss(
        (pred - model.y_mean) / model.y_std,
        (yv - model.y_mean) / model.y_std))
    hist.append({"epoch": ep, "train": tot / max(xt.shape[0], 1), "val": vl})
    if log and (ep + 1) % 10 == 0:
      log(f"    surrogate epoch {ep + 1}/{epochs}  train {hist[-1]['train']:.3e}"
          f"  val {vl:.3e}")
  model.eval()
  for p in model.parameters():
    p.requires_grad_(False)
  # Normalised MSE of 1.0 is "predicts the mean"; anything near it means the
  # surrogate learned nothing and no gradient it carries is worth trusting.
  return model, {"history": hist, "val_nmse": hist[-1]["val"],
                 "n_train": int(xt.shape[0]), "n_val": int(xv.shape[0]),
                 "params": sum(p.numel() for p in model.parameters())}
