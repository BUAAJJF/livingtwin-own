"""A deployable estimate of how close the arm is to tripping the safety shell.

    C_obs(history) -> P(the shell fires within the next H control steps)

Phase WM0 measured a mismatch whose entire cost is in the tail: at
`servo_damping_scale = 0.75` throughput falls 4.5% and safety-shell trips
multiply by 89. Any calibration score built on throughput, on reward, or on
how well a simulator reproduces a state trajectory reads that domain as nearly
harmless. This head exists so that a score can read it as what it is.

**What it may see.** Only :data:`piper_push.wm_data.DEPLOYABLE` channels --
the encoder latent, the recurrent state, joint states, the commanded action and
the gripper's servo error. Every one of those is computed on a real robot from
its own sensors and its own policy weights.

**What it may not see, ever.** The true damping, the privileged critic, reward,
success, or the trip labels *in the target domain*. It is *trained* on
simulator trip labels, which the phase specification allows and which is the
only way a rare-event head can be fitted at all; the discipline that makes that
sound is that training happens offline in simulation, and the head is frozen
before a single target session is read.

**Why the score is a difference and not a level.** `C_obs` on the target
session alone says how dangerous the session was, which is a fact about the
session and not about which simulator matches it. The score compares the head's
output on the *observed* next latent against its output on the latent the
dynamics model predicts under a candidate domain:

    S_risk(theta) = | C_obs(... z_real) - C_obs(... z_pred(theta)) |

so a candidate is penalised for predicting a future the policy would find
differently dangerous than the one that happened. That is the same shape as
`S_action`, with the safety head in place of the actor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from piper_push import wm_data

HORIZON = 10
"""Control steps ahead the head predicts over: 0.2 s at 50 Hz.

Chosen by measuring, not by argument.  Sweeping history length in
{25, 50, 100} against horizon in {2, 5, 10, 25} and scoring the best simple
velocity feature *within* the target domain -- so the answer is about which
windows trip rather than which domain they came from -- gives:

    horizon  2 steps   AUC 0.79-0.83, AP 0.008-0.021  (base 0.0024-0.0056)
    horizon  5 steps   AUC 0.63-0.67, AP 0.012-0.025
    horizon 10 steps   AUC 0.60-0.66, AP 0.026-0.035
    horizon 25 steps   AUC 0.57-0.60, AP 0.042-0.058  (base 0.028-0.031)

Two steps is 40 ms and is not prediction -- the joint is already at the
threshold -- so it is detection wearing a horizon.  Twenty-five is what the
phase specification suggests and is barely above chance.  Ten is where the
ratio of average precision to base rate is highest with a lead time still
worth acting on.  None of these is a strong signal, and the report says so
rather than picking the flattering pooled number."""


class RiskHead(nn.Module):
  """``C_obs``: history in, one probability out."""

  CHANNELS = ("enc", "proprio", "action", "servo")

  def __init__(self, dims: dict[str, int], hidden: int = 96) -> None:
    super().__init__()
    wm_data.assert_deployable(self.CHANNELS)
    self.dims = dims
    self.hidden_size = hidden
    in_dim = sum(dims[k] for k in self.CHANNELS)
    self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU())
    self.gru = nn.GRU(hidden, hidden)
    self.head = nn.Linear(hidden, 1)

  def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
    """``b[k]`` is ``(T, B, C)``.  Returns a logit per window, shape ``(B,)``."""
    x = torch.cat([b[k] for k in self.CHANNELS], dim=-1)
    out, _ = self.gru(self.inp(x))
    return self.head(out[-1]).squeeze(-1)

  @torch.no_grad()
  def probability(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sigmoid(self(b))

  def state(self) -> dict:
    return {"state_dict": self.state_dict(), "dims": self.dims,
            "hidden": self.hidden_size, "horizon": HORIZON}

  @classmethod
  def load(cls, d: dict, map_location="cpu") -> "RiskHead":
    m = cls(d["dims"], hidden=d["hidden"])
    m.load_state_dict(d["state_dict"])
    return m.eval().to(map_location)


def labels_within_horizon(trip: torch.Tensor, starts, length: int,
                          horizon: int = HORIZON) -> torch.Tensor:
    """Did the shell fire within ``horizon`` steps of each window's end?

    ``trip`` is ``(T,)`` for one session.  A window ``[t0, t0+length)`` is
    labelled by what happens in ``(t0+length-1, t0+length-1+horizon]`` -- the
    future, strictly after everything the head is shown.  Off the end of the
    session the label is whatever is available, which is the honest thing: a
    window whose future is not recorded cannot be a positive.
    """
    n = int(trip.shape[0])
    out = torch.zeros(len(starts), dtype=torch.float32)
    for i, t0 in enumerate(starts):
      lo = min(t0 + length, n)
      hi = min(lo + horizon, n)
      if hi > lo:
        out[i] = float(bool(trip[lo:hi].any()))
    return out


# ---------------------------------------------------------------------------
# Metrics for a rare-event classifier
# ---------------------------------------------------------------------------


def average_precision(prob: torch.Tensor, label: torch.Tensor) -> float:
  """Area under the precision-recall curve.

  Not ROC-AUC: with a base rate in the low percent, ROC-AUC is dominated by
  how the head orders the overwhelming majority of negatives and looks
  impressive for a head that never finds a positive.
  """
  if label.sum() == 0:
    return float("nan")
  order = torch.argsort(prob, descending=True)
  y = label[order]
  tp = torch.cumsum(y, 0)
  precision = tp / torch.arange(1, len(y) + 1, dtype=torch.float32)
  return float((precision * y).sum() / y.sum())


def roc_auc(prob: torch.Tensor, label: torch.Tensor) -> float:
  pos, neg = prob[label > 0.5], prob[label <= 0.5]
  if len(pos) == 0 or len(neg) == 0:
    return float("nan")
  order = torch.argsort(prob)
  ranks = torch.empty_like(order, dtype=torch.float64)
  ranks[order] = torch.arange(1, len(prob) + 1, dtype=torch.float64)
  return float((ranks[label > 0.5].sum() - len(pos) * (len(pos) + 1) / 2)
               / (len(pos) * len(neg)))


def brier(prob: torch.Tensor, label: torch.Tensor) -> float:
  return float(((prob - label) ** 2).mean())


def calibration(prob: torch.Tensor, label: torch.Tensor, bins: int = 10
                ) -> list[dict]:
  """Predicted against observed frequency, per decile of predicted risk."""
  out = []
  for i in range(bins):
    lo, hi = i / bins, (i + 1) / bins
    sel = (prob > lo) & (prob <= hi) if i else (prob <= hi)
    n = int(sel.sum())
    if not n:
      continue
    out.append({"bin": [lo, hi], "n": n,
                "predicted": float(prob[sel].mean()),
                "observed": float(label[sel].mean())})
  return out
