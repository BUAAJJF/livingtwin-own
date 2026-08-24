"""A categorical distribution over one simulator parameter's candidate values.

Phase WM1-A built this for observation latency and Phase WM1-B needs the same
algebra for servo damping: the same mixing with a source prior, the same
temperature-scaled construction from a cost vector, the same fingerprint for
de-duplicating adaptation runs. Only the value set differs, so only the value
set is a subclass.

The one thing that does *not* generalise is how a draw reaches the simulator.
Observation latency is a per-environment lag written into an observation term's
delay buffer; servo damping is a per-environment field of the compiled model.
Those live in :mod:`piper_push.latency` and :mod:`piper_push.damping`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch


def risk_lambda(costs, odds: float = 9.0) -> float:
  """The pre-registered tilt strength for :meth:`CategoricalPrior.tilt`.

  ``ln(odds) / (max cost - min cost)``, with ``odds = 9``.  Fixed before any
  adaptation was run, and stated as a rule rather than a number so that it
  cannot be tuned towards the answer.

  What it means: applied to a posterior that is *undecided between* the two
  extremes of the cost range, it produces 9:1 odds in favour of the dangerous
  one.  That is a deliberate, bounded amount of pessimism -- enough that a
  method which is genuinely uncertain spends most of its PPO budget where the
  safety cost is, and little enough that a candidate the data have argued
  against does not come back.  Being a ratio of costs it is scale-free, so the
  same rule carries to an axis whose costs are measured in other units.
  """
  lo = min(float(c) for c in costs)
  hi = max(float(c) for c in costs)
  if hi <= lo:
    return 0.0
  return math.log(odds) / (hi - lo)


@dataclass(frozen=True)
class CategoricalPrior:
  """A distribution over :attr:`VALUES`, which a subclass supplies."""

  probs: tuple[float, ...]

  VALUES: ClassVar[tuple[float, ...]] = ()
  UNIT: ClassVar[str] = ""

  def __post_init__(self) -> None:
    n = len(type(self).VALUES)
    if len(self.probs) != n:
      raise ValueError(f"expected {n} probabilities, got {len(self.probs)}")
    if any(p < 0.0 for p in self.probs):
      raise ValueError(f"negative probability in {self.probs}")
    total = sum(self.probs)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
      raise ValueError(f"probabilities sum to {total}, not 1")

  # -- constructors ---------------------------------------------------------

  @classmethod
  def point(cls, value):
    if value not in cls.VALUES:
      raise ValueError(f"{value} is not one of {cls.VALUES}")
    return cls(tuple(1.0 if v == value else 0.0 for v in cls.VALUES))

  @classmethod
  def uniform(cls):
    n = len(cls.VALUES)
    return cls(tuple(1.0 / n for _ in range(n)))

  @classmethod
  def from_scores(cls, scores, temperature: float = 1.0, prior=None):
    """``q(theta) ~ p(theta) * exp(-S(theta) / temperature)``.

    Scores are costs, so the sign is flipped relative to logits.  The base
    measure defaults to uniform, which is the honest one for inference: a base
    measure peaked at the source domain would make a method look better exactly
    where it should look worse.
    """
    s = torch.as_tensor(scores, dtype=torch.float64).flatten()
    logp = -s / max(temperature, 1e-12)
    if prior is not None:
      logp = logp + torch.log(
        torch.as_tensor(prior.probs, dtype=torch.float64).clamp_min(1e-300))
    return cls(tuple(float(x) for x in torch.softmax(logp, dim=0)))

  @classmethod
  def from_logits(cls, logits, temperature: float = 1.0):
    t = torch.as_tensor(logits, dtype=torch.float64).flatten()
    return cls(tuple(float(x) for x in torch.softmax(t / temperature, dim=0)))

  # -- combination ----------------------------------------------------------

  def tilt(self, cost, lam: float):
    """Reweight by ``exp(lam * cost)`` and renormalise -- a risk-averse tilt.

    This is where WM1-B's risk-awareness lives, and it is worth being exact
    about why it lives here rather than in a learned head.

    The phase specification asks for ``C_obs(history) -> P(a safety event in
    the next H steps)``, and that head was built, trained and measured. Inside
    the target domain, with no cross-domain base-rate shortcut available, it
    scored 1.02x its base rate against a shuffled-label control at 1.16x: the
    label is not predictable from deployable observations at any lead time
    tested. So there is no honest per-step risk signal to put in a score.

    What is available without any target label at all is ``cost[i]``: how
    often the safety shell fires under candidate ``i`` *in simulation*. Tilting
    an ordinary posterior by it produces a distribution that is deliberately
    not the best estimate of the domain -- it is the estimate a planner should
    train against when being wrong towards danger is cheaper than being wrong
    towards safety. At ``lam = 0`` it is the posterior unchanged; as ``lam``
    grows it concentrates on the most dangerous candidate the data have not
    ruled out.

    The consequence, which the report has to state rather than bury: when the
    posterior is already a point mass the tilt does nothing, because there is
    no surviving candidate to move mass towards. It can only help a posterior
    that is genuinely uncertain -- which is the comparison the phase is
    actually about.

    ``cost`` is in whatever units the caller measured; ``lam`` carries the
    reciprocal unit. Both are recorded in :meth:`to_json` so a tilted prior can
    never be mistaken for a fitted one.
    """
    values = type(self).VALUES
    if len(cost) != len(values):
      raise ValueError(f"cost has {len(cost)} entries, expected {len(values)}")
    if lam < 0:
      raise ValueError("lam must be non-negative: a negative tilt would move "
                       "mass towards the safest candidate, which is the "
                       "opposite of risk aversion")
    m = max(float(c) for c in cost)
    w = [p * math.exp(lam * (float(c) - m))
         for p, c in zip(self.probs, cost)]
    total = sum(w)
    if total <= 0.0:
      # every candidate the posterior allows has been tilted into underflow;
      # returning the posterior unchanged is the only defensible answer, and
      # it is loud rather than silent.
      raise ValueError("the tilt underflowed to zero mass; lam is too large "
                       f"for costs spanning {min(cost)} to {m}")
    return type(self)(tuple(v / total for v in w))

  def mix(self, other, alpha: float):
    """``alpha * self + (1 - alpha) * other``."""
    if not 0.0 <= alpha <= 1.0:
      raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return type(self)(tuple(alpha * a + (1.0 - alpha) * b
                            for a, b in zip(self.probs, other.probs)))

  # -- summaries ------------------------------------------------------------

  @property
  def values(self) -> tuple[float, ...]:
    return type(self).VALUES

  @property
  def argmax(self):
    v = type(self).VALUES
    return v[max(range(len(v)), key=lambda i: self.probs[i])]

  @property
  def entropy_bits(self) -> float:
    return float(-sum(p * math.log2(p) for p in self.probs if p > 0.0))

  @property
  def mean(self) -> float:
    return float(sum(p * v for p, v in zip(self.probs, type(self).VALUES)))

  def mass(self, value) -> float:
    return float(self.probs[type(self).VALUES.index(value)])

  def is_point_at(self, value, tol: float = 1e-9) -> bool:
    return abs(self.mass(value) - 1.0) <= tol

  def total_variation(self, other) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(self.probs, other.probs))

  def fingerprint(self, places: int = 3) -> tuple[float, ...]:
    """Rounded probabilities, for de-duplicating adaptation runs.

    Two methods whose adaptation distributions round to the same vector produce
    the *same* PPO run, not a similar one.  Training both and reporting them as
    independent evidence would be inventing agreement.
    """
    return tuple(round(p, places) for p in self.probs)

  @property
  def support(self) -> tuple[float, ...]:
    return tuple(v for v, p in zip(type(self).VALUES, self.probs) if p > 0.0)

  # -- use ------------------------------------------------------------------

  def sample_indices(self, n: int, generator: torch.Generator | None = None
                     ) -> torch.Tensor:
    w = torch.tensor(self.probs, dtype=torch.float64)
    return torch.multinomial(w, n, replacement=True, generator=generator)

  def sample(self, n: int, generator: torch.Generator | None = None,
             device="cpu") -> torch.Tensor:
    idx = self.sample_indices(n, generator)
    return torch.tensor(type(self).VALUES)[idx].to(device)

  def to_json(self) -> dict:
    return {
      "values": list(type(self).VALUES),
      "unit": type(self).UNIT,
      "probs": list(self.probs),
      "argmax": self.argmax,
      "mean": self.mean,
      "entropy_bits": self.entropy_bits,
    }
