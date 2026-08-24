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
