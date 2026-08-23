"""Observation latency as a domain parameter, on mjlab's own delay buffer.

Phase WM0 measured this axis through a ring buffer inside the camera
observation term, and only afterwards found that mjlab already ships the same
thing: ``ObservationTermCfg.delay_min_lag`` / ``delay_max_lag`` build a
:class:`~mjlab.utils.buffers.DelayBuffer` and serve the term's output from
``t - lag``.  WM1 uses the shipped one.  ``perturb.py`` now routes
``obs_latency_steps`` there too, so there is exactly one implementation in the
tree; ``tests/test_latency.py`` pins the two to the same output sequence and
``results/wm1_latency/equivalence/`` carries the empirical re-measurement.

What is added on top of it is *which* lag each environment gets.  mjlab samples
uniformly on ``[min_lag, max_lag]``; posterior-guided adaptation needs an
arbitrary categorical, because the adaptation distribution is

    p_adapt(theta) = alpha * q(theta | D_target) + (1 - alpha) * p_source(theta)

and ``q`` is whatever the inference returned.  So the buffer is built with
``delay_hold_prob=1.0`` -- "keep the previous lag with probability 1", i.e.
never resample -- and :class:`LatencyScene` writes the lags itself, once per
episode, before the manager's delay stage runs on that same step's output.

The lag is a property of the *plant*, so it is constant within an episode and
redrawn at the episode boundary.  One control step is 20 ms at the task's
50 Hz, which is the only place that conversion is written down.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

CONTROL_HZ = 50.0
STEP_MS = 1000.0 / CONTROL_HZ

LAGS: tuple[int, ...] = (0, 1, 2, 3, 4)
"""The candidate set.  Zero is the domain the deployed policy was trained in;
four is 80 ms, past which the WM0 sweep measured the policy as unusable rather
than degraded (`docs/sim2real_sweep_phase_wm0.md`, section 5)."""

TARGET_LAG = 3
"""The hidden target for WM1-A: 60 ms.  Named here only so that scripts which
must *not* see it can be checked against a single symbol; the inference path
never imports it."""


@dataclass(frozen=True)
class LatencyPrior:
  """A categorical distribution over :data:`LAGS`, in control steps."""

  probs: tuple[float, ...]

  def __post_init__(self) -> None:
    if len(self.probs) != len(LAGS):
      raise ValueError(f"expected {len(LAGS)} probabilities, got {len(self.probs)}")
    if any(p < 0.0 for p in self.probs):
      raise ValueError(f"negative probability in {self.probs}")
    total = sum(self.probs)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
      raise ValueError(f"probabilities sum to {total}, not 1")

  # -- constructors ---------------------------------------------------------

  @classmethod
  def point(cls, lag: int) -> "LatencyPrior":
    if lag not in LAGS:
      raise ValueError(f"{lag} is not one of {LAGS}")
    return cls(tuple(1.0 if v == lag else 0.0 for v in LAGS))

  @classmethod
  def uniform(cls) -> "LatencyPrior":
    return cls(tuple(1.0 / len(LAGS) for _ in LAGS))

  @classmethod
  def from_logits(cls, logits, temperature: float = 1.0) -> "LatencyPrior":
    t = torch.as_tensor(logits, dtype=torch.float64).flatten()
    p = torch.softmax(t / temperature, dim=0)
    return cls(tuple(float(x) for x in p))

  @classmethod
  def from_scores(cls, scores, temperature: float = 1.0,
                  prior: "LatencyPrior | None" = None) -> "LatencyPrior":
    """``q(theta) ~ p(theta) * exp(-S(theta) / temperature)``.

    Scores are costs, so the sign is flipped relative to
    :meth:`from_logits`.  The base measure defaults to uniform, which is the
    honest one for inference: a base measure peaked at the source domain would
    make the method look better exactly where it should look worse.
    """
    s = torch.as_tensor(scores, dtype=torch.float64).flatten()
    logp = -s / max(temperature, 1e-12)
    if prior is not None:
      logp = logp + torch.log(torch.as_tensor(prior.probs, dtype=torch.float64)
                              .clamp_min(1e-300))
    return cls(tuple(float(x) for x in torch.softmax(logp, dim=0)))

  # -- combination ----------------------------------------------------------

  def mix(self, other: "LatencyPrior", alpha: float) -> "LatencyPrior":
    """``alpha * self + (1 - alpha) * other``."""
    if not 0.0 <= alpha <= 1.0:
      raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return LatencyPrior(tuple(alpha * a + (1.0 - alpha) * b
                              for a, b in zip(self.probs, other.probs)))

  # -- summaries ------------------------------------------------------------

  @property
  def argmax(self) -> int:
    return LAGS[max(range(len(LAGS)), key=lambda i: self.probs[i])]

  @property
  def entropy_bits(self) -> float:
    return float(-sum(p * math.log2(p) for p in self.probs if p > 0.0))

  @property
  def mean_lag(self) -> float:
    return float(sum(p * v for p, v in zip(self.probs, LAGS)))

  def mass(self, lag: int) -> float:
    return float(self.probs[LAGS.index(lag)])

  def is_point_at(self, lag: int, tol: float = 1e-9) -> bool:
    return abs(self.mass(lag) - 1.0) <= tol

  def total_variation(self, other: "LatencyPrior") -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(self.probs, other.probs))

  def fingerprint(self, places: int = 3) -> tuple[float, ...]:
    """Rounded probabilities, for de-duplicating adaptation runs.

    Two methods whose adaptation distributions round to the same vector would
    produce the *same* PPO run, not merely a similar one.  Training both and
    reporting them as independent evidence would be inventing agreement.
    """
    return tuple(round(p, places) for p in self.probs)

  # -- use ------------------------------------------------------------------

  def sample(self, n: int, generator: torch.Generator | None = None,
             device: str | torch.device = "cpu") -> torch.Tensor:
    w = torch.tensor(self.probs, dtype=torch.float64)
    idx = torch.multinomial(w, n, replacement=True, generator=generator)
    return torch.tensor(LAGS, dtype=torch.long)[idx].to(device)

  @property
  def support(self) -> tuple[int, ...]:
    return tuple(v for v, p in zip(LAGS, self.probs) if p > 0.0)

  @property
  def max_lag(self) -> int:
    return max(self.support)

  def to_json(self) -> dict:
    return {
      "lags": list(LAGS),
      "probs": list(self.probs),
      "argmax": self.argmax,
      "mean_lag_steps": self.mean_lag,
      "mean_lag_ms": self.mean_lag * STEP_MS,
      "entropy_bits": self.entropy_bits,
    }


P_SOURCE = LatencyPrior.point(0)
"""The distribution the deployed policy was actually trained under.

Written as ``delta(0)`` because that is what it is: every rollout that produced
``f3/model_1500`` ran with no observation delay.  Calling it a broad latency
prior would make the retention half of the gate meaningless -- there would be
nothing to retain."""


# ---------------------------------------------------------------------------
# The observation term
# ---------------------------------------------------------------------------


class LatencyScene:
  """The camera term, plus the per-environment lag assignment.

  Subclasses the WM0 perturbation term so that the depth axes still compose;
  with no depth mismatch configured that parent is a straight pass-through to
  ``pick_mdp.camera_scene``.

  Where the lag write lands in the step: the observation manager runs
  ``func -> noise -> clip -> scale -> nan-check -> delay``, so a lag written
  inside ``__call__`` is the lag the delay stage uses for this very step's
  output.  The manager zeroes the buffer's lags on reset, which is exactly why
  they are rewritten every step rather than only at the episode boundary.
  """

  def __init__(self, cfg, env) -> None:
    from piper_push.perturb import PerturbedCameraScene

    self._inner = PerturbedCameraScene(cfg, env)
    probs = tuple(float(x) for x in cfg.params["latency_probs"])
    self._prior = LatencyPrior(probs)
    self._env = env
    self._gen = torch.Generator().manual_seed(
      int(cfg.params.get("latency_seed", 0)) ^ 0x1A7E)
    self._lags = self._prior.sample(env.num_envs, self._gen, "cpu").to(env.device)
    self._buffer = None

  # -- lag plumbing ---------------------------------------------------------

  def _find_buffer(self):
    """The DelayBuffer the manager built for *this* term.

    Located by identity rather than by name so that renaming the group or the
    term cannot silently detach the lags from the buffer they steer.  Returns
    None during ``_prepare_terms``, which calls every term once to measure its
    output shape before any delay buffer exists.
    """
    om = getattr(self._env, "observation_manager", None)
    if om is None:
      return None
    buffers = getattr(om, "_group_obs_term_delay_buffer", None)
    if not buffers:
      return None
    for group, cfgs in om._group_obs_term_cfgs.items():
      for name, cfg in zip(om._group_obs_term_names[group], cfgs):
        if cfg.func is self:
          return buffers.get(group, {}).get(name)
    return None

  def __call__(self, env, *args, **kwargs) -> torch.Tensor:
    # The manager hands every entry of ``params`` to the func as a keyword, so
    # the two this class added have to come back off before the camera term,
    # which knows nothing about them, is called.
    kwargs.pop("latency_probs", None)
    kwargs.pop("latency_seed", None)
    obs = self._inner(env, *args, **kwargs)
    if self._buffer is None:
      self._buffer = self._find_buffer()
    if self._buffer is not None:
      self._buffer.set_lags(self._lags)
    return obs

  def reset(self, env_ids=None) -> None:
    self._inner.reset(env_ids)
    if env_ids is None:
      idx = slice(None)
      n = self._env.num_envs
    else:
      idx = env_ids
      n = len(env_ids)
    if n:
      self._lags[idx] = self._prior.sample(n, self._gen, "cpu").to(self._lags.device)

  # -- for instrumentation --------------------------------------------------

  @property
  def lags(self) -> torch.Tensor:
    """Per-environment lag in control steps.  Recorded as the simulator label;
    never an input to inference."""
    return self._lags


def apply_latency_prior(env_cfg, prior: LatencyPrior, seed: int = 0) -> dict:
  """Install ``prior`` on the task's camera observation term.

  Returns a provenance dict.  A point mass at zero leaves the config untouched
  and returns ``{}`` -- the nominal path stays byte-identical to the one every
  earlier result was measured on.
  """
  if prior.is_point_at(0):
    return {}
  grp = env_cfg.observations.get("camera")
  if grp is None:
    raise ValueError("observation latency requested on a task with no camera")
  term = grp.terms["scene"]
  params = dict(term.params)
  params["latency_probs"] = prior.probs
  params["latency_seed"] = seed
  grp.terms["scene"] = dataclasses.replace(
    term,
    func=LatencyScene,
    params=params,
    # min=0 rather than min=support-minimum: set_lags clamps to [min, max] and
    # a prior with mass at 0 must be able to express it.
    delay_min_lag=0,
    delay_max_lag=prior.max_lag,
    delay_per_env=True,
    delay_hold_prob=1.0,  # this term owns the lags; the buffer never resamples
    delay_update_period=0,
  )
  return {"latency_prior": prior.to_json(), "latency_seed": seed}


def add_latency_args(parser) -> None:
  parser.add_argument(
    "--latency-probs", default=None,
    help="comma-separated categorical over lags 0..4, e.g. '0,0,0,1,0'")
  parser.add_argument(
    "--latency-lag", type=int, default=None,
    help="shorthand for a point mass at this lag")
  parser.add_argument("--latency-seed", type=int, default=0)


def prior_from_args(args) -> LatencyPrior:
  if getattr(args, "latency_probs", None):
    raw = [float(x) for x in args.latency_probs.split(",")]
    total = sum(raw)
    if total <= 0:
      raise ValueError("--latency-probs sums to zero")
    return LatencyPrior(tuple(x / total for x in raw))
  if getattr(args, "latency_lag", None) is not None:
    return LatencyPrior.point(int(args.latency_lag))
  return P_SOURCE
