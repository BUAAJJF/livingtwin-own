"""Observation latency as a domain parameter, on mjlab's own delay buffer.

``ObservationTermCfg.delay_min_lag`` / ``delay_max_lag`` build a
:class:`~mjlab.utils.buffers.DelayBuffer` and serve the term's output from
``t - lag``.  What is added here is *which* lag each environment gets: mjlab
samples uniformly on ``[min_lag, max_lag]``, the robust task wants an
arbitrary categorical (``HEAVY_DR_PROFILE["timing"]["observation_latency_probs"]``).
So the buffer is built with ``delay_hold_prob=1.0`` -- never resample -- and
:class:`LatencyScene` writes the lags itself, once per episode, before the
manager's delay stage runs on that same step's output.

The lag is a property of the *plant*, so it is constant within an episode and
redrawn at the episode boundary.  One control step is 20 ms at the task's
50 Hz, which is the only place that conversion is written down.

The point-cloud tasks do not use this term: ``piper_push.pc.cloud`` owns its
own capture cadence and latency ring, reading the same probability vector.
"""

from __future__ import annotations

import dataclasses
import math

import torch

CONTROL_HZ = 50.0
STEP_MS = 1000.0 / CONTROL_HZ

LAGS: tuple[int, ...] = (0, 1, 2, 3, 4)
"""The candidate set, in control steps.  Four is 80 ms, past which the policy
was measured as unusable rather than degraded."""


@dataclasses.dataclass(frozen=True)
class LatencyPrior:
  """A categorical distribution over :data:`LAGS`."""

  probs: tuple[float, ...]

  def __post_init__(self) -> None:
    if len(self.probs) != len(LAGS):
      raise ValueError(f"expected {len(LAGS)} probabilities, got {len(self.probs)}")
    if any(p < 0.0 for p in self.probs):
      raise ValueError(f"negative probability in {self.probs}")
    total = sum(self.probs)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
      raise ValueError(f"probabilities sum to {total}, not 1")

  @classmethod
  def point(cls, lag: int) -> "LatencyPrior":
    if lag not in LAGS:
      raise ValueError(f"{lag} is not one of {LAGS}")
    return cls(tuple(1.0 if v == lag else 0.0 for v in LAGS))

  @classmethod
  def uniform(cls) -> "LatencyPrior":
    return cls(tuple(1.0 / len(LAGS) for _ in LAGS))

  @property
  def argmax(self) -> int:
    return LAGS[max(range(len(LAGS)), key=lambda i: self.probs[i])]

  @property
  def entropy_bits(self) -> float:
    return float(-sum(p * math.log2(p) for p in self.probs if p > 0.0))

  @property
  def mean(self) -> float:
    return float(sum(p * v for p, v in zip(self.probs, LAGS)))

  @property
  def mean_lag(self) -> float:
    return self.mean

  @property
  def support(self) -> tuple[int, ...]:
    return tuple(v for v, p in zip(LAGS, self.probs) if p > 0.0)

  @property
  def max_lag(self) -> int:
    return int(max(self.support))

  def mass(self, lag: int) -> float:
    return float(self.probs[LAGS.index(lag)])

  def is_point_at(self, lag: int, tol: float = 1e-9) -> bool:
    return abs(self.mass(lag) - 1.0) <= tol

  def sample(self, n: int, generator: torch.Generator | None = None,
             device="cpu") -> torch.Tensor:
    w = torch.tensor(self.probs, dtype=torch.float64)
    idx = torch.multinomial(w, n, replacement=True, generator=generator)
    return torch.tensor(LAGS)[idx].to(device)

  def to_json(self) -> dict:
    return {
      "lags": list(LAGS),
      "unit": "control steps",
      "probs": list(self.probs),
      "argmax": self.argmax,
      "mean": self.mean,
      "mean_lag_steps": self.mean,
      "mean_lag_ms": self.mean * STEP_MS,
    }


P_SOURCE = LatencyPrior.point(0)
"""No delay: the nominal task's camera."""


# ---------------------------------------------------------------------------
# The observation term
# ---------------------------------------------------------------------------


class LatencyScene:
  """``pick_mdp.CameraScene`` plus the per-environment lag assignment.

  Where the lag write lands in the step: the observation manager runs
  ``func -> noise -> clip -> scale -> nan-check -> delay``, so a lag written
  inside ``__call__`` is the lag the delay stage uses for this very step's
  output.  The manager zeroes the buffer's lags on reset, which is exactly why
  they are rewritten every step rather than only at the episode boundary.
  """

  def __init__(self, cfg, env) -> None:
    from piper_push.tasks.pick_place import mdp as pick_mdp

    self._inner = pick_mdp.CameraScene(cfg, env)
    probs = tuple(float(x) for x in cfg.params["latency_probs"])
    self._prior = LatencyPrior(probs)
    self._env = env
    self._gen = torch.Generator().manual_seed(
      int(cfg.params.get("latency_seed", 0)) ^ 0x1A7E)
    self._lags = self._prior.sample(env.num_envs, self._gen, "cpu").to(env.device)
    self._buffer = None

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
    # The camera term holds a sensor (frozen noise field, per-surface quality
    # drawn at reset); swallowing the reset would freeze one environment's
    # fixed-pattern error for the whole run.
    inner_reset = getattr(self._inner, "reset", None)
    if callable(inner_reset):
      inner_reset(env_ids)
    if env_ids is None:
      idx = slice(None)
      n = self._env.num_envs
    else:
      idx = env_ids
      n = len(env_ids)
    if n:
      self._lags[idx] = self._prior.sample(n, self._gen, "cpu").to(self._lags.device)

  @property
  def lags(self) -> torch.Tensor:
    """Per-environment lag in control steps.  Recorded as the simulator label;
    never an input to inference."""
    return self._lags


def apply_latency_prior(env_cfg, prior: LatencyPrior, seed: int = 0) -> dict:
  """Install ``prior`` on the task's camera observation term.

  Returns a provenance dict.  A point mass at zero leaves the config untouched
  and returns ``{}`` -- the nominal path stays byte-identical.
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
