"""The structural actuator mismatch Phase RA-Sim-0 hides in the target domain.

Everything in :mod:`piper_push.perturb` is a *scalar*: a transport delay in
whole steps, a constant fraction of each commanded step the servo completes, a
symmetric rate deadband, a damping multiplier.  Those are the axes a
calibration can search over, and Phases WM1-A and WM1-B measured what happens
when the mismatch really is one of them -- the answer was that identification
is easy and adaptation buys little.

This module is the other case.  It is a mismatch that **no setting of any
existing axis can express**, built from two effects a geared position servo
actually has:

**Backlash (direction-reversal hysteresis), per joint and asymmetric.**  A
gear train has play.  Driving the output up, the input leads by one flank
width; reversing, the command must cross the whole backlash band before the
output moves at all.  In steady motion this is a constant offset -- which
*is* expressible, as an encoder bias -- but the transient at every reversal is
not, and a pick-and-place policy reverses constantly.  The two flanks are not
the same width on a real reducer and are not the same across joints.

**Command-magnitude-dependent response time.**  A servo asked for a small step
completes most of it inside one 20 ms control period; asked for a large one it
runs into its current limit and completes proportionally less.  ``perturb``'s
``joint_response_scale`` is exactly this fraction held *constant*, so the
magnitude dependence is the part that is out of reach.

The constants below were fixed before any result run and are not tuned.  They
are quoted against the hardware they stand for: 0.006-0.022 rad is 0.34-1.26
degrees of play, which is a low-cost planetary reducer rather than a harmonic
drive; ``BETA0 = 0.85`` with ``KAPPA`` of 1.6 means a 0.04 rad step completes
33% of the way in one control period against 85% for a step near zero.

Nothing in the training or evaluation path constructs this.  It is applied by
``apply_hidden_plant`` and by nothing else, and the state it carries
(``_flank``, ``_lag``) is available to *oracle* evaluations only -- never to
the residual learner, which sees what a robot could measure and no more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Per-joint half-widths of the backlash band, radians, joint1..joint6.
# Asymmetric, and which flank is wider changes with the joint: a global
# constant cannot be right about more than one of them at a time.
BACKLASH_UP: tuple[float, ...] = (0.012, 0.018, 0.015, 0.008, 0.006, 0.010)
BACKLASH_DOWN: tuple[float, ...] = (0.020, 0.009, 0.022, 0.005, 0.011, 0.006)

# Fraction of the remaining command a servo completes in one control period,
# at zero step size.
BETA0: float = 0.85
# How fast that fraction falls with the size of the step asked for.  Larger on
# the three proximal joints, which carry the arm's mass and reach their
# current limit sooner.
KAPPA: tuple[float, ...] = (1.6, 1.6, 1.6, 0.9, 0.9, 0.9)
# The step size at which KAPPA is quoted, radians.  0.04 rad in 20 ms is
# 2 rad/s, mid-range for this arm's command limits.
S_REF: float = 0.04


@dataclass
class HiddenPlantCfg:
  """One frozen structural mismatch.  Inert only if never installed."""

  backlash_up: tuple[float, ...] = BACKLASH_UP
  backlash_down: tuple[float, ...] = BACKLASH_DOWN
  beta0: float = BETA0
  kappa: tuple[float, ...] = KAPPA
  s_ref: float = S_REF
  name: str = "backlash+rate_lag"

  def build(self, action_term) -> "HiddenPlant":
    return HiddenPlant(self, action_term)

  def to_json(self) -> dict:
    return {
      "name": self.name,
      "backlash_up": list(self.backlash_up),
      "backlash_down": list(self.backlash_down),
      "beta0": self.beta0,
      "kappa": list(self.kappa),
      "s_ref": self.s_ref,
    }


class HiddenPlant:
  """``u -> y``: current-limited first-order lag, then gear backlash.

  Two per-environment states, both reset at the episode boundary:

  ``_lag``    the servo's own commanded position, which chases ``u`` at a rate
              that falls with how far it has to go;
  ``_flank``  which side of the backlash band the gear teeth are resting on,
              carried as the output position itself.

  Batched over environments and joints; no Python loop over environments.
  """

  def __init__(self, cfg: HiddenPlantCfg, action_term) -> None:
    dev = action_term.device
    n = action_term._num_targets
    for name, seq in (("backlash_up", cfg.backlash_up),
                      ("backlash_down", cfg.backlash_down),
                      ("kappa", cfg.kappa)):
      if len(seq) != n:
        raise ValueError(
          f"{name} has {len(seq)} entries for {n} controlled joints")
      if any(float(v) < 0.0 for v in seq):
        raise ValueError(f"{name} must be non-negative, got {seq}")
    if not 0.0 < cfg.beta0 <= 1.0:
      raise ValueError(f"beta0 must be in (0, 1], got {cfg.beta0}")
    if cfg.s_ref <= 0.0:
      raise ValueError(f"s_ref must be positive, got {cfg.s_ref}")
    self.cfg = cfg
    self._term = action_term
    self._up = torch.tensor(cfg.backlash_up, device=dev).unsqueeze(0)
    self._dn = torch.tensor(cfg.backlash_down, device=dev).unsqueeze(0)
    self._kappa = torch.tensor(cfg.kappa, device=dev).unsqueeze(0)
    self._beta0 = float(cfg.beta0)
    self._s_ref = float(cfg.s_ref)
    # Persistent, and copied into rather than rebound.  A rollout runs under
    # inference mode; a state rebound to a tensor made inside one cannot be
    # written by the reset that follows it outside one.  actions.py carries
    # the same note for the same reason.
    self._lag = action_term._default.clone()
    self._flank = action_term._default.clone()

  def __call__(self, target: torch.Tensor, action_term) -> torch.Tensor:
    del action_term
    # 1. Current-limited lag.  beta falls with the size of the step demanded,
    #    so a large move starts slower than a small one -- the part a constant
    #    response scale cannot say.
    gap = target - self._lag
    beta = self._beta0 / (1.0 + self._kappa * gap.abs() / self._s_ref)
    lag = self._lag + beta * gap
    # 2. Backlash.  The output holds until the input has crossed the band,
    #    then follows one flank width behind, and which width depends on the
    #    direction of travel.
    out = torch.minimum(torch.maximum(self._flank, lag - self._up),
                        lag + self._dn)
    self._lag.copy_(lag)
    self._flank.copy_(out)
    return out

  def reset(self, env_ids=None) -> None:
    """Both states go to where the reset actually left the arm.

    ``RateLimitedJointPositionAction.reset`` has already set
    ``_previous_target`` to the fresh posture by the time this runs, so the
    teeth start resting at the joint's own position and the lag state points
    at it.  Seeding from the default posture instead would hand the first
    step of every episode a command step the size of the reset offset.
    """
    if env_ids is None:
      env_ids = slice(None)
    rest = self._term._previous_target[env_ids]
    self._lag[env_ids] = rest
    self._flank[env_ids] = rest

  @property
  def state(self) -> tuple[torch.Tensor, torch.Tensor]:
    """The hidden variables, for ORACLE evaluation only.

    Reading these into anything the residual learner trains on would be
    reading the answer.  Phase RA-Sim-0 forbids it; the only consumer is
    ``scripts/ra_sim0_eval.py``'s oracle arm, which installs the target
    itself rather than inspecting it.
    """
    return self._lag, self._flank


def apply_hidden_plant(env_cfg, cfg: HiddenPlantCfg | None) -> dict:
  """Install the structural target on the arm's command path.

  The gripper is left alone.  Its command path is a different one -- a slide
  with its own rate limit -- and adding a second, differently-shaped mismatch
  would make the phase's question two questions.
  """
  if cfg is None:
    return {}
  arm = env_cfg.actions["arm"]
  arm.command_hooks = tuple(arm.command_hooks) + (cfg,)
  return cfg.to_json()


def add_hidden_target_args(parser) -> None:
  parser.add_argument(
    "--hidden-target", action="store_true",
    help="install Phase RA-Sim-0's structural actuator mismatch (backlash "
         "plus a current-limited lag).  Off by default; nothing in the "
         "training or evaluation path sets it except an explicit oracle or "
         "target-domain run.")


def hidden_from_args(args) -> HiddenPlantCfg | None:
  return HiddenPlantCfg() if getattr(args, "hidden_target", False) else None
