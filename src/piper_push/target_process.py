"""When the rig can see its target, as a process in time rather than a rate.

``blind_when_held`` modelled the carrying phase as an independent coin flip per
frame.  The marginal it was fitted to was real; the process was not, and for a
recurrent policy the process is most of the signal.  An 8% IID keep re-exposes
the object every ~12 frames and essentially never hides it for a second.  What
the rig does, measured from 20 recorded sessions' control logs
(``scripts/measure_target_gaps.py``, 26010 control steps):

    gap length, consecutive steps with no target
      jaws open    n= 69   mean 62.5   p50  14   p90 104   max 1168  (23.4 s)
      jaws closed  n=174   mean 71.0   p50  40   p90  85   max 1463  (29.3 s)

A median gap of 40 steps against a geometric model's 9.  So this replaces the
coin flip with the simplest process that has the right shape: a two-state
Markov chain per environment, with rates fitted to the measured mean gap and
marginal, and separate rates for the approach and the carry.

Two further departures from the old model, both from the same measurement:

* **The marginal was wrong as well as the process.**  0.08 came from one
  session, ``v4_fixedseg_try6``, which is the worst of the twenty (6.2%).
  Across all of them the target is present in 37.4% of jaws-closed steps.

* **Sessions differ enormously** -- 6% to 86% visible while held -- because
  they span a segmenter fix, two depth backends and different table layouts.
  A single number would train against a rig that does not exist, so the rates
  are drawn per episode from the observed session spread.  That range IS the
  domain randomisation; it is not a hand-set width.

What this deliberately does NOT model, and why: identity swaps, where the
tracker hands the policy a confident mask of the *wrong* object.  Those are
real and common -- 326 of them while the jaws were closed, across 142 grasps,
about 2.3 per carry -- but the robust training task has a single object on the
table, so there is no bystander to swap to.  Modelling it needs multi-object
scenes, which is a change to the task rather than to the sensor.  Left out
rather than faked, because a decoy mask with no object under it disagrees with
the depth channel and teaches the policy to detect the simulator.
"""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class TargetProcessCfg:
  """Per-episode ranges for the two-state chain, from the session spread.

  ``visible_*`` are marginal probabilities that the target is reported at all;
  ``mean_gap_*`` are the mean lengths of a no-target run, in control steps.
  Defaults are the measured quantiles over the twelve sessions with more than
  200 jaws-closed steps: held 0.06-0.86 (median 0.41), approach 0.14-0.93
  (median 0.66).  The extremes are kept -- they are sessions that happened.
  """

  visible_held: tuple[float, float] = (0.10, 0.85)
  visible_approach: tuple[float, float] = (0.15, 0.90)
  mean_gap_held: tuple[float, float] = (20.0, 120.0)
  mean_gap_approach: tuple[float, float] = (15.0, 110.0)
  confirm_frames: int = 3
  """Steps a re-acquired target stays invisible before the mask returns.

  ``mask.TargetTracker`` requires an instance to survive confirmation before it
  is eligible, so a recovery is never instantaneous on the rig.  Three is its
  configured minimum; it is not randomised because the tracker's value is
  fixed, not a property of the scene."""

  enabled: bool = True


class TargetProcess:
  """Per-environment visibility state for the target mask.

  Holds one bit and two counters per environment.  ``reset`` redraws the rates
  -- once per episode, because a session's detection quality is a property of
  its lighting, layout and backend, and redrawing it every step would be the
  IID model again one level up.
  """

  def __init__(self, cfg: TargetProcessCfg, num_envs: int, device) -> None:
    self.cfg = cfg
    self.device = device
    z = torch.zeros(num_envs, device=device)
    self.visible = torch.ones(num_envs, dtype=torch.bool, device=device)
    self.confirm = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._p_lose_held = z.clone()
    self._p_recover_held = z.clone()
    self._p_lose_appr = z.clone()
    self._p_recover_appr = z.clone()
    self.reset()

  @staticmethod
  def _rates(visible: torch.Tensor, mean_gap: torch.Tensor):
    """Chain rates from a marginal and a mean gap.

    For a two-state chain the stationary probability of being visible is
    ``mean_visible / (mean_visible + mean_gap)``, so the visible run length
    that reproduces a marginal ``v`` is ``v / (1 - v) * mean_gap``.  Both are
    clamped away from zero: a marginal of exactly 1 would divide by zero, and
    a gap under one step is not a gap.
    """
    v = visible.clamp(1e-3, 1.0 - 1e-3)
    g = mean_gap.clamp_min(1.0)
    mean_visible = (v / (1.0 - v) * g).clamp_min(1.0)
    return 1.0 / mean_visible, 1.0 / g

  def _draw(self, lo_hi, n):
    lo, hi = lo_hi
    return torch.rand(n, device=self.device) * (hi - lo) + lo

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.visible.numel(), device=self.device)
    env_ids = env_ids.to(self.device)
    n = int(env_ids.numel())
    if n == 0:
      return
    c = self.cfg
    pl, pr = self._rates(self._draw(c.visible_held, n),
                         self._draw(c.mean_gap_held, n))
    self._p_lose_held[env_ids], self._p_recover_held[env_ids] = pl, pr
    pl, pr = self._rates(self._draw(c.visible_approach, n),
                         self._draw(c.mean_gap_approach, n))
    self._p_lose_appr[env_ids], self._p_recover_appr[env_ids] = pl, pr
    # Start seeing it.  An episode that opened blind would charge the policy
    # for a failure it had no chance to avoid.
    self.visible[env_ids] = True
    self.confirm[env_ids] = 0

  def step(self, held: torch.Tensor) -> torch.Tensor:
    """Advance one control step and return the per-environment visible bit.

    ``held`` selects which pair of rates applies, so the carry and the
    approach are different processes rather than one process with a different
    constant -- which is what the measurement shows them to be.
    """
    p_lose = torch.where(held, self._p_lose_held, self._p_lose_appr)
    p_recover = torch.where(held, self._p_recover_held, self._p_recover_appr)
    u = torch.rand_like(p_lose)
    n = int(self.cfg.confirm_frames)

    lost = self.visible & (u < p_lose)

    # A recovery already in flight: tick it down, and the step it reaches zero
    # is the step the mask comes back.
    counting = (~self.visible) & (self.confirm > 0)
    self.confirm = torch.where(counting, self.confirm - 1, self.confirm)
    became = counting & (self.confirm == 0)

    # Otherwise, roll for a recovery and start the tracker's confirmation.
    idle = (~self.visible) & (self.confirm == 0) & ~became
    start = idle & (u < p_recover)
    self.confirm = torch.where(start, torch.full_like(self.confirm, n),
                               self.confirm)

    self.visible = (self.visible & ~lost) | became | (start & (n <= 0))
    # Losing it abandons any confirmation in flight, which is what the tracker
    # does: the instance stopped being confirmed.
    self.confirm = torch.where(lost, torch.zeros_like(self.confirm),
                               self.confirm)
    return self.visible
