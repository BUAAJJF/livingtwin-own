"""The target's visibility as a process, not a rate.

The model this replaces was an independent per-frame coin flip, and every
property below is one it would fail.  That matters because the failure is
silent: both models produce a mask that is sometimes absent, both report a
plausible marginal, and only a recurrent policy trained on them can tell the
difference -- a day later, on hardware.

The numbers quoted are from ``scripts/measure_target_gaps.py`` over 20
recorded sessions, 26010 control steps.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from piper_push.target_process import TargetProcess, TargetProcessCfg


def _runs(mask: np.ndarray) -> np.ndarray:
  """Lengths of the True runs, DROPPING any that touch an end.

  A trace that stops mid-gap leaves a truncated run, and counting it is how
  this file first "found" a one-step gap under an eight-step confirmation --
  a censoring artefact, not a bug in the process.  Right-censored runs are
  shorter than the thing they sample and belong in neither tail.
  """
  d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
  start, stop = np.flatnonzero(d > 0), np.flatnonzero(d < 0)
  keep = (start > 0) & (stop < mask.size)
  return (stop - start)[keep]


def _trace(cfg, held: bool, steps=4000, n=512, seed=0):
  torch.manual_seed(seed)
  tp = TargetProcess(cfg, n, "cpu")
  h = torch.full((n,), bool(held), dtype=torch.bool)
  return np.stack([tp.step(h).clone().numpy() for _ in range(steps)])


def test_the_gaps_are_long_which_is_the_whole_point():
  """A median gap near the rig's 40 steps, not the coin flip's 9.

  The IID model at the same marginal has geometric gaps: median
  ``ln(2)/p_recover`` with ``p_recover = 1 - marginal``, which for anything
  like the measured 0.37 is under two steps.  This asserts the property that
  separates them rather than a fitted constant.
  """
  v = _trace(TargetProcessCfg(), held=True)
  gaps = np.concatenate([_runs(~v[:, b]) for b in range(0, v.shape[1], 4)])
  assert gaps.size > 200, "not enough gaps to say anything"
  assert np.median(gaps) > 15, f"median gap {np.median(gaps)} is IID-short"
  assert gaps.max() > 200, "no long tail: a stall of seconds must be possible"


def test_visibility_is_correlated_step_to_step():
  """Consecutive steps mostly agree.  Under IID they would not.

  For an independent process the chance of a change between two steps is
  ``2 p (1 - p)``, about 0.47 at a marginal near a half.  A correlated one
  changes on the order of its transition rates, which are a few percent.
  """
  v = _trace(TargetProcessCfg(), held=True)
  change = np.abs(np.diff(v.astype(np.int8), axis=0)).mean()
  assert change < 0.10, f"per-step change {change:.3f} looks independent"


def test_the_carry_and_the_approach_are_different_processes():
  """Different rates, because the rig measures them differently.

  Jaws closed the target is present in 37.4% of steps with a median gap of
  40; jaws open, 31.2% with a median gap of 14.  One process with one constant
  cannot be both.
  """
  cfg = TargetProcessCfg(visible_held=(0.10, 0.15),
                         visible_approach=(0.85, 0.90))
  held = _trace(cfg, held=True, steps=2000).mean()
  appr = _trace(cfg, held=False, steps=2000).mean()
  assert held < 0.35 < appr, f"held {held:.2f} approach {appr:.2f} not separated"


def test_the_marginal_lands_inside_the_configured_range():
  """The range is the measured session spread, so the mean must sit in it."""
  cfg = TargetProcessCfg(visible_held=(0.30, 0.50), mean_gap_held=(40.0, 60.0))
  m = _trace(cfg, held=True, steps=6000).mean()
  assert 0.25 < m < 0.55, f"marginal {m:.3f} outside the drawn range"


def test_a_recovery_waits_for_the_tracker_to_confirm():
  """``TargetTracker`` needs an instance to survive confirmation frames.

  With a long confirmation nothing can come back quickly, so no gap may be
  shorter than it.  A model that let the mask flicker back for one frame would
  be describing a tracker the deployment does not have.
  """
  cfg = TargetProcessCfg(visible_held=(0.5, 0.5), mean_gap_held=(10.0, 10.0),
                         confirm_frames=8)
  v = _trace(cfg, held=True, steps=3000)
  gaps = np.concatenate([_runs(~v[:, b]) for b in range(0, v.shape[1], 4)])
  assert gaps.size > 100
  assert gaps.min() >= 8, f"a gap of {gaps.min()} skipped confirmation"


def test_reset_redraws_the_rates_per_episode():
  """Per episode, not per step.

  Drawing every step would restore the independence this exists to remove,
  one level up: the marginal would be right and the process would again be
  memoryless.
  """
  torch.manual_seed(1)
  cfg = TargetProcessCfg(visible_held=(0.05, 0.95), mean_gap_held=(10.0, 200.0))
  tp = TargetProcess(cfg, 256, "cpu")
  before = tp._p_lose_held.clone()
  tp.reset()
  after = tp._p_lose_held
  assert not torch.allclose(before, after), "reset did not redraw"
  # And a partial reset leaves the others alone.
  keep = tp._p_lose_held.clone()
  tp.reset(torch.arange(8))
  assert torch.allclose(keep[8:], tp._p_lose_held[8:]), "reset hit the wrong envs"


def test_every_environment_starts_able_to_see_its_target():
  """An episode that opens blind charges the policy for an unavoidable miss."""
  torch.manual_seed(2)
  tp = TargetProcess(TargetProcessCfg(), 128, "cpu")
  assert bool(tp.visible.all())


def test_the_rates_reproduce_a_requested_marginal_and_gap():
  """The closed form the fit relies on, checked against the simulation.

  ``mean_visible = v / (1 - v) * mean_gap`` is the only algebra in the module;
  if it is wrong every number downstream is wrong by the same factor.
  """
  v = torch.tensor([0.374])
  g = torch.tensor([71.0])
  p_lose, p_recover = TargetProcess._rates(v, g)
  mean_visible = 1.0 / float(p_lose)
  assert float(1.0 / p_recover) == pytest.approx(71.0)
  assert mean_visible / (mean_visible + 71.0) == pytest.approx(0.374, abs=1e-3)
