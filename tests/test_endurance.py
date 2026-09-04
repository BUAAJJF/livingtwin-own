"""The arithmetic behind the endurance gate.

It exists because the quantity it computes inverted a conclusion: the mean
placement rate said `strong_teacher` was "slower than v5", and the ratio said
it stops working.  A gate that decides which teacher gets distilled must not
have an off-by-one in it, and none of this needs a simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_endurance import summarise  # noqa: E402

W, B = 12, 100
DT, WIN = 0.02, 100


def flat(n=2.0):
  return np.full((W, B), n / B)


def test_a_steady_policy_scores_one():
  s = summarise(flat(), np.full(B, 0.03), WIN, DT)
  assert abs(s["late_over_early"] - 1.0) < 1e-9
  assert s["stopped_envs"] == 0 and s["survivor_fraction"] == 1.0


def test_a_policy_that_halves_scores_one_half():
  p = flat()
  p[W // 2:] *= 0.5
  s = summarise(p, np.full(B, 0.03), WIN, DT)
  assert abs(s["late_over_early"] - 0.5) < 1e-9


def test_dying_environments_and_a_slowing_one_are_told_apart():
  """The same mean, two diagnoses.

  Half the environments stopping is a different fault from all of them
  halving, and only ``survivor_placements_last`` separates them.  This is the
  distinction that identified the real failure: the survivors were placing
  *more* than at the start.
  """
  dying = np.ones((W, B)) * 0.02
  dying[W // 2:, : B // 2] = 0.0                 # half the envs stop dead
  slowing = np.ones((W, B)) * 0.02
  slowing[W // 2:] *= 0.5                        # every env halves
  d = summarise(dying, np.full(B, 0.03), WIN, DT)
  s = summarise(slowing, np.full(B, 0.03), WIN, DT)
  assert abs(d["late_over_early"] - s["late_over_early"]) < 1e-9
  assert d["stopped_envs"] == B // 2 and s["stopped_envs"] == 0
  assert abs(d["survivor_placements_last"]
             - d["survivor_placements_first"]) < 1e-9
  assert s["survivor_placements_last"] < s["survivor_placements_first"]


def test_an_environment_that_never_started_is_not_counted_as_dead():
  """It has no decay to report.  Charging it to the ratio would make a policy
  that does nothing at all look like one that stopped."""
  p = np.ones((W, B)) * 0.02
  p[:, :10] = 0.0
  s = summarise(p, np.full(B, 0.03), WIN, DT)
  assert s["started_envs"] == B - 10 and s["stopped_envs"] == 0


def test_a_policy_that_never_places_is_not_scored_one():
  """0/0 must not read as healthy.  It is the failure mode a gate on this
  number would otherwise wave through."""
  s = summarise(np.zeros((W, B)), np.full(B, 0.03), WIN, DT)
  assert np.isnan(s["late_over_early"])


def test_the_jaw_split_reports_the_stopped_environments_separately():
  """The mean jaw opening is the one number that hid the mechanism: it is a
  blend of environments that latched shut and environments working normally."""
  p = np.ones((W, B)) * 0.02
  p[W // 2:, : B // 2] = 0.0
  jaw = np.full(B, 0.030)
  jaw[: B // 2] = 0.0001
  s = summarise(p, jaw, WIN, DT)
  assert s["jaw_mm_stopped"] < 1.0 and s["jaw_mm_running"] > 25.0


def test_the_rate_is_per_minute_of_simulated_time():
  """One placement per environment per window, at 50 Hz and 100 steps, is
  30 per minute; a wrong dt here rescales every number the gate reports."""
  s = summarise(np.ones((W, B)), np.full(B, 0.03), WIN, DT)
  assert abs(s["early_per_min"] - 30.0) < 1e-6
