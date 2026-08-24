"""The WM1-B gate must not be able to declare a safety win it did not measure.

Three things this file pins.

The **clustering unit**. Counts go into the rate model summed per training
seed, never per evaluation repeat. Repeats inside a seed agree closely; seeds
do not. A gate that clusters on the repeat measures how precisely one PPO run
was observed and reports it as how precisely the method was.

The **agreement rule**. A criterion passes only when the negative binomial and
the cluster-robust sandwich both put the ratio's upper bound below one. Where
they disagree, the difference is the negative binomial's variance assumption
and not the counts, and the gate should not spend that assumption on a pass.

**Holm**, over the two pre-registered comparisons and nothing else.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
  "wm1b_gate", ROOT / "scripts" / "wm1b_gate.py")
gate = importlib.util.module_from_spec(spec)
sys.modules["wm1b_gate"] = gate
spec.loader.exec_module(gate)


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------


def test_holm_scales_the_smallest_by_the_family_size():
  adj = gate.holm({"a": 0.01, "b": 0.04})
  assert adj["a"] == pytest.approx(0.02)
  assert adj["b"] == pytest.approx(0.04)


def test_holm_is_monotone():
  """Without the running maximum a later comparison can come out more
  significant than an earlier one, which is not a correction, it is a
  reordering."""
  adj = gate.holm({"a": 0.03, "b": 0.031})
  assert adj["a"] <= adj["b"]


def test_holm_never_exceeds_one():
  adj = gate.holm({"a": 0.7, "b": 0.9})
  assert all(v <= 1.0 for v in adj.values())


def test_holm_turns_a_marginal_pair_into_no_finding():
  """0.03 and 0.04 are each 'significant' alone and neither survives two."""
  adj = gate.holm({"a": 0.03, "b": 0.04})
  assert min(adj.values()) > 0.05


# ---------------------------------------------------------------------------
# Seeds are the unit
# ---------------------------------------------------------------------------


def test_repeats_are_summed_inside_a_seed():
  d = {42: [{"trips": 10, "hours": 6.8, "throughput": 50.0},
            {"trips": 12, "hours": 6.8, "throughput": 52.0},
            {"trips": 11, "hours": 6.8, "throughput": 51.0}],
       7: [{"trips": 30, "hours": 6.8, "throughput": 49.0}]}
  sl = gate.by_seed(d)
  assert sl["seeds"] == [7, 42]
  assert sl["trips"] == [30, 33]                 # summed, not listed
  assert sl["hours"] == [6.8, pytest.approx(20.4)]
  assert sl["throughput"] == [49.0, pytest.approx(51.0)]   # averaged
  assert sl["n_eval"] == 4


def test_three_repeats_of_one_seed_is_one_observation():
  """The failure this whole design exists to stop: nine tight evaluations of
  three seeds must not resolve a rate better than three seeds do."""
  one = gate.by_seed({42: [{"trips": 10, "hours": 6.8, "throughput": 50.0}] * 9})
  assert len(one["seeds"]) == 1
  assert one["n_eval"] == 9


def test_the_throughput_interval_is_over_seeds_not_evaluations():
  tight = gate.t_interval([50.0, 50.1, 49.9])
  spread = gate.t_interval([44.0, 50.0, 56.0])
  assert tight["mean"] == pytest.approx(spread["mean"], abs=0.1)
  assert (spread["hi"] - spread["lo"]) > 10 * (tight["hi"] - tight["lo"])
  assert gate.t_interval([50.0])["n"] == 1
  assert gate.t_interval([50.0])["lo"] != gate.t_interval([50.0])["lo"]  # nan


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _arm(trips, seeds=None):
  n = len(trips)
  return {"seeds": seeds or list(range(n)), "trips": list(trips),
          "hours": [6.8] * n, "throughput": [50.0] * n, "n_eval": n}


def test_a_real_reduction_is_found_by_both_methods():
  c = gate.compare(_arm([2, 3, 1, 2, 3, 2, 1, 2]),
                   _arm([20, 25, 22, 19, 24, 21, 23, 20]),
                   "risk_aware", "trajectory_matching")
  assert c["negative_binomial"]["ci"][1] < 1.0
  assert c["cluster_robust"]["ci"][1] < 1.0
  assert max(c["p_nb"], c["p_robust"]) < 0.01


def test_one_unlucky_seed_does_not_become_a_finding():
  """Pooled totals differ two-to-one and it is one seed.  A gate that clusters
  on the evaluation would call this overwhelming."""
  c = gate.compare(_arm([3, 2, 4, 3, 2, 3, 4, 5]),
                   _arm([1, 2, 1, 48, 2, 1, 3, 2]),
                   "risk_aware", "broad_dr")
  assert max(c["p_nb"], c["p_robust"]) > 0.05


def test_the_verdict_needs_seeds_as_well_as_significance():
  assert gate.verdict(True, True, True) == "GREEN"
  assert gate.verdict(True, True, False) == "YELLOW"     # too few seeds
  assert gate.verdict(False, True, True) == "YELLOW"     # point only
  assert gate.verdict(False, False, True) == "RED"


def test_the_roles_cover_every_method_the_plans_emit():
  """A method in a plan that no role claims is silently dropped from the gate,
  which is how a comparison quietly loses an arm."""
  named = {m for names in gate.ROLES.values() for m in names}
  for m in ("B3_state", "KNOWN_PARAM", "M2_broad", "M5_risk_aware",
            "B0_prior_refit", "KNOWN_PARAM_MIX"):
    assert m in named, f"{m} is in the plans but no gate role claims it"


def test_the_seed_floor_is_the_one_the_specification_asked_for():
  assert gate.MIN_SEEDS == 8
  assert gate.THROUGHPUT_BUDGET == 0.05
