"""Count data gets a count interval, and the approximations have to be checked.

Phase WM0 reported a nominal safety-trip rate of 2.29 per arm-hour with a
standard deviation over three repeats.  That rate comes from seventeen events
in 6.8 arm-hours, and a standard deviation of three Poisson draws is mostly
Poisson noise being reported as a measurement.  Re-running the identical
command in this phase gave 3.15/h -- a 37% "regression" that a rate-ratio test
puts at p = 0.13.  G3 is a threshold on this quantity, so the arithmetic under
it is worth pinning.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from analyze_wm1 import (chi2_quantile, hierarchical_bootstrap, holm,  # noqa: E402
                         paired_diff, poisson_rate, rate_ratio, t_interval)


# -- the approximations -----------------------------------------------------


@pytest.mark.parametrize("p,k,want", [
  (0.975, 20, 34.170), (0.025, 20, 9.591),
  (0.975, 28, 44.461), (0.025, 28, 15.308),
  (0.975, 100, 129.561), (0.025, 100, 74.222),
])
def test_chi2_quantile_matches_the_table(p, k, want):
  assert chi2_quantile(p, k) == pytest.approx(want, rel=3e-3)


@pytest.mark.parametrize("p,k,want", [
  (0.975, 2, 7.378), (0.025, 2, 0.0506), (0.975, 4, 11.143),
])
def test_chi2_quantile_is_exact_at_small_even_df(p, k, want):
  """Wilson-Hilferty is off by a factor of two at k = 2, which is exactly the
  interval on a condition that never tripped."""
  assert chi2_quantile(p, k) == pytest.approx(want, rel=1e-3)


# -- rates ------------------------------------------------------------------


def test_pooled_rate_is_events_over_exposure():
  r = poisson_rate([17, 14, 16], [6.8267] * 3)
  assert r["events"] == 47
  assert r["rate"] == pytest.approx(47 / (3 * 6.8267), rel=1e-9)
  assert r["ci_poisson"][0] < r["rate"] < r["ci_poisson"][1]


def test_the_wm0_and_wm1_nominal_trip_rates_are_not_distinguishable():
  """The check this file exists for.

  Identical command, identical checkpoint, different day: 47 events in 20.5
  arm-hours against 43 in 13.7.  Read as three-repeat means those are 2.29 and
  3.15 and look like a regression; read as counts they are one rate.
  """
  wm0 = poisson_rate([17, 14, 16], [6.8267] * 3)
  wm1 = poisson_rate([20, 23], [6.8267] * 2)
  rr = rate_ratio(wm1, wm0)
  assert rr["ratio"] == pytest.approx(1.37, abs=0.02)
  assert rr["ci"][0] < 1.0 < rr["ci"][1]
  assert rr["p"] > 0.05


def test_overdispersion_is_detected_not_absorbed():
  """Repeats that disagree by more than Poisson allows must widen the
  interval, not be averaged into a confident wrong one."""
  tight = poisson_rate([20, 20, 20], [1.0] * 3)
  wild = poisson_rate([2, 20, 60], [1.0] * 3)
  assert tight["dispersion"] < 2.0
  assert wild["dispersion"] > 10.0
  assert (wild["ci_quasi"][1] - wild["ci_quasi"][0]) > \
         (wild["ci_poisson"][1] - wild["ci_poisson"][0])
  # A well-behaved set is not widened.
  assert tight["ci_quasi"] == pytest.approx(tight["ci_poisson"], rel=0.3)


def test_a_rate_of_zero_does_not_divide_by_zero():
  r = poisson_rate([0, 0], [5.0, 5.0])
  assert r["rate"] == 0.0
  assert r["ci_poisson"][0] == 0.0


# -- throughput -------------------------------------------------------------


def _repeat(n_env, per_env, secs=60.0):
  return ([float(per_env)] * n_env, [secs] * n_env)


def test_bootstrap_recovers_the_pooled_ratio():
  reps = [_repeat(64, 10), _repeat(64, 12), _repeat(64, 11)]
  b = hierarchical_bootstrap(reps, draws=500)
  assert b["mean"] == pytest.approx(11.0, rel=1e-6)
  assert b["ci"][0] < b["mean"] < b["ci"][1]


def test_bootstrap_interval_is_not_a_point():
  """A hand-rolled resampler whose low bits cycle produces [x, x] and reads as
  certainty.  That happened once; it raises now."""
  reps = [_repeat(32, 8), _repeat(32, 14), _repeat(32, 11)]
  b = hierarchical_bootstrap(reps, draws=800)
  assert b["ci"][1] - b["ci"][0] > 0.1


def test_bootstrap_carries_the_between_repeat_spread():
  """Repeats that disagree must widen the interval even when every
  environment inside a repeat is identical."""
  same = hierarchical_bootstrap([_repeat(64, 11)] * 3, draws=800)
  spread = hierarchical_bootstrap(
    [_repeat(64, 6), _repeat(64, 11), _repeat(64, 16)], draws=800)
  assert (spread["ci"][1] - spread["ci"][0]) > (same["ci"][1] - same["ci"][0])


def test_throughput_is_a_pooled_ratio_not_a_mean_of_ratios():
  """An environment that lived four seconds must not count as much as one
  that ran the whole rollout."""
  reps = [(([10.0, 1.0]), ([60.0, 4.0]))]
  b = hierarchical_bootstrap(reps, draws=200)
  assert b["mean"] == pytest.approx(11.0 / 64.0 * 60.0, rel=1e-6)
  # The mean of the two per-environment rates would be (10/60 + 1/4)/2 * 60
  # = 12.5, which weights a environment that lived four seconds as much as one
  # that ran the whole rollout.
  assert b["mean"] != pytest.approx(12.5, rel=1e-3)


# -- comparisons ------------------------------------------------------------


def test_paired_difference_uses_the_pairing():
  a = [50.0, 52.0, 48.0]
  b = [45.0, 47.0, 43.0]
  d = paired_diff(a, b)
  assert d["diff"] == pytest.approx(5.0)
  assert d["ci"][0] > 0.0     # the shared run-to-run term cancelled


def test_holm_is_monotone_and_bounded():
  adj = holm({"a": 0.001, "b": 0.02, "c": 0.4})
  assert adj["a"] <= adj["b"] <= adj["c"] <= 1.0
  assert adj["a"] == pytest.approx(0.003)


def test_t_interval_on_one_sample_is_not_a_claim():
  s = t_interval([42.0])
  assert s["n"] == 1
  assert s["ci"][0] != s["ci"][0]     # NaN
