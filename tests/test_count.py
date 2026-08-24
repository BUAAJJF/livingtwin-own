"""The count model has to be right about seed-level variance, or it is worse
than the quasi-Poisson it replaces.

The whole reason this module exists is that WM1-A's trip interval was too
narrow: the dispersion was estimated over process repeats, which agree, and
used to speak about training seeds, which do not. So the tests that matter are
the ones that pin *how wide* an interval gets when the clusters disagree, and
the ones that check the likelihood is the negative binomial's and not something
that merely looks like it.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import math

import pytest
import torch

from piper_push import count


# ---------------------------------------------------------------------------
# The likelihood is the one it claims to be
# ---------------------------------------------------------------------------


def test_the_loglikelihood_matches_an_independent_implementation():
  """Against `torch.distributions.NegativeBinomial`, in the NB2 mapping
  ``r = 1/alpha``, ``p = alpha*mu / (1 + alpha*mu)``.

  Written out by hand, the NB2 log-likelihood is easy to get subtly wrong --
  swapping ``p`` for ``1-p`` still produces a smooth surface with a maximum,
  and it is not the maximum of the right function.
  """
  y = [0.0, 3.0, 11.0, 2.0, 40.0]
  mu = [1.5, 4.0, 9.0, 2.5, 30.0]
  for alpha in (0.01, 0.2, 1.0, 5.0):
    r = 1.0 / alpha
    # float64 throughout: at float32 the reference is only good to ~1e-6,
    # which is not a tight enough check to catch a p/(1-p) swap in one term.
    probs = torch.tensor([alpha * m / (1.0 + alpha * m) for m in mu],
                         dtype=torch.float64)
    ref = torch.distributions.NegativeBinomial(
      total_count=torch.tensor(r, dtype=torch.float64), probs=probs
    ).log_prob(torch.tensor(y, dtype=torch.float64)).sum().item()
    assert count.nb2_loglik(y, None, mu, alpha) == pytest.approx(ref, rel=1e-9)


def test_alpha_zero_is_the_poisson_limit():
  y = [0.0, 3.0, 11.0, 2.0]
  mu = [1.5, 4.0, 9.0, 2.5]
  poisson = sum(yi * math.log(m) - m - math.lgamma(yi + 1)
                for yi, m in zip(y, mu))
  assert count.nb2_loglik(y, None, mu, 0.0) == pytest.approx(poisson, rel=1e-12)
  # and approached continuously from above
  assert count.nb2_loglik(y, None, mu, 1e-7) == pytest.approx(poisson, abs=1e-3)


# ---------------------------------------------------------------------------
# The optimiser finds the maximum
# ---------------------------------------------------------------------------


def test_the_fit_beats_a_brute_force_grid():
  """A profile search plus Newton is only worth having if it lands at least as
  high as an exhaustive sweep of the same surface."""
  counts = [3, 19, 7, 25, 11, 4]
  exposure = [4.0] * 6
  fit = count.nb2_fit(counts, exposure)

  best = -1e18
  for i in range(400):
    b0 = math.log(0.2) + i * (math.log(30.0) - math.log(0.2)) / 399
    for j in range(200):
      a = 1e-4 * (10 ** (j * 5.0 / 199))
      mu = [math.exp(b0) * t for t in exposure]
      best = max(best, count.nb2_loglik(counts, None, mu, a))
  assert fit.loglik >= best - 1e-6


def test_equal_exposures_put_the_rate_at_the_mean_count():
  """With one intercept and a common exposure the NB2 mean is the sample mean
  whatever alpha is, so this isolates the beta step from the alpha search."""
  counts = [3, 19, 7, 25, 11, 4]
  t = 4.0
  fit = count.nb2_fit(counts, [t] * 6)
  assert math.exp(fit.beta[0]) == pytest.approx(sum(counts) / (6 * t), rel=1e-8)


# ---------------------------------------------------------------------------
# The point of the module: seed heterogeneity widens the interval
# ---------------------------------------------------------------------------


def test_dispersed_clusters_give_a_wider_interval_than_poisson():
  """Same total events, same total exposure; only the spread across clusters
  differs.  The negative binomial must notice."""
  agree = count.nb2_rate([15, 16, 15, 14], [4.0] * 4)
  disagree = count.nb2_rate([2, 30, 3, 25], [4.0] * 4)
  assert agree["rate"] == pytest.approx(disagree["rate"], rel=1e-9)
  assert agree["alpha"] < 1e-3
  assert disagree["alpha"] > 0.1
  w = lambda d: d["ci"][1] - d["ci"][0]
  assert w(disagree) > 2.5 * w(agree)


def test_it_widens_wm1a_past_the_gate_the_quasi_poisson_scraped():
  """The concrete regression this module was written for.

  WM1-A's three training seeds tripped at 3.22, 8.01 and 4.49 per arm-hour.
  Pooling gave 5.24 with a quasi-Poisson upper bound of 6.83 against a 5.99
  threshold -- a fail, but a narrow one, and the interval was built from a
  dispersion estimated inside seeds.  With the seed as the observation the
  upper bound has to move up, not down.
  """
  per_seed = [3.22, 8.01, 4.49]
  hours = 4.0
  r = count.nb2_rate([p * hours for p in per_seed], [hours] * 3)
  assert r["rate"] == pytest.approx(5.24, abs=0.02)
  assert r["ci"][1] > 6.83
  assert r["ci"][1] > 5.99          # still a fail, and now unambiguously


# ---------------------------------------------------------------------------
# Rate ratios
# ---------------------------------------------------------------------------


def test_a_real_difference_is_found_and_a_null_one_is_not():
  hours = [4.0] * 8
  low = [4, 6, 3, 5, 4, 7, 5, 4]
  high = [22, 25, 19, 28, 24, 21, 26, 23]
  hit = count.nb2_rate_ratio(low, hours, high, hours)
  assert hit["ratio"] == pytest.approx(sum(low) / sum(high), rel=0.05)
  assert hit["p_lrt"] < 1e-6
  assert hit["ci"][1] < 1.0

  same = count.nb2_rate_ratio(low, hours, [5, 4, 6, 4, 5, 3, 6, 5], hours)
  assert same["p_lrt"] > 0.05
  assert same["ci"][0] < 1.0 < same["ci"][1]


def test_heterogeneous_clusters_stop_a_difference_being_declared():
  """Eight clusters, a two-to-one difference in the pooled totals, and enough
  seed-to-seed spread that it could be one unlucky seed.  A Poisson model on
  the pooled counts calls this overwhelming; the seed-level model must not.
  """
  hours = [4.0] * 8
  a = [1, 2, 1, 48, 2, 1, 3, 2]
  b = [3, 2, 4, 3, 2, 3, 4, 5]
  res = count.nb2_rate_ratio(a, hours, b, hours)
  assert sum(a) > 2 * sum(b)
  assert res["alpha"] > 0.5
  assert res["p_lrt"] > 0.05
  # a naive Poisson on the same totals would have been certain of it
  naive_se = math.sqrt(1.0 / sum(a) + 1.0 / sum(b))
  assert abs(math.log(sum(a) / sum(b))) / naive_se > 3.0


def test_wald_and_likelihood_ratio_are_both_reported():
  hours = [4.0] * 8
  res = count.nb2_rate_ratio([4, 6, 3, 5, 4, 7, 5, 4], hours,
                             [22, 25, 19, 28, 24, 21, 26, 23], hours)
  for k in ("p_wald", "p_lrt", "lr_statistic", "alpha", "n_clusters"):
    assert k in res
  assert res["lr_statistic"] > 0


# ---------------------------------------------------------------------------
# Cluster-robust Poisson
# ---------------------------------------------------------------------------


def test_clustering_widens_the_interval_when_repeats_agree_within_a_seed():
  """Three repeats per seed, repeats nearly identical, seeds far apart.

  Treating the twenty-four numbers as independent is the mistake the module
  exists to stop; clustering on the seed must give a visibly larger standard
  error than pretending each repeat is its own draw.
  """
  seeds = [2, 2, 2, 30, 30, 30, 3, 3, 3, 26, 26, 26]
  counts, cl, t = [], [], []
  for i, v in enumerate(seeds):
    counts.append(v)
    cl.append(i // 3)
    t.append(4.0)
  X = [[1.0] for _ in counts]
  clustered = count.cluster_robust_poisson(counts, t, X, cl)
  independent = count.cluster_robust_poisson(counts, t, X, list(range(12)))
  assert clustered["n_clusters"] == 4
  assert clustered["se"][0] > 1.5 * independent["se"][0]


def test_the_robust_ratio_agrees_with_the_negative_binomial_on_easy_data():
  """Different assumptions, so they need not match exactly; on well-behaved
  data they must not disagree about the answer."""
  hours = [4.0] * 8
  a = [4, 6, 3, 5, 4, 7, 5, 4]
  b = [22, 25, 19, 28, 24, 21, 26, 23]
  nb = count.nb2_rate_ratio(a, hours, b, hours)
  rob = count.robust_rate_ratio(a, hours, b, hours,
                               list(range(8)), list(range(8)))
  assert rob["ratio"] == pytest.approx(nb["ratio"], rel=0.05)
  assert rob["p"] < 0.01
  assert rob["n_clusters"] == 16


def test_a_seed_shared_by_both_arms_is_not_merged_into_one_cluster():
  hours = [4.0] * 4
  seeds = [1, 2, 3, 4]
  rob = count.robust_rate_ratio([4, 6, 3, 5], hours, [22, 25, 19, 28], hours,
                                seeds, seeds)
  assert rob["n_clusters"] == 8


def test_too_few_clusters_is_an_error_not_a_number():
  with pytest.raises(ValueError, match="at least two clusters"):
    count.cluster_robust_poisson([3, 4], [1.0, 1.0], [[1.0], [1.0]], [0, 0])


def test_a_collinear_design_is_an_error_not_a_number():
  with pytest.raises(ValueError, match="singular"):
    count.nb2_fit([3, 4, 5], [1.0, 1.0, 1.0],
                  [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])


def test_zero_exposure_is_rejected():
  with pytest.raises(ValueError, match="positive exposure"):
    count.nb2_fit([3, 4], [1.0, 0.0])


def test_a_method_that_never_trips_gets_a_rule_of_three_bound():
  """The outcome the phase is hoping for must not be the one that crashes.

  With no events the negative binomial is degenerate -- the rate is zero, the
  log rate is minus infinity and the information matrix is singular -- so the
  answer is the exact one-sided Poisson bound, ``-2 ln(0.025) / 2T``, the rule
  of three.  Sixteen arm-hours of silence bounds the rate at 0.23/h and not at
  zero, which is the honest thing to report.
  """
  r = count.nb2_rate([0, 0, 0, 0], [4.0] * 4)
  assert r["rate"] == 0.0
  assert r["ci"][0] == 0.0
  assert r["ci"][1] == pytest.approx(3.6888794541139363 / 16.0, rel=1e-9)
  assert r["zero_events"] is True


def test_a_zero_event_arm_gets_an_exact_test_not_a_degenerate_fit():
  """Against the other arm, zero events is still evidence, and the conditional
  binomial test is how much.  Twenty-four events against none, at equal
  exposure, is decisive; three against none is not.
  """
  hours = [4.0] * 4
  decisive = count.nb2_rate_ratio([0] * 4, hours, [7, 5, 6, 6], hours)
  assert decisive["ratio"] == 0.0
  assert decisive["p"] < 1e-5
  assert decisive["method"] == "exact conditional binomial"
  assert decisive["ci"][1] < 0.3

  weak = count.nb2_rate_ratio([0] * 4, hours, [1, 1, 1, 0], hours)
  assert weak["p"] > 0.05


def test_the_exact_test_reproduces_a_hand_computable_case():
  """Three events, all in one arm, equal exposure: under equal rates that is
  ``2 * 0.5**3 = 0.25``."""
  r = count.exact_poisson_ratio_test(0, 8.0, 3, 8.0)
  assert r["p"] == pytest.approx(0.25, rel=1e-12)


def test_chi2_survival_matches_a_known_value():
  assert count.chi2_sf_1df(3.841458820694124) == pytest.approx(0.05, abs=1e-6)
  assert count.chi2_sf_1df(0.0) == pytest.approx(1.0)
