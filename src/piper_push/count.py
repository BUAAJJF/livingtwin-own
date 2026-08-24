"""Count models for safety trips, with the training seed as the cluster.

WM1-A reported trip rates with a quasi-Poisson interval: pooled rate, Pearson
dispersion over the process repeats, interval widened by its square root. That
was right as far as it went and wrong about where the variance lives. The
per-seed rates behind one of those pooled numbers were 3.22, 8.01 and 4.49 per
arm-hour. The repeats within a seed agree far better than the seeds do with
each other, so scaling by a dispersion estimated *within* seed understates the
uncertainty of anything the report says about the method rather than about one
particular training run.

Two models here, and the report is asked to show both.

**Negative binomial, seeds as observations.** Aggregate the repeats inside a
seed, then fit

    y_s ~ NB2(mu_s, alpha),   mu_s = t_s * exp(x_s . beta),
    Var(y_s) = mu_s + alpha * mu_s^2

by maximum likelihood. `alpha` is exactly the seed-to-seed heterogeneity the
quasi-Poisson scaling was standing in for, and unlike a dispersion multiplier
it is a parameter with a likelihood, so a rate ratio gets a Wald interval and a
likelihood-ratio test rather than a scaled Poisson interval.

**Cluster-robust Poisson.** The same log-linear mean, fitted by Poisson
likelihood, with a sandwich variance whose meat is summed over seeds. This
makes no assumption at all about the shape of the seed-level variance -- it
only assumes seeds are independent, which is true by construction -- so where
the two disagree, the negative binomial's extra assumption is what is doing
the work and the report says which number it is quoting.

Both need the seeds to be independent draws, which they are: a training seed
sets PPO's initialisation and its rollout stream and nothing else.

With few seeds the sandwich is anticonservative, so the small-sample
correction ``G/(G-1) * (N-1)/(N-p)`` is applied and the number of clusters is
reported next to every interval. At eight seeds this is a real limitation and
the report states it rather than hiding it in a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Z975 = 1.959963984540054


# ---------------------------------------------------------------------------
# Small dense linear algebra: p is 1 or 2 here
# ---------------------------------------------------------------------------


def _inv(m: list[list[float]]) -> list[list[float]]:
  """Gauss-Jordan with partial pivoting.  Raises on a singular matrix rather
  than returning a plausible-looking inverse of a rank-deficient design."""
  n = len(m)
  a = [list(row) + [1.0 if i == j else 0.0 for j in range(n)]
       for i, row in enumerate(m)]
  for col in range(n):
    piv = max(range(col, n), key=lambda r: abs(a[r][col]))
    if abs(a[piv][col]) < 1e-12:
      raise ValueError("singular design matrix: a column is collinear or "
                       "constant within every group")
    a[col], a[piv] = a[piv], a[col]
    d = a[col][col]
    a[col] = [v / d for v in a[col]]
    for r in range(n):
      if r == col:
        continue
      f = a[r][col]
      if f:
        a[r] = [v - f * w for v, w in zip(a[r], a[col])]
  return [row[n:] for row in a]


def _matvec(m, v):
  return [sum(mi * vi for mi, vi in zip(row, v)) for row in m]


# ---------------------------------------------------------------------------
# Negative binomial (NB2)
# ---------------------------------------------------------------------------


def nb2_loglik(y, offset, mu, alpha: float) -> float:
  """``alpha -> 0`` is Poisson, and the limit is taken rather than divided by."""
  del offset
  if alpha <= 1e-10:
    return sum(yi * math.log(max(mi, 1e-300)) - mi - math.lgamma(yi + 1.0)
               for yi, mi in zip(y, mu))
  r = 1.0 / alpha
  out = 0.0
  for yi, mi in zip(y, mu):
    out += (math.lgamma(yi + r) - math.lgamma(r) - math.lgamma(yi + 1.0)
            + yi * math.log(alpha * mi / (1.0 + alpha * mi))
            - r * math.log(1.0 + alpha * mi))
  return out


def _nb2_beta(y, log_t, X, alpha: float, beta=None, iters: int = 100):
  """Newton on ``beta`` at fixed ``alpha``.  The NB2 score and Fisher
  information for beta are

      U = sum x_i (y_i - mu_i) / (1 + alpha mu_i)
      I = sum x_i x_i' mu_i / (1 + alpha mu_i)

  which is Poisson's, downweighted by the overdispersion.  Concave in beta, so
  Newton from zero converges; the step is halved if the likelihood falls, which
  only happens on the first step of a badly scaled design.
  """
  p = len(X[0])
  b = list(beta) if beta else [0.0] * p
  def mus(bb):
    return [math.exp(min(lt + sum(xi * bi for xi, bi in zip(x, bb)), 700.0))
            for lt, x in zip(log_t, X)]
  ll = nb2_loglik(y, log_t, mus(b), alpha)
  for _ in range(iters):
    mu = mus(b)
    w = [mi / (1.0 + alpha * mi) for mi in mu]
    u = [sum(X[i][j] * (y[i] - mu[i]) / (1.0 + alpha * mu[i])
             for i in range(len(y))) for j in range(p)]
    info = [[sum(w[i] * X[i][j] * X[i][k] for i in range(len(y)))
             for k in range(p)] for j in range(p)]
    step = _matvec(_inv(info), u)
    t = 1.0
    for _ in range(30):
      cand = [bi + t * si for bi, si in zip(b, step)]
      ll_c = nb2_loglik(y, log_t, mus(cand), alpha)
      if ll_c >= ll - 1e-12:
        break
      t *= 0.5
    else:
      break
    if max(abs(t * s) for s in step) < 1e-10:
      b, ll = cand, ll_c
      break
    b, ll = cand, ll_c
  mu = mus(b)
  w = [mi / (1.0 + alpha * mi) for mi in mu]
  info = [[sum(w[i] * X[i][j] * X[i][k] for i in range(len(y)))
           for k in range(p)] for j in range(p)]
  return b, mu, info, ll


def _profile_alpha(y, log_t, X, lo: float = 0.0, hi: float = 50.0):
  """Golden-section on ``log(alpha)`` over the profile likelihood.

  ``alpha = 0`` is on the boundary and is checked separately: when the data are
  not overdispersed the maximum sits there, and a search on the log scale would
  approach it without reaching it.
  """
  best = None
  b0, mu0, i0, ll0 = _nb2_beta(y, log_t, X, 0.0)
  best = (0.0, b0, mu0, i0, ll0)

  def at(a):
    b, mu, info, ll = _nb2_beta(y, log_t, X, a)
    return ll, (a, b, mu, info, ll)

  grid = [1e-4 * (10 ** (i / 4.0)) for i in range(0, 25)]
  grid = [g for g in grid if lo <= g <= hi]
  vals = [at(g) for g in grid]
  k = max(range(len(vals)), key=lambda i: vals[i][0])
  if vals[k][0] > best[4]:
    best = vals[k][1]
  a_lo = grid[max(k - 1, 0)]
  a_hi = grid[min(k + 1, len(grid) - 1)]
  if a_hi > a_lo:
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = a_hi - phi * (a_hi - a_lo)
    x2 = a_lo + phi * (a_hi - a_lo)
    f1, r1 = at(x1)
    f2, r2 = at(x2)
    for _ in range(60):
      if f1 < f2:
        a_lo, x1, f1, r1 = x1, x2, f2, r2
        x2 = a_lo + phi * (a_hi - a_lo)
        f2, r2 = at(x2)
      else:
        a_hi, x2, f2, r2 = x2, x1, f1, r1
        x1 = a_hi - phi * (a_hi - a_lo)
        f1, r1 = at(x1)
      if a_hi - a_lo < 1e-9:
        break
    cand = r1 if f1 > f2 else r2
    if cand[4] > best[4]:
      best = cand
  return best


@dataclass
class NBFit:
  """A fitted NB2 log-linear rate model."""

  beta: list[float]
  alpha: float
  loglik: float
  cov: list[list[float]]
  n: int
  mu: list[float] = field(default_factory=list)

  def rate(self, x=None) -> float:
    x = x if x is not None else [1.0] + [0.0] * (len(self.beta) - 1)
    return math.exp(sum(xi * bi for xi, bi in zip(x, self.beta)))

  def se(self, j: int) -> float:
    return math.sqrt(max(self.cov[j][j], 0.0))

  def to_json(self) -> dict:
    return {"beta": self.beta, "alpha": self.alpha, "loglik": self.loglik,
            "se": [self.se(j) for j in range(len(self.beta))],
            "n_clusters": self.n}


def nb2_fit(counts, exposures, X=None) -> NBFit:
  """Maximum-likelihood NB2 with a log exposure offset.

  ``X`` defaults to an intercept, which makes this a pooled rate with a
  heterogeneity parameter.  Pass a two-column design to compare two methods.
  """
  y = [float(c) for c in counts]
  if any(t <= 0 for t in exposures):
    raise ValueError("every observation needs positive exposure")
  log_t = [math.log(float(t)) for t in exposures]
  X = X if X is not None else [[1.0] for _ in y]
  a, b, mu, info, ll = _profile_alpha(y, log_t, X)
  return NBFit(beta=b, alpha=a, loglik=ll, cov=_inv(info), n=len(y), mu=mu)


# ``-2 ln(0.025)``: the exact upper 97.5% chi-square quantile on 2 degrees of
# freedom, which for df=2 is closed form because the survival function is
# ``exp(-x/2)``.  Halved and divided by exposure it is the rule of three.
CHI2_2DF_975 = -2.0 * math.log(0.025)


def nb2_rate(counts, exposures) -> dict:
  """Pooled rate with a negative-binomial interval, seeds as observations.

  With no events at all the model is degenerate -- zero rate, minus-infinity
  log rate, singular information -- and the exact one-sided Poisson bound is
  reported instead.  A method that never tripped is the result this phase is
  looking for, so it must not be the one that raises.
  """
  total = sum(counts)
  T = sum(float(t) for t in exposures)
  if total == 0:
    return {"rate": 0.0, "ci": [0.0, 0.5 * CHI2_2DF_975 / T], "alpha": 0.0,
            "events": 0, "exposure_h": T, "n_clusters": len(counts),
            "per_cluster_rate": [0.0] * len(counts), "loglik": 0.0,
            "zero_events": True,
            "method": "exact one-sided Poisson (rule of three)"}
  fit = nb2_fit(counts, exposures)
  r = math.exp(fit.beta[0])
  se = fit.se(0)
  return {
    "rate": r,
    "ci": [r * math.exp(-Z975 * se), r * math.exp(Z975 * se)],
    "alpha": fit.alpha,
    "events": sum(counts), "exposure_h": sum(exposures),
    "n_clusters": len(counts),
    "per_cluster_rate": [c / t for c, t in zip(counts, exposures)],
    "loglik": fit.loglik, "zero_events": False,
    "method": "negative binomial (NB2), clusters as observations",
  }


def nb2_rate_ratio(counts_a, exp_a, counts_b, exp_b) -> dict:
  """Rate ratio ``a/b`` from a two-group NB2 model.

  Both a Wald interval and a likelihood-ratio test are returned. They are
  reported together because with eight clusters they can disagree, and when
  they do the likelihood ratio is the one to believe -- the Wald statistic is
  the one that depends on the log scale being quadratic near the maximum.
  """
  ca, cb = sum(counts_a), sum(counts_b)
  if ca == 0 or cb == 0:
    return exact_poisson_ratio_test(ca, sum(exp_a), cb, sum(exp_b))
  y = [float(c) for c in counts_a] + [float(c) for c in counts_b]
  t = [float(v) for v in exp_a] + [float(v) for v in exp_b]
  X = ([[1.0, 1.0] for _ in counts_a] + [[1.0, 0.0] for _ in counts_b])
  full = nb2_fit(y, t, X)
  null = nb2_fit(y, t, [[1.0] for _ in y])
  lr = 2.0 * (full.loglik - null.loglik)
  se = full.se(1)
  logr = full.beta[1]
  return {
    "ratio": math.exp(logr),
    "ci": [math.exp(logr - Z975 * se), math.exp(logr + Z975 * se)],
    "p_wald": 2 * 0.5 * math.erfc(abs(logr / se) / math.sqrt(2.0))
              if se > 0 else float("nan"),
    "p_lrt": chi2_sf_1df(max(lr, 0.0)),
    "lr_statistic": lr,
    "alpha": full.alpha,
    "n_clusters": [len(counts_a), len(counts_b)],
    "rate_a": math.exp(full.beta[0] + full.beta[1]),
    "rate_b": math.exp(full.beta[0]),
  }


def exact_poisson_ratio_test(ca: float, ta: float, cb: float, tb: float
                             ) -> dict:
  """Conditional binomial test for two Poisson rates -- used when an arm is
  empty and the likelihood-based fit is degenerate.

  Conditioning on the total ``N = ca + cb``, ``ca`` is binomial with
  ``theta = ra*ta / (ra*ta + rb*tb)``; under equal rates that is
  ``ta/(ta+tb)``, free of the unknown rate.  The interval on ``theta`` is
  Clopper-Pearson, mapped back to the ratio.

  It ignores clustering, and says so in what it returns. With zero events in
  one arm there is no between-cluster variation there to estimate, so no
  cluster-robust or negative-binomial method has anything more to use; the
  cost is that the *other* arm's seed heterogeneity is not reflected either,
  which makes this the optimistic end of the range.
  """
  n = int(round(ca + cb))
  k = int(round(ca))
  theta0 = ta / (ta + tb)
  if n == 0:
    return {"ratio": float("nan"), "ci": [0.0, float("inf")], "p": 1.0,
            "method": "exact conditional binomial",
            "clustered": False, "n_events": [ca, cb]}

  def binom_cdf(x: int) -> float:
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                        - math.lgamma(n - i + 1)
                        + (i * math.log(theta0) if i else 0.0)
                        + ((n - i) * math.log1p(-theta0) if n - i else 0.0))
               for i in range(0, x + 1))

  p = min(1.0, 2.0 * min(binom_cdf(k), 1.0 - binom_cdf(k - 1) if k else 1.0))

  def beta_q(q: float, a: float, b: float) -> float:
    """Clopper-Pearson endpoint by bisection on the beta CDF, itself summed as
    a binomial tail -- exact, and cheap at these counts."""
    if a <= 0:
      return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
      mid = 0.5 * (lo + hi)
      x = int(round(a)) - 1
      tail = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                          - math.lgamma(n - i + 1)
                          + (i * math.log(mid) if i and mid > 0 else
                             (0.0 if i == 0 else -1e300))
                          + ((n - i) * math.log1p(-mid) if n - i and mid < 1
                             else (0.0 if n == i else -1e300)))
                 for i in range(0, x + 1))
      if tail > q:
        lo = mid
      else:
        hi = mid
    return 0.5 * (lo + hi)

  th_lo = beta_q(0.975, k, n - k + 1) if k > 0 else 0.0
  th_hi = 1.0 - beta_q(0.975, n - k, k + 1) if k < n else 1.0
  scale = tb / ta

  def to_ratio(th):
    return float("inf") if th >= 1.0 else scale * th / (1.0 - th)

  return {"ratio": (cb and (ca / ta) / (cb / tb)) or 0.0,
          "ci": [to_ratio(th_lo), to_ratio(th_hi)],
          "p": p, "method": "exact conditional binomial",
          "clustered": False, "n_events": [ca, cb],
          "rate_a": ca / ta, "rate_b": cb / tb}


def chi2_sf_1df(x: float) -> float:
  """``P(chi2_1 > x)`` -- the complementary error function, exactly."""
  return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


# ---------------------------------------------------------------------------
# Cluster-robust Poisson
# ---------------------------------------------------------------------------


def cluster_robust_poisson(counts, exposures, X, clusters) -> dict:
  """Poisson log-linear fit with a sandwich variance summed over clusters.

  The point estimate is the Poisson MLE, which is consistent whatever the true
  variance function is. Only the standard error changes: the meat is built from
  cluster-summed scores, so any correlation between the repeats inside a
  training seed is absorbed instead of assumed away.
  """
  y = [float(c) for c in counts]
  log_t = [math.log(float(v)) for v in exposures]
  p = len(X[0])
  b, mu, info, ll = _nb2_beta(y, log_t, X, 0.0)
  bread = _inv(info)

  by_cluster: dict = {}
  for i, g in enumerate(clusters):
    acc = by_cluster.setdefault(g, [0.0] * p)
    for j in range(p):
      acc[j] += X[i][j] * (y[i] - mu[i])
  G, N = len(by_cluster), len(y)
  if G < 2:
    raise ValueError("a cluster-robust variance needs at least two clusters")
  meat = [[sum(u[j] * u[k] for u in by_cluster.values()) for k in range(p)]
          for j in range(p)]
  correction = (G / (G - 1.0)) * ((N - 1.0) / max(N - p, 1.0))
  cov = [[correction * sum(bread[j][a] * meat[a][bb] * bread[bb][k]
                           for a in range(p) for bb in range(p))
          for k in range(p)] for j in range(p)]
  return {"beta": b, "cov": cov, "n_clusters": G, "n": N, "loglik": ll,
          "se": [math.sqrt(max(cov[j][j], 0.0)) for j in range(p)]}


def robust_rate_ratio(counts_a, exp_a, counts_b, exp_b,
                      clusters_a, clusters_b) -> dict:
  """Rate ratio ``a/b``, Poisson point estimate, seed-clustered sandwich.

  Clusters are namespaced by group, so the same training seed appearing in both
  arms does not merge two independent fits into one cluster. Where the two arms
  really are paired on seed, the paired analysis belongs upstream in the
  comparison, not here.
  """
  y = list(counts_a) + list(counts_b)
  t = list(exp_a) + list(exp_b)
  X = [[1.0, 1.0] for _ in counts_a] + [[1.0, 0.0] for _ in counts_b]
  g = [f"a:{c}" for c in clusters_a] + [f"b:{c}" for c in clusters_b]
  fit = cluster_robust_poisson(y, t, X, g)
  logr, se = fit["beta"][1], fit["se"][1]
  return {
    "ratio": math.exp(logr),
    "ci": [math.exp(logr - Z975 * se), math.exp(logr + Z975 * se)],
    "p": 2 * 0.5 * math.erfc(abs(logr / se) / math.sqrt(2.0))
         if se > 0 else float("nan"),
    "n_clusters": fit["n_clusters"],
    "rate_a": math.exp(fit["beta"][0] + logr),
    "rate_b": math.exp(fit["beta"][0]),
  }
