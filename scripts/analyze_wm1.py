"""Summarise the WM1-A evaluations, with the intervals the data type deserves.

    python scripts/analyze_wm1.py --dirs results/wm1_latency/adapt \
        results/wm1_latency/equivalence --json results/wm1_latency/analysis.json

Two metrics, two different statistics, because treating them the same is how
the safety half of a gate gets decided by noise.

**Throughput** is a ratio of two large totals -- objects placed over
arm-minutes -- measured on 512 environments per repeat and three repeats per
condition.  Three numbers is too few for a t-interval to be worth much on its
own, so the interval is a *hierarchical* bootstrap: resample repeats, then
resample environments inside each resampled repeat.  That carries both the
process-level spread (which is what differs between repeats of an identical
command, since MuJoCo-Warp is not run-to-run reproducible) and the
environment-level spread.  The t-interval over repeats is printed next to it,
and where the two disagree the bootstrap is the one to believe.

**Safety-shell trips** are counts, and small ones: a nominal run produces
about seventeen events in 6.8 arm-hours.  A standard deviation over three
repeats of a rate that small is mostly Poisson noise being reported as if it
were a measurement, which is how Phase WM0's nominal trip rate (2.29/h from
47 events) and this phase's re-measurement of the same command (3.15/h from
43 events) can look like a 37% regression and be nothing at all.  So trips are
pooled as counts over exposure, given an exact Poisson interval, and checked
for overdispersion; where the repeats disagree by more than Poisson allows,
the interval is widened by the estimated dispersion rather than the
disagreement being ignored.

Comparisons between conditions are paired on the evaluation seed, and where
several are made at once the p-values are Holm-corrected.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

Z975 = 1.959963984540054


# ---------------------------------------------------------------------------
# Distributions, without scipy
# ---------------------------------------------------------------------------


def chi2_cdf_even(x: float, k: int) -> float:
  """Exact chi-square CDF for even degrees of freedom.

  ``P(X <= x) = 1 - sum_{j<k/2} e^{-x/2} (x/2)^j / j!`` -- the Erlang tail.
  Every degree of freedom this module needs is even, because a Poisson
  interval on a count ``C`` uses ``2C`` and ``2C + 2``.
  """
  if x <= 0:
    return 0.0
  m = k // 2
  term, total = 1.0, 1.0
  for j in range(1, m):
    term *= (x / 2.0) / j
    total += term
  return 1.0 - math.exp(-x / 2.0) * total


def chi2_quantile(p: float, k: float) -> float:
  """Exact for even ``k`` by bisection; Wilson-Hilferty otherwise.

  Exact rather than approximate because the counts here are small -- a nominal
  run trips about seventeen times -- and Wilson-Hilferty is off by a factor of
  two at ``k = 2``, which is the interval on a run that never tripped at all.
  Bisection over a monotone closed-form CDF costs sixty iterations and removes
  the question.
  """
  if k <= 0:
    return 0.0
  ki = int(round(k))
  if abs(k - ki) < 1e-9 and ki % 2 == 0:
    lo, hi = 0.0, max(4.0 * ki, 40.0)
    while chi2_cdf_even(hi, ki) < p:
      hi *= 2.0
    for _ in range(200):
      mid = 0.5 * (lo + hi)
      if chi2_cdf_even(mid, ki) < p:
        lo = mid
      else:
        hi = mid
    return 0.5 * (lo + hi)
  z = _ndtri(p)
  return k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3


def _ndtri(p: float) -> float:
  """Inverse standard normal CDF (Acklam's rational approximation)."""
  a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
       1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
  b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
       6.680131188771972e+01, -1.328068155288572e+01]
  c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
       -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
  d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
       3.754408661907416e+00]
  pl, ph = 0.02425, 1 - 0.02425
  if p < pl:
    q = math.sqrt(-2 * math.log(p))
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  if p > ph:
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  q = p - 0.5
  r = q * q
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def normal_sf(z: float) -> float:
  return 0.5 * math.erfc(z / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def poisson_rate(counts: list[float], exposures: list[float]) -> dict:
  """Pooled rate with an exact interval, plus an overdispersion check.

  The dispersion is Pearson's statistic over the repeats divided by its
  degrees of freedom.  Above one, the repeats vary more than a common Poisson
  rate allows -- which is what a simulator whose run-to-run behaviour is not
  reproducible should do -- and the interval is scaled by its square root.
  Reported either way, so that "the repeats agreed" is a statement in the
  output rather than an assumption in the method.
  """
  c, t = sum(counts), sum(exposures)
  if t <= 0:
    return {"rate": float("nan")}
  rate = c / t
  lo = 0.5 * chi2_quantile(0.025, 2 * c) / t if c > 0 else 0.0
  hi = 0.5 * chi2_quantile(0.975, 2 * c + 2) / t
  n = len(counts)
  if n > 1 and rate > 0:
    x2 = sum((ci - rate * ti) ** 2 / max(rate * ti, 1e-9)
             for ci, ti in zip(counts, exposures))
    phi = x2 / (n - 1)
  else:
    phi = 1.0
  scale = math.sqrt(max(phi, 1.0))
  return {
    "rate": rate, "events": c, "exposure_h": t,
    "ci_poisson": [lo, hi],
    "ci_quasi": [rate - (rate - lo) * scale, rate + (hi - rate) * scale],
    "dispersion": phi, "n_repeats": n,
    "per_repeat_rate": [ci / ti for ci, ti in zip(counts, exposures)],
  }


def rate_ratio(a: dict, b: dict) -> dict:
  """Ratio of two pooled rates, on the log scale, widened by the larger of the
  two dispersions.  ``a`` over ``b``."""
  ca, cb = a.get("events", 0), b.get("events", 0)
  if not ca or not cb:
    return {"ratio": float("nan")}
  r = a["rate"] / b["rate"]
  phi = max(a.get("dispersion", 1.0), b.get("dispersion", 1.0), 1.0)
  se = math.sqrt(phi * (1.0 / ca + 1.0 / cb))
  return {"ratio": r,
          "ci": [r * math.exp(-Z975 * se), r * math.exp(Z975 * se)],
          "p": 2 * normal_sf(abs(math.log(r)) / se) if se > 0 else float("nan"),
          "dispersion_used": phi}


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def hierarchical_bootstrap(repeats: list[tuple[list[float], list[float]]],
                           draws: int = 4000, seed: int = 20260824) -> dict:
  """Resample repeats, then environments within each resampled repeat.

  ``repeats[i]`` is ``(placed_per_env, seconds_per_env)``.  The statistic is
  the pooled ratio, which is what "objects per minute" means -- not the mean of
  per-environment ratios, which would weight a environment that spent four
  seconds alive the same as one that ran the whole rollout.
  """
  rng = random.Random(seed)
  point = (sum(sum(p) for p, _ in repeats) /
           max(sum(sum(s) for _, s in repeats), 1e-9) * 60.0)
  vals = []
  n = len(repeats)
  for _ in range(draws):
    num = den = 0.0
    for _ in range(n):
      placed, secs = repeats[rng.randrange(n)]
      m = len(placed)
      for _ in range(m):
        j = rng.randrange(m)
        num += placed[j]
        den += secs[j]
    vals.append(num / max(den, 1e-9) * 60.0)
  vals.sort()
  lo = vals[int(0.025 * len(vals))]
  hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)]
  # A zero-width interval is either a broken resampler -- a hand-rolled
  # generator whose low bits cycled produced exactly this once, and it reads as
  # certainty -- or genuinely constant input.  Only the first is an error.
  spread = {round(x, 9) for p_, _ in repeats for x in p_}
  if hi - lo < 1e-9 and len(spread) > 1:
    raise RuntimeError("degenerate bootstrap interval; the resampler is broken")
  return {"mean": point, "ci": [lo, hi], "draws": draws, "n_repeats": n}


def t_interval(xs: list[float]) -> dict:
  n = len(xs)
  if n < 2:
    return {"mean": xs[0] if xs else float("nan"), "ci": [float("nan")] * 2,
            "n": n}
  m = sum(xs) / n
  sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
  # Two-sided 95% t quantiles for the small n this ever sees.
  tq = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 2.262)
  h = tq * sd / math.sqrt(n)
  return {"mean": m, "sd": sd, "ci": [m - h, m + h], "n": n}


def paired_diff(a, b) -> dict:
  """Paired on the evaluation seed, which is what makes three repeats usable:
  the run-to-run term is shared and cancels.

  ``a`` and ``b`` are ``{seed: [values]}``.  Pairing by seed rather than by
  position matters here because the conditions do not all have the same
  repeats: the anchors were re-measured at six evaluation seeds and the adapted
  runs at three, and zipping those in order would pair a run against a
  different rollout and call the difference an effect.  Where a condition has
  several values for one seed -- three training seeds evaluated at the same
  evaluation seed -- they are averaged first, so the pairing is one number per
  seed on each side.
  """
  shared = sorted(set(a) & set(b))
  if len(shared) < 2:
    return {"diff": float("nan"), "n": len(shared)}
  d = [sum(a[s]) / len(a[s]) - sum(b[s]) / len(b[s]) for s in shared]
  s = t_interval(d)
  n = len(d)
  se = s["sd"] / math.sqrt(n) if s.get("sd") else 0.0
  t = s["mean"] / se if se > 0 else float("inf")
  # Normal tail: with three pairs a t-distribution p-value is not meaningful
  # to two digits either way, and this is only used for ordering before the
  # Holm correction.
  return {"diff": s["mean"], "ci": s["ci"], "n": n, "seeds": shared,
          "p_approx": 2 * normal_sf(abs(t)) if math.isfinite(t) else 0.0}


def holm(pvals: dict[str, float]) -> dict[str, float]:
  items = sorted(pvals.items(), key=lambda kv: kv[1])
  m = len(items)
  out, running = {}, 0.0
  for i, (k, p) in enumerate(items):
    adj = min(1.0, (m - i) * p)
    running = max(running, adj)
    out[k] = running
  return out


# ---------------------------------------------------------------------------


SEED_SUFFIX = re.compile(r"_s\d+(?=_(?:target|retention)$)")


def load(dirs: list[Path], pool_seeds: bool = True) -> dict[str, list[dict]]:
  """Group result files by ``<tag>_<kind>``, keeping the repeat order.

  With ``pool_seeds`` an extra group per configuration is emitted with the
  training seed stripped from the tag.  That pooled group is the formal
  result: a gate criterion on three process repeats of one training seed would
  be a claim about that seed, and the trip counts in particular need the
  exposure of all nine evaluations before an interval on them means anything.
  The per-seed groups stay, because "which seed" is exactly what a reader
  wants when the pooled number is close to a threshold.
  """
  groups: dict[str, list[dict]] = defaultdict(list)
  for d in dirs:
    for f in sorted(d.glob("*__r*.json")):
      base, _, _rep = f.stem.rpartition("__r")
      groups[f"{d.name}/{base}"].append(json.loads(f.read_text()))
      if pool_seeds:
        pooled = SEED_SUFFIX.sub("", base)
        if pooled != base:
          groups[f"{d.name}/pooled:{pooled}"].append(
            json.loads(f.read_text()))
  return dict(groups)


def _seconds_per_env(r: dict) -> list[float]:
  """Every environment runs the whole rollout, so this is one number.

  It is stored as a scalar; the bootstrap resamples environments and needs the
  denominator per environment, so it is expanded here rather than the
  resampler being taught about two shapes."""
  secs = r["per_env_seconds"]
  n = len(r["per_env"]["placed"])
  return list(secs) if isinstance(secs, list) else [float(secs)] * n


def summarise(runs: list[dict]) -> dict:
  reps = [(r["per_env"]["placed"], _seconds_per_env(r)) for r in runs]
  thr = [r["metrics"]["throughput_per_min"] for r in runs]
  counts = [r["metrics"]["trips_total"] for r in runs]
  hours = [r["config"]["arm_hours"] for r in runs]
  return {
    "n_repeats": len(runs),
    "labels": [r["label"] for r in runs],
    "throughput": {"bootstrap": hierarchical_bootstrap(reps),
                   "t_over_repeats": t_interval(thr),
                   "per_repeat": thr},
    "trips": poisson_rate(counts, hours),
    "success": t_interval([r["metrics"]["overall_success"] for r in runs])
    if "overall_success" in runs[0]["metrics"] else {},
  }


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--dirs", nargs="+", required=True)
  p.add_argument("--json", default=None)
  p.add_argument("--no-pool", action="store_true",
                 help="do not emit the seed-pooled groups")
  p.add_argument("--baseline", default=None,
                 help="group name to compare every other group against")
  a = p.parse_args()

  groups = load([Path(d) for d in a.dirs], pool_seeds=not a.no_pool)
  if not groups:
    raise SystemExit("no result files found")
  out = {"groups": {}}
  for name in sorted(groups):
    out["groups"][name] = summarise(groups[name])

  print()
  print(f"  {'group':44s} {'n':>2s} {'obj/min':>8s} {'95% CI':>16s} "
        f"{'trips/h':>8s} {'events':>7s} {'disp':>5s}")
  for name, s in out["groups"].items():
    b = s["throughput"]["bootstrap"]
    t = s["trips"]
    print(f"  {name:44s} {s['n_repeats']:2d} {b['mean']:8.2f} "
          f"[{b['ci'][0]:6.2f},{b['ci'][1]:6.2f}] "
          f"{t.get('rate', float('nan')):8.2f} "
          f"{int(t.get('events', 0)):7d} {t.get('dispersion', 1.0):5.2f}")

  if a.baseline and a.baseline in out["groups"]:
    def by_seed(runs):
      m: dict[int, list[float]] = defaultdict(list)
      for r in runs:
        m[int(r["config"]["seed_effective"])].append(
          r["metrics"]["throughput_per_min"])
      return dict(m)

    base_thr = by_seed(groups[a.baseline])
    base_rate = out["groups"][a.baseline]["trips"]
    comps, pvals = {}, {}
    for name in sorted(groups):
      if name == a.baseline:
        continue
      d = paired_diff(by_seed(groups[name]), base_thr)
      comps[name] = {"throughput_diff": d,
                     "trip_rate_ratio": rate_ratio(out["groups"][name]["trips"],
                                                   base_rate)}
      if math.isfinite(d.get("p_approx", float("nan"))):
        pvals[name] = d["p_approx"]
    for k, v in holm(pvals).items():
      comps[k]["throughput_p_holm"] = v
    out["comparisons"] = {"baseline": a.baseline, "against": comps}
    print()
    print(f"  paired against {a.baseline}")
    print(f"  {'group':44s} {'d obj/min':>10s} {'95% CI':>16s} "
          f"{'p(holm)':>8s} {'trip ratio':>10s}")
    for name, c in comps.items():
      d = c["throughput_diff"]
      rr = c["trip_rate_ratio"]
      if "ci" not in d:
        # Fewer than two evaluation seeds in common: there is no paired
        # comparison to make, and printing a NaN interval as though there were
        # one is worse than saying so.
        print(f"  {name:44s} {'-- fewer than two shared evaluation seeds':>47s}")
        continue
      print(f"  {name:44s} {d['diff']:10.2f} "
            f"[{d['ci'][0]:6.2f},{d['ci'][1]:6.2f}] "
            f"{c.get('throughput_p_holm', float('nan')):8.3f} "
            f"{rr.get('ratio', float('nan')):10.2f}")

  if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {a.json}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
