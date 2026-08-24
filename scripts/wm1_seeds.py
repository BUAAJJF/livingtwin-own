"""The seed-extended WM1-A analysis, with the training seed as the cluster.

    python scripts/wm1_seeds.py --adapt results/wm1_latency/adapt

WM1-A ran three training seeds and reported trips with a quasi-Poisson
interval scaled by the dispersion across process repeats. That was the wrong
denominator. The repeats inside a seed agree; the seeds do not -- 3.22, 8.01
and 4.49 trips per arm-hour for the same method -- so an interval built from
within-seed variation says how precisely we measured *that PPO run*, not how
precisely we measured *the method*.

This script re-answers every trip question with the seed as the unit:

* repeats are summed inside a seed, so each seed contributes one count and one
  exposure;
* the pooled rate is a negative binomial whose ``alpha`` is the seed-to-seed
  heterogeneity, reported next to it;
* every rate ratio also gets a cluster-robust Poisson sandwich, because the
  negative binomial assumes a particular shape for that heterogeneity and the
  sandwich does not;
* throughput and retention get the same treatment -- seed means first, then a
  t interval on the seeds, so a method with eight repeats of one seed cannot
  look better resolved than a method with eight seeds.

Nothing here reads the target reward to decide anything; it reads evaluations
that were already produced under a fixed posterior, and only re-aggregates
them.

`ORACLE` in the stored plans is displayed as **known-parameter target-only
adaptation**. The old name asserted an upper bound that the WM1-A numbers do
not support -- the posterior-guided run beat it in both domains -- and the
plans are machine artefacts from before the rename, so the translation happens
here rather than by rewriting them.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from piper_push import count

TAG = re.compile(r"^(q[0-9a-f]+)_a([0-9.]+)_s(\d+)_(target|retention)__r(\d+)$")

DISPLAY = {
  "ORACLE": "known-parameter target-only adaptation",
  "ORACLE_MIX": "known-parameter mixture",
  "DA": "decision-aware",
  "B1b": "broad posterior (B1b)",
  "SOURCE": "source-prior refit control",
}


def display(m: str) -> str:
  return DISPLAY.get(m, m)


def load(adapt: Path) -> tuple[dict, dict]:
  """Returns ``runs[(method, alpha, domain)][seed] = [record, ...]`` and the
  tag-to-method map recovered from whichever plans are present."""
  method = {}
  for plan in sorted(adapt.glob("*.json")):
    try:
      d = json.loads(plan.read_text())
    except json.JSONDecodeError:
      continue
    for job in d.get("jobs", []) if isinstance(d, dict) else []:
      ms = job.get("methods") or []
      if job.get("tag") and ms:
        # a tag can be shared by methods whose posteriors coincide
        method.setdefault(job["tag"].rsplit("_s", 1)[0], sorted(ms))

  runs: dict = defaultdict(lambda: defaultdict(list))
  for f in sorted(adapt.glob("*__r*.json")):
    m = TAG.match(f.stem)
    if not m:
      continue
    qhash, alpha, seed, domain, _rep = m.groups()
    key = f"{qhash}_a{alpha}"
    d = json.loads(f.read_text())
    runs[(key, float(alpha), domain)][int(seed)].append({
      "trips": float(d["metrics"]["trips_total"]),
      "arm_hours": float(d["config"]["arm_hours"]),
      "throughput": float(d["metrics"]["throughput_per_min"]),
      "success": float(d["metrics"]["success"]),
      "file": f.name,
    })
  return runs, method


def seed_level(by_seed: dict) -> dict:
  """One count, one exposure and one mean per seed."""
  seeds = sorted(by_seed)
  return {
    "seeds": seeds,
    "trips": [sum(r["trips"] for r in by_seed[s]) for s in seeds],
    "hours": [sum(r["arm_hours"] for r in by_seed[s]) for s in seeds],
    "throughput": [sum(r["throughput"] for r in by_seed[s]) / len(by_seed[s])
                   for s in seeds],
    "repeats": [len(by_seed[s]) for s in seeds],
  }


def t_interval(xs: list[float], conf: float = 0.95) -> dict:
  """Seeds are the observations, so the interval is over seeds and its width
  is governed by how many *seeds* there are, not how many evaluations."""
  n = len(xs)
  mean = sum(xs) / n
  if n < 2:
    return {"mean": mean, "ci": [float("nan")] * 2, "n": n, "sd": float("nan")}
  sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
  # Student t, two-sided 95%, by df; beyond the table the normal is close
  T = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131}
  t = T.get(n - 1, 1.96)
  h = t * sd / math.sqrt(n)
  return {"mean": mean, "ci": [mean - h, mean + h], "n": n, "sd": sd}


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--adapt", default="results/wm1_latency/adapt")
  p.add_argument("--out", default="results/wm1_latency/adapt/seed_counts.json")
  p.add_argument("--gate-trips", type=float, default=5.99,
                 help="G3's ceiling on the target-domain trip rate")
  p.add_argument("--min-seeds", type=int, default=8,
                 help="the specification's floor; configurations below it are "
                      "printed with their count and excluded from the "
                      "headline comparison")
  a = p.parse_args()
  adapt = Path(a.adapt)
  runs, method = load(adapt)
  if not runs:
    raise SystemExit(f"no evaluations under {adapt}")

  report: dict = {"configurations": {}, "comparisons": [],
                  "gate_trips_per_arm_hour": a.gate_trips}

  print()
  print(f"  {'configuration':44s} {'dom':9s} {'seeds':>5s} {'ev':>3s} "
        f"{'trips':>6s} {'h':>6s} {'rate':>6s} {'NB 95% CI':>16s} "
        f"{'alpha':>6s}")
  print("  " + "-" * 112)

  rows = {}
  for (key, alpha, domain), by_seed in sorted(runs.items()):
    ms = method.get(key, [key])
    name = " / ".join(display(m) for m in ms)
    sl = seed_level(by_seed)
    nb = count.nb2_rate(sl["trips"], sl["hours"])
    thr = t_interval(sl["throughput"])
    rows[(key, alpha, domain)] = {"name": name, "methods": ms, "seed": sl,
                                 "nb": nb, "throughput": thr}
    label = f"{name} a={alpha:.2f}"[:44]
    ci = f"[{nb['ci'][0]:.2f}, {nb['ci'][1]:.2f}]"
    print(f"  {label:44s} {domain:9s} {len(sl['seeds']):5d} "
          f"{sum(sl['repeats']):3d} {sum(sl['trips']):6.0f} "
          f"{sum(sl['hours']):6.1f} {nb['rate']:6.2f} {ci:>16s} "
          f"{nb['alpha']:6.3f}")
    report["configurations"][f"{key}|a{alpha}|{domain}"] = {
      "name": name, "methods": ms, "alpha": alpha, "domain": domain,
      "n_seeds": len(sl["seeds"]), "seeds": sl["seeds"],
      "n_evaluations": sum(sl["repeats"]),
      "trips_per_seed": sl["trips"], "hours_per_seed": sl["hours"],
      "negative_binomial": nb,
      "per_seed_rate": [t / h for t, h in zip(sl["trips"], sl["hours"])],
      "throughput_over_seeds": thr,
      "quasi_poisson_would_have_said": _quasi(by_seed),
    }

  # -- the gate, re-decided on the seed-level interval -----------------------
  print()
  print("  G3 -- target-domain trips, ceiling "
        f"{a.gate_trips:.2f}/arm-hour, decided on the upper bound")
  for (key, alpha, domain), r in sorted(rows.items()):
    if domain != "target":
      continue
    nb, n = r["nb"], len(r["seed"]["seeds"])
    ok = nb["ci"][1] <= a.gate_trips
    short = " (under the seed floor)" if n < a.min_seeds else ""
    print(f"    {'PASS' if ok else 'FAIL'}  {r['name'][:52]:52s} "
          f"a={alpha:.2f}  {nb['rate']:5.2f} "
          f"[{nb['ci'][0]:.2f}, {nb['ci'][1]:.2f}]  {n} seeds{short}")
    report["configurations"][f"{key}|a{alpha}|{domain}"]["g3"] = {
      "pass": ok, "upper": nb["ci"][1], "enough_seeds": n >= a.min_seeds}

  # -- pairwise comparisons within a domain ---------------------------------
  print()
  print("  Rate ratios within a domain -- negative binomial, and a "
        "cluster-robust\n  Poisson sandwich beside it. Where they disagree "
        "the sandwich is the\n  weaker assumption and the one to quote.")
  keys = sorted(rows)
  for i, ka in enumerate(keys):
    for kb in keys[i + 1:]:
      if ka[2] != kb[2]:
        continue
      A, B = rows[ka], rows[kb]
      if min(len(A["seed"]["seeds"]), len(B["seed"]["seeds"])) < 2:
        continue
      nb = count.nb2_rate_ratio(A["seed"]["trips"], A["seed"]["hours"],
                                B["seed"]["trips"], B["seed"]["hours"])
      try:
        rob = count.robust_rate_ratio(
          A["seed"]["trips"], A["seed"]["hours"],
          B["seed"]["trips"], B["seed"]["hours"],
          A["seed"]["seeds"], B["seed"]["seeds"])
      except ValueError:
        rob = None
      entry = {"domain": ka[2],
               "a": {"name": A["name"], "alpha": ka[1],
                     "n_seeds": len(A["seed"]["seeds"])},
               "b": {"name": B["name"], "alpha": kb[1],
                     "n_seeds": len(B["seed"]["seeds"])},
               "negative_binomial": nb, "cluster_robust": rob}
      report["comparisons"].append(entry)
      pn = nb.get("p_lrt", nb.get("p", float("nan")))
      pr = rob["p"] if rob else float("nan")
      print(f"    [{ka[2]:9s}] {A['name'][:30]:30s} / {B['name'][:30]:30s}  "
            f"ratio {nb['ratio']:5.2f} "
            f"[{nb['ci'][0]:.2f},{nb['ci'][1]:.2f}]  "
            f"p_NB {pn:.3f}  p_robust {pr:.3f}")

  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  Path(a.out).write_text(json.dumps(report, indent=1))
  print(f"\n  wrote {a.out}")

  short = [k for k, r in report["configurations"].items()
           if r["n_seeds"] < a.min_seeds]
  if short:
    print(f"\n  {len(short)} configuration(s) still under {a.min_seeds} "
          f"training seeds; the comparison above is provisional.")
  return 0


def _quasi(by_seed: dict) -> dict:
  """What WM1-A's method would have reported on the same data, so the report
  can show the difference rather than assert it."""
  counts, hours = [], []
  for s in sorted(by_seed):
    for r in by_seed[s]:
      counts.append(r["trips"])
      hours.append(r["arm_hours"])
  c, t = sum(counts), sum(hours)
  if t <= 0 or c <= 0:
    return {"rate": 0.0}
  rate = c / t
  n = len(counts)
  phi = (sum((ci - rate * ti) ** 2 / max(rate * ti, 1e-9)
             for ci, ti in zip(counts, hours)) / (n - 1)) if n > 1 else 1.0
  hi = 0.5 * _chi2q(0.975, 2 * int(c) + 2) / t
  scale = math.sqrt(max(phi, 1.0))
  return {"rate": rate, "upper_quasi": rate + (hi - rate) * scale,
          "dispersion_within_repeats": phi, "n_evaluations": n}


def _chi2q(q: float, df: int) -> float:
  """Exact quantile for even df by bisection on the closed-form CDF."""
  k = df // 2

  def cdf(x):
    s, term = 0.0, 1.0
    for i in range(k):
      if i:
        term *= x / (2.0 * i)
      s += term
    return 1.0 - math.exp(-x / 2.0) * s

  lo, hi = 0.0, max(4.0 * df, 20.0)
  while cdf(hi) < q:
    hi *= 2
  for _ in range(200):
    mid = 0.5 * (lo + hi)
    if cdf(mid) < q:
      lo = mid
    else:
      hi = mid
  return 0.5 * (lo + hi)


if __name__ == "__main__":
  raise SystemExit(main())
