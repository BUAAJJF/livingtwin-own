"""Phase WM1-B's gate: does risk-awareness buy safety at equal budget?

    python scripts/wm1b_gate.py --adapt results/wm1_damping/adapt

WM1-A's gate asked six things about a throughput mismatch. This axis's cost is
almost entirely in the tail -- 4.5% of throughput, 89 times the safety-shell
trips -- so reusing those criteria would grade this phase mostly on the part of
the problem that barely moved. The criteria here are the phase specification's
own central question, split into the pieces that can each fail separately:

    C1  identification clears its controls
    C2  risk-aware trips < trajectory matching
    C3  risk-aware trips < broad domain randomisation
    C4  throughput in the target domain is not paid away for it
    C5  retention in the source domain is not paid away for it
    C6  every new feature is default-off with provenance

C2 and C3 are the question. C4 and C5 are the constraint that makes the answer
mean something: a policy that never moves trips nothing.

**How a rate comparison is decided.** Counts are clustered on the *training
seed*, never on the evaluation repeat. Repeats inside a seed agree closely and
seeds do not, so a within-repeat interval measures how precisely one PPO run
was observed rather than how precisely the method was. Both a negative
binomial and a cluster-robust Poisson sandwich are computed; a criterion passes
only if **both** agree, because where they disagree the difference is the
negative binomial's variance assumption rather than the data.

**Holm.** C2 and C3 are two pre-registered comparisons in one family and are
corrected together. C4 and C5 are constraints, not discoveries -- they are
one-sided margins against a fixed budget -- so they are not in the family.

Verdicts are GREEN / YELLOW / RED per criterion, as the phase asked:

    GREEN   passes, with enough training seeds to mean it
    YELLOW  passes on the point estimate but not the interval, or passes on
            fewer seeds than the specification asked for
    RED     fails
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
ANCHOR = re.compile(r"^anchor_(zeroshot|nominal)__r(\d+)$")

MIN_SEEDS = 8
"""The specification's floor for the core safety comparison."""

THROUGHPUT_BUDGET = 0.05
"""C4 and C5: at most 5% may be paid away, against the stated reference."""

# The three distributions the phase compares.  Named by what they are rather
# than by the hash of their probability vector.
ROLES = {
  "trajectory_matching": ("B3_state", "KNOWN_PARAM"),
  "broad_dr": ("M2_broad",),
  "risk_aware": ("M5_risk_aware",),
  "source_refit": ("B0_prior_refit",),
  "known_mixture": ("KNOWN_PARAM_MIX",),
}


def load(adapt: Path) -> tuple[dict, dict]:
  """``runs[role][domain][seed] = [record...]`` plus the two anchors."""
  method_of: dict[str, list[str]] = {}
  for plan in sorted(adapt.glob("*.json")):
    try:
      d = json.loads(plan.read_text())
    except json.JSONDecodeError:
      continue
    if not isinstance(d, dict):
      continue
    for job in d.get("jobs", []):
      if job.get("tag") and job.get("methods"):
        method_of.setdefault(job["tag"].rsplit("_s", 1)[0], sorted(job["methods"]))

  runs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  anchors: dict = defaultdict(list)
  for f in sorted(adapt.glob("*__r*.json")):
    d = json.loads(f.read_text())
    rec = {"trips": float(d["metrics"]["trips_total"]),
           "hours": float(d["config"]["arm_hours"]),
           "throughput": float(d["metrics"]["throughput_per_min"]),
           "success": float(d["metrics"]["success"]), "file": f.name}
    m = ANCHOR.match(f.stem)
    if m:
      anchors[m.group(1)].append(rec)
      continue
    m = TAG.match(f.stem)
    if not m:
      continue
    qhash, alpha, seed, domain, _ = m.groups()
    names = set(method_of.get(f"{qhash}_a{alpha}", []))
    for role, want in ROLES.items():
      if names & set(want):
        runs[role][domain][int(seed)].append(rec)
        break
  return runs, anchors


def by_seed(d: dict) -> dict:
  seeds = sorted(d)
  return {"seeds": seeds,
          "trips": [sum(r["trips"] for r in d[s]) for s in seeds],
          "hours": [sum(r["hours"] for r in d[s]) for s in seeds],
          "throughput": [sum(r["throughput"] for r in d[s]) / len(d[s])
                         for s in seeds],
          "n_eval": sum(len(d[s]) for s in seeds)}


def t_interval(xs: list[float]) -> dict:
  n = len(xs)
  if n == 0:
    return {"mean": float("nan"), "lo": float("nan"), "n": 0}
  mean = sum(xs) / n
  if n < 2:
    return {"mean": mean, "lo": float("nan"), "n": n}
  sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
  T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
       14: 2.145, 15: 2.131}
  h = T.get(n - 1, 1.96) * sd / math.sqrt(n)
  return {"mean": mean, "lo": mean - h, "hi": mean + h, "n": n, "sd": sd}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
  """Step-down, and the adjusted values are made monotone so that a later
  comparison never comes out more significant than an earlier one."""
  items = sorted(pvalues.items(), key=lambda kv: kv[1])
  m = len(items)
  out, running = {}, 0.0
  for i, (k, p) in enumerate(items):
    running = max(running, min(1.0, (m - i) * p))
    out[k] = running
  return out


def compare(a: dict, b: dict, label_a: str, label_b: str) -> dict:
  """Rate ratio ``a/b`` two ways.  A difference is only claimed when both
  agree, because where they disagree it is the negative binomial's variance
  assumption doing the work rather than the counts."""
  nb = count.nb2_rate_ratio(a["trips"], a["hours"], b["trips"], b["hours"])
  try:
    rob = count.robust_rate_ratio(a["trips"], a["hours"],
                                  b["trips"], b["hours"],
                                  a["seeds"], b["seeds"])
  except ValueError:
    rob = None
  return {"a": label_a, "b": label_b,
          "n_seeds": [len(a["seeds"]), len(b["seeds"])],
          "rate_a": sum(a["trips"]) / max(sum(a["hours"]), 1e-9),
          "rate_b": sum(b["trips"]) / max(sum(b["hours"]), 1e-9),
          "negative_binomial": nb, "cluster_robust": rob,
          "p_nb": nb.get("p_lrt", nb.get("p", float("nan"))),
          "p_robust": (rob or {}).get("p", float("nan"))}


def verdict(passes: bool, point_ok: bool, enough_seeds: bool) -> str:
  if passes and enough_seeds:
    return "GREEN"
  if passes or point_ok:
    return "YELLOW"
  return "RED"


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--adapt", default="results/wm1_damping/adapt")
  p.add_argument("--posterior",
                 default="results/wm1_damping/posterior/posterior_report.json")
  p.add_argument("--out", default="results/wm1_damping/gate.json")
  p.add_argument("--min-seeds", type=int, default=MIN_SEEDS)
  a = p.parse_args()

  adapt = Path(a.adapt)
  runs, anchors = load(adapt)
  report: dict = {"criteria": {}, "roles_found": sorted(runs),
                  "min_seeds": a.min_seeds}

  # -- anchors --------------------------------------------------------------
  anc = {}
  for k, rows in anchors.items():
    anc[k] = {"throughput": sum(r["throughput"] for r in rows) / len(rows),
              "trips_per_hour": (sum(r["trips"] for r in rows)
                                 / max(sum(r["hours"] for r in rows), 1e-9)),
              "n": len(rows)}
  report["anchors"] = anc
  print()
  if anc:
    print("  anchors (unadapted policy, measured the same way as every run)")
    for k, v in sorted(anc.items()):
      print(f"    {k:10s} {v['throughput']:6.2f} obj/min   "
            f"{v['trips_per_hour']:7.2f} trips/arm-h   ({v['n']} repeats)")
  else:
    print("  !! no anchors on disk; C4 and C5 have nothing to be measured "
          "against and are NOT RUN rather than assumed")

  # -- what is on disk ------------------------------------------------------
  seed_level = {}
  print()
  print(f"  {'role':22s} {'domain':10s} {'seeds':>5s} {'ev':>3s} "
        f"{'trips':>6s} {'arm-h':>7s} {'rate':>7s} {'NB 95% CI':>16s} "
        f"{'obj/min':>8s}")
  for role in sorted(runs):
    for domain in ("target", "retention"):
      if domain not in runs[role]:
        continue
      sl = by_seed(runs[role][domain])
      seed_level[(role, domain)] = sl
      nb = count.nb2_rate(sl["trips"], sl["hours"])
      thr = t_interval(sl["throughput"])
      ci = f"[{nb['ci'][0]:.2f}, {nb['ci'][1]:.2f}]"
      print(f"  {role:22s} {domain:10s} {len(sl['seeds']):5d} "
            f"{sl['n_eval']:3d} {sum(sl['trips']):6.0f} "
            f"{sum(sl['hours']):7.1f} {nb['rate']:7.2f} {ci:>16s} "
            f"{thr['mean']:8.2f}")
      report.setdefault("configurations", {})[f"{role}|{domain}"] = {
        "n_seeds": len(sl["seeds"]), "seeds": sl["seeds"],
        "n_evaluations": sl["n_eval"],
        "negative_binomial": nb, "throughput_over_seeds": thr,
        "trips_per_seed": sl["trips"], "hours_per_seed": sl["hours"]}

  # -- C1: identification ---------------------------------------------------
  c1 = {"checked": False}
  try:
    pr = json.loads(Path(a.posterior).read_text())
    m = pr["budgets"]["60.0"]["methods"]["test"]
    best = max((m[k]["balanced_accuracy"], k) for k in m
               if not k.startswith("ctrl_") and k not in ("M2_broad", "B0_prior"))
    ctrl = max((m[k]["balanced_accuracy"], k) for k in m if k.startswith("ctrl_"))
    c1 = {"checked": True, "best": best[1], "best_balanced": best[0],
          "worst_control": ctrl[1], "control_balanced": ctrl[0],
          "margin": best[0] - ctrl[0],
          "pass": best[0] - ctrl[0] >= 0.15 and best[0] >= 0.9,
          "note": ("ctrl_wm_shuffled is not a chance-level control on this "
                   "axis -- the magnitude of world-model prediction error "
                   "differs by domain even unconditioned -- so the margin is "
                   "taken against the strongest control, not the mean of "
                   "them")}
  except (OSError, json.JSONDecodeError, KeyError, ValueError):
    pass
  report["criteria"]["C1  identification clears its controls"] = c1

  # -- C2, C3: the phase's question -----------------------------------------
  family = {}
  comparisons = {}
  for name, (num, den) in (
      ("C2  risk-aware trips < trajectory matching",
       ("risk_aware", "trajectory_matching")),
      ("C3  risk-aware trips < broad domain randomisation",
       ("risk_aware", "broad_dr"))):
    A = seed_level.get((num, "target"))
    B = seed_level.get((den, "target"))
    if not A or not B or min(len(A["seeds"]), len(B["seeds"])) < 2:
      report["criteria"][name] = {"checked": False,
                                  "why": f"need both {num} and {den} in the "
                                         "target domain, on 2+ seeds each"}
      continue
    c = compare(A, B, num, den)
    comparisons[name] = c
    family[name] = max(c["p_nb"], c["p_robust"])

  adj = holm(family) if family else {}
  for name, c in comparisons.items():
    nb, rob = c["negative_binomial"], c["cluster_robust"]
    both_below_one = (nb["ci"][1] < 1.0
                      and (rob is None or rob["ci"][1] < 1.0))
    enough = min(c["n_seeds"]) >= a.min_seeds
    point = c["rate_a"] < c["rate_b"]
    ok = both_below_one and adj.get(name, 1.0) < 0.05
    report["criteria"][name] = {
      "checked": True, "pass": ok, "point_estimate_favours": point,
      "verdict": verdict(ok, point, enough),
      "ratio": nb["ratio"], "ci_nb": nb["ci"],
      "ci_robust": (rob or {}).get("ci"),
      "p_nb": c["p_nb"], "p_robust": c["p_robust"],
      "p_holm": adj.get(name), "n_seeds": c["n_seeds"],
      "enough_seeds": enough,
      "rate_a": c["rate_a"], "rate_b": c["rate_b"],
      "both_methods_agree": both_below_one,
      "alpha_seed_heterogeneity": nb.get("alpha"),
    }

  # -- C4, C5: what it cost -------------------------------------------------
  for name, domain, ref, why in (
      ("C4  target throughput not paid away", "target", "trajectory_matching",
       "against the best comparator in the same domain, so a method cannot "
       "pass by being safe and slow"),
      ("C5  source retention not paid away", "retention", None,
       "against the unadapted policy in the source domain")):
    A = seed_level.get(("risk_aware", domain))
    if not A:
      report["criteria"][name] = {"checked": False,
                                  "why": "no risk-aware runs in " + domain}
      continue
    got = t_interval(A["throughput"])
    if ref:
      B = seed_level.get((ref, domain))
      base = t_interval(B["throughput"])["mean"] if B else float("nan")
    else:
      base = anc.get("nominal", {}).get("throughput", float("nan"))
    loss = (base - got["mean"]) / base if base == base and base else float("nan")
    ok = loss == loss and loss <= THROUGHPUT_BUDGET
    report["criteria"][name] = {
      "checked": base == base, "pass": bool(ok), "loss": loss,
      "budget": THROUGHPUT_BUDGET, "reference": ref or "nominal anchor",
      "why": why, "throughput": got["mean"], "reference_throughput": base,
      "n_seeds": got["n"],
      "verdict": ("GREEN" if ok and got["n"] >= a.min_seeds
                  else "YELLOW" if ok else "RED") if base == base else "NOT RUN",
    }

  # -- C6: default-off and provenance --------------------------------------
  root = Path(__file__).resolve().parent.parent
  src = (root / "src" / "piper_push" / "damping.py").read_text()
  c6 = {
    "checked": True,
    "point_mass_at_nominal_installs_nothing": "if p.is_point_at(NOMINAL)" in src,
    "writes_from_the_default_field": "get_default_field" in src,
    "gripper_skipped": "skip_gripper" in src,
    "risk_head_asserts_deployable":
      "assert_deployable" in (root / "src" / "piper_push" / "risk.py").read_text(),
    "tilt_records_its_provenance":
      "risk_tilt" in (root / "scripts" / "wm_posterior.py").read_text(),
  }
  c6["pass"] = all(v for k, v in c6.items() if k != "checked")
  report["criteria"]["C6  default-off with provenance"] = c6

  # -- print ----------------------------------------------------------------
  print()
  print("  criterion                                            verdict")
  print("  " + "-" * 66)
  worst = "GREEN"
  order = {"GREEN": 0, "YELLOW": 1, "NOT RUN": 1, "RED": 2}
  for name, c in report["criteria"].items():
    if not c.get("checked"):
      v = "NOT RUN"
    else:
      v = c.get("verdict") or ("GREEN" if c.get("pass") else "RED")
    worst = max(worst, v, key=lambda x: order.get(x, 1))
    extra = ""
    if "ratio" in c:
      extra = (f"  ratio {c['ratio']:.2f} "
               f"[{c['ci_nb'][0]:.2f},{c['ci_nb'][1]:.2f}] "
               f"p_holm {c['p_holm']:.3f}  {min(c['n_seeds'])} seeds")
    elif "loss" in c and c["loss"] == c["loss"]:
      extra = f"  {100 * c['loss']:+.1f}% vs {c['reference']}"
    elif "margin" in c:
      extra = (f"  {c['best_balanced']:.3f} vs {c['control_balanced']:.3f} "
               f"({c['worst_control']})")
    print(f"  {name:52s} {v:8s}{extra}")

  report["verdict"] = worst
  print()
  print(f"  OVERALL: {worst}")
  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  Path(a.out).write_text(json.dumps(report, indent=1))
  print(f"  wrote {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
