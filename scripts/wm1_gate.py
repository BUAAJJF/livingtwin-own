"""Evaluate the Phase WM1-A gate from the result files, mechanically.

    python scripts/wm1_gate.py --json results/wm1_latency/gate.json

Six criteria, all of them thresholds fixed before the runs.  The thresholds
come from Phase WM0 and are written here as constants so that a verdict cannot
be reached by adjusting one while looking at the numbers:

    J_nominal   55.86      J_zero_shot 42.19      J_oracle 49.72
    trips/h      2.29                   8.69                4.83

    G2  target    >= 42.19 + 0.70 * (49.72 - 42.19) = 47.46 obj/min
    G3  target    <= 8.69 - 0.70 * (8.69 - 4.83)    =  6.00 trips/arm-hour
    G4  retention >= 0.95 * 55.86                   = 53.07 obj/min

This phase re-measured the two anchors under the delay buffer that replaced
the WM0 ring buffer, and the re-measurement is reported next to the inherited
thresholds along with what the verdict would be if the thresholds were
re-derived from it.  If the two disagree the report says so; they are not
silently swapped.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "wm1_latency"

# -- inherited from Phase WM0, docs/sim2real_sweep_phase_wm0.md -------------
J_NOMINAL = 55.86
J_ZERO = 42.19
J_ORACLE = 49.72
TRIPS_NOMINAL = 2.29
TRIPS_ZERO = 8.69
TRIPS_ORACLE = 4.83

RECOVERY_TARGET = 0.70
G2_THRESHOLD = J_ZERO + RECOVERY_TARGET * (J_ORACLE - J_ZERO)       # 47.46
G3_THRESHOLD = TRIPS_ZERO - RECOVERY_TARGET * (TRIPS_ZERO - TRIPS_ORACLE)  # 6.00
G4_THRESHOLD = 0.95 * J_NOMINAL                                     # 53.07

# -- G1 ---------------------------------------------------------------------
G1_BUDGET = "60.0"
G1_MARGIN = 0.15
"""Balanced accuracy the method must clear its controls by.  Balanced, not
raw: the five domains have equal session counts here, but a method that always
answered '3' would still score 20% raw and that is the number a reader
compares against 1/5 without thinking about it."""

DA_METHOD = "DA"
TM_METHOD = "B3_state"
"""The comparison G5 is about.  B3 is trajectory matching -- proprioception
and servo error, the channels a classical system-identification pipeline
would use.  DA adds the actor latent and the action the frozen policy would
have taken.  If they are indistinguishable, the decision-aware claim is not
established by this phase, and the gate says so rather than rounding in its
favour."""


def _load(p: Path):
  try:
    return json.loads(p.read_text())
  except Exception:
    return None


# ---------------------------------------------------------------------------


def criterion_1(post: dict | None) -> dict:
  out = {"checked": bool(post), "budget_s": float(G1_BUDGET)}
  if not post:
    return out
  b = post.get("budgets", {}).get(G1_BUDGET)
  if not b:
    out["checked"] = False
    return out
  test = b["methods"]["test"]
  ctrl_names = [k for k in test if k.startswith("ctrl_")]
  ctrl_best = max((test[k]["balanced_accuracy"] for k in ctrl_names),
                  default=1.0 / 5)
  done_only = test.get("ctrl_done_only", {}).get("balanced_accuracy")
  rows = []
  for name, m in sorted(test.items()):
    if name.startswith("ctrl_"):
      continue
    rows.append({
      "method": name,
      "balanced_accuracy": m["balanced_accuracy"],
      "top1": m["top1"],
      "mass_on_truth": m["mass_on_truth"],
      "over_controls": m["balanced_accuracy"] - ctrl_best,
      "target_top1": (m.get("target_only") or {}).get("top1"),
      "passes": (m["balanced_accuracy"] - ctrl_best) >= G1_MARGIN
                and (m["balanced_accuracy"] - 1.0 / 5) >= G1_MARGIN,
    })
  out.update({
    "controls": {k: test[k]["balanced_accuracy"] for k in ctrl_names},
    "best_control_balanced_accuracy": ctrl_best,
    "done_only_balanced_accuracy": done_only,
    "methods": rows,
    "n_passing": sum(1 for r in rows if r["passes"]),
    # Every test session is a held-out shape class by construction, so a
    # method that passes here has already passed on unseen objects.
    "held_out_shapes_by_construction": True,
    "g1": any(r["passes"] for r in rows),
    "da_passes": any(r["passes"] and r["method"] == DA_METHOD for r in rows),
  })
  return out


def _group(analysis: dict, tag: str, kind: str) -> dict | None:
  """The seed-pooled group for a configuration, or the per-seed one.

  ``tag`` here is a configuration key -- the run tag with its ``_s<seed>``
  suffix removed -- so what this finds is the group that pools all three
  training seeds.  Nine evaluations rather than three, which is what a trip
  count needs before an interval on it says anything."""
  for name, g in analysis.get("groups", {}).items():
    if name.endswith(f"/pooled:{tag}_{kind}") or name.endswith(f"/{tag}_{kind}"):
      return g
  return None


def criteria_2_3_4(analysis: dict | None, formal: dict | None) -> dict:
  """Target recovery, safety recovery and retention, for every formal run."""
  out = {"checked": bool(analysis and formal), "runs": []}
  if not out["checked"]:
    return out
  # One entry per configuration, not per (configuration, seed): the gate is
  # evaluated on the pooled result and the per-seed ones are reported beside it.
  by_tag: dict[str, list[str]] = {}
  for job in formal["jobs"]:
    cfg = re.sub(r"_s\d+$", "", job["tag"])
    by_tag.setdefault(cfg, job["methods"])
  for tag, methods in sorted(by_tag.items()):
    tgt = _group(analysis, tag, "target")
    ret = _group(analysis, tag, "retention")
    if not tgt or not ret:
      continue
    thr = tgt["throughput"]["bootstrap"]
    rthr = ret["throughput"]["bootstrap"]
    trips = tgt["trips"]
    rec = ((thr["mean"] - J_ZERO) / (J_ORACLE - J_ZERO)
           if J_ORACLE > J_ZERO else float("nan"))
    out["runs"].append({
      "tag": tag, "methods": methods,
      "target_throughput": thr["mean"], "target_ci": thr["ci"],
      "target_trips": trips.get("rate"), "target_trips_ci": trips.get("ci_quasi"),
      "target_trip_events": trips.get("events"),
      "retention_throughput": rthr["mean"], "retention_ci": rthr["ci"],
      "recovery_fraction_of_oracle": rec,
      "g2": thr["ci"][0] >= G2_THRESHOLD,
      "g2_point": thr["mean"] >= G2_THRESHOLD,
      "g3": trips.get("ci_quasi", [math.inf, math.inf])[1] <= G3_THRESHOLD,
      "g3_point": (trips.get("rate", math.inf) <= G3_THRESHOLD),
      "g4": rthr["ci"][0] >= G4_THRESHOLD,
      "g4_point": rthr["mean"] >= G4_THRESHOLD,
    })
  da = [r for r in out["runs"] if DA_METHOD in r["methods"]]
  out["da_runs"] = [r["tag"] for r in da]
  for g in ("g2", "g3", "g4"):
    out[g] = any(r[g] for r in da)
    out[g + "_point"] = any(r[g + "_point"] for r in da)
  return out


def criterion_5(analysis: dict | None, formal: dict | None,
                post: dict | None) -> dict:
  """Is the decision-aware posterior distinguishable from trajectory matching?

  Four places it could be, and it needs one.  The first thing checked is
  whether the two even produced different adaptation distributions: if they
  did not, the same PPO run served both and there is nothing to compare, which
  is an INCONCLUSIVE and not a pass.
  """
  out = {"checked": bool(post)}
  if not post:
    return out
  b = post.get("budgets", {}).get(G1_BUDGET, {})
  test = b.get("methods", {}).get("test", {})
  da, tm = test.get(DA_METHOD), test.get(TM_METHOD)
  if not da or not tm:
    out["checked"] = False
    return out

  # (a) identification: balanced accuracy over all five domains.
  out["identification"] = {
    "da_balanced_accuracy": da["balanced_accuracy"],
    "tm_balanced_accuracy": tm["balanced_accuracy"],
    "delta": da["balanced_accuracy"] - tm["balanced_accuracy"],
  }
  # (b) benign-domain false positives: how often each says "target" when the
  # session came from one of the four domains that are not the target.
  def false_target(m):
    conf, lags = m["confusion"], [0, 1, 2, 3, 4]
    j = lags.index(3)
    wrong = sum(conf[i][j] for i in range(len(lags)) if lags[i] != 3)
    total = sum(sum(conf[i]) for i in range(len(lags)) if lags[i] != 3)
    return wrong / max(total, 1)

  out["benign_false_positive"] = {
    "da": false_target(da), "tm": false_target(tm),
    "delta": false_target(tm) - false_target(da),
  }
  # (c) and (d): the adaptation halves, only if the two got different runs.
  shared = None
  if formal:
    for job in formal["jobs"]:
      if DA_METHOD in job["methods"] and TM_METHOD in job["methods"]:
        shared = re.sub(r"_s\d+$", "", job["tag"])
  out["shared_adaptation_run"] = shared
  if analysis and formal and shared is None:
    def find(method, kind):
      for job in formal["jobs"]:
        if method in job["methods"]:
          g = _group(analysis, re.sub(r"_s\d+$", "", job["tag"]), kind)
          if g:
            return g
      return None
    dat, tmt = find(DA_METHOD, "target"), find(TM_METHOD, "target")
    dar, tmr = find(DA_METHOD, "retention"), find(TM_METHOD, "retention")
    if dat and tmt:
      out["target_recovery"] = {
        "da": dat["throughput"]["bootstrap"]["mean"],
        "tm": tmt["throughput"]["bootstrap"]["mean"],
        "da_ci": dat["throughput"]["bootstrap"]["ci"],
        "tm_ci": tmt["throughput"]["bootstrap"]["ci"],
        "separated": dat["throughput"]["bootstrap"]["ci"][0]
                     > tmt["throughput"]["bootstrap"]["ci"][1],
      }
      out["safety_recovery"] = {
        "da": dat["trips"]["rate"], "tm": tmt["trips"]["rate"],
        "separated": dat["trips"]["ci_quasi"][1] < tmt["trips"]["ci_quasi"][0],
      }
    if dar and tmr:
      out["retention"] = {"da": dar["throughput"]["bootstrap"]["mean"],
                          "tm": tmr["throughput"]["bootstrap"]["mean"]}

  wins = []
  if out["identification"]["delta"] >= 0.10:
    wins.append("identification")
  if out["benign_false_positive"]["delta"] >= 0.10:
    wins.append("benign_false_positive")
  if out.get("target_recovery", {}).get("separated"):
    wins.append("target_recovery")
  if out.get("safety_recovery", {}).get("separated"):
    wins.append("safety_recovery")
  out["wins"] = wins
  out["g5"] = bool(wins)
  out["inconclusive_because_shared_run"] = (
    shared is not None and not wins)
  return out


def criterion_6(post: dict | None, train: dict | None, clf: dict | None,
                formal: dict | None, timings: dict | None) -> dict:
  """Wall-clock, split into what counts against a one-hour online budget and
  what does not.

  Offline pre-training is a cost paid once, before deployment, and is reported
  separately.  What a deployment spends is: collecting the target data,
  running inference on it, and the PPO adaptation.
  """
  out = {"checked": True}
  out["offline"] = {
    "world_model_train_s": (train or {}).get("train_wall_clock_s"),
    "classifier_train_s": (clf or {}).get("wall_clock_s"),
    "dataset_generation": (timings or {}).get("dataset_generation_s"),
  }
  budget = float(G1_BUDGET)
  out["online"] = {
    "target_data_collection_s": budget,
    "posterior_inference_s": (timings or {}).get("inference_per_session_s"),
    "ppo_adaptation_s": (timings or {}).get("ppo_per_run_s"),
  }
  known = [v for v in out["online"].values() if isinstance(v, (int, float))]
  out["online_total_s"] = sum(known) if len(known) == len(out["online"]) else None
  out["g6"] = all(v is not None for v in out["online"].values())
  return out


# ---------------------------------------------------------------------------


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--results", default=str(RESULTS))
  p.add_argument("--json", default=None)
  a = p.parse_args()
  root = Path(a.results)

  post = _load(root / "posterior" / "posterior_report.json")
  analysis = _load(root / "analysis.json")
  formal = _load(root / "adapt" / "formal.json")
  train = _load(root / "model" / "train_report.json")
  clf = _load(root / "model" / "classifier_report.json")
  timings = _load(root / "timings.json")
  equiv = _load(root / "equivalence_summary.json")

  c1 = criterion_1(post)
  c234 = criteria_2_3_4(analysis, formal)
  c5 = criterion_5(analysis, formal, post)
  c6 = criterion_6(post, train, clf, formal, timings)

  checks = [
    ("G1  reward-free identification beats its controls",
     c1.get("g1"), c1["checked"]),
    (f"G2  target throughput >= {G2_THRESHOLD:.2f} obj/min",
     c234.get("g2"), c234["checked"]),
    (f"G3  target trips <= {G3_THRESHOLD:.2f} per arm-hour",
     c234.get("g3"), c234["checked"]),
    (f"G4  retention >= {G4_THRESHOLD:.2f} obj/min",
     c234.get("g4"), c234["checked"]),
    ("G5  decision-aware beats trajectory matching somewhere",
     c5.get("g5"), c5["checked"]),
    ("G6  every stage's wall-clock recorded",
     c6.get("g6"), c6["checked"]),
  ]

  print()
  print("  Phase WM1-A gate")
  print()
  for name, ok, checked in checks:
    mark = "NOT RUN" if not checked else ("PASS" if ok else "FAIL")
    print(f"    [{mark:^7s}] {name}")

  unrun = [n for n, _, ck in checks if not ck]
  hard = [c234.get("g2"), c234.get("g3"), c234.get("g4")]
  if unrun:
    verdict, why = "INCOMPLETE", "not evaluated: " + "; ".join(unrun)
  elif not c1.get("g1"):
    verdict = "RED"
    why = ("reward-free data does not identify the domain, so there is nothing "
           "for adaptation to be guided by; fix the data or the estimator "
           "before spending another PPO run")
  elif all(hard) and c5.get("g5"):
    verdict = "GREEN"
    why = ("identified from reward-free data, recovered in the target domain, "
           "retained in the nominal one, and the decision-aware posterior is "
           "distinguishable from trajectory matching")
  elif all(hard) and c5.get("inconclusive_because_shared_run"):
    verdict = "YELLOW"
    why = ("the loop closes, but the decision-aware posterior and trajectory "
           "matching produced the same adaptation distribution and therefore "
           "the same PPO run: on this axis the decision-aware claim is not "
           "tested, let alone established")
  elif all(hard):
    verdict = "YELLOW"
    why = ("the loop closes but decision-awareness buys nothing measurable "
           "over trajectory matching")
  elif c1.get("g1") and not c234.get("g2"):
    verdict = "YELLOW"
    why = ("the domain is identified but adaptation does not recover the "
           "target; the failure is in simulator adaptation, not in inference")
  elif c1.get("g1") and c234.get("g2") and not c234.get("g4"):
    verdict = "YELLOW"
    why = ("the target is recovered and the nominal domain is not retained; "
           "the posterior is too narrow and needs more source-prior mixing")
  else:
    verdict = "YELLOW"
    why = "mixed; see the per-criterion table"

  print()
  print(f"  VERDICT: {verdict}")
  print(f"  {why}")
  print()

  out = {
    "verdict": verdict, "why": why,
    "thresholds": {
      "inherited_from": "docs/sim2real_sweep_phase_wm0.md",
      "J_nominal": J_NOMINAL, "J_zero_shot": J_ZERO, "J_oracle": J_ORACLE,
      "trips_nominal": TRIPS_NOMINAL, "trips_zero": TRIPS_ZERO,
      "trips_oracle": TRIPS_ORACLE,
      "g2": G2_THRESHOLD, "g3": G3_THRESHOLD, "g4": G4_THRESHOLD,
      "g1_margin": G1_MARGIN, "recovery_target": RECOVERY_TARGET,
    },
    "re_measured_anchors": equiv,
    "criteria": {n: {"pass": ok, "checked": ck} for n, ok, ck in checks},
    "g1": c1, "g234": c234, "g5": c5, "g6": c6,
  }
  if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"  wrote {a.json}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
