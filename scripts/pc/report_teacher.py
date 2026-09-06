"""Tabulate one or more teacher evaluations written by scripts/pc/eval_teacher.sh.

    python scripts/pc/report_teacher.py results/pc/teacher_eval/v10c_sight_7400 results/pc/teacher_eval/v10c_nosight_7400

Per seed and as median [min, max]; the screening rules of the yf/pc brief are
applied and printed with the numbers they were applied to.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

SEEDS = (101, 202, 303)


def load(path):
  try:
    return json.load(open(path))
  except Exception:
    return None


def fmt(xs, nd=2):
  xs = [x for x in xs if x is not None and np.isfinite(x)]
  if not xs:
    return "-"
  if len(xs) == 1:
    return f"{xs[0]:.{nd}f}"
  return f"{np.median(xs):.{nd}f} [{min(xs):.{nd}f}, {max(xs):.{nd}f}]"


def arm(out):
  r = {"dir": out}
  man = load(os.path.join(out, "manifest.json")) or {}
  r["checkpoint"] = man.get("checkpoint")
  r["sha256"] = man.get("checkpoint_sha256")
  r["code_commit_local"] = man.get("code_commit_local")
  per = {}
  for s in SEEDS:
    acc = load(os.path.join(out, f"accept_s{s}.json"))
    end = load(os.path.join(out, f"endurance_s{s}.json"))
    full = load(os.path.join(out, f"endurance_fullreset_s{s}.json"))
    occ = load(os.path.join(out, f"occlusion_s{s}.json"))
    act = load(os.path.join(out, f"actions_s{s}.json"))
    d = {}
    if acc:
      m = acc["metrics"]
      d.update(placed_per_min=m.get("throughput_per_min"), success=m.get("success"), drops=m.get("drop_rate"),
               trips_per_arm_hour=m.get("trips_per_arm_hour"), p95_s=m.get("p95_s"), verdict=acc.get("verdict"),
               api=((acc.get("domain") or {}).get("weights") or {}).get("action_api", {}).get("status"),
               sensor=((acc.get("domain") or {}).get("sensor") or {}).get("setting"))
    if end:
      d.update(late_over_early=end.get("late_over_early"), early=end.get("early_per_min"), late=end.get("late_per_min"),
               stopped=f"{end.get('stopped_envs')}/{end.get('started_envs')}", jaw_stopped_mm=end.get("jaw_mm_stopped"),
               jaw_running_mm=end.get("jaw_mm_running"))
    if full:
      d.update(full_late_over_early=full.get("late_over_early"), full_early=full.get("early_per_min"), full_late=full.get("late_per_min"))
    if occ:
      d.update(approach_blocked=(occ.get("by_phase") or {}).get("approach", {}).get("blocked_rate"),
               engaged_blocked=(occ.get("engaged") or {}).get("blocked_rate"),
               engaged_fraction=(occ.get("engaged_meta") or occ.get("engaged") or {}).get("engaged_fraction"),
               occ_placed_per_min=occ.get("placed_per_min"), action_rate=occ.get("action_rate"))
    if act:
      d.update(mean_abs_da=act.get("mean_abs_da_all"), sat99=act.get("frac_sat99"), sat999=act.get("frac_sat999"),
               mean_abs_u=act.get("mean_abs_u"), max_abs_u=act.get("max_abs_u"), safe_env_fraction=act.get("safe_env_fraction"),
               object_lost_per_arm_min=(act.get("terminations_per_arm_minute") or {}).get("object_lost"),
               over_speed_per_arm_min=(act.get("terminations_per_arm_minute") or {}).get("over_speed"),
               nan_terms=(act.get("terminations") or {}).get("nan"), nonfinite=act.get("nonfinite_action_steps"),
               jaw_end_mm=act.get("jaw_end_mm_median"), jaw_end_p10_mm=act.get("jaw_end_mm_p10"))
    per[s] = d
  r["per_seed"] = per
  return r


def screen(r):
  per = r["per_seed"]
  g = lambda k: [per[s].get(k) for s in SEEDS]
  placed = g("placed_per_min"); loe = g("late_over_early")
  checks = {}
  checks["three_seeds_place"] = all(x is not None and x > 0 for x in placed) and all(
    per[s].get("late") not in (None, 0.0) and per[s]["late"] > 0 for s in SEEDS)
  checks["late_over_early_ge_0.85_median"] = (np.median([x for x in loe if x is not None]) >= 0.85) if any(x is not None for x in loe) else False
  checks["late_over_early_per_seed"] = loe
  jaw = g("jaw_stopped_mm")
  checks["no_jaw_latch"] = all((x is None) or (not np.isfinite(x)) or x > 5.0 for x in jaw)
  checks["no_nonfinite"] = all((per[s].get("nonfinite") or 0) == 0 and (per[s].get("nan_terms") or 0) == 0 for s in SEEDS)
  sat = g("sat999")
  checks["max_sat999_any_dim"] = max((max(x) for x in sat if x), default=None)
  checks["action_api_ok"] = all(per[s].get("api") == "ok" for s in SEEDS)
  checks["sensor_measured"] = all(per[s].get("sensor") == "measured" for s in SEEDS)
  vals = [x for x in placed if x is not None]
  checks["seed_spread_placed"] = (max(vals) - min(vals)) if vals else None
  checks["seed_spread_rel"] = ((max(vals) - min(vals)) / max(np.median(vals), 1e-9)) if vals else None
  checks["no_single_seed_fluke"] = (checks["seed_spread_rel"] is not None and checks["seed_spread_rel"] < 0.35)
  checks["usable_for_screening"] = all(checks[k] for k in ("three_seeds_place", "late_over_early_ge_0.85_median", "no_jaw_latch",
                                                            "no_nonfinite", "action_api_ok", "no_single_seed_fluke"))
  return checks


def main():
  dirs = [a for a in sys.argv[1:] if os.path.isdir(a)]
  if not dirs or len(dirs) != len(sys.argv[1:]):
    print(__doc__)
    raise SystemExit("usage: report_teacher.py <teacher_eval dir> [...]")
  rows = []
  for out in dirs:
    r = arm(out)
    r["screen"] = screen(r)
    rows.append(r)
  cols = [("placed/min", "placed_per_min"), ("success", "success"), ("drops", "drops"), ("trips/h", "trips_per_arm_hour"),
          ("early", "early"), ("late", "late"), ("late/early", "late_over_early"), ("full-reset l/e", "full_late_over_early"),
          ("full-reset early", "full_early"), ("stopped", "stopped"), ("jaw stop mm", "jaw_stopped_mm"),
          ("approach blocked", "approach_blocked"), ("engaged blocked", "engaged_blocked"), ("|da|", "mean_abs_da"),
          ("safe env", "safe_env_fraction"), ("lost/arm-min", "object_lost_per_arm_min"), ("overspeed/arm-min", "over_speed_per_arm_min")]
  for r in rows:
    print(f"\n=== {r['dir']}\n    {r['checkpoint']}\n    sha256 {r['sha256']}  code {r['code_commit_local']}")
    print(f"{'metric':18s} " + " ".join(f"{'s' + str(s):>10s}" for s in SEEDS) + "   median [min, max]")
    for label, key in cols:
      vals = [r["per_seed"][s].get(key) for s in SEEDS]
      cells = []
      for v in vals:
        if isinstance(v, str):
          cells.append(f"{v:>10s}")
        elif v is None:
          cells.append(f"{'-':>10s}")
        else:
          cells.append(f"{v:10.3f}")
      med = fmt([v for v in vals if not isinstance(v, str)], 3) if not any(isinstance(v, str) for v in vals) else ""
      print(f"{label:18s} " + " ".join(cells) + f"   {med}")
    for s in SEEDS:
      d = r["per_seed"][s]
      if d.get("sat99"):
        print(f"  s{s} sat>0.99 per dim {[round(x, 3) for x in d['sat99']]}  sat>0.999 {[round(x, 3) for x in d['sat999']]}  mean|u| {[round(x, 2) for x in d['mean_abs_u']]}  max|u| {[round(x, 1) for x in d['max_abs_u']]}")
    print("  screen:", json.dumps(r["screen"], default=str))
  json.dump(rows, open("results/pc/teacher_eval/report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
  main()
