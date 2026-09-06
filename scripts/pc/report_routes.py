"""One table over the vision routes written by scripts/pc/run_route.sh, plus the deployment gate.

    python scripts/pc/report_routes.py results/pc/routes/pc_screen_*_20260905T1842 --teacher-placed 19.7

Per route: the student after distillation (one seed), the final policy after
fine-tuning (three seeds, median [min, max]), held-out objects, actions,
occlusion, export, the smoke's inference cost.  The gate is the brief's:
three seeds all placing, >= 70% of the teacher's throughput, late/early >= 0.85,
no jaw latch, no NaN, held-out non-zero, export agreeing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

SEEDS = (101, 202, 303)


def load(p):
  try:
    return json.load(open(p))
  except Exception:
    return None


def med(xs, nd=2):
  xs = [x for x in xs if x is not None and np.isfinite(x)]
  if not xs:
    return "-"
  if len(xs) == 1:
    return f"{xs[0]:.{nd}f}"
  return f"{np.median(xs):.{nd}f} [{min(xs):.{nd}f}, {max(xs):.{nd}f}]"


def export_status(d):
  log = os.path.join(d, "export.log")
  if not os.path.exists(log):
    return "no export"
  txt = open(log, errors="replace").read()
  m = re.search(r"max \|torch - onnx\| over \d+ steps: ([0-9.e+-]+)", txt)
  m2 = re.search(r"max \|torch - jit \| over \d+ steps: ([0-9.e+-]+)", txt)
  if "OK" in txt.splitlines()[-1:] or txt.rstrip().endswith("OK"):
    return f"OK onnx {m.group(1) if m else '?'} jit {m2.group(1) if m2 else '?'}"
  if "MISMATCH" in txt:
    return f"MISMATCH onnx {m.group(1) if m else '?'} jit {m2.group(1) if m2 else '?'}"
  if "Traceback" in txt:
    last = [l for l in txt.splitlines() if l.strip()][-1]
    return f"FAILED: {last[:90]}"
  return "unknown"


def route(d, teacher_placed):
  man = load(os.path.join(d, "manifest.json")) or {}
  smoke = load(os.path.join(d, "smoke.json")) or {}
  r = {"dir": d, "route": man.get("route"), "gpu": man.get("gpu"), "teacher_sha256": (man.get("teacher_sha256") or "")[:16],
       "distill_iters": man.get("distill_iters"), "finetune_iters": man.get("finetune_iters"), "vision_envs": man.get("vision_envs"),
       "encoders": smoke.get("encoders"), "infer_ms_single_env": smoke.get("infer_ms_single_env"),
       "learn_s_per_iter_8env": smoke.get("learn_s_per_iter"), "fresh_fraction": smoke.get("fresh_fraction")}
  st = load(os.path.join(d, "accept_student_s101.json"))
  se = load(os.path.join(d, "endurance_student_s101.json"))
  r["student_placed"] = st["metrics"]["throughput_per_min"] if st else None
  r["student_success"] = st["metrics"]["success"] if st else None
  r["student_loe"] = se.get("late_over_early") if se else None
  fin = {s: load(os.path.join(d, f"accept_final_s{s}.json")) for s in SEEDS}
  end = {s: load(os.path.join(d, f"endurance_final_s{s}.json")) for s in SEEDS}
  r["final_placed"] = [fin[s]["metrics"]["throughput_per_min"] if fin[s] else None for s in SEEDS]
  r["final_success"] = [fin[s]["metrics"]["success"] if fin[s] else None for s in SEEDS]
  r["final_drops"] = [fin[s]["metrics"]["drop_rate"] if fin[s] else None for s in SEEDS]
  r["final_trips"] = [fin[s]["metrics"]["trips_per_arm_hour"] if fin[s] else None for s in SEEDS]
  r["final_loe"] = [end[s].get("late_over_early") if end[s] else None for s in SEEDS]
  r["final_early"] = [end[s].get("early_per_min") if end[s] else None for s in SEEDS]
  r["final_late"] = [end[s].get("late_per_min") if end[s] else None for s in SEEDS]
  r["final_jaw_stopped"] = [end[s].get("jaw_mm_stopped") if end[s] else None for s in SEEDS]
  r["final_api"] = [((fin[s].get("domain") or {}).get("weights") or {}).get("action_api", {}).get("status") if fin[s] else None for s in SEEDS]
  ho = load(os.path.join(d, "accept_heldout_s101.json"))
  r["heldout_placed"] = ho["metrics"]["throughput_per_min"] if ho else None
  r["heldout_success"] = ho["metrics"]["success"] if ho else None
  act = load(os.path.join(d, "actions_final_s101.json"))
  if act:
    r["mean_abs_da"] = act.get("mean_abs_da_all"); r["sat999_max"] = max(act.get("frac_sat999") or [0])
    r["nonfinite"] = act.get("nonfinite_action_steps"); r["nan_terms"] = (act.get("terminations") or {}).get("nan")
    r["safe_env"] = act.get("safe_env_fraction"); r["jaw_end_mm"] = act.get("jaw_end_mm_median")
    t = act.get("terminations_per_arm_minute") or {}
    r["object_lost_per_min"] = t.get("object_lost"); r["over_speed_per_min"] = t.get("over_speed")
  occ = load(os.path.join(d, "occlusion_final_s101.json"))
  if occ:
    r["approach_blocked"] = (occ.get("by_phase") or {}).get("approach", {}).get("blocked_rate")
    r["engaged_blocked"] = (occ.get("engaged") or {}).get("blocked_rate")
  r["export"] = export_status(d)
  # -- gate
  placed = [x for x in r["final_placed"] if x is not None]
  loe = [x for x in r["final_loe"] if x is not None and np.isfinite(x)]
  g = {}
  g["three_seeds_place"] = len(placed) == 3 and all(x > 0 for x in placed) and all((r["final_late"][i] or 0) > 0 for i in range(3))
  g["throughput_ge_70pct_teacher"] = bool(placed) and np.median(placed) >= 0.70 * teacher_placed if teacher_placed else None
  g["late_over_early_ge_0.85"] = bool(loe) and np.median(loe) >= 0.85
  g["no_jaw_latch"] = all((x is None) or (not np.isfinite(x)) or x > 5.0 for x in r["final_jaw_stopped"])
  g["no_nan"] = (r.get("nonfinite") or 0) == 0 and (r.get("nan_terms") or 0) == 0
  g["heldout_nonzero"] = (r.get("heldout_placed") or 0) > 0
  g["export_ok"] = r["export"].startswith("OK")
  g["action_api_ok"] = all(x == "ok" for x in r["final_api"])
  g["seed_spread_rel"] = ((max(placed) - min(placed)) / max(np.median(placed), 1e-9)) if placed else None
  g["deployable"] = all(bool(g[k]) for k in ("three_seeds_place", "throughput_ge_70pct_teacher", "late_over_early_ge_0.85",
                                              "no_jaw_latch", "no_nan", "heldout_nonzero", "export_ok", "action_api_ok"))
  r["gate"] = g
  return r


def main():
  p = argparse.ArgumentParser()
  p.add_argument("dirs", nargs="+")
  p.add_argument("--teacher-placed", type=float, default=None, help="the chosen teacher's median placed/min")
  p.add_argument("--json", default=None)
  a = p.parse_args()
  rows = [route(d, a.teacher_placed) for d in a.dirs]
  print(f"{'route':6s} {'student/min':>11s} {'stud l/e':>8s} | {'final placed/min':>24s} {'success':>18s} {'late/early':>22s} {'held-out':>9s} {'|da|':>6s} {'blocked eng':>11s} {'export':>18s} {'GO?':>4s}")
  for r in rows:
    print(f"{str(r['route']):6s} {str(round(r['student_placed'], 2)) if r['student_placed'] is not None else '-':>11s} "
          f"{str(round(r['student_loe'], 2)) if r['student_loe'] is not None else '-':>8s} | {med(r['final_placed']):>24s} {med(r['final_success'], 3):>18s} "
          f"{med(r['final_loe']):>22s} {str(round(r['heldout_placed'], 2)) if r['heldout_placed'] is not None else '-':>9s} "
          f"{str(round(r['mean_abs_da'], 3)) if r.get('mean_abs_da') is not None else '-':>6s} "
          f"{str(round(r['engaged_blocked'], 3)) if r.get('engaged_blocked') is not None else '-':>11s} {r['export'][:18]:>18s} {'GO' if r['gate']['deployable'] else 'no':>4s}")
  for r in rows:
    print(f"\n-- {r['route']} {r['dir']}\n   encoders {r['encoders']}  infer {r['infer_ms_single_env']} ms/env  teacher {r['teacher_sha256']}  budget {r['distill_iters']}+{r['finetune_iters']} @ {r['vision_envs']} envs")
    print(f"   final per seed: placed {r['final_placed']} success {r['final_success']} drops {r['final_drops']} trips/h {r['final_trips']}")
    print(f"   endurance per seed: early {r['final_early']} late {r['final_late']} l/e {r['final_loe']} jaw_stopped {r['final_jaw_stopped']}")
    print(f"   held-out placed {r['heldout_placed']} success {r['heldout_success']}  safe_env {r.get('safe_env')}  lost/min {r.get('object_lost_per_min')}  overspeed/min {r.get('over_speed_per_min')}  sat999 {r.get('sat999_max')}  nonfinite {r.get('nonfinite')}")
    print(f"   occlusion approach {r.get('approach_blocked')} engaged {r.get('engaged_blocked')}   export {r['export']}")
    print(f"   gate {json.dumps(r['gate'], default=str)}")
  if a.json:
    json.dump(rows, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
  main()
