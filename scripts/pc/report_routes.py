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
    r["p2"] = act.get("p2")
  occ = load(os.path.join(d, "occlusion_final_s101.json"))
  if occ:
    r["approach_blocked"] = (occ.get("by_phase") or {}).get("approach", {}).get("blocked_rate")
    r["engaged_blocked"] = (occ.get("engaged") or {}).get("blocked_rate")
  r["export"] = export_status(d)
  r["oracle_only"] = bool(man.get("oracle_only", False))
  # -- second generation: initiation (36 s, three seeds) and the long no-reset run
  ini = {s: load(os.path.join(d, f"initiation_final_s{s}.json")) for s in SEEDS}
  lng = {s: load(os.path.join(d, f"long_final_s{s}.json")) for s in SEEDS}
  def pick(dd, *keys):
    out = []
    for s in SEEDS:
      x = dd[s]
      for k in keys:
        x = x.get(k) if isinstance(x, dict) else None
      out.append(x)
    return out
  if any(ini.values()):
    r["ini_attempts_per_min"] = pick(ini, "attempts_per_min")
    r["ini_success_per_attempt"] = pick(ini, "success_per_attempt")
    r["ini_placed_per_min"] = pick(ini, "placed_per_min")
    r["ini_loe_attempts"] = pick(ini, "late_over_early_attempts")
    r["ini_wait_success_p50"] = pick(ini, "wait_after_success_s", "p50")
    r["ini_wait_success_p90"] = pick(ini, "wait_after_success_s", "p90")
    r["ini_first_attempt_p50"] = pick(ini, "time_to_first_attempt_s", "p50")
    r["ini_stalled_fraction"] = pick(ini, "stalls", "stalled_step_fraction")
    r["ini_stalls_per_min"] = pick(ini, "stalls", "per_arm_minute")
    r["ini_target_pts_approach_re"] = pick(ini, "by_phase", "approach_re", "target_points_mean")
    r["ini_target_pts_approach_first"] = pick(ini, "by_phase", "approach_first", "target_points_mean")
    r["ini_target_pts_engaged"] = pick(ini, "by_phase", "engaged", "target_points_mean")
    r["ini_zero_target_approach_re"] = pick(ini, "by_phase", "approach_re", "frac_frames_zero_target_points")
    r["ini_drops"] = pick(ini, "drops")
  if any(lng.values()):
    r["long_seconds"] = pick(lng, "seconds")
    r["long_placed_per_min"] = pick(lng, "placed_per_min")
    r["long_attempts_per_min"] = pick(lng, "attempts_per_min")
    r["long_loe_placed"] = pick(lng, "late_over_early_placed")
    r["long_loe_attempts"] = pick(lng, "late_over_early_attempts")
    r["long_stalled_fraction"] = pick(lng, "stalls", "stalled_step_fraction")
    r["long_resets"] = pick(lng, "resets")
    r["long_terminations"] = pick(lng, "terminations")
    # first and last 36 s of the long run, from the placements per window
    def edge(dd):
      out = []
      for s in SEEDS:
        x = dd[s]
        if not x:
          out.append(None); continue
        pw, w, dt = x["placed_per_window"], x["window"], x["dt"]
        k = int(round(36.0 / (w * dt)))
        if len(pw) < 2 * k:
          out.append(None); continue
        e, l = float(np.mean(pw[:k])), float(np.mean(pw[-k:]))
        out.append(l / e if e > 1e-9 else None)
      return out
    r["long_last36_over_first36"] = edge(lng)
  tim = os.path.join(d, "timing.jsonl")
  if os.path.exists(tim):
    r["timing_s"] = {}
    for line in open(tim):
      try:
        j = json.loads(line); r["timing_s"][j["stage"]] = j["seconds"]
      except Exception:
        pass
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
  g["not_oracle"] = not r["oracle_only"]
  g["deployable"] = all(bool(g[k]) for k in ("three_seeds_place", "throughput_ge_70pct_teacher", "late_over_early_ge_0.85",
                                              "no_jaw_latch", "no_nan", "heldout_nonzero", "export_ok", "action_api_ok", "not_oracle"))
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
    if r.get("p2"):
      print(f"   p2 {json.dumps(r['p2'])}")
    if r.get("ini_attempts_per_min"):
      print(f"   initiation (36 s): attempts/min {med(r['ini_attempts_per_min'])}  success/attempt {med(r['ini_success_per_attempt'], 3)}  "
            f"l/e attempts {med(r['ini_loe_attempts'])}  wait after success p50 {med(r['ini_wait_success_p50'])} s p90 {med(r['ini_wait_success_p90'])} s  "
            f"first attempt p50 {med(r['ini_first_attempt_p50'])} s")
      print(f"   stalls: stalled fraction {med(r['ini_stalled_fraction'], 3)}  per arm-min {med(r['ini_stalls_per_min'])}  drops {r['ini_drops']}")
      print(f"   target points (sampled, fresh): approach_first {med(r['ini_target_pts_approach_first'], 1)}  approach_re {med(r['ini_target_pts_approach_re'], 1)}  "
            f"engaged {med(r['ini_target_pts_engaged'], 1)}  zero-target frames on re-approach {med(r['ini_zero_target_approach_re'], 3)}")
    if r.get("long_placed_per_min"):
      print(f"   long no-reset ({r['long_seconds'][0]} s): placed/min {med(r['long_placed_per_min'])}  attempts/min {med(r['long_attempts_per_min'])}  "
            f"l/e placed {med(r['long_loe_placed'])}  l/e attempts {med(r['long_loe_attempts'])}  last36/first36 {med(r['long_last36_over_first36'])}  "
            f"stalled {med(r['long_stalled_fraction'], 3)}  resets {r['long_resets']}  terminations {r['long_terminations']}")
    if r.get("timing_s"):
      print(f"   timing s {json.dumps(r['timing_s'])}")
    if r.get("oracle_only"):
      print("   ORACLE-ONLY route: simulation diagnostic, never a deployment candidate")
    print(f"   gate {json.dumps(r['gate'], default=str)}")
  if a.json:
    json.dump(rows, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
  main()
