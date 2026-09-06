"""E0 against E2, on the rows the second generation is judged on, with the pre-registered rule applied.

    python scripts/pc/report_gen2.py results/pc/routes/pc_gen2_P1BZ_<stamp> results/pc/routes/pc_gen2_P1BT_<stamp> \\
        --teacher-ref results/pc/gen2/teacher_v11_<stamp> --gen1 results/pc/gen2/audit --json results/pc/gen2/compare.json

Per route: the final policy over three evaluation seeds (median [min, max]) on the accept ruler,
the endurance ruler, the initiation ruler (36 s) and the 180 s no-reset run; the teacher under the
same initiation ruler beside them; the gen-1 P1B audit for scale.  The rule from
results/pc/gen2/REPORT.md section 4: "clearly better" = the three-seed intervals do not overlap AND
the difference is at least 30 % of E0's value, on grasps per attempt and on late/early (raw and
live).  The script prints which of the four outcomes the numbers fall into; it does not decide
anything else.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

SEEDS = (101, 202, 303)


def load(p):
  try:
    return json.load(open(p))
  except Exception:
    return None


def dig(d, *keys):
  for k in keys:
    d = d.get(k) if isinstance(d, dict) else None
  return d


def stats(xs):
  xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
  if not xs:
    return None
  return {"median": float(np.median(xs)), "min": min(xs), "max": max(xs), "n": len(xs)}


def fmt(s, nd=2):
  if not s:
    return "-"
  if s["n"] == 1:
    return f"{s['median']:.{nd}f}"
  return f"{s['median']:.{nd}f} [{s['min']:.{nd}f}, {s['max']:.{nd}f}]"


def route_rows(d, prefix_ini="initiation_final", prefix_long="long_final", prefix_acc="accept_final", prefix_end="endurance_final"):
  acc = [load(os.path.join(d, f"{prefix_acc}_s{s}.json")) for s in SEEDS]
  end = [load(os.path.join(d, f"{prefix_end}_s{s}.json")) for s in SEEDS]
  ini = [load(os.path.join(d, f"{prefix_ini}_s{s}.json")) for s in SEEDS]
  lng = [load(os.path.join(d, f"{prefix_long}_s{s}.json")) for s in SEEDS]
  g = lambda lst, *k: stats([dig(x, *k) for x in lst if x])
  rows = {
    "accept placed/min": g(acc, "metrics", "throughput_per_min"),
    "accept success": g(acc, "metrics", "success"),
    "accept drop rate": g(acc, "metrics", "drop_rate"),
    "accept trips/h": g(acc, "metrics", "trips_per_arm_hour"),
    "endurance l/e (24 s)": g(end, "late_over_early"),
    "endurance jaw stopped mm": g(end, "jaw_mm_stopped"),
    "ini placed/min": g(ini, "placed_per_min"),
    "ini placed/min live": g(ini, "placed_per_live_min"),
    "ini attempts/min": g(ini, "attempts_per_min"),
    "ini grasps/min": g(ini, "grasps_per_min"),
    "ini grasps per attempt": stats([(x["grasps_per_min"] / x["attempts_per_min"]) if x and x.get("attempts_per_min") else None for x in ini]),
    "ini success per grasp": g(ini, "success_per_grasp"),
    "ini l/e placed": g(ini, "late_over_early_placed"),
    "ini l/e placed live": g(ini, "late_over_early_placed_live"),
    "ini l/e attempts": g(ini, "late_over_early_attempts"),
    "ini wait after success p50 s": g(ini, "wait_after_success_s", "p50"),
    "ini stalled fraction": g(ini, "stalls", "stalled_step_fraction"),
    "ini envs stalled at end": g(ini, "end_state_env_fraction", "stalled"),
    "ini stuck-object fraction": g(ini, "stuck_object", "step_fraction"),
    "ini engaged fraction": g(ini, "by_phase", "engaged", "steps_fraction"),
    "ini carry fraction": g(ini, "by_phase", "carry", "steps_fraction"),
    "ini jaw cmd engaged mm": g(ini, "by_phase", "engaged", "jaw_cmd_mm"),
    "ini target pts engaged": g(ini, "by_phase", "engaged", "target_points_mean"),
    "ini zero-target frames engaged": g(ini, "by_phase", "engaged", "frac_frames_zero_target_points"),
    "ini drops": g(ini, "drops"),
    "ini object_lost/min": g(ini, "terminations_per_arm_minute", "object_lost"),
    "long placed/min": g(lng, "placed_per_min"),
    "long placed/min live": g(lng, "placed_per_live_min"),
    "long l/e placed": g(lng, "late_over_early_placed"),
    "long l/e attempts": g(lng, "late_over_early_attempts"),
    "long stalled fraction": g(lng, "stalls", "stalled_step_fraction"),
    "long stuck-object fraction": g(lng, "stuck_object", "step_fraction"),
  }
  # first vs last 36 s of the long run
  def edge(x):
    if not x:
      return None
    pw, w, dt = x["placed_per_window"], x["window"], x["dt"]
    k = int(round(36.0 / (w * dt)))
    if len(pw) < 2 * k:
      return None
    e, l = float(np.mean(pw[:k])), float(np.mean(pw[-k:]))
    return (l / e) if e > 1e-9 else None
  rows["long last36/first36"] = stats([edge(x) for x in lng])
  rows["long blocks placed/min (s101)"] = None
  if lng[0]:
    pw, w, dt = lng[0]["placed_per_window"], lng[0]["window"], lng[0]["dt"]
    k = int(round(36.0 / (w * dt)))
    rows["long blocks placed/min (s101)"] = [round(float(np.mean(pw[i:i + k])), 2) for i in range(0, len(pw), k)]
  man = load(os.path.join(d, "manifest.json")) or {}
  tim = {}
  tp = os.path.join(d, "timing.jsonl")
  if os.path.exists(tp):
    for line in open(tp):
      try:
        j = json.loads(line); tim[j["stage"]] = j["seconds"]
      except Exception:
        pass
  return rows, man, tim


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("e0")
  p.add_argument("e2")
  p.add_argument("--teacher-ref", default=None)
  p.add_argument("--gen1", default=None, help="results/pc/gen2/audit (initiation_gen1_P1B_s*.json)")
  p.add_argument("--json", default=None)
  a = p.parse_args()
  r0, m0, t0 = route_rows(a.e0)
  r2, m2, t2 = route_rows(a.e2)
  cols = [("E0 " + str(m0.get("route")), r0), ("E2 " + str(m2.get("route")), r2)]
  if a.teacher_ref:
    rt, _, _ = route_rows(a.teacher_ref, prefix_ini="initiation_teacher", prefix_long="long_teacher", prefix_acc="none", prefix_end="none")
    cols.append(("teacher v11", rt))
  if a.gen1:
    rg, _, _ = route_rows(a.gen1, prefix_ini="initiation_gen1_P1B", prefix_long="long_gen1_P1B", prefix_acc="none", prefix_end="none")
    cols.append(("gen-1 P1B (bug-trained)", rg))
  print(f"| metric | " + " | ".join(c for c, _ in cols) + " |")
  print("|---|" + "---|" * len(cols))
  for key in r0:
    cells = []
    for _, r in cols:
      v = r.get(key)
      cells.append(str(v) if isinstance(v, list) else fmt(v, 3 if ("fraction" in key or "success" in key or "per attempt" in key or "rate" in key) else 2))
    print(f"| {key} | " + " | ".join(cells) + " |")
  print()
  for name, m, t in (("E0", m0, t0), ("E2", m2, t2)):
    print(f"{name}: route {m.get('route')} oracle_only {m.get('oracle_only')} teacher {str(m.get('teacher_sha256'))[:16]} seed {m.get('seed')} "
          f"envs {m.get('vision_envs')} budget {m.get('distill_iters')}+{m.get('finetune_iters')} "
          f"env steps {dig(m, 'budget', 'distill_env_steps')}+{dig(m, 'budget', 'finetune_env_steps')} "
          f"updates {dig(m, 'budget', 'distill_optimizer_updates')}+{dig(m, 'budget', 'finetune_optimizer_updates')} "
          f"gpu {m.get('gpu_name')} commit {m.get('code_commit_local')} timing s {t}")
  # -- the pre-registered rule
  def better(key, higher=True):
    s0, s2 = r0.get(key), r2.get(key)
    if not s0 or not s2:
      return None
    sep = (s2["min"] > s0["max"]) if higher else (s2["max"] < s0["min"])
    diff = (s2["median"] - s0["median"]) / max(abs(s0["median"]), 1e-9)
    return {"separated": bool(sep), "rel_diff": float(diff), "clear": bool(sep and abs(diff) >= 0.30 and (diff > 0) == higher)}
  rule = {k: better(k) for k in ("ini grasps per attempt", "ini l/e placed", "ini l/e placed live", "accept placed/min", "ini placed/min", "long l/e placed")}
  rule["grasps per attempt worse"] = better("ini grasps per attempt", higher=False)
  clear_keys = [k for k in ("ini grasps per attempt", "ini l/e placed", "ini l/e placed live") if rule[k] and rule[k]["clear"]]
  print("\nrule:", json.dumps(rule, indent=1))
  if any(rule[k] is None for k in ("ini grasps per attempt", "ini l/e placed", "ini l/e placed live")):
    print("\nverdict: INCOMPLETE -- the initiation rows are not in yet for both routes")
    if a.json:
      json.dump({"e0": {"rows": r0, "manifest": m0, "timing": t0}, "e2": {"rows": r2, "manifest": m2, "timing": t2},
                 "rule": rule, "verdict": "incomplete"}, open(a.json, "w"), indent=1, default=str)
    return
  gate_placed = 0.70 * 19.3
  e0_close = bool(r0.get("accept placed/min") and r0["accept placed/min"]["median"] >= gate_placed and r0.get("endurance l/e (24 s)") and r0["endurance l/e (24 s)"]["median"] >= 0.85)
  if len(clear_keys) == 3:
    verdict = "E2 clearly better on all three rows -> next: a deployable target selection + persistence (two more training seeds first)"
  elif clear_keys:
    verdict = f"E2 clearly better on {clear_keys} only -> partial; read the per-phase rows before deciding"
  elif e0_close:
    verdict = "E0 near the gate -> quantify the recovery against gen-1 P1B; decide whether a target mechanism is needed at all"
  else:
    verdict = "E2 ~ E0 -> next: the mask-line control (same teacher, budget, 36 s, fixed cadence)"
  print("\nverdict:", verdict)
  if a.json:
    json.dump({"e0": {"rows": r0, "manifest": m0, "timing": t0}, "e2": {"rows": r2, "manifest": m2, "timing": t2},
               "rule": rule, "verdict": verdict}, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
  main()
