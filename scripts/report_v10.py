"""The morning report for the v10 arms: one table, medians with spread, from the JSONs only.

    python scripts/report_v10.py results/d455_heavy_dr/v10_sight results/d455_heavy_dr/v10_nosight
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime

import numpy as np


def load(path):
  try:
    return json.load(open(path))
  except Exception:
    return None


def med_spread(xs):
  xs = [x for x in xs if x is not None and np.isfinite(x)]
  if not xs:
    return "-"
  if len(xs) == 1:
    return f"{xs[0]:.2f} (n=1)"
  return f"{np.median(xs):.2f} [{min(xs):.2f}, {max(xs):.2f}] n={len(xs)}"


def stage_times(out):
  rows = []
  for m in sorted(glob.glob(os.path.join(out, "stage_*.done")), key=os.path.getmtime):
    rows.append((os.path.basename(m)[6:-5], datetime.fromtimestamp(os.path.getmtime(m)).strftime("%m-%d %H:%M")))
  return rows


def arm(out):
  r = {"out": out}
  man = load(os.path.join(out, "manifest.json")) or {}
  r["final"] = man.get("final_checkpoint", "")
  r["finetune_ran"] = man.get("finetune_ran")
  r["stages"] = stage_times(out)
  for label in ("teacher", "student", "final"):
    files = sorted(glob.glob(os.path.join(out, f"endurance_{label}_s*.json")))
    ds = [load(f) for f in files]
    ds = [d for d in ds if d]
    r[f"{label}_late_over_early"] = med_spread([d.get("late_over_early") for d in ds])
    r[f"{label}_placed_per_min"] = med_spread([float(np.mean(d["placed_per_min"])) for d in ds if "placed_per_min" in d])
    r[f"{label}_jaw_stopped_mm"] = med_spread([d.get("jaw_mm_stopped") for d in ds])
    r[f"{label}_started"] = med_spread([d.get("started_envs") for d in ds])
  for dom in ("robust", "nominal"):
    ds = [load(f) for f in sorted(glob.glob(os.path.join(out, f"accept_final_{dom}_s*.json")))]
    ds = [d for d in ds if d]
    m = [d.get("metrics", {}) for d in ds]
    r[f"accept_{dom}_success"] = med_spread([x.get("success") for x in m])
    r[f"accept_{dom}_per_min"] = med_spread([x.get("throughput_per_min") for x in m])
    r[f"accept_{dom}_verdict"] = ",".join(str(d.get("verdict", d.get("pass", "?"))) for d in ds) or "-"
  for label in ("teacher", "final"):
    d = load(os.path.join(out, f"occlusion_{label}.json"))
    if d and d.get("engaged"):
      r[f"occ_{label}_engaged_blocked"] = f"{100 * d['engaged']['blocked_rate']:.1f}%"
      r[f"occ_{label}_over_rating"] = f"{100 * d['joint_speed_rad_s']['over_rating']:.2f}%"
    else:
      r[f"occ_{label}_engaged_blocked"] = "-"
      r[f"occ_{label}_over_rating"] = "-"
  # teacher curve from the log
  log = os.path.join(out, "teacher.log")
  if os.path.exists(log):
    vals = []
    for line in open(log, errors="replace"):
      if "objects_placed" in line:
        try:
          vals.append(float(line.strip().split()[-1]))
        except ValueError:
          pass
    r["teacher_objects_placed_curve"] = " ".join(f"{v:.2f}" for v in vals[:: max(1, len(vals) // 8)][-9:])
  return r


def main():
  outs = sys.argv[1:] or ["results/d455_heavy_dr/v10_sight", "results/d455_heavy_dr/v10_nosight"]
  arms = [arm(o) for o in outs]
  keys = [
    ("final checkpoint", "final"), ("finetune ran", "finetune_ran"),
    ("teacher objects_placed (log samples)", "teacher_objects_placed_curve"),
    ("teacher late/early (1 seed)", "teacher_late_over_early"),
    ("teacher engaged blocked", "occ_teacher_engaged_blocked"),
    ("student (distilled) late/early", "student_late_over_early"),
    ("final accept robust success", "accept_robust_success"),
    ("final accept robust /min", "accept_robust_per_min"),
    ("final accept nominal success", "accept_nominal_success"),
    ("final accept nominal /min", "accept_nominal_per_min"),
    ("final endurance late/early", "final_late_over_early"),
    ("final endurance placed/min (mean of windows)", "final_placed_per_min"),
    ("final jaw stopped mm", "final_jaw_stopped_mm"),
    ("final engaged blocked", "occ_final_engaged_blocked"),
    ("final joint speed over rating", "occ_final_over_rating"),
  ]
  print("| metric | " + " | ".join(os.path.basename(a["out"]) for a in arms) + " |")
  print("|---|" + "---|" * len(arms))
  for name, k in keys:
    print(f"| {name} | " + " | ".join(str(a.get(k, "-")) for a in arms) + " |")
  print()
  for a in arms:
    print(os.path.basename(a["out"]) + " stages: " + ", ".join(f"{n}@{t}" for n, t in a["stages"]))


if __name__ == "__main__":
  main()
