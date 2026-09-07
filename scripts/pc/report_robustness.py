"""Tabulate scripts/pc/robustness_sweep.sh: each perturbation's placed/min, grasps per attempt, stalls, against 'none'.

    python scripts/pc/report_robustness.py results/pc/gen4/robustness/R8 results/pc/gen4/robustness/C1
"""
import glob, json, os, sys

ORDER = ["none", "plane_0.004", "plane_-0.004", "plane_0.008", "drop_0.3", "drop_0.6", "offset_0.01", "offset_0.02", "jitter_0.005", "noise_1.5", "noise_2.0", "campos_0.02-camrot_2", "campos_0.04-camrot_4"]


def load_dir(d):
  out = {}
  for f in glob.glob(os.path.join(d, "ini_*.json")):
    name = os.path.basename(f)[4:-5]
    x = json.load(open(f))
    out[name] = {"placed": x["placed_per_min"], "gpa": x["grasps_per_min"] / max(x["attempts_per_min"], 1e-9), "stalled": x["stalls"]["stalled_step_fraction"],
                 "astray": x["terminations_per_arm_minute"].get("object_astray", 0.0), "success": x.get("success_per_grasp")}
  return out


def main():
  dirs = sys.argv[1:]
  data = {d: load_dir(d) for d in dirs}
  names = [os.path.basename(d.rstrip("/")) for d in dirs]
  print("| perturbation | " + " | ".join(f"{n} placed/min (vs none) | grasps/att | stalled" for n in names) + " |")
  print("|---|" + "---|---|---|" * len(names))
  for lv in ORDER:
    cells = []
    for d in dirs:
      x = data[d].get(lv); base = data[d].get("none")
      if not x: cells += ["-", "-", "-"]; continue
      rel = f" ({100 * (x['placed'] / base['placed'] - 1):+.0f}%)" if base and lv != "none" else ""
      cells += [f"{x['placed']:.2f}{rel}", f"{x['gpa']:.3f}", f"{x['stalled']:.3f}"]
    print(f"| {lv} | " + " | ".join(cells) + " |")
  summary = {}
  for d, n in zip(dirs, names):
    base = data[d].get("none")
    if not base: continue
    rels = [data[d][lv]["placed"] / base["placed"] for lv in ORDER[1:] if lv in data[d]]
    summary[n] = {"none": base["placed"], "mean_retained": sum(rels) / len(rels) if rels else None, "worst_retained": min(rels) if rels else None}
  print("\nretained throughput over the sweep (mean / worst):", json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
