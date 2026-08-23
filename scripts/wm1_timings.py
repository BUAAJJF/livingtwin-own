"""Assemble the WM1-A wall-clock accounting that G6 asks for.

    python scripts/wm1_timings.py --json results/wm1_latency/timings.json

The split that matters is between what a deployment pays and what was paid
once, beforehand, on a machine that is not the robot:

**Online** -- against a "one hour to tidy" budget: the seconds of arm time spent
collecting the target session, the inference run on it, and the PPO
adaptation.  The last of those is simulator time, not robot time, but it is
time the robot spends waiting, so it counts.

**Offline** -- the dataset generation and the two model fits.  Reported in full,
with the data volume, because "the online loop is fast" is only interesting
next to what it cost to make it possible.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(p: Path):
  try:
    return json.loads(p.read_text())
  except Exception:
    return None


def dataset_totals(data: Path) -> dict:
  by_split: dict[str, dict] = {}
  for f in sorted(data.glob("*.json")):
    d = _load(f)
    if not d or "split" not in d:
      continue          # the manifest lives here too
    e = by_split.setdefault(d["split"], {"files": 0, "arm_seconds": 0.0,
                                         "bytes": 0, "wall_clock_s": 0.0})
    e["files"] += 1
    e["arm_seconds"] += d["arm_seconds"]
    e["bytes"] += d["bytes"]
  return by_split


def collection_wall_clock() -> dict:
  """Generation time from the collection logs.

  Read from the logs rather than from the tensor files' metadata, which would
  mean loading four gigabytes of fp16 to recover one float each.
  """
  total, files = 0.0, 0
  for log in sorted((ROOT / "logs" / "wm1_collect").glob("*.log")):
    text = log.read_text(errors="ignore")
    rates = re.findall(r"step (\d+)/(\d+)\s+([\d,]+) env-steps/s", text)
    if not rates:
      continue
    last, total_steps, rate = rates[-1]
    envs = re.search(r"(\d+) envs x", text)
    n_env = int(envs.group(1)) if envs else 0
    r = float(rate.replace(",", ""))
    if r > 0 and n_env:
      total += int(total_steps) * n_env / r
      files += 1
  return {"files": files, "seconds": total, "hours": total / 3600.0}


def ppo_wall_clock(adapt: Path) -> dict:
  """Seconds per adaptation run, from the checkpoint timestamps it wrote."""
  out = {}
  for ck in sorted(adapt.glob("*.ckpt")):
    path = Path(ck.read_text().strip())
    run_dir = path.parent
    models = sorted(run_dir.glob("model_*.pt"),
                    key=lambda p: p.stat().st_mtime)
    if len(models) < 2:
      continue
    span = models[-1].stat().st_mtime - models[0].stat().st_mtime
    first = int(re.search(r"model_(\d+)", models[0].name).group(1))
    last = int(re.search(r"model_(\d+)", models[-1].name).group(1))
    out[ck.stem] = {"seconds": span, "iterations": last - first,
                    "seconds_per_iteration": span / max(last - first, 1)}
  return out


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--results", default="results/wm1_latency")
  p.add_argument("--json", default="results/wm1_latency/timings.json")
  a = p.parse_args()
  root = Path(a.results)

  train = _load(root / "model" / "train_report.json") or {}
  clf = _load(root / "model" / "classifier_report.json") or {}
  inf = _load(root / "posterior" / "inference_timing.json") or {}
  ppo = ppo_wall_clock(root / "adapt")
  ds = dataset_totals(root / "data")
  gen = collection_wall_clock()

  per_run = [v["seconds"] for v in ppo.values()]
  out = {
    "offline": {
      "dataset": ds,
      "dataset_generation_s": gen["seconds"],
      "dataset_generation": gen,
      "world_model_train_s": train.get("train_wall_clock_s"),
      "world_model_windows": (train.get("n_windows") or {}).get("train"),
      "classifier_train_s": clf.get("wall_clock_s"),
    },
    "online": {
      "target_data_collection_s": 60.0,
      "inference_per_session_s": inf.get("world_model_scores_s"),
      "inference_breakdown_s": inf,
      "ppo_per_run_s": (sum(per_run) / len(per_run)) if per_run else None,
      "ppo_runs": ppo,
    },
  }
  known = [out["online"]["target_data_collection_s"],
           out["online"]["inference_per_session_s"],
           out["online"]["ppo_per_run_s"]]
  out["online_total_s"] = sum(known) if all(k is not None for k in known) else None
  if out["online_total_s"]:
    out["online_total_min"] = out["online_total_s"] / 60.0

  Path(a.json).parent.mkdir(parents=True, exist_ok=True)
  Path(a.json).write_text(json.dumps(out, indent=1))
  print(json.dumps(out["online"], indent=1)[:800])
  print(f"  wrote {a.json}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
