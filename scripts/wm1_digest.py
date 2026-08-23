"""Slim every evaluation down to what the analysis and the provenance need.

    python scripts/wm1_digest.py

`accept_s1.py` writes about 700 kB per run: per-instance cycle times, joint
speed histograms, per-phase trip attribution, the full git diff. Phase WM1-A
produces well over a hundred of those, which is seventy-five megabytes of git
history for a repository whose entire source tree is under a megabyte. Phase
WM0 hit the same wall and gitignored its raw sweep outputs, keeping only the
summaries -- which meant the intervals in that report could not be recomputed
from what was committed.

This keeps both. Every field the analysis reads survives, **including the
512-element per-environment placement vector the hierarchical bootstrap
resamples**, along with the provenance that says which code and which weights
produced it. That is about 3 kB per run. The fat originals stay on the
training server and are gitignored here.

`scripts/analyze_wm1.py` reads the digest directory unchanged, because the
digest is a subset of the same schema rather than a new format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Everything analyze_wm1.py touches, plus what a reader needs to know which
# run this was.  Anything not named here is dropped.
CONFIG_KEYS = ("num_envs", "steps", "seed_requested", "seed_effective",
               "sim_seconds", "arm_hours", "control_dt", "episode_length_s",
               "redraw_on_place", "recurrent")
METRIC_KEYS = ("throughput_per_min", "trips_total", "trips_per_arm_hour",
               "trips_per_100_placed", "overall_success", "drop_rate",
               "p95_s", "stuck_fraction")
PROV_KEYS = ("git_commit", "git_branch", "checkpoint", "checkpoint_sha256",
             "argv", "mjlab", "rsl_rl", "torch", "mujoco", "mujoco_warp")


def slim(d: dict) -> dict:
  out = {
    "label": d.get("label"),
    "task": d.get("task"),
    "verdict": d.get("verdict"),
    "config": {k: d.get("config", {}).get(k) for k in CONFIG_KEYS
               if k in d.get("config", {})},
    "metrics": {k: d.get("metrics", {}).get(k) for k in METRIC_KEYS
                if k in d.get("metrics", {})},
    # The bootstrap resamples environments, so this vector is not a summary --
    # it is the data.
    "per_env": {"placed": d.get("per_env", {}).get("placed", [])},
    "per_env_seconds": d.get("per_env_seconds"),
    "mismatch": d.get("mismatch"),
    "provenance": {k: d.get("provenance", {}).get(k) for k in PROV_KEYS
                   if k in d.get("provenance", {})},
  }
  return out


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--src", nargs="+",
                 default=["results/wm1_latency/adapt",
                          "results/wm1_latency/equivalence"])
  p.add_argument("--out", default="results/wm1_latency/runs")
  a = p.parse_args()

  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  n, big, small = 0, 0, 0
  for src in a.src:
    for f in sorted(Path(src).glob("*__r*.json")):
      raw = f.read_text()
      big += len(raw)
      d = slim(json.loads(raw))
      if not d["per_env"]["placed"]:
        print(f"  !! {f.name} has no per-environment placements; "
              "the bootstrap cannot be recomputed from the digest")
      text = json.dumps(d, separators=(",", ":"))
      (out / f.name).write_text(text)
      small += len(text)
      n += 1
  print(f"  {n} runs: {big / 1e6:.1f} MB -> {small / 1e6:.2f} MB "
        f"({big / max(small, 1):.0f}x)")
  print(f"  wrote {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
