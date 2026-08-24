"""Read the sensor-model evaluation matrix and print the two comparisons it holds.

Two students -- one distilled against the fitted D405 model, one against clean
depth -- evaluated under both sensors, three rollout seeds each.  That gives
two questions and one sanity check:

  the sim-to-real question   under the *measured* sensor, does the student that
                             was shown it beat the one that was not?
  the price of realism       under *clean* depth, how much does training
                             against the noise cost, if anything?
  the sanity check           the clean-trained student under clean depth should
                             be the best cell in the table; if it is not,
                             something other than the sensor is moving.

Spread is over rollout seeds only.  There is one training seed per cell, so
this cannot separate "the sensor model did this" from "these two runs differed";
it is a signal, not a result, and it is labelled as one wherever it is printed.

    micromamba run -n mjlab python scripts/analyze_depth_sensor.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

METRICS = (
  ("throughput_per_min", "objects/min", 2),
  ("success", "success", 3),
  ("drop_rate", "post-grasp drop", 3),
  ("trips_per_arm_hour", "trips/arm-hour", 1),
)


def load(root: pathlib.Path) -> dict:
  out: dict[tuple[str, str], list[dict]] = {}
  for f in sorted(root.glob("*.json")):
    parts = f.stem.split("_")
    if len(parts) < 3:
      continue
    student, sensor = parts[0], parts[1]
    out.setdefault((student, sensor), []).append(
      json.loads(f.read_text())["metrics"])
  return out


def cell(rows: list[dict], key: str) -> tuple[float, float, int]:
  vals = [r[key] for r in rows if r.get(key) is not None]
  if not vals:
    return float("nan"), float("nan"), 0
  spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
  return statistics.median(vals), spread, len(vals)


def ablation(root: pathlib.Path) -> None:
  """Does the policy use its camera?  Read off throughput, not activations.

  ``shuffled`` is the arm that answers it: a real, correctly normalised depth
  image of another environment's table.  ``blank`` is out of distribution and a
  network may do anything with it, so it is reported and not relied on.
  """
  d = root / "ablation"
  rows: dict[str, dict[str, float]] = {}
  for f in sorted(d.glob("*.json")):
    who, cam = f.stem.rsplit("_", 1)
    m = json.loads(f.read_text())["metrics"]
    rows.setdefault(who, {})[cam] = m["throughput_per_min"]
  if not rows:
    return
  print("\ncamera ablation (objects/min, measured sensor)")
  print(f"{'':10s} {'real':>8s} {'blank':>8s} {'shuffled':>10s}   shuffled/real")
  for who, r in rows.items():
    real = r.get("real", float("nan"))
    frac = r.get("shuffled", float("nan")) / max(real, 1e-9)
    print(f"{who:10s} {real:8.2f} {r.get('blank', float('nan')):8.2f} "
          f"{r.get('shuffled', float('nan')):10.2f}   {100 * frac:8.1f}%")
  print("A policy that scored the same on another table's image would not be "
        "using its camera.")


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--root", default="results/depth_sensor")
  a = p.parse_args()
  root = pathlib.Path(a.root)
  data = load(root)
  if not data:
    print(f"no evaluations under {root}")
    return 1

  print(f"{'':22s}  " + "  ".join(f"{lbl:>22s}" for _, lbl, _ in METRICS))
  order = [("d405", "measured"), ("clean", "measured"),
           ("d405", "clean"), ("clean", "clean")]
  for student, sensor in order:
    rows = data.get((student, sensor))
    if not rows:
      continue
    name = f"{student:5s} under {sensor:8s}"
    cells = []
    for key, _, dp in METRICS:
      med, spread, n = cell(rows, key)
      cells.append(f"{med:>{14}.{dp}f} +-{spread:<6.{dp}f}")
    print(f"{name:22s}  " + "  ".join(cells) + f"   (n={len(rows)})")

  print()
  for sensor in ("measured", "clean"):
    a_rows, b_rows = data.get(("d405", sensor)), data.get(("clean", sensor))
    if not (a_rows and b_rows):
      continue
    ta, sa, _ = cell(a_rows, "throughput_per_min")
    tb, sb, _ = cell(b_rows, "throughput_per_min")
    gap = ta - tb
    # The spreads are ranges over rollout seeds; if the gap is inside them it
    # is not distinguishable from which seeds happened to be drawn.
    verdict = ("distinguishable" if abs(gap) > 0.5 * (sa + sb)
               else "INSIDE the seed spread")
    print(f"under the {sensor} sensor: the D405-trained student is "
          f"{gap:+.2f} objects/min against the clean-trained one -- {verdict}")

  ablation(root)

  print("\nOne training seed per cell.  The spreads are over rollout seeds and "
        "say nothing about\nrun-to-run variance in training, which this "
        "repository measures at 2.9% for a single\ncommand.  Treat the "
        "comparison as a signal, not a result.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
