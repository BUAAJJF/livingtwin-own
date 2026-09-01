#!/usr/bin/env python3
"""Summarize the matched nominal/robust evaluation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def metrics(path: Path) -> dict:
  doc = json.loads(path.read_text())
  return doc["metrics"]


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("directory", type=Path)
  a = p.parse_args()
  names = (
    "nominal_on_nominal", "nominal_on_stress",
    "robust_on_nominal", "robust_on_stress",
  )
  matrix = {name: metrics(a.directory / f"{name}.json") for name in names}
  nn, ns = matrix["nominal_on_nominal"], matrix["nominal_on_stress"]
  rn, rs = matrix["robust_on_nominal"], matrix["robust_on_stress"]

  def rel_loss(reference: float, value: float) -> float:
    return 0.0 if reference == 0 else (reference - value) / reference

  report = {
    "evaluation_matrix": matrix,
    "degrade": {
      "nominal_success_pp": 100.0 * (nn["success"] - rn["success"]),
      "nominal_throughput_fraction": rel_loss(
        nn["throughput_per_min"], rn["throughput_per_min"]),
      "nominal_p95_fraction": (
        (rn["p95_s"] - nn["p95_s"]) / nn["p95_s"]),
    },
    "stress_gain": {
      "success_pp": 100.0 * (rs["success"] - ns["success"]),
      "throughput_fraction": (
        0.0 if ns["throughput_per_min"] == 0 else
        (rs["throughput_per_min"] - ns["throughput_per_min"])
        / ns["throughput_per_min"]),
      "trip_rate_fraction": rel_loss(
        ns["trips_per_arm_hour"], rs["trips_per_arm_hour"]),
    },
    "robust_domain_gap": {
      "success_pp": 100.0 * (rn["success"] - rs["success"]),
      "throughput_fraction": rel_loss(
        rn["throughput_per_min"], rs["throughput_per_min"]),
    },
  }
  out = a.directory / "dr_degrade_report.json"
  out.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
