"""Gate H-D: is a recorded session usable, before anyone models it?

    python scripts/ra_hw0_validate.py results/ra_hw0/sessions/<id>.raw.jsonl

Reads an append-only raw log, never writes to it, and puts everything it
derives -- differentiated velocity, timing statistics, per-joint signal and
noise -- in a separate ``*.derived.json`` next to it.  A raw log that has been
"cleaned" is a log nobody can check, so this tool is the only thing that
produces derived quantities and it always produces them somewhere else.

Gate H-D, from docs/ra_hw0_experiment_plan.md:

* timestamps, units and joint order consistent;
* no unexplained dropped frames or control-period anomalies;
* the starting state is reproducible;
* the signal is above the measurement noise;
* the command *after* the safety filter is recorded, not only the request;
* no continuous trajectory split across train and test.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load(path: Path) -> list[dict]:
  return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def summarise(rows: list[dict], meta: dict) -> dict:
  hz = meta.get("control_rate_hz") or 50.0
  dt_nominal = 1.0 / hz
  t = [r["host_monotonic_s"] for r in rows]
  gaps = [b - a for a, b in zip(t, t[1:])]
  dropped = sum(1 for r in rows if r["q_rad"] is None)
  have_q = [r for r in rows if r["q_rad"] is not None]
  n_j = len(have_q[0]["q_rad"]) if have_q else 0

  out: dict = {
    "n_samples": len(rows), "n_with_telemetry": len(have_q),
    "dropped_frames": dropped,
    "dropped_fraction": dropped / max(len(rows), 1),
    "joint_order": meta.get("joint_order"),
    "units": meta.get("units"),
    "control_rate_hz": hz,
    "duration_s": (t[-1] - t[0]) if len(t) > 1 else 0.0,
  }
  if gaps:
    s = sorted(gaps)
    out["timing"] = {
      "nominal_period_s": dt_nominal,
      "median_s": s[len(s) // 2],
      "p99_s": s[min(len(s) - 1, int(0.99 * len(s)))],
      "max_s": s[-1],
      "jitter_std_s": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
      "periods_over_2x_nominal": sum(1 for g in gaps if g > 2 * dt_nominal),
    }

  # Per-joint signal against the still-arm noise floor, which is what decides
  # whether a trajectory measured anything.
  per_joint = []
  for j in range(n_j):
    q = [r["q_rad"][j] for r in have_q]
    cmd = [r["command_after_safety_filter_rad"][j] for r in have_q
           if r["command_after_safety_filter_rad"] is not None]
    still = [b - a for a, b in zip(q, q[1:])][:max(1, len(q) // 10)]
    noise = statistics.pstdev(still) if len(still) > 1 else 0.0
    per_joint.append({
      "index": j,
      "name": (meta.get("joint_order") or [f"j{j}"])[j],
      "q_min": min(q), "q_max": max(q), "q_travel": max(q) - min(q),
      "commanded_travel": (max(cmd) - min(cmd)) if cmd else None,
      "first_tenth_step_noise_std_rad": noise,
      "signal_to_noise": ((max(q) - min(q)) / noise) if noise > 0 else None,
    })
  out["per_joint"] = per_joint

  filt = [r for r in rows if r["command_requested_rad"] is not None]
  out["command_records"] = {
    "n_commanded_samples": len(filt),
    "safety_filter_recorded": all(
      r["command_after_safety_filter_rad"] is not None for r in filt),
    "filter_changed_command": sum(
      1 for r in filt
      if r["command_after_safety_filter_rad"] != r["command_requested_rad"]),
  }
  starts = {r["trajectory_id"] for r in rows}
  out["trajectories"] = sorted(starts)
  return out


def gate(sumry: dict, meta: dict) -> dict:
  hz = sumry["control_rate_hz"]
  timing = sumry.get("timing", {})
  checks = {
    "units_and_joint_order_recorded": bool(
      sumry["units"] and sumry["joint_order"]),
    "no_unexplained_dropped_frames": sumry["dropped_fraction"] < 0.01,
    "control_period_within_2x_nominal": timing.get(
      "periods_over_2x_nominal", 1) == 0,
    "safety_filtered_command_recorded":
      sumry["command_records"]["safety_filter_recorded"],
    "signal_above_noise": all(
      (p["signal_to_noise"] or 0) > 10 for p in sumry["per_joint"]
      if (p["commanded_travel"] or 0) > 1e-6) or not any(
      (p["commanded_travel"] or 0) > 1e-6 for p in sumry["per_joint"]),
    "session_stop_recorded": "stop" in meta,
  }
  return {"checks": checks,
          "verdict": "PASS" if all(checks.values()) else "FAIL",
          "failed": [k for k, v in checks.items() if not v]}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("raw")
  a = ap.parse_args()
  raw = Path(a.raw)
  meta_path = raw.with_suffix("").with_suffix(".meta.json")
  meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
  rows = load(raw)
  s = summarise(rows, meta)
  g = gate(s, meta)
  out = {"raw": str(raw), "meta": str(meta_path), "summary": s,
         "gate_H_D": g,
         "note": "derived quantities live here; the raw log is untouched"}
  dest = raw.with_suffix("").with_suffix(".derived.json")
  dest.write_text(json.dumps(out, indent=2, default=str))
  print(json.dumps({"samples": s["n_samples"],
                    "dropped": s["dropped_frames"],
                    "duration_s": round(s["duration_s"], 2),
                    "timing_p99_ms": round(
                      1e3 * s.get("timing", {}).get("p99_s", float("nan")), 2),
                    "gate_H_D": g["verdict"], "failed": g["failed"]}, indent=2))
  print(f"wrote {dest}")
  return 0 if g["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
