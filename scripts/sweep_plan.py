"""The Phase WM0 sweep, written down once and emitted as job lists.

Levels are chosen against what the policy was actually trained on, which
``perturb.AXES`` records: for an axis the training distribution covered, the
sweep goes to its edge and past it; for an axis the simulator did not model at
all -- nine of the seventeen, including every kind of latency -- there is no
edge, so the levels are set from what the hardware plausibly does.

Emitting the plan rather than typing shell lines keeps three things true: the
same levels are used at screening and formal scale, every run's label encodes
exactly what it was, and the report can print the plan next to the results
without either being transcribed.

    python scripts/sweep_plan.py s1 > s1.jobs
    python scripts/sweep_plan.py s2 --axes depth_scale,obs_latency_steps
    python scripts/sweep_plan.py s3 --pair depth_scale:obs_latency_steps
"""

from __future__ import annotations

import argparse
import json

from piper_push.perturb import AXES

TASK = "Mjlab-Pick-Place-PiperX-Vision"
CKPT = ("logs/rsl_rl/piperx_pick_place_vision/"
        "2026-08-22_17-15-09_f3/model_1500.pt")
"""The best honest-cadence vision PPO policy, 55.8 objects/min.

The DAgger student is a diagnostic control, not a second subject: it is 15%
slower before any perturbation, so running the whole sweep on both would
double the cost to compare two policies that differ in more than the thing
under test."""

# Screening and formal levels per axis.  The first entry of each list is the
# nominal and is only run once per stage, as the shared reference point.
LEVELS: dict[str, list[float]] = {
  # -- camera and depth ----------------------------------------------------
  # Trained jitter is +-2 deg, so 1.5 is inside it and 4 is twice its edge.
  "cam_pitch_deg": [0.0, -4.0, -1.5, 1.5, 4.0],
  "cam_yaw_deg": [0.0, -4.0, -1.5, 1.5, 4.0],
  # Trained jitter is +-20 mm.
  "cam_pos_x_m": [0.0, -0.05, -0.02, 0.02, 0.05],
  "cam_pos_z_m": [0.0, -0.05, -0.02, 0.02, 0.05],
  # Never modelled.  A 3% scale error at 0.8 m is 24 mm, which is about one
  # object half-height -- the level at which "how tall is it" starts to move.
  "depth_scale": [1.0, 0.93, 0.97, 1.03, 1.07],
  "depth_bias_m": [0.0, -0.03, -0.01, 0.01, 0.03],
  # Trained at 0.02 i.i.d.; one-sided because less dropout is not a mismatch
  # worth calibrating.
  "depth_dropout": [0.02, 0.06, 0.12, 0.20],
  "depth_dropout_blob": [0.0, 0.05, 0.10, 0.20],
  # One step is 20 ms.  A USB depth camera at 30 fps plus a copy plus a
  # forward pass is 2-4.
  "obs_latency_steps": [0, 1, 2, 3, 4],
  # -- robot ---------------------------------------------------------------
  "action_latency_steps": [0, 1, 2, 3, 4],
  "joint_response_scale": [1.0, 0.65, 0.80, 0.90, 1.10],
  "servo_damping_scale": [1.0, 0.50, 0.75, 1.50, 2.00],
  "action_deadband_rad": [0.0, 0.002, 0.005, 0.010],
  # -- gripper and contact --------------------------------------------------
  # The 0.10 m/s the simulator assumes is flagged in robot.py as NOT measured.
  "gripper_rate_scale": [1.0, 0.35, 0.50, 0.70, 1.50],
  "gripper_latency_steps": [0, 1, 2, 4],
  # Trained band is [0.55, 1.15] about 0.85, i.e. roughly +-35%.
  "pad_friction_scale": [1.0, 0.60, 0.80, 1.20],
  "table_friction_scale": [1.0, 0.60, 0.80, 1.30],
}

STAGES = {
  # Screening: enough to rule an axis out, not enough to publish.
  "s1": dict(num_envs=256, steps=1200, repeats=1),
  # Formal: the protocol every other number in this repository uses, and
  # three independent processes per point because the simulator is not
  # reproducible run to run (novelty_validation_phase_0_2.md section 7.5).
  "s2": dict(num_envs=512, steps=2400, repeats=3),
  "s3": dict(num_envs=512, steps=2400, repeats=3),
}

SEEDS = [20260823, 31415926, 27182818]


def _flag(axis: str) -> str:
  return "--" + axis.replace("_", "-")


def _fmt(axis: str, v: float) -> str:
  if "steps" in axis:
    return str(int(v))
  return f"{v:g}"


def _label(axis: str, v: float, rep: int) -> str:
  tag = _fmt(axis, v).replace("-", "m").replace(".", "p")
  return f"{axis}__{tag}__r{rep}"


def emit(stage: str, axes: list[str], pair: str | None) -> list[str]:
  spec = STAGES[stage]
  reps, lines = spec["repeats"], []

  def job(label: str, flags: str) -> str:
    return f"{label} {TASK} {CKPT} {flags}".rstrip()

  if stage == "s3":
    a, b = pair.split(":")
    # Three levels each: nominal, moderate, large -- taken from the same
    # LEVELS lists so the interaction points sit on the main-effect grid and
    # the two can be read against each other.
    la = [LEVELS[a][0], LEVELS[a][2], LEVELS[a][-1]]
    lb = [LEVELS[b][0], LEVELS[b][2], LEVELS[b][-1]]
    for va in la:
      for vb in lb:
        for r in range(reps):
          lab = (f"pair__{a}__{_fmt(a, va).replace('-', 'm').replace('.', 'p')}"
                 f"__{b}__{_fmt(b, vb).replace('-', 'm').replace('.', 'p')}__r{r}")
          lines.append(job(lab, f"--seed {SEEDS[r]} "
                                f"{_flag(a)} {_fmt(a, va)} "
                                f"{_flag(b)} {_fmt(b, vb)}"))
    return lines

  # The shared nominal, once per repeat rather than once per axis.
  for r in range(reps):
    lines.append(job(f"nominal__r{r}", f"--seed {SEEDS[r]}"))
  for axis in axes:
    for v in LEVELS[axis][1:]:
      for r in range(reps):
        lines.append(job(_label(axis, v, r),
                         f"--seed {SEEDS[r]} {_flag(axis)} {_fmt(axis, v)}"))
  return lines


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("stage", choices=list(STAGES))
  p.add_argument("--axes", default=None,
                 help="comma-separated subset; default is every axis")
  p.add_argument("--pair", default=None, help="s3 only: 'axis_a:axis_b'")
  p.add_argument("--manifest", default=None,
                 help="also write the plan, with levels and provenance, here")
  a = p.parse_args()

  axes = list(LEVELS) if a.axes is None else [s.strip() for s in a.axes.split(",")]
  unknown = set(axes) - set(LEVELS)
  if unknown:
    p.error(f"unknown axes {sorted(unknown)}")
  if a.stage == "s3" and not a.pair:
    p.error("s3 needs --pair axis_a:axis_b")

  lines = emit(a.stage, axes, a.pair)
  print("\n".join(lines))

  if a.manifest:
    spec = STAGES[a.stage]
    man = {
      "stage": a.stage, "task": TASK, "checkpoint": CKPT,
      "protocol": spec, "seeds": SEEDS[:spec["repeats"]],
      "n_runs": len(lines),
      "axes": {
        ax: {
          "levels": LEVELS[ax],
          "nominal": AXES[ax].nominal,
          "unit": AXES[ax].unit,
          "group": AXES[ax].group,
          "trained_range": list(AXES[ax].trained_range)
          if AXES[ax].trained_range else None,
          "outside_training": [
            _outside(ax, v) for v in LEVELS[ax]
          ],
          "hardware": AXES[ax].hardware,
        }
        for ax in (axes if a.stage != "s3" else a.pair.split(":"))
      },
    }
    with open(a.manifest, "w") as f:
      json.dump(man, f, indent=1)
  return 0


def _outside(axis: str, v: float) -> str:
  """Where this level sits relative to what the policy was trained across."""
  tr = AXES[axis].trained_range
  if tr is None:
    return "not-modelled-in-training"
  lo, hi = tr
  if lo <= v <= hi:
    return "inside" if (v != lo and v != hi) else "at-edge"
  span = hi - lo
  over = (v - hi) if v > hi else (lo - v)
  return f"outside-by-{over / span:.2f}x-range"


if __name__ == "__main__":
  raise SystemExit(main())
