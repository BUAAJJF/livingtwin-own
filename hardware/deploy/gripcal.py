"""Measure what the gripper drive reports, so ``pad_contact`` stops being a guess.

Two of the policy's thirty-six proprioceptive values do not exist on this arm
and are reconstructed from the gripper drive's reported effort:

* ``pad_contact`` -- two contact sensors in simulation, one bit here, thresholded
  at :attr:`proprio.ProprioBuilder.contact_effort`;
* ``squeeze`` -- how far the gripper is being asked to close past where it is.

The threshold has never been measured.  The repository says so plainly: *"The
threshold is a guess until someone squeezes something and reads the number."*
And ``GRIPPER_TORQUE_NM = 1.5``, which the reported effort is divided by, is
documented as "a starting point and not a measurement".

That guess now has a symptom.  On the arm, in a run where the policy reached
the object and closed on it, the reconstructed bit toggled almost every other
control step::

    0 1 1 1 0 1 1 1 1 0 0 1 0 0 1 1 1 1 0 0 0 1 0 0 0

In simulation ``pad_contact`` is a contact sensor: once the pads touch it stays
on.  A policy that learned "contact holds, so keep closing and lift" is being
told "contact, gone, contact, gone" and lets go.

So this measures the two things that decide the threshold, without moving the
arm:

**free** closes and opens the empty gripper.  That is the effort of the drive
moving itself -- friction, acceleration, the transient at each end of travel --
and it is the floor any threshold has to sit above.

**held** does the same with something between the fingers.  That is the effort
of the drive pushing against a thing that will not move.

The two distributions are what a threshold is for.  The tool reports them, the
separation between them, and what the bit would do at a range of thresholds --
including how often it would flicker, which is the failure that prompted this.

    python -m hardware.deploy.gripcal --free
    python -m hardware.deploy.gripcal --held      # with an object in the jaws
    python -m hardware.deploy.gripcal --report    # both, and a recommendation

The arm is never commanded.  Only ``GripperCtrl`` is sent, and the six joint
targets are held at the measured pose for the whole session.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from . import config, robot

HERE = pathlib.Path(__file__).resolve().parent
OUT_FILE = HERE / "gripper_effort.json"


def sweep(arm, low_m: float, high_m: float, cycles: int, rate_hz: float,
          hold_s: float, on_sample=None) -> list[dict]:
  """Close and open between two openings, recording what the drive reports.

  The arm's six joints are commanded to the pose they are already in, on every
  message, because ``PiperArm.command`` sends the whole seven-vector: leaving
  the joint part out is not an option, and sending anything but the measured
  pose would be moving the arm.
  """
  period = 1.0 / float(rate_hz)
  st = arm.read()
  hold_q = np.asarray(st.q, dtype=np.float64).copy()
  rows: list[dict] = []
  deadline = time.monotonic()

  def step(target_m: float, phase: str, seconds: float) -> None:
    nonlocal deadline
    n = max(1, int(seconds * rate_hz))
    for _ in range(n):
      arm.command(np.array([*hold_q, target_m]), period)
      deadline += period
      time.sleep(max(0.0, deadline - time.monotonic()))
      s = arm.read()
      rows.append({
        "t": time.time(), "phase": phase,
        "target_m": float(target_m),
        "gap_m": float(s.gripper) * 2.0,
        "finger_m": float(s.gripper),
        "effort": float(s.gripper_effort),
        "joint_drift_rad": float(np.max(np.abs(np.asarray(s.q) - hold_q))),
      })
      if on_sample is not None:
        on_sample(rows[-1])

  for k in range(cycles):
    step(high_m, "open", hold_s)
    step(low_m, "close", hold_s)
  step(high_m, "open", hold_s)
  return rows


def describe(rows: list[dict], label: str) -> dict:
  e = np.abs(np.asarray([r["effort"] for r in rows], dtype=np.float64))
  gap = np.asarray([r["gap_m"] for r in rows]) * 1000.0
  closing = np.asarray([r["phase"] == "close" for r in rows])
  # "Settled" is the part that matters: the drive at the end of its travel,
  # not the transient of starting to move.  A contact sensor would be on for
  # all of it.
  settled = np.zeros(len(rows), dtype=bool)
  if closing.any():
    idx = np.flatnonzero(closing)
    keep = idx[idx > 0]
    settled[keep[np.diff(np.concatenate([[0], closing.astype(int)]))[keep] == 0]] = True
  d = {
    "label": label,
    "n": len(rows),
    "gap_mm": {"min": float(gap.min()), "max": float(gap.max())},
    "effort": {
      "min": float(e.min()), "p50": float(np.median(e)),
      "p90": float(np.percentile(e, 90)), "p99": float(np.percentile(e, 99)),
      "max": float(e.max()),
    },
    "effort_while_closing": (
      None if not closing.any() else {
        "p50": float(np.median(e[closing])),
        "p90": float(np.percentile(e[closing], 90)),
        "max": float(e[closing].max()),
      }),
    "max_joint_drift_deg": float(np.degrees(
      max(r["joint_drift_rad"] for r in rows))),
  }
  return d


def flicker(rows: list[dict], threshold: float) -> dict:
  """How the bit would behave at this threshold, in runs rather than counts.

  A contact sensor produces one long run per contact.  The number that says
  whether a reconstruction is usable is not how often it is on, it is how often
  it turns off again while the gripper is still shut.
  """
  e = np.abs(np.asarray([r["effort"] for r in rows]))
  bit = (e > threshold).astype(int)
  runs, k = [], 0
  for b in bit:
    if b:
      k += 1
    elif k:
      runs.append(k)
      k = 0
  if k:
    runs.append(k)
  r = np.asarray(runs) if runs else np.zeros(1)
  return {
    "threshold": float(threshold),
    "on_fraction": float(bit.mean()),
    "runs": int(len(runs)),
    "median_run": float(np.median(r)),
    "single_step_runs": float((r == 1).mean()) if runs else 0.0,
  }


def main() -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--free", action="store_true",
                 help="cycle the empty gripper")
  p.add_argument("--held", action="store_true",
                 help="cycle with an object between the fingers")
  p.add_argument("--report", action="store_true",
                 help="read both saved sweeps and recommend a threshold")
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--cycles", type=int, default=4)
  p.add_argument("--rate-hz", type=float, default=25.0)
  p.add_argument("--hold-s", type=float, default=1.2,
                 help="seconds at each end of travel")
  p.add_argument("--open-mm", type=float, default=90.0)
  p.add_argument("--close-mm", type=float, default=0.0,
                 help="jaw gap to command when closing.  0 asks the drive to "
                      "shut fully, which against an object is what a grasp "
                      "does")
  p.add_argument("--out", default=str(OUT_FILE))
  a = p.parse_args()

  out = pathlib.Path(a.out)
  saved = json.loads(out.read_text()) if out.exists() else {}

  if a.free or a.held:
    label = "free" if a.free else "held"
    if a.free and a.held:
      p.error("--free and --held are separate sweeps; run them one at a time")
    if not sys.stdin.isatty():
      raise SystemExit("refusing to drive the gripper without a terminal")
    print(f"gripper {label} sweep: {a.cycles} cycles, "
          f"{a.open_mm:.0f} to {a.close_mm:.0f} mm jaw gap.")
    print("The six arm joints are held at their measured pose and are never "
          "commanded anywhere else.")
    if label == "held":
      print("Put the object between the fingers now.")
    if input("type 'go' to move the gripper: ").strip() != "go":
      return 1

    arm = robot.PiperArm(a.can)
    arm.connect()
    from .run import _wait_for_feedback
    st = _wait_for_feedback(arm)
    print("start joints (deg): "
          + " ".join(f"{x:+.1f}" for x in np.degrees(st.q)))
    gap, ok = arm.check_gripper_range()
    trained = robot.GRIPPER_JAW_GAP_M * 1000.0
    verdict = "matches" if ok else (
      "THESE DO NOT MATCH -- the top of the policy's gripper command does "
      "nothing and the opening it reads back never reaches what it asked for")
    print(f"configured jaw gap {gap * 1000:.0f} mm; trained against "
          f"{trained:.0f} mm -- {verdict}")
    arm.enable()
    try:
      rows = sweep(arm, a.close_mm / 2000.0, a.open_mm / 2000.0,
                   a.cycles, a.rate_hz, a.hold_s)
    finally:
      arm.hold()
      arm.close()
    d = describe(rows, label)
    d["samples"] = rows
    d["configured_jaw_gap_m"] = float(gap)
    saved[label] = d
    out.write_text(json.dumps(saved, indent=1) + "\n")
    print(f"\n{label}: effort p50 {d['effort']['p50']:.3f}  "
          f"p90 {d['effort']['p90']:.3f}  max {d['effort']['max']:.3f}")
    print(f"  jaw gap reached {d['gap_mm']['min']:.1f} .. "
          f"{d['gap_mm']['max']:.1f} mm")
    print(f"  arm drift during the sweep {d['max_joint_drift_deg']:.2f} deg")
    print(f"wrote {out}")
    return 0

  if a.report:
    if "free" not in saved or "held" not in saved:
      raise SystemExit(f"need both sweeps in {out}; run --free and --held")
    free, held = saved["free"], saved["held"]
    fe = np.abs(np.asarray([r["effort"] for r in free["samples"]]))
    he = np.abs(np.asarray([r["effort"] for r in held["samples"]]))
    print("gripper effort, normalised by GRIPPER_TORQUE_NM = "
          f"{robot.GRIPPER_TORQUE_NM}")
    print(f"  free  p50 {np.median(fe):.3f}  p90 {np.percentile(fe, 90):.3f}  "
          f"p99 {np.percentile(fe, 99):.3f}  max {fe.max():.3f}")
    print(f"  held  p50 {np.median(he):.3f}  p90 {np.percentile(he, 90):.3f}  "
          f"p99 {np.percentile(he, 99):.3f}  max {he.max():.3f}")
    sep = np.percentile(he, 10) - np.percentile(fe, 90)
    print(f"  separation (held p10 - free p90): {sep:+.3f}")
    print()
    print("what the bit would do, per threshold:")
    print("%9s  %-28s  %s" % ("threshold", "free (should stay off)",
                              "held (should stay on)"))
    best = None
    for t in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
      f = flicker(free["samples"], t)
      h = flicker(held["samples"], t)
      print("%9.2f  on %4.0f%% runs %3d med %4.1f      "
            "on %4.0f%% runs %3d med %5.1f  1-step %2.0f%%"
            % (t, 100 * f["on_fraction"], f["runs"], f["median_run"],
               100 * h["on_fraction"], h["runs"], h["median_run"],
               100 * h["single_step_runs"]))
      score = h["on_fraction"] - f["on_fraction"] - h["single_step_runs"]
      if best is None or score > best[1]:
        best = (t, score)
    print()
    print(f"best separation at threshold {best[0]:.2f}")
    print("Read the run lengths, not only the fractions: a contact sensor "
          "produces one long run per contact, and a reconstruction that turns "
          "off while the gripper is still shut is what makes a policy let go.")
    return 0

  p.print_help()
  return 0


if __name__ == "__main__":
  sys.exit(main())
