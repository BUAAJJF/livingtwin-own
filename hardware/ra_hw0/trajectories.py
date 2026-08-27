"""The H0-H3 command streams, and the only place their numbers are written.

Every trajectory is a function of time returning a full six-joint position
command in radians, built from the start posture in
``hardware.ra_hw0.limits``.  They are generated, previewed and envelope-checked
offline; nothing here talks to a robot.

The ordering across joints is by how much energy a mistake puts into the room,
not by how interesting the joint is for identification.  Joint 5 is the wrist
pitch: the least mass, the shortest lever, an arc of a few centimetres at the
gripper, and no cable twist.  The joints that matter most for actuator
identification -- 1, 2 and 3, which carry the arm -- are last, and only after
H1 has passed on all of the distal ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .limits import (H1_COMMANDED_SPEED_RAD_S, H1_MAX_SEGMENT_S,
                     H1_START_POSE_RAD, ARM_JOINTS)

CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ

H1_JOINT_ORDER = ("joint5", "joint4", "joint6", "joint3", "joint2", "joint1")
"""Increasing energy.  H1 runs the first of these and stops."""

H1_STEP_RAD = 0.02
"""The smallest step this phase will try to resolve.  The encoder reports
0.001 degrees (1.75e-5 rad), so 0.02 rad is about 1150 counts -- far above
quantisation, and 1.15 degrees is a motion a person watching can see, which
matters more than the arithmetic."""

H1_TRIANGLE_AMPLITUDE_RAD = 0.05
"""2.9 degrees peak-to-start.  At the commanded speed this is a 1 s ramp."""


@dataclass
class Segment:
  """One command stream, with everything a preview needs to describe it."""

  name: str
  stage: str
  joint: str | None
  duration_s: float
  amplitude_rad: float
  peak_speed_rad_s: float
  q: list[list[float]] = field(default_factory=list)

  @property
  def n(self) -> int:
    return len(self.q)

  def envelope(self) -> dict[str, tuple[float, float]]:
    """Empty for a segment that transmits nothing, which H0 does not."""
    if not self.q:
      return {}
    return {j: (min(row[i] for row in self.q), max(row[i] for row in self.q))
            for i, j in enumerate(ARM_JOINTS)}

  def describe(self) -> str:
    if not self.q:
      return (f"{self.name} [{self.stage}] {self.duration_s:.2f} s of "
              "telemetry with NO command transmitted")
    env = self.envelope()
    moved = {j: v for j, v in env.items() if v[1] - v[0] > 1e-9}
    parts = ", ".join(f"{j} {lo:+.4f}..{hi:+.4f} rad" for j, (lo, hi) in moved.items())
    return (f"{self.name} [{self.stage}] {self.duration_s:.2f} s, "
            f"{self.n} commands at {CONTROL_HZ:.0f} Hz; "
            f"amplitude {self.amplitude_rad:.4f} rad, peak commanded speed "
            f"{self.peak_speed_rad_s:.4f} rad/s; moves: {parts or 'nothing'}")


def _start() -> list[float]:
  return [H1_START_POSE_RAD[j] for j in ARM_JOINTS]


def _build(name, stage, joint, profile, duration_s) -> Segment:
  """``profile(t) -> offset in rad`` applied to one joint of the start pose."""
  # Inclusive of the endpoint.  Without the final sample a segment stops one
  # control period short of its own profile -- for the triangle that left the
  # arm 0.001 rad from the start pose, so the next segment began with a jump
  # the preview did not show.  A test pins it.
  n = int(round(duration_s * CONTROL_HZ)) + 1
  base = _start()
  idx = ARM_JOINTS.index(joint) if joint else None
  q = []
  for k in range(n):
    row = list(base)
    if idx is not None:
      row[idx] = base[idx] + profile(k * DT)
    q.append(row)
  amp = 0.0
  peak = 0.0
  if idx is not None and n > 1:
    vals = [row[idx] - base[idx] for row in q]
    amp = max(abs(v) for v in vals)
    peak = max(abs(b - a) for a, b in zip(vals, vals[1:])) / DT
  return Segment(name, stage, joint, duration_s, amp, peak, q)


def h0_hold(duration_s: float = 30.0) -> Segment:
  """H0 sends nothing.  This exists so the recorder has a length and the
  preview has something to print; the collector for H0 issues no command at
  all and the segment's ``q`` is never transmitted."""
  seg = _build("H0-static", "H0", None, lambda t: 0.0, duration_s)
  seg.q = []                      # nothing is sent, and the object says so
  return seg


def h1_hold(duration_s: float = 3.0) -> Segment:
  """The first thing that is ever transmitted: the pose the arm is already in.

  Zero commanded displacement.  It exercises the command path, the watchdog,
  the logger and the stop machine with the least energy that exists, and a
  joint that moves during it is a finding, not a trajectory."""
  return _build("H1-A-hold", "H1", None, lambda t: 0.0, duration_s)


def h1_step(joint: str, step_rad: float = H1_STEP_RAD, hold_s: float = 1.0,
            speed_rad_s: float = H1_COMMANDED_SPEED_RAD_S) -> Segment:
  """A set-point change, reached at H1's commanded rate, held, then undone.

  **Not** a single-period step.  Moving 0.02 rad in one 20 ms control period
  asks for 1.0 rad/s, which is twenty times what H1 commands and three times
  its own stop threshold -- ``check`` refuses it, and it refused this function
  before it was rewritten.  A true step is the more informative experiment and
  it belongs in H2, where the speed cap is re-derived rather than inherited.

  The ramp is 0.4 s for the default 0.02 rad, so the segment is a 0.4 s ramp,
  a hold, a 0.4 s ramp back, and a hold.
  """
  ramp = step_rad / speed_rad_s

  def profile(t):
    if t < hold_s:
      return 0.0
    if t < hold_s + ramp:
      return speed_rad_s * (t - hold_s)
    if t < 2 * hold_s + ramp:
      return step_rad
    if t < 2 * hold_s + 2 * ramp:
      return step_rad - speed_rad_s * (t - 2 * hold_s - ramp)
    return 0.0
  return _build(f"H1-B-step-{joint}", "H1", joint, profile,
                3 * hold_s + 2 * ramp)


def h1_triangle(joint: str, amplitude_rad: float = H1_TRIANGLE_AMPLITUDE_RAD,
                speed_rad_s: float = H1_COMMANDED_SPEED_RAD_S,
                cycles: int = 2) -> Segment:
  """Up and back to the start, twice, all on one side of the start pose.

  Deliberately never crosses the start: this is the *same-direction* response,
  and keeping it one-sided is what makes H1-D's crossing an experiment rather
  than a repetition.  Two cycles so the second can be compared with the first
  on an arm that has just been moved."""
  ramp = amplitude_rad / speed_rad_s
  period = 2 * ramp

  def profile(t):
    u = t % period
    return speed_rad_s * u if u < ramp else amplitude_rad - speed_rad_s * (u - ramp)
  return _build(f"H1-C-triangle-{joint}", "H1", joint, profile, cycles * period)


def h1_reversal(joint: str, amplitude_rad: float = H1_TRIANGLE_AMPLITUDE_RAD,
                speed_rad_s: float = H1_COMMANDED_SPEED_RAD_S,
                hold_s: float = 0.5) -> Segment:
  """+A, then straight through the start to -A, then back.

  The one thing the whole programme is about: what the drive does when the
  command changes sign.  Backlash, stiction and a direction-dependent gain all
  live in the crossing and nowhere else, so this segment crosses twice and
  ends stationary."""
  ramp = amplitude_rad / speed_rad_s

  def profile(t):
    if t < ramp:                       # 0 -> +A
      return speed_rad_s * t
    if t < 3 * ramp:                   # +A -> -A, through the start
      return amplitude_rad - speed_rad_s * (t - ramp)
    if t < 4 * ramp:                   # -A -> 0
      return -amplitude_rad + speed_rad_s * (t - 3 * ramp)
    return 0.0
  return _build(f"H1-D-reversal-{joint}", "H1", joint, profile,
                4 * ramp + hold_s)


def h1_plan(joint: str = H1_JOINT_ORDER[0]) -> list[Segment]:
  """The whole of H1 for one joint, in order, each ending stationary."""
  return [h1_hold(), h1_step(joint), h1_triangle(joint), h1_reversal(joint)]


def check(seg: Segment, table: dict) -> list[str]:
  """Every reason this segment must not be transmitted."""
  reasons = []
  if seg.duration_s > H1_MAX_SEGMENT_S + 1e-9 and seg.stage == "H1":
    reasons.append(f"{seg.name}: {seg.duration_s:.2f} s exceeds the "
                   f"{H1_MAX_SEGMENT_S} s first-motion ceiling")
  for i, row in enumerate(table["joints"]):
    lo, hi = row["envelope_min_rad"]["value"], row["envelope_max_rad"]["value"]
    if lo is None or hi is None:
      reasons.append(f"{row['joint']}: envelope is UNKNOWN")
      continue
    for k, cmd in enumerate(seg.q):
      if not (lo <= cmd[i] <= hi):
        reasons.append(f"{seg.name}: {row['joint']} command {cmd[i]:.4f} rad "
                       f"at sample {k} outside envelope [{lo:.4f}, {hi:.4f}]")
        break
  if seg.q:
    for i, j in enumerate(ARM_JOINTS):
      vals = [row[i] for row in seg.q]
      peak = max((abs(b - a) for a, b in zip(vals, vals[1:])), default=0.0) / DT
      cap = table["joints"][i]["max_speed_rad_s"]["value"]
      if cap is not None and peak > cap:
        reasons.append(f"{seg.name}: {j} commanded speed {peak:.3f} rad/s "
                       f"over the {cap} rad/s stop threshold")
  return reasons
