"""Per-joint absolute limits, derived from sources rather than typed.

Every number carries where it came from.  Three sources are trusted:

``vendor_sdk``   the AgileX ``piper_sdk`` parameter table and message
                 documentation, read out of the installed package;
``device``       the arm's own answer to an enquiry frame -- its configured
                 angle limits, maximum joint speed and maximum acceleration.
                 Absent until an arm is attached, which is why several rows
                 below are ``UNKNOWN``;
``repo_sim``     ``piper_push.robot``, the envelope the policy was trained
                 inside.

A row whose source is ``UNKNOWN`` sets ``blocks_motion`` and no amount of code
here can clear it.  Phase RA-HW-0's rule is that uncertainty stops motion
rather than being resolved by trying it.

**The joint-5 finding.**  The simulator's ``SAFE_TARGET_CLIP`` for joint 5 is
+-1.3090 rad; the vendor's limit is +-1.2200.  The simulated envelope is
**0.089 rad (5.1 degrees) wider than the arm allows at each end**, so a policy
trained in it can command an angle the drive will refuse and report as
``arm_status = 0x04, target angle over limit``.  Joints 1, 2, 3 and 6 exceed
the vendor table by 0.0001-0.0016 rad, which is the table's own rounding.
Joint 4's simulated clip is 0.19 rad *tighter* than the vendor's, deliberately.

Every absolute bound below is therefore the **intersection** of the two, and
the intersection is what any command path in this phase clips to.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

# -- vendor, read out of piper_sdk 0.6.2's parameter manager -----------------
VENDOR_JOINT_LIMIT_RAD = {
  "joint1": (-2.6179, 2.6179),
  "joint2": (0.0, 3.14),
  "joint3": (-2.967, 0.0),
  "joint4": (-1.745, 1.745),
  "joint5": (-1.22, 1.22),
  "joint6": (-2.09439, 2.09439),
}
VENDOR_GRIPPER_RANGE_M = (0.0, 0.07)
"""``piper_param_manager``'s default.  The simulator assumes a 100 mm jaw gap;
this arm's configured stroke is 70 mm unless the teaching-pendant parameter
says otherwise, which ``PiperArm.check_gripper_range`` asks and firmware
before V1.5-2 does not answer."""

UNKNOWN = "UNKNOWN"

# -- Phase RA-HW-0's own conservatism ---------------------------------------
LIMIT_MARGIN_RAD = 0.15
"""Kept clear of every absolute bound.  8.6 degrees: larger than any plausible
encoder or command rounding, small enough to leave the working range usable."""

H1_START_POSE_RAD = {
  "joint1": 0.0,
  "joint2": 1.05,
  "joint3": -1.05,
  "joint4": 0.0,
  "joint5": 0.0,
  "joint6": 0.0,
}
"""Proposed start posture for the first motion, pending the simulated envelope
check and a human's eyes.

Chosen for distance, not for the task: the closest joint sits **1.05 rad (60
degrees) from its nearest limit**, which is joint 2's lower bound and joint 3's
upper.  It is not the policy's home pose -- that one is a task posture a
few centimetres above the table, which is the wrong place to make a robot move
for the first time."""

H1_COMMANDED_SPEED_RAD_S = 0.05
"""What the first trajectories ask for: 1/39th of joint 1's trained command
rate limit and 1/63rd of its safety-shell trip speed."""

H1_STOP_SPEED_RAD_S = 0.35
"""Measured |qdot| at which the session stops.  Seven times what H1 commands,
and 9-11% of the trip speeds the simulator uses, so noise cannot trip it and a
runaway cannot hide under it."""

H1_STOP_ACCEL_RAD_S2 = 8.0
"""Measured |qddot| stop threshold.  A joint that reaches H1's stop speed
inside one 20 ms control period would show 17.5 rad/s^2; this catches half of
that.  It is *not* a vendor number -- the vendor's own maximum acceleration is
readable from the arm (``GetAllMotorMaxAccLimit``, 0.001 rad/s^2) and until it
has been read this row is provisional."""

H1_STOP_TRACKING_ERROR_RAD = 0.05
"""Commanded minus measured, at which the session stops.  2.9 degrees, against
H1 steps of 0.02-0.05 rad: a joint that is a whole commanded step behind is
either stalled or being driven by something other than the command."""

H1_MAX_SEGMENT_S = 5.0
"""No first-motion segment runs longer than this, and each one ends in a
stationary arm awaiting a human."""


@dataclass
class Limit:
  """One threshold, its unit, its provenance, and whether it blocks motion."""

  value: float | None
  unit: str
  source: str
  note: str = ""

  @property
  def blocks_motion(self) -> bool:
    return self.value is None or self.source == UNKNOWN

  def to_json(self) -> dict:
    d = asdict(self)
    d["blocks_motion"] = self.blocks_motion
    return d


@dataclass
class JointLimits:
  joint: str
  abs_min_rad: Limit
  abs_max_rad: Limit
  envelope_min_rad: Limit
  envelope_max_rad: Limit
  max_speed_rad_s: Limit
  max_accel_rad_s2: Limit
  max_tracking_error_rad: Limit
  device_max_speed_rad_s: Limit
  device_max_accel_rad_s2: Limit

  def to_json(self) -> dict:
    return {"joint": self.joint,
            **{k: v.to_json() for k, v in self.__dict__.items()
               if isinstance(v, Limit)}}


def intersect_absolute(joint: str, sim_clip: tuple[float, float]
                       ) -> tuple[float, float, str]:
  """The tighter of the vendor's bound and the simulator's, per end."""
  vlo, vhi = VENDOR_JOINT_LIMIT_RAD[joint]
  slo, shi = sim_clip
  lo, hi = max(vlo, slo), min(vhi, shi)
  which = []
  if lo > slo + 1e-9:
    which.append("vendor tightens the lower bound")
  if hi < shi - 1e-9:
    which.append("vendor tightens the upper bound")
  return lo, hi, "; ".join(which) or "simulator and vendor agree"


def build(sim_clip: dict[str, tuple[float, float]],
          device: dict | None = None) -> dict:
  """The whole table.  ``device`` is the arm's own answer, or ``None``."""
  rows = []
  for j in ARM_JOINTS:
    lo, hi, note = intersect_absolute(j, sim_clip[j])
    env_lo = max(lo + LIMIT_MARGIN_RAD, H1_START_POSE_RAD[j] - 0.20)
    env_hi = min(hi - LIMIT_MARGIN_RAD, H1_START_POSE_RAD[j] + 0.20)
    dev = (device or {}).get(j, {})
    rows.append(JointLimits(
      joint=j,
      abs_min_rad=Limit(lo, "rad", "vendor_sdk+repo_sim", note),
      abs_max_rad=Limit(hi, "rad", "vendor_sdk+repo_sim", note),
      envelope_min_rad=Limit(
        env_lo, "rad", "derived",
        "H1 envelope: within 0.20 rad of the start pose and "
        f"{LIMIT_MARGIN_RAD} rad clear of every absolute bound"),
      envelope_max_rad=Limit(env_hi, "rad", "derived", ""),
      max_speed_rad_s=Limit(H1_STOP_SPEED_RAD_S, "rad/s", "derived",
                            "measured |qdot| stop threshold for H1"),
      max_accel_rad_s2=Limit(H1_STOP_ACCEL_RAD_S2, "rad/s^2", "derived",
                             "provisional until the device's own maximum "
                             "acceleration has been read"),
      max_tracking_error_rad=Limit(H1_STOP_TRACKING_ERROR_RAD, "rad",
                                   "derived", "commanded minus measured"),
      device_max_speed_rad_s=Limit(
        dev.get("max_speed_rad_s"), "rad/s",
        "device" if dev.get("max_speed_rad_s") is not None else UNKNOWN,
        "GetAllMotorAngleLimitMaxSpd; needs an attached arm"),
      device_max_accel_rad_s2=Limit(
        dev.get("max_accel_rad_s2"), "rad/s^2",
        "device" if dev.get("max_accel_rad_s2") is not None else UNKNOWN,
        "GetAllMotorMaxAccLimit, unit 0.001 rad/s^2; needs an attached arm"),
    ))

  env = {
    "driver_temp_stop_c": Limit(
      None, "degC", UNKNOWN,
      "no vendor maximum found in the SDK; must be taken from the "
      "manufacturer's specification or set from an H0 baseline plus a "
      "documented margin before any motion"),
    "motor_temp_stop_c": Limit(None, "degC", UNKNOWN, "as above"),
    "bus_voltage_min_v": Limit(None, "V", UNKNOWN,
                               "GetArmLowSpdInfoMsgs reports 0.1 V units; the "
                               "undervoltage threshold is a vendor number"),
    "joint_current_stop_a": Limit(
      None, "A", UNKNOWN,
      "GetArmHighSpdInfoMsgs reports 0.001 A; the stop threshold is H0's "
      "measured still-arm baseline plus a margin, and H0 has not run"),
    "telemetry_gap_stop_s": Limit(
      0.10, "s", "derived",
      "five missed 50 Hz control periods.  The SDK timestamps frames with "
      "python-can's kernel receive time, so this is measurable"),
    "command_watchdog_s": Limit(
      0.10, "s", "derived",
      "host-side: if the control loop cannot issue a command within this, it "
      "stops issuing them and holds"),
    "gripper_disabled": Limit(
      1.0, "bool", "derived",
      "no gripper command is issued in any RA-HW-0 stage; the field records "
      "that the decision is deliberate rather than accidental"),
  }

  unknowns = [j.joint + "." + k for j in rows for k, v in j.__dict__.items()
              if isinstance(v, Limit) and v.blocks_motion]
  unknowns += [k for k, v in env.items() if v.blocks_motion]
  return {
    "phase": "RA-HW-0",
    "units": {"angle": "rad", "speed": "rad/s", "accel": "rad/s^2",
              "temperature": "degC", "current": "A", "voltage": "V",
              "time": "s"},
    "joint_order": list(ARM_JOINTS),
    "coordinate_note": "joint angles are the SDK's, converted from its "
                       "0.001-degree feedback unit; positive senses are the "
                       "vendor's and have NOT been verified against the "
                       "physical arm",
    "start_pose_rad": dict(H1_START_POSE_RAD),
    "h1": {"commanded_speed_rad_s": H1_COMMANDED_SPEED_RAD_S,
           "max_segment_s": H1_MAX_SEGMENT_S},
    "gripper": {"vendor_range_m": list(VENDOR_GRIPPER_RANGE_M),
                "commanded_in_this_phase": False},
    "joints": [r.to_json() for r in rows],
    "environment": {k: v.to_json() for k, v in env.items()},
    "unknowns_blocking_motion": sorted(set(unknowns)),
    "motion_authorised": False,
    "authorisation_note": "set only by a human replying exactly "
                          "'APPROVE RA-HW-0 MOTION', never by code",
  }


def write(path: str | Path, sim_clip: dict, device: dict | None = None) -> dict:
  table = build(sim_clip, device)
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(table, indent=2))
  return table


def load(path: str | Path) -> dict:
  return json.loads(Path(path).read_text())


def blocks_motion(table: dict) -> list[str]:
  """Everything that must stop a motion request, as a list of reasons."""
  reasons = []
  if not table.get("motion_authorised"):
    reasons.append("no human authorisation recorded")
  if table.get("unknowns_blocking_motion"):
    reasons.append(
      f"{len(table['unknowns_blocking_motion'])} limits are UNKNOWN: "
      + ", ".join(table["unknowns_blocking_motion"][:6])
      + ("..." if len(table["unknowns_blocking_motion"]) > 6 else ""))
  return reasons


def clip_to_envelope(q: list[float], table: dict) -> list[float]:
  """The last arithmetic before any command would leave this process."""
  out = []
  for v, row in zip(q, table["joints"]):
    lo = row["envelope_min_rad"]["value"]
    hi = row["envelope_max_rad"]["value"]
    if lo is None or hi is None:
      raise ValueError(f"{row['joint']} has no envelope; motion is blocked")
    out.append(min(max(float(v), lo), hi))
  return out


def check_state(q, qd, cmd, table: dict, *, qdd=None, temps=None,
                fault_bits=None, gap_s=None, require_known: bool = True
                ) -> list[str]:
  """Every stop condition, evaluated on one sample.  Returns the reasons.

  ``require_known=False`` is for H0 only, where nothing is transmitted: an
  absent threshold is then a finding to record rather than a reason to stop
  looking.  Every stage that commands the arm uses the default.

  Deliberately returns *all* of them rather than the first: a log line that
  says only "over speed" when the joint was also outside its envelope and the
  driver was reporting a fault describes a different failure from the one that
  happened.
  """
  reasons = []
  for i, row in enumerate(table["joints"]):
    j = row["joint"]
    lo, hi = row["envelope_min_rad"]["value"], row["envelope_max_rad"]["value"]
    if lo is not None and not (lo <= q[i] <= hi):
      reasons.append(f"{j}: position {q[i]:.4f} rad outside envelope "
                     f"[{lo:.4f}, {hi:.4f}]")
    vmax = row["max_speed_rad_s"]["value"]
    if vmax is not None and abs(qd[i]) > vmax:
      reasons.append(f"{j}: speed {qd[i]:+.4f} rad/s over {vmax} rad/s")
    amax = row["max_accel_rad_s2"]["value"]
    if qdd is not None and amax is not None and abs(qdd[i]) > amax:
      reasons.append(f"{j}: acceleration {qdd[i]:+.2f} rad/s^2 over {amax}")
    emax = row["max_tracking_error_rad"]["value"]
    if cmd is not None and emax is not None and abs(cmd[i] - q[i]) > emax:
      reasons.append(f"{j}: tracking error {cmd[i] - q[i]:+.4f} rad over {emax}")
    if any(not math.isfinite(x) for x in (q[i], qd[i])):
      reasons.append(f"{j}: non-finite telemetry")
  gapmax = table["environment"]["telemetry_gap_stop_s"]["value"]
  if gap_s is not None and gapmax is not None and gap_s > gapmax:
    reasons.append(f"telemetry gap {gap_s * 1e3:.1f} ms over {gapmax * 1e3:.0f} ms")
  for name, bits in (fault_bits or {}).items():
    for bit, on in (bits or {}).items():
      if on and bit != "driver_enable_status":
        reasons.append(f"{name}: fault bit {bit}")
  for name, t in (temps or {}).items():
    key = "driver_temp_stop_c" if "foc" in name or "driver" in name \
      else "motor_temp_stop_c"
    limit = table["environment"][key]["value"]
    if limit is None:
      # An unknown limit blocks *motion*, not *observation*.  H0 exists to
      # measure the baseline that makes this limit knowable, and a stage that
      # transmits nothing cannot be made safer by refusing to look; so the
      # unknown is recorded by the caller and stops H1, not H0.
      if require_known:
        reasons.append(f"{name}: temperature limit is UNKNOWN, cannot be checked")
    elif t > limit:
      reasons.append(f"{name}: {t} degC over {limit}")
  return reasons
