"""Turning the policy's seven numbers into joint targets, and sending them.

The action mapping is not a detail.  The policy emits values in roughly
[-1, 1]; what the arm receives is ``raw * scale + offset``, clipped to the
safety envelope, and then slew-limited so no single control step asks for more
travel than the joint can make in 20 ms.  All of that happens inside
``piper_push.actions.RateLimitedJointPositionAction`` during training, which
means it is part of what the policy learned to work with, not a wrapper around
it.  Reproduced wrongly here, the policy is being asked to drive a different
robot.

So the constants are imported from ``piper_push.robot`` rather than copied, and
``selftest.py`` steps this mapper and the simulator's action term through the
same random actions and requires them to agree to a micro-radian.

The within-step interpolation the simulator does -- ten physics substeps ramping
towards the target -- is deliberately *not* reproduced.  On the robot that job
belongs to the servo's own inner loop, which runs far faster than this one and
does the same thing.  Reproducing it here would mean sending ten commands per
control step over CAN, and the arm would interpolate them again.

Two backends.  ``DryRunArm`` accepts commands and pretends, with a first-order
lag so a dry run is not trivially perfect; it is what lets the whole stack be
exercised on a desk with nothing plugged in.  ``PiperArm`` talks to the real
one over CAN and is the only part of this directory that has never been run
against hardware -- there is no robot on this machine.  It is written from the
SDK's documented interface and it is marked, here and in the README, as
unverified.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from piper_push import robot as sim_robot

from . import config
from .proprio import JointFeedback

RESPONSE_PLAIN = 0x00
RESPONSE_MIT = 0xAD
"""``MotionCtrl_2``'s fourth field: the drives' response law.

The SDK calls ``0xAD`` "MIT mode" and its ``piper_set_mit.py`` demo describes it
as "设置机械臂为mit控制模式，这个模式下，机械臂相应最快" -- the ordinary position
path, executed with the fastest response the drives have.  It is NOT the
per-joint impedance interface (``JointMitCtrl``, see ``mit.py``), which this
arm's S-V1.8-9 firmware accepts on the bus and then ignores.

Measured on this rig, joint6 tracking a 1.2 rad/s triangle at 50 Hz, with the
speed field held at 100 so the flag is the only thing that changed:

    plain   lag p50 0.190   p95 0.329   |dq| p50 0.50
    0xAD    lag p50 0.089   p95 0.106   |dq| p50 0.97
    plain   lag p50 0.190   p95 0.329   |dq| p50 0.51   (repeat, to bracket it)

Default is PLAIN so that the calibration and teaching tools keep the behaviour
they were tested against; ``run.py --control mit`` turns it on for the policy.
"""

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINT = "gripper_joint1"

GRIPPER_TORQUE_NM = 1.5
"""What to ask the PiPER's gripper for, in its own units.

Deliberately not derived from ``sim_robot.GRIPPER_FORCE_N``.  That constant is
10 N -- a *force*, the URDF's effort limit on a prismatic finger joint -- and
the CAN message wants a *torque*, in 0.001 N.m over a documented range of
0-5000.  The two are not convertible without the gripper's internal lever arm,
which is not published, so the previous code's ``int(force_n * 1000)`` was not
a wrong conversion so much as no conversion: it produced 10000, which is twice
the maximum the SDK accepts, and ``ArmMsgGripperCtrl`` raises ``ValueError`` on
construction.  Every gripper command would have thrown.

1.5 N.m is 30% of the drive's range and a starting point, not a measurement.
The number that would justify itself is a force-vs-command curve measured on
the rig, which ``piper_push.robot`` already asks for in its own note about
``GRIPPER_STIFFNESS``; until that exists this is a rig parameter to turn up
until objects stop slipping and no further.
"""

GRIPPER_TORQUE_MAX_NM = 5.0
"""``arm_gripper_ctrl.py``: "Range 0-5000, corresponse 0-5N/m"."""

GRIPPER_JAW_GAP_M = 2.0 * sim_robot.GRIPPER_OPEN_M
"""The simulator's jaw gap, 100 mm, which ``piper_push.robot`` records as
measured.  The PiPER's own maximum stroke is a *configured* value and the SDK's
default is 70 -- see ``PiperArm.check_gripper_range``, because a 70 mm arm
running a 100 mm policy saturates over the top 30% of the command range and
reports a gripper opening the policy has never seen.
"""


class ActionMapper:
  """The policy's action to a joint target, exactly as in training.

  Holds one piece of state -- the previous target -- because the slew limit is
  relative to it and not to where the arm currently is.  That distinction
  matters: limiting against the measured position would let a stalled joint
  wind its target up without bound, and would make the command path depend on
  the arm's dynamics in a way the trained one does not.
  """

  def __init__(self, spec: dict, dt: float = 1.0 / config.CONTROL_HZ,
               clip_actions: float | None = None,
               accel_limit: float | None = None,
               gripper_accel_limit: float | None = None):
    self.dt = float(dt)
    self.clip_actions = clip_actions
    self.accel_limit = (None if accel_limit is None else np.asarray(
      [float(accel_limit)] * 6 + [float(
        gripper_accel_limit if gripper_accel_limit is not None else accel_limit
      )], dtype=np.float64))
    self.previous_velocity = np.zeros(7, dtype=np.float64)
    names = list(spec["joint_names"])
    default = np.asarray(spec["default_joint_pos"], dtype=np.float64)

    self.arm_idx = [names.index(j) for j in ARM_JOINTS]
    self.grip_idx = names.index(GRIPPER_JOINT)

    # The convention the policy was trained under, from the spec written next
    # to it.  Exports before 2026-09-05 carry no "action_spec": that is the v1
    # convention (PICK_ARM_SCALE about the default pose, nothing bounding a).
    # Exports since carry the bounded one (a = +-1 is the safe clip, tanh head);
    # driving either with the other's constants is driving a different robot.
    aspec = spec.get("action_spec")
    if aspec is None:
      aspec = sim_robot.action_spec("v1", dict(zip(names, default.tolist())))
    if list(aspec["joints"]) != list(ARM_JOINTS) + [GRIPPER_JOINT]:
      raise ValueError(f"action_spec joints {aspec['joints']} are not {list(ARM_JOINTS) + [GRIPPER_JOINT]}")
    self.convention = str(aspec["convention"])
    self.scale = np.asarray(aspec["scale"], dtype=np.float64)
    self.offset = np.asarray(aspec["offset"], dtype=np.float64)
    lo, hi = [], []
    for j in ARM_JOINTS:
      a, b = sim_robot.SAFE_TARGET_CLIP[j]
      lo.append(a)
      hi.append(b)
    a, b = sim_robot.GRIPPER_CLIP[GRIPPER_JOINT]
    lo.append(a)
    hi.append(b)
    self.lo, self.hi = np.array(lo), np.array(hi)

    rate = [sim_robot.COMMAND_RATE_LIMIT_RAD_S[j] for j in ARM_JOINTS]
    rate.append(sim_robot.GRIPPER_RATE_LIMIT_M_S)
    self.max_step = np.asarray(rate) * self.dt

    self.default_target = np.array(
      [default[i] for i in self.arm_idx] + [default[self.grip_idx]]
    )
    self.previous = self.default_target.copy()

  def reset(self, position: np.ndarray | None = None) -> None:
    """Start slewing from where the arm actually is, not from the nominal pose.

    The simulator's action term does exactly this, and says why: reset events
    run first, so the arm is already in its fresh posture, and a limiter that
    started from the default would hand the servo the whole reset offset as a
    single step.  In simulation that tripped the safety shell in 30% of
    episodes.  On the robot the arm is wherever the last run left it, which can
    be most of a radian away, and the first command would be a lunge.

    ``position`` is the measured joint state, ``(8,)`` in the simulator's joint
    order or ``(7,)`` as six arm joints and a finger.
    """
    self.previous_velocity[:] = 0.0
    if position is None:
      self.previous = self.default_target.copy()
      return
    p = np.asarray(position, dtype=np.float64).reshape(-1)
    if p.size == 7:
      self.previous = p.copy()
    else:
      self.previous = np.array([p[i] for i in self.arm_idx] + [p[self.grip_idx]])

  def __call__(self, action: np.ndarray,
               dt: float | None = None) -> np.ndarray:
    """``(7,)`` policy output to ``(7,)`` joint targets: six radians, one metre."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    step_dt = self.dt if dt is None else float(dt)
    if not np.isfinite(step_dt) or step_dt <= 0.0:
      raise ValueError("action mapping dt must be finite and positive")
    if a.size != 7:
      raise ValueError(f"expected 7 actions, got {a.size}")
    if self.clip_actions is not None:
      a = np.clip(a, -self.clip_actions, self.clip_actions)
    target = np.clip(a * self.scale + self.offset, self.lo, self.hi)
    delta = target - self.previous
    if self.accel_limit is not None:
      requested_velocity = delta / step_dt
      dv = self.accel_limit * step_dt
      velocity = np.clip(requested_velocity,
                         self.previous_velocity - dv,
                         self.previous_velocity + dv)
      delta = velocity * step_dt
    max_step = self.max_step * (step_dt / self.dt)
    delta = np.clip(delta, -max_step, max_step)
    self.previous = self.previous + delta
    self.previous_velocity = delta / step_dt
    return self.previous.copy()


@dataclasses.dataclass
class ArmState:
  q: np.ndarray            # (6,) radians
  dq: np.ndarray           # (6,) rad/s
  gripper: float           # metres, one finger
  gripper_vel: float
  gripper_effort: float    # normalised, 1.0 = stall
  stamp: float


class DryRunArm:
  """A stand-in that moves the way a position servo roughly does.

  First-order lag rather than an exact follower, so that a dry run does not
  silently depend on the arm being perfect.  The time constant is a guess and
  is labelled as one; nothing downstream should be tuned against it.
  """

  def __init__(self, spec: dict, tau: float = 0.06):
    names = list(spec["joint_names"])
    default = np.asarray(spec["default_joint_pos"], dtype=np.float64)
    self._q = np.array([default[names.index(j)] for j in ARM_JOINTS])
    self._dq = np.zeros(6)
    self._g = float(default[names.index(GRIPPER_JOINT)])
    self._gv = 0.0
    self.tau = float(tau)
    self.connected = True

  def connect(self) -> None:
    self.connected = True

  def enable(self) -> None:
    pass

  def read(self) -> ArmState:
    return ArmState(q=self._q.copy(), dq=self._dq.copy(), gripper=self._g,
                    gripper_vel=self._gv, gripper_effort=0.0, stamp=time.time())

  def command(self, target: np.ndarray, dt: float = 1.0 / config.CONTROL_HZ) -> None:
    alpha = dt / (dt + self.tau)
    nq = self._q + alpha * (np.asarray(target[:6]) - self._q)
    self._dq = (nq - self._q) / dt
    self._q = nq
    ng = self._g + alpha * (float(target[6]) - self._g)
    self._gv = (ng - self._g) / dt
    self._g = ng

  def hold(self) -> None:
    pass

  def close(self, disable: bool = False) -> None:
    del disable
    self.connected = False


class PiperArm:
  """The AgileX PiPER over CAN.

  The message conversions were originally written from the SDK documentation;
  the calibration GUI is now running against the real arm.  That bring-up
  exposed one safety-critical lifecycle fact: ``DisableArm`` removes holding
  torque and must never be an ordinary software-close operation.  ``close``
  therefore holds and disconnects unless deliberate disable is requested.

  Units are the trap.  The SDK takes joint angles in units of 0.001 degrees and
  the gripper in 0.001 mm, while everything above this line is radians and
  metres, and a missed conversion is a factor of 57000.
  """

  RAD_TO_MDEG = 180.0 / np.pi * 1000.0
  M_TO_UM = 1e6

  response_mode = RESPONSE_PLAIN
  """Which response law ``command`` asks the drives for.  A class attribute so
  that it exists however the object was constructed; ``run.py --control`` sets
  it per session."""

  def __init__(self, can: str = config.CAN_INTERFACE,
               gripper_torque_nm: float = GRIPPER_TORQUE_NM):
    try:
      from piper_sdk import C_PiperInterface_V2
    except ImportError as e:                       # pragma: no cover
      raise RuntimeError(
        "piper_sdk is not installed.  `pip install piper_sdk`, bring the CAN "
        "interface up with the SDK's can_activate.sh, and check `ip -br link` "
        "shows it before trying again."
      ) from e
    self._iface = C_PiperInterface_V2(can)
    self.gripper_torque_nm = float(
      np.clip(gripper_torque_nm, 0.0, GRIPPER_TORQUE_MAX_NM))
    self.connected = False
    self._prev: tuple[float, np.ndarray, float] | None = None

  def connect(self) -> None:
    self._iface.ConnectPort()
    self.connected = True

  @property
  def iface(self):
    """The SDK interface, for command paths that are not this class.

    ``mit.MitDriver`` streams impedance frames to the same drives over the same
    connection, and one process must not open two.  Exposed deliberately rather
    than reached for through the private name, so that "something else is also
    commanding this arm" is visible in the code that does it.
    """
    return self._iface

  def enable(self, timeout_s: float = 5.0) -> None:
    """Enable every joint, and refuse to continue if any of them did not.

    The SDK's enable is a request, not an acknowledgement.  A run that starts
    with one joint still disabled looks like a policy that cannot reach.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      self._iface.EnableArm(7)
      msgs = self._iface.GetArmLowSpdInfoMsgs()
      states = [getattr(msgs, f"motor_{i}").foc_status.driver_enable_status
                for i in range(1, 7)]
      if all(states):
        return
      time.sleep(0.1)
    raise RuntimeError(f"arm did not enable within {timeout_s} s")

  def read(self) -> ArmState:
    """Position and gripper load from the drives; velocity by differencing.

    The drives do publish a speed, and it is deliberately not used.  Its unit
    is not the one the joint angles are in and the SDK's documentation does not
    pin it down, so a conversion here would be a guess feeding straight into
    ``joint_vel`` -- eight of the policy's thirty-six inputs, at a scale nobody
    has checked.  A difference over the control period needs no unit at all,
    and it is closer to what the observation means anyway: ``joint_vel_rel`` in
    simulation is the velocity across a control step, not an instantaneous one.

    The cost is one period of lag and the quantisation of the encoder divided
    by 20 ms.  Worth revisiting once someone can put the two side by side on a
    moving arm; until then this one cannot be wrong by a factor.
    """
    now = time.time()
    j = self._iface.GetArmJointMsgs().joint_state
    g = self._iface.GetArmGripperMsgs().gripper_state
    q = np.array([j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5,
                  j.joint_6], dtype=np.float64) / self.RAD_TO_MDEG
    # The SDK reports the gripper as the full opening; the simulator's joint is
    # one finger, which is half of it.
    gripper = float(g.grippers_angle) / self.M_TO_UM / 2.0

    dq = np.zeros(6)
    gripper_vel = 0.0
    if self._prev is not None:
      t_prev, q_prev, g_prev = self._prev
      dt = now - t_prev
      # A stale or duplicated CAN sample gives dt near zero and a velocity of
      # whatever the encoder noise was, divided by nothing.
      if dt > 1e-3:
        dq = (q - q_prev) / dt
        gripper_vel = (gripper - g_prev) / dt
    self._prev = (now, q.copy(), gripper)

    # Reported in 0.001 N.m, the same unit as the command, so the ratio is
    # dimensionless and 1.0 means "the drive is doing what it was told".  The
    # channel it feeds is ``pad_contact``, whose deployable content is one bit
    # -- the drive is loaded -- so what matters is that the scale is stable and
    # in the right kind of unit, not that it is a force.
    effort = float(g.grippers_effort) / 1000.0 / max(self.gripper_torque_nm,
                                                     1e-6)
    return ArmState(q=q, dq=dq, gripper=gripper, gripper_vel=gripper_vel,
                    gripper_effort=effort, stamp=now)

  def command(self, target: np.ndarray, dt: float = 1.0 / config.CONTROL_HZ) -> None:
    """The last thing before the motors, so it clamps rather than trusts.

    ``ActionMapper`` already clips to ``SAFE_TARGET_CLIP`` and slew-limits, and
    this repeats the clip anyway.  ``ArmMsgJointCtrl`` has no range check of
    its own -- unlike the gripper message, which does -- so a units error
    upstream reaches the drives as a number they will try to achieve.  Six
    comparisons is a cheap thing to put between that and the table.
    """
    del dt
    lo = np.array([sim_robot.SAFE_TARGET_CLIP[j][0] for j in ARM_JOINTS])
    hi = np.array([sim_robot.SAFE_TARGET_CLIP[j][1] for j in ARM_JOINTS])
    q = np.clip(np.asarray(target[:6], dtype=np.float64), lo, hi)
    q = np.round(q * self.RAD_TO_MDEG).astype(int)
    self._iface.MotionCtrl_2(0x01, 0x01, 100, self.response_mode)
    self._iface.JointCtrl(*q.tolist())
    self.command_gripper(target[6])

  def command_gripper(self, one_finger_m: float) -> None:
    """The gripper alone.

    Its own method because MIT mode covers motors 1-6 and nothing else: an
    impedance command path still has to drive the jaw through this message, and
    duplicating the unit conversion in two places is how the two drift apart.
    """
    # The policy's gripper value is one finger; the CAN message is the jaw gap.
    opening = float(np.clip(float(one_finger_m) * 2.0, 0.0, GRIPPER_JAW_GAP_M))
    effort = int(round(self.gripper_torque_nm * 1000.0))
    self._iface.GripperCtrl(int(round(opening * self.M_TO_UM)),
                            int(np.clip(effort, 0, 5000)), 0x01, 0)

  def check_gripper_range(self, timeout_s: float = 1.0) -> tuple[float, bool]:
    """The jaw gap this arm is configured for, against the one trained.

    ``GripperTeachingPendantParamConfig``'s ``max_range_config`` defaults to 70
    in the SDK and the simulator's measured gap is 100.  On a 70 mm arm the top
    30% of the policy's gripper command does nothing and the opening it reads
    back never exceeds 70 -- so the policy commands a gap it never observes,
    which is not a failure any log would name.  Returns the configured gap in
    metres and whether it matches.
    """
    # The feedback frame is not periodic -- it is answered on request, and the
    # request is enquiry 0x04 ("query gripper/teaching pendant parameter
    # index").  Reading the accessor without asking first returns the SDK's
    # zero-initialised message, which is indistinguishable from an arm that
    # says its jaw gap is zero.
    self._iface.ArmParamEnquiryAndConfig(param_enquiry=0x04)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      fb = self._iface.GetGripperTeachingPendantParamFeedback()
      mm = float(getattr(fb, "max_range_config", 0.0) or 0.0)
      if mm > 0.0:
        return mm / 1000.0, abs(mm / 1000.0 - GRIPPER_JAW_GAP_M) < 1e-3
      time.sleep(0.05)
    # Firmware before V1.5-2 does not answer this enquiry at all, which is not
    # the same as a wrong gap -- so it is reported as unknown, not as a fault.
    return float("nan"), False

  def hold(self) -> None:
    """Stop moving but stay enabled.  What to do when the loop falls behind."""
    st = self.read()
    self.command(np.concatenate([st.q, [st.gripper]]))

  def disconnect(self) -> None:
    """Release the SDK connection without changing drive state or targets.

    Read-only tools use this instead of ``close``: they never enabled or
    commanded the arm, so issuing even a same-pose hold command on exit would
    violate that contract.  Disconnecting CAN does not disable the drives.
    """
    if not self.connected:
      return
    try:
      self._iface.DisconnectPort()
    finally:
      self.connected = False

  def close(self, disable: bool = False) -> None:
    """Release CAN without dropping gravity support.

    ``DisableArm(7)`` makes every joint lose holding torque immediately.  The
    old unconditional call here was observed on the real rig as the arm
    falling whenever calibgui exited or restarted.  Closing a software client
    is not authorization to remove actuator power: by default command the
    measured pose once, then disconnect the SDK while the drives remain
    enabled.  Deliberate power-down must be explicit with ``disable=True`` and
    should only be used while the arm is physically supported or parked.
    """
    if not self.connected:
      return
    try:
      if disable:
        self._iface.DisableArm(7)
      else:
        try:
          self.hold()
        except Exception:
          # A failed final read/hold must still never fall through to disable.
          # The drive retains its last position target.
          pass
    finally:
      self.disconnect()


# The guided-calibration motion parameters, which are the only ones on this
# rig that have driven the arm across the workspace without incident.  Reused
# rather than re-chosen: a second set of numbers for the same job is a second
# thing to get wrong.
AUTO_SPEED_RAD_S = 0.22
AUTO_RATE_HZ = 30.0
AUTO_TRACKING_ERROR_RAD = 0.45


def joint_trajectory(q0, q1, speed_rad_s: float = AUTO_SPEED_RAD_S,
                     rate_hz: float = AUTO_RATE_HZ) -> np.ndarray:
  """A rest-to-rest joint path whose peak speed is bounded.

  ``3 u^2 - 2 u^3`` peaks at 1.5 times its average speed, hence the 1.5 in the
  duration.  The first row is the measured position, so enabling the arm is
  immediately followed by a hold at exactly where it already is.
  """
  a = np.asarray(q0, dtype=np.float64).reshape(6)
  b = np.asarray(q1, dtype=np.float64).reshape(6)
  distance = float(np.max(np.abs(b - a)))
  duration = max(0.8, 1.5 * distance / max(float(speed_rad_s), 1e-3))
  n = max(2, int(np.ceil(duration * float(rate_hz))) + 1)
  u = np.linspace(0.0, 1.0, n)
  blend = 3.0 * u ** 2 - 2.0 * u ** 3
  return a[None, :] + blend[:, None] * (b - a)[None, :]


def feedback(state: ArmState, target: np.ndarray) -> JointFeedback:
  """Arm telemetry into the shape ``ProprioBuilder`` wants."""
  return JointFeedback.from_arm(
    q6=state.q, dq6=state.dq, gripper_m=state.gripper,
    gripper_vel=state.gripper_vel, target6=target[:6],
    gripper_target_m=float(target[6]), gripper_effort=state.gripper_effort,
  )
