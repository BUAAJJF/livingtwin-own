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

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINT = "gripper_joint1"


class ActionMapper:
  """The policy's action to a joint target, exactly as in training.

  Holds one piece of state -- the previous target -- because the slew limit is
  relative to it and not to where the arm currently is.  That distinction
  matters: limiting against the measured position would let a stalled joint
  wind its target up without bound, and would make the command path depend on
  the arm's dynamics in a way the trained one does not.
  """

  def __init__(self, spec: dict, dt: float = 1.0 / config.CONTROL_HZ,
               clip_actions: float | None = None):
    self.dt = float(dt)
    self.clip_actions = clip_actions
    names = list(spec["joint_names"])
    default = np.asarray(spec["default_joint_pos"], dtype=np.float64)

    self.arm_idx = [names.index(j) for j in ARM_JOINTS]
    self.grip_idx = names.index(GRIPPER_JOINT)

    self.scale = np.array(
      [sim_robot.PICK_ARM_SCALE[j] for j in ARM_JOINTS] + [sim_robot.GRIPPER_SCALE]
    )
    self.offset = np.array(
      [default[i] for i in self.arm_idx] + [sim_robot.GRIPPER_OFFSET]
    )
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
    if position is None:
      self.previous = self.default_target.copy()
      return
    p = np.asarray(position, dtype=np.float64).reshape(-1)
    if p.size == 7:
      self.previous = p.copy()
    else:
      self.previous = np.array([p[i] for i in self.arm_idx] + [p[self.grip_idx]])

  def __call__(self, action: np.ndarray) -> np.ndarray:
    """``(7,)`` policy output to ``(7,)`` joint targets: six radians, one metre."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size != 7:
      raise ValueError(f"expected 7 actions, got {a.size}")
    if self.clip_actions is not None:
      a = np.clip(a, -self.clip_actions, self.clip_actions)
    target = np.clip(a * self.scale + self.offset, self.lo, self.hi)
    delta = np.clip(target - self.previous, -self.max_step, self.max_step)
    self.previous = self.previous + delta
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

  def close(self) -> None:
    self.connected = False


class PiperArm:
  """The AgileX PiPER over CAN.

  UNVERIFIED AGAINST HARDWARE.  There is no arm on the machine this was written
  on and no CAN interface in ``ip link``, so every line below is from the SDK's
  documented interface and none of it has been executed.  Treat the first run
  as a bring-up: ``python -m hardware.deploy.jointcheck --joint 1`` moves one
  joint five degrees and prints what came back, which is the smallest motion
  that can tell a working conversion from a broken one.

  Units are the trap.  The SDK takes joint angles in units of 0.001 degrees and
  the gripper in 0.001 mm, while everything above this line is radians and
  metres, and a missed conversion is a factor of 57000.
  """

  RAD_TO_MDEG = 180.0 / np.pi * 1000.0
  M_TO_UM = 1e6

  def __init__(self, can: str = config.CAN_INTERFACE,
               gripper_force_n: float = sim_robot.GRIPPER_FORCE_N):
    try:
      from piper_sdk import C_PiperInterface_V2
    except ImportError as e:                       # pragma: no cover
      raise RuntimeError(
        "piper_sdk is not installed.  `pip install piper_sdk`, bring the CAN "
        "interface up with the SDK's can_activate.sh, and check `ip -br link` "
        "shows it before trying again."
      ) from e
    self._iface = C_PiperInterface_V2(can)
    self.gripper_force_n = float(gripper_force_n)
    self.connected = False
    self._prev: tuple[float, np.ndarray, float] | None = None

  def connect(self) -> None:
    self._iface.ConnectPort()
    self.connected = True

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

    effort = float(g.grippers_effort) / 1000.0 / max(self.gripper_force_n, 1e-6)
    return ArmState(q=q, dq=dq, gripper=gripper, gripper_vel=gripper_vel,
                    gripper_effort=effort, stamp=now)

  def command(self, target: np.ndarray, dt: float = 1.0 / config.CONTROL_HZ) -> None:
    del dt
    q = (np.asarray(target[:6], dtype=np.float64) * self.RAD_TO_MDEG).astype(int)
    self._iface.MotionCtrl_2(0x01, 0x01, 100, 0x00)
    self._iface.JointCtrl(*q.tolist())
    opening = int(float(target[6]) * 2.0 * self.M_TO_UM)
    self._iface.GripperCtrl(abs(opening), int(self.gripper_force_n * 1000),
                            0x01, 0)

  def hold(self) -> None:
    """Stop moving but stay enabled.  What to do when the loop falls behind."""
    st = self.read()
    self.command(np.concatenate([st.q, [st.gripper]]))

  def close(self) -> None:
    try:
      self._iface.DisableArm(7)
    finally:
      self.connected = False


def feedback(state: ArmState, target: np.ndarray) -> JointFeedback:
  """Arm telemetry into the shape ``ProprioBuilder`` wants."""
  return JointFeedback.from_arm(
    q6=state.q, dq6=state.dq, gripper_m=state.gripper,
    gripper_vel=state.gripper_vel, target6=target[:6],
    gripper_target_m=float(target[6]), gripper_effort=state.gripper_effort,
  )
