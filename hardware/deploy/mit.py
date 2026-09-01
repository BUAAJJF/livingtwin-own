"""The MIT-mode boundary: impedance commands, clamped, and the drives' own view.

The deployment has been driving the arm with ``MotionCtrl_2(0x01, 0x01, 100,
0x00)`` + ``JointCtrl`` -- MOVE J, which hands the controller a target and lets
it re-plan its own trajectory every 20 ms.  The simulator models something
else entirely: ``BuiltinPositionActuatorCfg`` with the identified gains, i.e.

    tau = kp (p_ref - p) - kd v + tau_gravcomp

which is exactly what MIT mode exposes.  ``get_pick_spec`` even sets
``body.gravcomp`` with the comment "the production command path streams t_ff =
full inverse dynamics" -- a command path that did not exist until this module.
So MIT is not a workaround for the tracking gap measured on 2026-08-30; it is
the plant the policy was trained against.

**The SDK will not protect you here.**  ``JointMitCtrl``'s docstring advertises
``t_ref`` in [-18, 18] N.m; the encoder behind it packs 8 bits over [-8, 8] and
``FloatToUint`` does not clamp, so the byte wraps.  Asking for -12 N.m lands on
**+4.11**.  Every field is clamped here, on the way out, and clamping is
counted rather than silent.

Nothing in this module knows what the numbers mean physically.  ``kp``, ``kd``
and ``t_ref`` go to the drives as written; whether a written kp of 10 is 10
N.m/rad at the joint is what ``sysid.py`` measures.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import numpy as np

from . import config

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))

# The ENCODER's ranges, from ``C_PiperParserV2.FloatToUint`` as called by
# ``__JointMitCtrl`` -- not the public docstring's, which disagree on torque.
FIELD_RANGE = {
  "pos": (-12.5, 12.5),
  "vel": (-45.0, 45.0),
  "kp": (0.0, 500.0),
  "kd": (-5.0, 5.0),
  "torque": (-8.0, 8.0),
}

MIT_MODE = 0xAD
POSITION_MODE = 0x00

# The move mode matters as much as the MIT flag, and getting it wrong fails
# silently.  The SDK ships two demos:
#
#   piper_set_mit.py             MotionCtrl_2(1, 1, 0, 0xAD)     + JointCtrl
#     "set the arm to MIT control mode, in which it responds fastest" -- the
#     ordinary position path, executed with MIT-speed response.
#   V2_piper_ctrl_joint_mit.py   MotionCtrl_2(1, 4, 0, 0xAD)     + JointMitCtrl
#     "set MIT control for an individual motor" -- per-joint impedance.
#
# This module wants the second.  Sent with MOVE_J instead, the controller stays
# in its own joint planner and every ``JointMitCtrl`` frame is discarded: the
# first attempt on this rig streamed 2569 of them at a joint commanded +-3 deg
# and the joint moved 0.0000 deg, while the hold test read a flawless zero sag
# because nothing was ever under impedance control at all.
MOVE_J = 0x01
MOVE_M = 0x04

GAINS_FILE = pathlib.Path(__file__).with_name("mit_gains.json")
"""Where the measured written gains live.  Absent until ``sysid`` has run,
which is deliberate: there is no defensible default for a number the drives
amplify by a factor this rig has not measured."""

# The manufacturer's suggested starting impedance, from the SDK docstring
# ("参考值---10" / "参考值---0.8").  Deliberately the default here: until the
# written-to-effective scale is measured, the only defensible starting point is
# the one the vendor names.
VENDOR_KP = 10.0
VENDOR_KD = 0.8


@dataclasses.dataclass(frozen=True)
class Gains:
  """Per-joint written kp/kd, in whatever units the drives interpret them in."""

  kp: np.ndarray
  kd: np.ndarray

  @staticmethod
  def uniform(kp: float = VENDOR_KP, kd: float = VENDOR_KD) -> "Gains":
    return Gains(np.full(6, float(kp)), np.full(6, float(kd)))

  @staticmethod
  def load(path: pathlib.Path) -> "Gains":
    d = json.loads(pathlib.Path(path).read_text())
    return Gains(np.array([d["kp"][j] for j in ARM_JOINTS], dtype=np.float64),
                 np.array([d["kd"][j] for j in ARM_JOINTS], dtype=np.float64))

  def save(self, path: pathlib.Path, **meta) -> None:
    pathlib.Path(path).write_text(json.dumps({
      "kp": {j: round(float(v), 4) for j, v in zip(ARM_JOINTS, self.kp)},
      "kd": {j: round(float(v), 4) for j, v in zip(ARM_JOINTS, self.kd)},
      **meta,
    }, indent=1) + "\n")

  def scaled(self, f: float) -> "Gains":
    return Gains(self.kp * f, self.kd * f)


@dataclasses.dataclass
class MotorFeedback:
  """What the drives report about themselves, at their own rate."""

  stamp: float
  motor_pos: np.ndarray
  """(6,) rad, the drive's own position register -- NOT the joint angle the
  0x2A5-7 messages carry.  Their ratio is the gear reduction, and measuring it
  is the first thing ``sysid probe`` does."""
  motor_speed: np.ndarray
  """(6,) rad/s as reported, scaled from the SDK's 0.001 rad/s."""
  current: np.ndarray
  """(6,) A."""
  effort: np.ndarray
  """(6,) N.m as reported, scaled from the SDK's 0.001 N.m.  The SDK calls it
  "torque converted using a fixed coefficient", so it is a current reading in
  torque clothing until ``sysid`` cross-checks it."""


def read_motors(iface) -> MotorFeedback:
  m = iface.GetArmHighSpdInfoMsgs()
  get = lambda i: getattr(m, f"motor_{i}")
  return MotorFeedback(
    stamp=time.time(),
    motor_pos=np.array([float(get(i).pos) for i in range(1, 7)]),
    motor_speed=np.array([get(i).motor_speed / 1000.0 for i in range(1, 7)]),
    current=np.array([get(i).current / 1000.0 for i in range(1, 7)]),
    effort=np.array([get(i).effort / 1000.0 for i in range(1, 7)]),
  )


class ClampCounter:
  """How often a command asked for something the CAN field cannot carry.

  A clamp is not an error -- the gravity feedforward on joint2 legitimately
  wants more than 8 N.m at some postures -- but it IS a silent change of what
  the policy asked for, so it is counted and reported rather than absorbed.
  """

  def __init__(self) -> None:
    self.counts = {k: 0 for k in FIELD_RANGE}
    self.worst = {k: 0.0 for k in FIELD_RANGE}

  def clamp(self, field: str, x: np.ndarray) -> np.ndarray:
    lo, hi = FIELD_RANGE[field]
    out = np.clip(x, lo, hi)
    over = np.abs(out - x)
    n = int((over > 1e-9).sum())
    if n:
      self.counts[field] += n
      self.worst[field] = max(self.worst[field], float(over.max()))
    return out

  def report(self) -> dict:
    return {k: {"clamped": self.counts[k], "worst_excess": round(self.worst[k], 4)}
            for k in FIELD_RANGE if self.counts[k]}


class MitDriver:
  """Impedance commands to the six arm drives, and the mode switch around them.

  Deliberately not a subclass of ``robot.PiperArm``: this owns a command path
  with a different failure surface, and the two have to be comparable side by
  side on the same arm within one session.
  """

  def __init__(self, iface, gains: Gains | None = None,
               torque_limit: float = 2.0, dry_run: bool = False) -> None:
    self._iface = iface
    self.gains = gains or Gains.uniform()
    # Not the field's 8 N.m.  Until the written-to-delivered torque scale is
    # measured, a feedforward that turns out to be motor-referred rather than
    # joint-referred would be multiplied by the gear ratio, so the first number
    # this class is allowed to send is small enough to be survivable either way.
    self.torque_limit = float(torque_limit)
    self.dry_run = bool(dry_run)
    self.clamps = ClampCounter()
    self.active = False
    self._sent = 0

  # -- mode ----------------------------------------------------------------

  def enter(self, hold_q: np.ndarray) -> None:
    """Switch to MIT and immediately hold the pose the arm is already in.

    The first MIT frame is the dangerous one: whatever ``pos_ref`` it carries
    becomes a step input against the full impedance.  So it is the measured
    pose, with zero velocity reference and zero feedforward, and nothing else
    happens until the caller asks for it.
    """
    q = np.asarray(hold_q, dtype=np.float64).reshape(6)
    if not np.isfinite(q).all():
      raise ValueError("refusing to enter MIT on a non-finite pose")
    self.active = True
    self.send(q, np.zeros(6), np.zeros(6))

  def leave(self, hold_q: np.ndarray) -> None:
    """Back to position mode, holding the measured pose.

    Called from every exit path including exceptions.  Leaving the drives in
    MIT with nobody streaming is the one state this module must never end in:
    the impedance stays whatever the last frame said, indefinitely.
    """
    q = np.asarray(hold_q, dtype=np.float64).reshape(6)
    self.active = False
    if self.dry_run:
      return
    self._iface.MotionCtrl_2(0x01, MOVE_J, 100, POSITION_MODE)
    self._iface.JointCtrl(*np.round(q * (180000.0 / np.pi)).astype(int).tolist())

  # -- command -------------------------------------------------------------

  def send(self, q_ref: np.ndarray, dq_ref: np.ndarray,
           tau_ff: np.ndarray, gains: Gains | None = None) -> dict:
    """One impedance frame per joint.  Returns what was actually sent."""
    if not self.active:
      raise RuntimeError("send() before enter()")
    g = gains or self.gains
    q = self.clamps.clamp("pos", np.asarray(q_ref, dtype=np.float64).reshape(6))
    dq = self.clamps.clamp("vel", np.asarray(dq_ref, dtype=np.float64).reshape(6))
    t = np.asarray(tau_ff, dtype=np.float64).reshape(6)
    t = np.clip(t, -self.torque_limit, self.torque_limit)
    t = self.clamps.clamp("torque", t)
    kp = self.clamps.clamp("kp", np.asarray(g.kp, dtype=np.float64).reshape(6))
    kd = self.clamps.clamp("kd", np.asarray(g.kd, dtype=np.float64).reshape(6))
    if not (np.isfinite(q).all() and np.isfinite(dq).all() and np.isfinite(t).all()):
      raise ValueError("non-finite MIT command")
    if not self.dry_run:
      self._iface.MotionCtrl_2(0x01, MOVE_M, 0, MIT_MODE)
      for i in range(6):
        self._iface.JointMitCtrl(i + 1, float(q[i]), float(dq[i]),
                                 float(kp[i]), float(kd[i]), float(t[i]))
    self._sent += 1
    return {"q_ref": q, "dq_ref": dq, "tau_ff": t, "kp": kp, "kd": kd}

  @property
  def frames_sent(self) -> int:
    return self._sent


def encoded(x: float, field: str) -> float:
  """What the drives will actually receive for ``x``, after quantisation.

  Round-trips one value through the SDK's fixed-point packing so a test can
  assert on the number the hardware sees rather than the number Python held.
  """
  bits = {"pos": 16, "vel": 12, "kp": 12, "kd": 12, "torque": 8}[field]
  lo, hi = FIELD_RANGE[field]
  n = int((float(x) - lo) * ((1 << bits) - 1) / (hi - lo))
  n &= (1 << bits) - 1                       # the wrap FloatToUint does not stop
  return lo + n * (hi - lo) / ((1 << bits) - 1)


def resolution(field: str) -> float:
  bits = {"pos": 16, "vel": 12, "kp": 12, "kd": 12, "torque": 8}[field]
  lo, hi = FIELD_RANGE[field]
  return (hi - lo) / ((1 << bits) - 1)


class GravityFeedforward:
  """The ``t_ff`` the simulator already assumes is being streamed.

  ``robot.get_pick_spec`` sets ``body.gravcomp = 1.0`` on every link, with the
  comment "the production command path streams t_ff = full inverse dynamics,
  so the real arm is gravity compensated for every mass its model knows
  about".  In MOVE J that was aspirational -- the command path streamed joint
  targets and nothing else.  Under MIT it is a thing this class computes.

  Bias, not gravity alone: ``qfrc_bias`` is ``C(q, v) v + g(q)``, so passing
  the measured velocity carries the Coriolis terms the simulator's actuator
  also does not have to fight.  Pass zeros to get pure gravity.
  """

  def __init__(self, joint_names=ARM_JOINTS) -> None:
    import mujoco

    from piper_push import robot as sim_robot

    self._mj = mujoco
    self.model = sim_robot.get_pick_spec().compile()
    self.data = mujoco.MjData(self.model)
    self._qadr = np.array([
      self.model.jnt_qposadr[mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names])
    self._dadr = np.array([
      self.model.jnt_dofadr[mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names])

  def __call__(self, q: np.ndarray, dq: np.ndarray | None = None) -> np.ndarray:
    """(6,) N.m at the joints, in the model's own units."""
    self.data.qpos[:] = 0.0
    self.data.qvel[:] = 0.0
    self.data.qpos[self._qadr] = np.asarray(q, dtype=np.float64).reshape(6)
    if dq is not None:
      self.data.qvel[self._dadr] = np.asarray(dq, dtype=np.float64).reshape(6)
    self._mj.mj_forward(self.model, self.data)
    return np.array([self.data.qfrc_bias[a] for a in self._dadr])


class MitCommandPath:
  """``PiperArm.command``'s signature, over the impedance path.

  Two differences from ``JointCtrl``, both deliberate copies of what
  ``actions.RateLimitedJointPositionAction.apply_actions`` does in simulation:

  * **the ramp.**  "Hand the servo a ramp, not a stair" -- the simulator
    interpolates the control step's target across its physics substeps with a
    smoothstep, so the joint sees a constant average rate instead of a 50 Hz
    staircase whose every step is a full period of travel.  Sending one MIT
    frame per control step would reintroduce exactly the staircase that comment
    was written about.
  * **the feedforward.**  ``get_pick_spec`` sets ``gravcomp = 1.0``, so the
    simulated actuator never fights gravity.  ``t_ff`` is what makes that true
    on the real arm.

  ``dq_ref`` is zero on purpose.  MIT's law is ``kp (p* - p) + kd (v* - v)``
  and the simulator's actuator is ``kp (p* - p) - kd v``; a velocity reference
  would be a lead term the policy never trained against.  Tracking better than
  the simulator is not the goal.
  """

  def __init__(self, driver: MitDriver, gravity: "GravityFeedforward | None",
               read, substeps: int = 4, gravity_scale: float = 1.0) -> None:
    self._drv = driver
    self._ff = gravity
    self._read = read
    self.substeps = max(1, int(substeps))
    self.gravity_scale = float(gravity_scale)
    self._previous: np.ndarray | None = None

  def start(self, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=np.float64).reshape(-1)[:6]
    self._previous = q.copy()
    self._drv.enter(q)

  def command(self, target: np.ndarray, dt: float) -> None:
    """One control step, streamed as ``substeps`` interpolated MIT frames."""
    q = np.asarray(target, dtype=np.float64).reshape(-1)[:6]
    if self._previous is None:
      self.start(q)
    prev = self._previous
    step = float(dt) / self.substeps
    for k in range(1, self.substeps + 1):
      alpha = k / self.substeps
      alpha = alpha * alpha * (3.0 - 2.0 * alpha)      # smoothstep, as in sim
      here = prev + (q - prev) * alpha
      st = self._read()
      tau = (self._ff(st.q, st.dq) * self.gravity_scale
             if self._ff is not None else np.zeros(6))
      self._drv.send(here, np.zeros(6), tau)
      if k < self.substeps:
        time.sleep(max(0.0, step))
    self._previous = q.copy()

  def hold(self, q: np.ndarray) -> None:
    self._previous = np.asarray(q, dtype=np.float64).reshape(-1)[:6].copy()

  def stop(self, q: np.ndarray) -> None:
    self._drv.leave(np.asarray(q, dtype=np.float64).reshape(-1)[:6])
    self._previous = None


class MitArm:
  """``robot.PiperArm``'s surface, with the six arm joints on the MIT path.

  A wrapper rather than a mode flag inside ``PiperArm``: the two command paths
  have different failure surfaces -- one can leave the drives holding an
  impedance forever, the other cannot -- and a session has to be able to run
  both against the same arm to compare them.  Everything that is not a joint
  command is delegated, including the gripper, which MIT does not cover.
  """

  def __init__(self, arm, gains: Gains, gravity: "GravityFeedforward | None" = None,
               substeps: int = 4, gravity_scale: float = 1.0,
               torque_limit: float = 2.0) -> None:
    self._arm = arm
    self.driver = MitDriver(arm.iface, gains, torque_limit=torque_limit)
    self.path = MitCommandPath(self.driver, gravity, arm.read,
                               substeps=substeps, gravity_scale=gravity_scale)
    self._started = False

  def __getattr__(self, name):
    return getattr(self._arm, name)

  @property
  def connected(self) -> bool:
    return self._arm.connected

  def read(self):
    return self._arm.read()

  def command(self, target, dt: float = 1.0 / config.CONTROL_HZ) -> None:
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if not self._started:
      # The MIT path may only start from where the arm is.  Entering on the
      # first policy target would make it a step against the full impedance.
      self.path.start(self._arm.read().q)
      self._started = True
    self._arm.command_gripper(t[6] if t.size > 6 else 0.0)
    self.path.command(t, dt)

  def hold(self) -> None:
    st = self._arm.read()
    if not self._started:
      self._arm.hold()
      return
    self.path.hold(st.q)
    self.command(np.concatenate([st.q, [st.gripper]]))

  def close(self, disable: bool = False) -> None:
    try:
      if self._started:
        self.path.stop(self._arm.read().q)
        self._started = False
    finally:
      self._arm.close(disable=disable)

  def disconnect(self) -> None:
    try:
      if self._started:
        self.path.stop(self._arm.read().q)
        self._started = False
    finally:
      self._arm.disconnect()
