"""The MIT command boundary, and the fits that read its recordings.

Everything here runs without an arm.  That is the point: the one thing a
hardware session must not be spent on is finding out that a fit is wrong.
"""

import math

import numpy as np
import pytest

from hardware.deploy import mit


class FakeIface:
  """Records what would have gone on the bus, and nothing else."""

  def __init__(self):
    self.mode = None
    self.mit_frames = []
    self.joint_frames = []
    self.gripper_frames = []

  def MotionCtrl_2(self, ctrl, move, spd, is_mit, *a, **k):
    self.mode = (ctrl, move, spd, is_mit)

  def JointMitCtrl(self, n, pos, vel, kp, kd, t):
    self.mit_frames.append((n, pos, vel, kp, kd, t))

  def JointCtrl(self, *q):
    self.joint_frames.append(q)

  def GripperCtrl(self, opening, effort, mode, zero):
    self.gripper_frames.append((opening, effort, mode, zero))


def test_torque_field_wraps_and_the_driver_clamps_before_it_can():
  # FloatToUint does not clamp: the SDK's own docstring advertises t_ref in
  # [-18, 18] while the encoder packs 8 bits over [-8, 8], so -12 N.m arrives
  # as +4.11 -- a sign flip, on a torque command, on a real arm.
  assert mit.encoded(-12.0, "torque") == pytest.approx(4.11, abs=0.02)
  assert mit.encoded(+10.0, "torque") == pytest.approx(-6.12, abs=0.02)
  # Which is why nothing outside the field's range may reach it.
  iface = FakeIface()
  drv = mit.MitDriver(iface, mit.Gains.uniform(), torque_limit=20.0)
  drv.enter(np.zeros(6))
  drv.send(np.zeros(6), np.zeros(6), np.full(6, -12.0))
  sent = [f[5] for f in iface.mit_frames[-6:]]
  assert all(t == -8.0 for t in sent)
  assert drv.clamps.counts["torque"] == 6


def test_torque_limit_binds_before_the_field_does():
  iface = FakeIface()
  drv = mit.MitDriver(iface, mit.Gains.uniform(), torque_limit=2.0)
  drv.enter(np.zeros(6))
  out = drv.send(np.zeros(6), np.zeros(6), np.full(6, 7.0))
  assert np.allclose(out["tau_ff"], 2.0)
  # The field never saw an out-of-range value, so it never counted a clamp.
  assert "torque" not in drv.clamps.report()


def test_gains_outside_the_field_are_clamped_and_counted():
  iface = FakeIface()
  # SYSID_GAINS asks for kd 6.5 on joints 1-3; the kd field stops at 5.0.
  drv = mit.MitDriver(iface, mit.Gains.uniform(kp=600.0, kd=6.5))
  drv.enter(np.zeros(6))
  drv.clamps = mit.ClampCounter()          # enter() sent a frame of its own
  out = drv.send(np.zeros(6), np.zeros(6), np.zeros(6))
  assert np.allclose(out["kp"], 500.0)
  assert np.allclose(out["kd"], 5.0)
  assert drv.clamps.counts["kp"] == 6 and drv.clamps.counts["kd"] == 6
  # Not a detail: kd 6.5 is what the simulator's actuator presents on joints
  # 1-3, and the field cannot carry it.  Either the written kd is a smaller
  # number that the drives amplify -- which is what `sysid step` measures -- or
  # the trained damping is unreachable and the profile has to move.
  assert mit.FIELD_RANGE["kd"][1] < 6.5


def test_enter_holds_the_measured_pose_with_no_feedforward():
  iface = FakeIface()
  drv = mit.MitDriver(iface, mit.Gains.uniform())
  q = np.array([0.1, 1.6, -1.2, 0.3, 0.0, -0.2])
  drv.enter(q)
  # All four fields, not just the MIT flag.  MOVE_J with the flag set is a
  # different feature -- the position path with a faster response -- and it
  # discards JointMitCtrl frames without complaining.
  assert iface.mode == (0x01, mit.MOVE_M, 0, mit.MIT_MODE)
  frames = iface.mit_frames[-6:]
  assert [f[0] for f in frames] == [1, 2, 3, 4, 5, 6]
  assert np.allclose([f[1] for f in frames], q, atol=1e-9)
  assert all(f[2] == 0.0 and f[5] == 0.0 for f in frames)


def test_enter_refuses_a_non_finite_pose():
  drv = mit.MitDriver(FakeIface(), mit.Gains.uniform())
  with pytest.raises(ValueError):
    drv.enter(np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0]))


def test_send_before_enter_is_an_error():
  drv = mit.MitDriver(FakeIface(), mit.Gains.uniform())
  with pytest.raises(RuntimeError):
    drv.send(np.zeros(6), np.zeros(6), np.zeros(6))


def test_leave_returns_to_position_mode_at_the_measured_pose():
  iface = FakeIface()
  drv = mit.MitDriver(iface, mit.Gains.uniform())
  q = np.array([0.0, 1.57, -1.2, 0.0, 0.0, 0.0])
  drv.enter(q)
  drv.leave(q)
  assert iface.mode == (0x01, mit.MOVE_J, 100, mit.POSITION_MODE)
  assert not drv.active
  got = np.array(iface.joint_frames[-1]) * math.pi / 180000.0
  assert np.allclose(got, q, atol=2e-5)


def test_gains_round_trip_through_a_file(tmp_path):
  g = mit.Gains(np.arange(6.0) + 1.0, np.arange(6.0) / 10.0)
  f = tmp_path / "mit_gains.json"
  g.save(f, source="test")
  back = mit.Gains.load(f)
  assert np.allclose(back.kp, g.kp) and np.allclose(back.kd, g.kd)


def test_gravity_feedforward_matches_a_static_hold():
  """qfrc_bias at rest IS the torque that holds the pose, so a joint released
  from it must not accelerate."""
  import mujoco
  ff = mit.GravityFeedforward()
  q = np.array([0.3, 1.4, -1.1, 0.2, 0.1, 0.0])
  tau = ff(q)
  d = ff.data
  d.qfrc_applied[:] = 0.0
  for k, a in enumerate(ff._dadr):
    d.qfrc_applied[a] = tau[k]
  # gravcomp is on in the pick spec, so cancel it here or it is counted twice.
  saved = ff.model.body_gravcomp.copy()
  ff.model.body_gravcomp[:] = 0.0
  try:
    mujoco.mj_forward(ff.model, d)
    held = np.array([d.qacc[a] for a in ff._dadr])
    d.qfrc_applied[:] = 0.0
    mujoco.mj_forward(ff.model, d)
    free = np.array([d.qacc[a] for a in ff._dadr])
  finally:
    ff.model.body_gravcomp[:] = saved
  # Solver residue leaves a few mrad/s^2; what matters is that the
  # feedforward accounts for essentially all of the sag.
  assert np.abs(free).max() > 5.0
  assert np.abs(held).max() < 0.01 * np.abs(free).max()


def test_the_mit_pose_field_is_coarser_than_the_move_j_one():
  """MIT quantises position 22x more coarsely than ``JointCtrl`` does.

  ``JointCtrl`` takes thousandths of a degree; the MIT frame packs 16 bits over
  +-12.5 rad, which is 0.0219 deg.  That is 1% of the policy's 2.2 deg slew
  step and about 0.15 mm at full reach, so it is not a reason to stay in MOVE
  J -- but it is the one axis on which MIT is worse, and it should be a
  measured number rather than a surprise.
  """
  assert mit.resolution("pos") == pytest.approx(math.radians(0.0219), rel=0.02)
  assert mit.resolution("pos") > math.radians(0.001)
  slew_step = 0.039                       # rad, robot.COMMAND_RATE_LIMIT / 50 Hz
  assert mit.resolution("pos") < 0.02 * slew_step


class _St:
  def __init__(self, q):
    self.q = np.asarray(q, dtype=np.float64)
    self.dq = np.zeros(6)


def test_command_path_ramps_across_the_control_step():
  """A single frame per control step is the 50 Hz staircase the simulator's
  substep interpolation exists to avoid."""
  iface = FakeIface()
  drv = mit.MitDriver(iface, mit.Gains.uniform())
  q0 = np.zeros(6)
  path = mit.MitCommandPath(drv, None, lambda: _St(q0), substeps=4)
  path.start(q0)
  iface.mit_frames.clear()
  goal = np.array([0.04, 0, 0, 0, 0, 0])
  path.command(np.concatenate([goal, [0.05]]), 0.02)
  j1 = [f[1] for f in iface.mit_frames if f[0] == 1]
  assert len(j1) == 4
  assert j1 == sorted(j1)                       # monotone towards the target
  assert j1[-1] == pytest.approx(goal[0], abs=1e-6)
  assert 0.0 < j1[0] < goal[0] / 2              # smoothstep starts slowly
  # The next step must ramp from the target just reached, not from the joint.
  iface.mit_frames.clear()
  path.command(np.concatenate([goal * 2, [0.05]]), 0.02)
  j1 = [f[1] for f in iface.mit_frames if f[0] == 1]
  assert j1[0] > goal[0]


def test_command_path_sends_no_velocity_reference():
  """MIT's kd acts on (v* - v); the simulated actuator's acts on (0 - v).
  A velocity feedforward would track better than the trained plant."""
  iface = FakeIface()
  drv = mit.MitDriver(iface, mit.Gains.uniform())
  path = mit.MitCommandPath(drv, None, lambda: _St(np.zeros(6)), substeps=2)
  path.start(np.zeros(6))
  path.command(np.array([0.1] * 6 + [0.05]), 0.02)
  assert all(f[2] == 0.0 for f in iface.mit_frames)


def test_command_path_hands_the_arm_back_on_stop():
  iface = FakeIface()
  drv = mit.MitDriver(iface, mit.Gains.uniform())
  path = mit.MitCommandPath(drv, None, lambda: _St(np.zeros(6)), substeps=1)
  path.start(np.zeros(6))
  path.stop(np.zeros(6))
  assert iface.mode[3] == mit.POSITION_MODE


class FakeBaseArm:
  """The part of ``robot.PiperArm``'s surface ``MitArm`` delegates to."""

  def __init__(self, iface, q):
    self.iface = iface
    self._q = np.asarray(q, dtype=np.float64)
    self.connected = True
    self.gripper_cmds = []
    self.joint_cmds = []
    self.closed = False

  def read(self):
    return _St(self._q)

  def command(self, target, dt=0.02):
    self.joint_cmds.append(np.asarray(target, dtype=np.float64).copy())

  def command_gripper(self, one_finger_m):
    self.gripper_cmds.append(float(one_finger_m))

  def hold(self):
    self.command(np.concatenate([self._q, [0.05]]))

  def close(self, disable=False):
    self.closed = True

  def disconnect(self):
    self.closed = True


def test_mit_arm_starts_from_the_measured_pose_not_the_first_target():
  """Entering MIT on the policy's first target would be a step against the
  full impedance, from wherever the arm happens to be standing."""
  iface = FakeIface()
  q = np.array([0.1, 1.55, -1.2, 0.0, 0.0, 0.0])
  base = FakeBaseArm(iface, q)
  arm = mit.MitArm(base, mit.Gains.uniform())
  far = np.concatenate([q + 0.5, [0.05]])
  arm.command(far, 0.02)
  first = [f[1] for f in iface.mit_frames[:6]]
  assert np.allclose(first, q, atol=1e-6)


def test_mit_arm_still_drives_the_gripper_through_the_ordinary_message():
  iface = FakeIface()
  base = FakeBaseArm(iface, np.zeros(6))
  arm = mit.MitArm(base, mit.Gains.uniform())
  arm.command(np.array([0.0] * 6 + [0.031]), 0.02)
  assert base.gripper_cmds[-1] == pytest.approx(0.031)
  assert not base.joint_cmds          # joints did NOT go through JointCtrl


def test_mit_arm_leaves_mit_before_releasing_the_bus():
  iface = FakeIface()
  base = FakeBaseArm(iface, np.zeros(6))
  arm = mit.MitArm(base, mit.Gains.uniform())
  arm.command(np.zeros(7), 0.02)
  arm.close()
  assert iface.mode[3] == mit.POSITION_MODE
  assert base.closed


def test_mit_arm_delegates_what_it_does_not_own():
  base = FakeBaseArm(FakeIface(), np.zeros(6))
  base.gripper_torque_nm = 0.7
  arm = mit.MitArm(base, mit.Gains.uniform())
  assert arm.gripper_torque_nm == 0.7
  assert arm.connected


def test_run_mit_selects_the_drives_fast_response_law():
  """--control mit is MotionCtrl_2's 0xAD on the ordinary position path.

  Measured on this rig with the speed field held at 100 so the flag was the
  only change: p95 tracking lag 0.329 -> 0.106 rad on a 1.2 rad/s triangle,
  bracketed by two plain runs that agreed to 0.0006.
  """
  import argparse
  from hardware.deploy import robot as robot_mod, run as run_mod
  base = FakeBaseArm(FakeIface(), np.zeros(6))
  base.response_mode = robot_mod.RESPONSE_PLAIN
  a = argparse.Namespace(control="mit")
  assert run_mod._wrap_control(base, a) is base
  assert base.response_mode == robot_mod.RESPONSE_MIT


def test_piper_arm_sends_its_response_mode_on_every_command():
  from hardware.deploy import robot as robot_mod
  iface = FakeIface()
  arm = robot_mod.PiperArm.__new__(robot_mod.PiperArm)
  arm._iface = iface
  arm.gripper_torque_nm = 0.5
  arm.response_mode = robot_mod.RESPONSE_MIT
  arm.command(np.array([0.0, 1.5, -1.2, 0.0, 0.0, 0.0, 0.03]))
  assert iface.mode == (0x01, 0x01, 100, robot_mod.RESPONSE_MIT)


def test_run_refuses_impedance_without_measured_gains(tmp_path):
  """"From now on use MIT" cannot mean "guess the gains".

  ``run.py`` defaults to the impedance path, and the impedance path needs a
  number the drives amplify by a factor this rig has not measured.  So the
  default refuses rather than inventing one, and names the way to get it.
  """
  import argparse
  from hardware.deploy import run as run_mod

  base = FakeBaseArm(FakeIface(), np.zeros(6))
  a = argparse.Namespace(control="impedance",
                         mit_gains=str(tmp_path / "absent.json"),
                         mit_gravity=0.0, mit_substeps=4, mit_torque_limit=3.0)
  with pytest.raises(RuntimeError, match="no measured impedance"):
    run_mod._wrap_control(base, a)
  assert base.closed                       # and it let go of the bus


def test_run_wraps_the_arm_when_the_gains_exist(tmp_path):
  import argparse
  from hardware.deploy import run as run_mod

  f = tmp_path / "mit_gains.json"
  mit.Gains.uniform(25.0, 1.3).save(f)
  base = FakeBaseArm(FakeIface(), np.zeros(6))
  a = argparse.Namespace(control="impedance", mit_gains=str(f), mit_gravity=0.0,
                         mit_substeps=4, mit_torque_limit=3.0)
  wrapped = run_mod._wrap_control(base, a)
  assert isinstance(wrapped, mit.MitArm)
  assert np.allclose(wrapped.driver.gains.kp, 25.0)


def test_run_movej_returns_the_arm_untouched():
  import argparse
  from hardware.deploy import run as run_mod
  base = FakeBaseArm(FakeIface(), np.zeros(6))
  a = argparse.Namespace(control="movej")
  assert run_mod._wrap_control(base, a) is base
