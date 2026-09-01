"""The identification's arithmetic, checked against plants whose answer is known.

A hardware session produces one recording per experiment and cannot be repeated
cheaply, so every fit that will be pointed at that recording is first pointed at
a simulated drive whose gains were chosen in this file.
"""

import argparse
import json
import math
import time

import numpy as np
import pytest

from hardware.deploy import mit, sysid


def second_order_step(wn, zeta, delay, t):
  """Unit step of a second-order plant with transport delay."""
  tt = np.clip(t - delay, 0.0, None)
  if zeta < 1.0:
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    y = 1.0 - np.exp(-zeta * wn * tt) * (
      np.cos(wd * tt) + (zeta * wn / wd) * np.sin(wd * tt))
  else:
    y = 1.0 - np.exp(-wn * tt) * (1.0 + wn * tt)
  return np.where(t < delay, 0.0, y)


@pytest.mark.parametrize("wn,zeta,delay", [
  (18.0, 0.35, 0.010),      # the identified plant: kp 125 / kd 6.5, underdamped
  (40.0, 0.70, 0.000),
  (8.0, 1.10, 0.030),       # overdamped, so there is no overshoot to read off
])
def test_step_fit_recovers_a_known_plant(wn, zeta, delay):
  t = np.arange(0.0, 1.2, 1.0 / 200.0)
  y = 0.5 + 0.08 * second_order_step(wn, zeta, delay, t)
  gw, gz, gd = sysid._fit_second_order(t, y, 0.5, 0.58)
  assert gw == pytest.approx(wn, rel=0.15)
  assert gz == pytest.approx(zeta, rel=0.25)
  assert abs(gd - delay) < 0.012


def test_step_fit_survives_encoder_noise():
  t = np.arange(0.0, 1.2, 1.0 / 200.0)
  rng = np.random.default_rng(0)
  y = 0.5 + 0.08 * second_order_step(18.0, 0.35, 0.01, t)
  y = y + rng.normal(0.0, math.radians(0.002), t.size)   # 0.002 deg encoder
  gw, gz, _ = sysid._fit_second_order(t, y, 0.5, 0.58)
  assert gw == pytest.approx(18.0, rel=0.15)
  assert gz == pytest.approx(0.35, rel=0.30)


def test_step_fit_reports_nan_for_a_step_that_did_not_happen():
  t = np.arange(0.0, 0.5, 0.005)
  assert all(math.isnan(v) for v in sysid._fit_second_order(t, t * 0 + 1.0, 1.0, 1.0))


def test_sine_fit_recovers_amplitude_and_phase():
  f = 3.0
  t = np.arange(0.0, 2.0, 1.0 / 200.0)
  amp, phase = 0.035, -0.8
  y = 0.2 + amp * np.sin(2 * math.pi * f * t + phase)
  a, p = sysid._sine_fit(t, y - y.mean(), f)
  assert a == pytest.approx(amp, rel=1e-3)
  assert math.atan2(math.sin(p - phase), math.cos(p - phase)) == pytest.approx(
    0.0, abs=1e-3)


def test_sine_fit_ignores_a_neighbouring_tone():
  t = np.arange(0.0, 4.0, 1.0 / 400.0)
  y = 0.03 * np.sin(2 * math.pi * 2.0 * t) + 0.05 * np.sin(2 * math.pi * 5.0 * t)
  a, _ = sysid._sine_fit(t, y, 2.0)
  assert a == pytest.approx(0.03, rel=0.05)


def test_slew_stats_match_a_hand_computed_lag():
  t = np.arange(0.0, 1.0, 0.02)
  qr = 2.0 * t                       # 2 rad/s ramp
  q = qr - 0.15                      # a constant 0.15 rad behind
  s = sysid._slew_stats(t, qr, q)
  assert s["lag_p50"] == pytest.approx(0.15)
  assert s["lag_p95"] == pytest.approx(0.15)
  assert s["dq_p50"] == pytest.approx(2.0, rel=1e-6)


def test_target_gains_come_from_the_simulator_not_a_copy():
  from piper_push import robot as sim_robot
  assert sysid._target_gains(2) == sim_robot.SYSID_GAINS["joint[1-3]"]
  assert sysid._target_gains(6) == sim_robot.SYSID_GAINS["joint6"]


def test_joint_inertia_is_positive_and_ordered():
  ff = mit.GravityFeedforward()
  q = np.array([0.0, 1.57, -1.2, 0.0, 0.0, 0.0])
  inertias = [sysid._joint_inertia(ff, q, j) for j in range(6)]
  assert all(i > 0 for i in inertias)
  # The shoulder carries the whole arm; the wrist roll carries the gripper.
  assert inertias[1] > 10 * inertias[5]
  # And every one of them is physically possible.  Computing this by inverse
  # dynamics instead read 1031 kg.m^2 for joint1 -- the constraint force at a
  # touching contact, not an inertia -- and the ordering assertion above was
  # perfectly happy with it.
  total_mass = float(ff.model.body_mass.sum())
  reach = 0.6
  assert max(inertias) < total_mass * reach ** 2


# ---------------------------------------------------------------------------
# the streaming loop, against a drive whose impedance this file chose
# ---------------------------------------------------------------------------

class FakeDrive:
  """Six independent second-order joints under the MIT law.

  ``g_kp``/``g_kd`` are the written-to-effective scales the identification
  exists to measure; setting them here to something other than 1 is what makes
  the end-to-end test meaningful.
  """

  def __init__(self, q0, inertia=0.05, g_kp=5.0, g_kd=5.0, dt=1.0 / 1000.0):
    self.q = np.asarray(q0, dtype=np.float64).copy()
    self.dq = np.zeros(6)
    self.I, self.g_kp, self.g_kd, self.dt = inertia, g_kp, g_kd, dt
    self.mode = None
    self.cmd = None
    self.t = 0.0

  # -- the SDK surface the driver uses
  def MotionCtrl_2(self, ctrl, move, spd, is_mit, *a, **k):
    self.mode = is_mit

  def JointMitCtrl(self, n, pos, vel, kp, kd, t):
    if self.cmd is None:
      self.cmd = np.zeros((6, 5))
    self.cmd[n - 1] = (pos, vel, kp, kd, t)

  def JointCtrl(self, *q):
    self.cmd = None

  def GetArmHighSpdInfoMsgs(self):
    drive = self

    class M:
      def __init__(self, i):
        self.pos = drive.q[i]
        self.motor_speed = int(drive.dq[i] * 1000)
        self.current = 0
        self.effort = 0
    return type("msgs", (), {f"motor_{i+1}": M(i) for i in range(6)})()

  # -- the arm surface
  iface = property(lambda self: self)

  def advance(self, wall):
    while self.t < wall:
      if self.cmd is not None:
        tau = (self.g_kp * self.cmd[:, 2] * (self.cmd[:, 0] - self.q)
               + self.g_kd * self.cmd[:, 3] * (self.cmd[:, 1] - self.dq)
               + self.cmd[:, 4])
        self.dq += (tau / self.I) * self.dt
        self.q += self.dq * self.dt
      self.t += self.dt

    class S:
      pass
    s = S(); s.q = self.q.copy(); s.dq = self.dq.copy()
    s.gripper = 0.05; s.gripper_vel = 0.0; s.gripper_effort = 0.0
    s.stamp = wall
    return s


class FakeArm:
  def __init__(self, drive):
    self.drive = drive
    self.iface = drive
    self.t0 = None

  def read(self):
    if self.t0 is None:
      self.t0 = time.time()
    return self.drive.advance(time.time() - self.t0)

  def close(self, disable=False):
    pass


def test_mit_loop_drives_a_fake_plant_and_records_both_sides():
  q0 = np.array([0.0, 1.5, -1.2, 0.0, 0.0, 0.0])
  drive = FakeDrive(q0)
  arm = FakeArm(drive)
  drv = mit.MitDriver(drive, mit.Gains.uniform(kp=10.0, kd=0.8))
  rec = sysid.Recorder("unit_loop", {})
  amp = math.radians(3.0)
  ref = lambda t: (q0 + np.array([0, 0, 0, 0, 0, amp * math.sin(6.0 * t)]),
                   np.zeros(6))
  stop = sysid._mit_loop(arm, drv, None if False else _NoGravity(), ref,
                         0.4, 400.0, rec, gravity=0.0)
  assert stop == "done"
  assert len(rec.rows) > 50
  q6 = sysid._col(rec, "q6")
  r6 = sysid._col(rec, "qref6")
  assert np.abs(r6).max() > 0.9 * amp                 # the reference moved
  assert np.abs(q6 - q0[5]).max() > 0.2 * amp         # and so did the joint
  assert drive.mode == mit.POSITION_MODE              # and it was handed back


def test_mit_loop_aborts_and_leaves_mit_when_a_joint_falls_behind():
  q0 = np.array([0.0, 1.5, -1.2, 0.0, 0.0, 0.0])      # not all-zero: see _wait_feedback
  drive = FakeDrive(q0, g_kp=0.0, g_kd=0.0)           # a drive that does nothing
  arm = FakeArm(drive)
  drv = mit.MitDriver(drive, mit.Gains.uniform())
  rec = sysid.Recorder("unit_abort", {})
  ref = lambda t: (q0 + t * np.array([3.0, 0, 0, 0, 0, 0]), np.zeros(6))
  stop = sysid._mit_loop(arm, drv, _NoGravity(), ref, 5.0, 200.0, rec)
  assert stop.startswith("tracking error")
  assert drive.mode == mit.POSITION_MODE


class _NoGravity:
  def __call__(self, q, dq=None):
    return np.zeros(6)


def test_step_command_recovers_the_written_to_effective_scale(tmp_path, monkeypatch, capsys):
  """End to end: a drive whose gains this test chose, identified by the command
  the hardware session will run.

  The number the session is after is the ratio between the kp written into a
  MIT frame and the stiffness the joint presents.  The SDK's note puts it near
  5 from one static sag measurement; this checks that the dynamic path can see
  a 5 when there is a 5 to see.
  """
  monkeypatch.setattr(sysid, "LOG_ROOT", tmp_path)
  ff = mit.GravityFeedforward()
  q0 = np.array([0.4, 1.50, -1.20, 0.10, 0.05, 0.00])
  joint = 5                                   # light, so the step settles fast
  inertia = sysid._joint_inertia(ff, q0, joint - 1)
  written_kp, written_kd, g_kp = 10.0, 0.8, 5.0
  # Critical-ish damping at the effective stiffness, so the fit has both a rise
  # and a settle to work with.
  g_kd = 0.9 * 2 * math.sqrt(g_kp * written_kp * inertia) / written_kd

  drive = FakeDrive(q0, inertia=inertia, g_kp=g_kp, g_kd=g_kd)
  monkeypatch.setattr(sysid, "_connect", lambda can: FakeArm(drive))
  monkeypatch.setattr(sysid, "_confirm", lambda *a, **k: True)

  a = argparse.Namespace(
    can="fake", yes=True, rate=400.0, kp=written_kp, kd=written_kd, gains=None,
    gravity=0.0, torque_limit=2.0, joint=joint, degrees=4.0, dwell=0.6)
  assert sysid.cmd_step(a) == 0
  out = capsys.readouterr().out
  ratio = float(out.split("ratio kp_eff/kp")[1].split()[0])
  assert ratio == pytest.approx(g_kp, rel=0.30), out



def test_gains_command_inverts_the_measured_scale(tmp_path, capsys):
  out = tmp_path / "g.json"
  a = argparse.Namespace(g_kp=5.0, g_kd=None, from_runs=None, out=str(out),
                         force=False)
  assert sysid.cmd_gains(a) == 0
  g = mit.Gains.load(out)
  # joint1-3 target 125 / 6.5; at a scale of 5 that is the vendor's own
  # suggested neighbourhood, which is the first sign the 5x is not nonsense.
  assert g.kp[0] == pytest.approx(25.0)
  assert g.kd[0] == pytest.approx(1.3)
  assert g.kp[5] == pytest.approx(8.0)


def test_gains_command_refuses_a_gain_the_field_cannot_carry(tmp_path, capsys):
  # A scale of 1 means the drives amplify nothing, so joints 1-3 would need a
  # written kd of 6.5 against a field that stops at 5.
  a = argparse.Namespace(g_kp=1.0, g_kd=None, from_runs=None,
                         out=str(tmp_path / "g.json"), force=False)
  assert sysid.cmd_gains(a) == 2
  assert "UNREACHABLE" in capsys.readouterr().out
  assert not (tmp_path / "g.json").exists()


def test_gains_command_reads_a_step_run_directory(tmp_path):
  run = tmp_path / "20260101_000000_step_j2"
  run.mkdir()
  (run / "run.json").write_text(json.dumps({
    "meta": {"joint": 2, "kp": [10.0] * 6, "kd": [0.8] * 6},
    "summary": {"kp_eff": 40.0, "kd_eff": 3.2},
  }))
  out = tmp_path / "g.json"
  a = argparse.Namespace(g_kp=5.0, g_kd=None, from_runs=[str(run)],
                         out=str(out), force=False)
  assert sysid.cmd_gains(a) == 0
  g = mit.Gains.load(out)
  # joint2 measured 4x, not the 5x assumed for everything else.
  assert g.kp[1] == pytest.approx(125.0 / 4.0)
  assert g.kp[0] == pytest.approx(125.0 / 5.0)



class SpyArm:
  """Records the order of connect / command / enable, which is the whole point."""

  def __init__(self, q=None, gripper=0.05):
    self.log = []
    self._q = np.array([0.0, 1.55, -1.2, 0.0, 0.0, 0.0] if q is None else q)
    self.gripper = float(gripper)
    self.connected = False
    self.iface = None

  def connect(self):
    self.connected = True
    self.log.append("connect")

  def read(self):
    self.log.append("read")
    s = argparse.Namespace()
    s.q = self._q.copy(); s.dq = np.zeros(6)
    s.gripper = self.gripper; s.gripper_vel = 0.0
    s.gripper_effort = 0.0; s.stamp = time.time()
    return s

  def command(self, target, dt=0.02):
    self.log.append("command")

  def enable(self):
    self.log.append("enable")

  def close(self, disable=False):
    self.log.append("close")

  def disconnect(self):
    self.log.append("disconnect")


def test_readonly_connect_never_energises_the_arm(monkeypatch):
  """`probe` is advertised as risk-free and has to actually be.

  Enabling is not passive: PiPER keeps its last position target across client
  restarts, so energising without preloading the measured pose can drive the
  arm at a stale target the instant it powers up.
  """
  spy = SpyArm()
  monkeypatch.setattr(sysid.robot, "PiperArm", lambda can: spy)
  monkeypatch.setattr(sysid, "_wait_feedback", lambda arm, *a, **k: arm.read())
  sysid._connect_readonly("can0")
  assert "enable" not in spy.log
  assert "command" not in spy.log


def test_motion_connect_preloads_the_measured_pose_before_enable(monkeypatch):
  spy = SpyArm()
  monkeypatch.setattr(sysid.robot, "PiperArm", lambda can: spy)
  monkeypatch.setattr(sysid, "_wait_feedback", lambda arm, *a, **k: arm.read())
  sysid._connect("can0")
  order = [e for e in spy.log if e in ("connect", "command", "enable")]
  assert order == ["connect", "command", "enable", "command"]


def test_motion_connect_refuses_a_start_pose_the_preload_would_clip(monkeypatch):
  """A joint outside SAFE_TARGET_CLIP cannot be preloaded truthfully.

  The preload would be clipped to the boundary, so enable would energise the
  arm against a target it is not standing at.  (The other start-pose hazard,
  the all-zero "no CAN frame yet" signature, is caught upstream in
  ``_wait_for_feedback`` -- joint3 at +1.0 rad is outside its [-2.967, 0]
  range, which is this check's own.)
  """
  spy = SpyArm(q=np.array([0.0, 1.55, +1.0, 0.0, 0.0, 0.0]))
  monkeypatch.setattr(sysid.robot, "PiperArm", lambda can: spy)
  monkeypatch.setattr(sysid, "_wait_feedback", lambda arm, *a, **k: arm.read())
  with pytest.raises(RuntimeError):
    sysid._connect("can0")
  assert "enable" not in spy.log
  assert "disconnect" in spy.log
