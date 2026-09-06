"""Identify the real arm's MIT impedance, then pick the gains that match the
simulator.

The 2026-08-30 bench runs failed with the hand sweeping past the object: the
arm trailed its command by 0.60 rad at p95 where the training domain tops out
near 0.28, and moved at 64% of the simulated speed.  The command path at the
time was MOVE J, which hands the controller a target and lets it re-plan every
20 ms.  The simulator models something else -- a PD position actuator with the
identified gains and full gravity compensation -- and MIT mode is the interface
that law is expressed in.

So the target of this identification is **match, not minimum error**.  The
policy was trained against ``robot.SYSID_GAINS`` (kp 125 / kd 6.5 on J1-3, and
so on) with ``gravcomp = 1.0``.  A real arm tuned to track better than that is
as out of distribution as one that tracks worse.  What has to be measured is
the map from the numbers written into a MIT frame to the impedance the joint
actually presents, because the SDK's own note says they differ by roughly 5x
and one static sag measurement on one joint is where that 5x came from.

Run the experiments in this order.  Each one answers something the next one
assumes, and the first three do not move the arm at all.

    python -m hardware.deploy.sysid probe                  # read-only
    python -m hardware.deploy.sysid probe --sweep          # slow, position mode
    python -m hardware.deploy.sysid hold                   # MIT, holds still
    python -m hardware.deploy.sysid gravity --joint 2      # t_ff scale
    python -m hardware.deploy.sysid impedance --joint 2    # kp/kd scale
    python -m hardware.deploy.sysid step --joint 2
    python -m hardware.deploy.sysid chirp --joint 2
    python -m hardware.deploy.sysid slew --joint 2 --mode mit
    python -m hardware.deploy.sysid slew --joint 2 --mode movej      # the A/B
    python -m hardware.deploy.sysid replay recordings/<session>

Everything writes ``logs/sysid/<stamp>_<what>/`` with the full time series, so
a fit can be redone without touching the robot again.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

from . import config, mit, robot

LOG_ROOT = pathlib.Path("logs/sysid")
ARM = mit.ARM_JOINTS

# Every motion experiment perturbs one joint about where it already is.  These
# are the caps, not the defaults: an identification that needs more than 15
# degrees of travel is measuring the workspace, not the drive.
MAX_DEG = 15.0
DEFAULT_DEG = 4.0

# Abort thresholds for the MIT loop.  Tracking error is the one that matters:
# under impedance control a joint that is 20 degrees behind its reference is
# either colliding with something or unstable, and both want the drives back in
# position mode within one control period.
ABORT_TRACKING_RAD = math.radians(20.0)
ABORT_SPEED_RAD_S = 2.5


class Recorder:
  def __init__(self, name: str, meta: dict) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    self.dir = LOG_ROOT / f"{stamp}_{name}"
    self.dir.mkdir(parents=True, exist_ok=True)
    self.meta = dict(meta)
    self.rows: list[dict] = []

  def add(self, **kw) -> None:
    self.rows.append(kw)

  def save(self, **summary) -> pathlib.Path:
    keys = sorted({k for r in self.rows for k in r})
    cols = {k: np.array([r.get(k, np.nan) for r in self.rows]) for k in keys}
    np.savez_compressed(self.dir / "trace.npz", **cols)
    (self.dir / "run.json").write_text(json.dumps(
      {"meta": self.meta, "summary": summary, "n": len(self.rows)},
      indent=1, default=float) + "\n")
    return self.dir


def _confirm(what: str, yes: bool) -> bool:
  if yes:
    return True
  print(f"\n{what}\nWorkspace clear?  estop in reach?")
  return input("type 'go' to continue: ").strip() == "go"


def _drives_enabled(iface) -> list:
  """Whether each drive is energised.

  Without this a resting ``effort`` of exactly zero looks like "the drives
  carry gravity upstream" when it actually means "there is no current because
  nothing is enabled".
  """
  try:
    msgs = iface.GetArmLowSpdInfoMsgs()
    return [bool(getattr(msgs, f"motor_{i}").foc_status.driver_enable_status)
            for i in range(1, 7)]
  except Exception:
    return [False] * 6


def _connect_readonly(can: str) -> robot.PiperArm:
  """Open CAN and read.  Never enable, never command.

  ``probe`` is the first thing anyone runs and it is advertised as risk-free,
  so it has to actually be that.  ``enable`` is not passive: the drives retain
  their last position target across client restarts, so energising without
  preloading the measured pose is exactly the jump ``run.py`` documents at its
  own enable.  Callers pair this with ``disconnect``, not ``close`` -- ``close``
  commands the measured pose on the way out, which a read-only tool must not.
  """
  arm = robot.PiperArm(can)
  arm.connect()
  _wait_feedback(arm)
  return arm


def _connect(can: str) -> robot.PiperArm:
  """Open CAN, preload the measured pose, then enable.

  The order is ``run.py``'s and calibgui's, and it is not decorative: PiPER
  keeps its last position target across client restarts, so enabling before
  the preload can drive the arm toward a stale target from a previous session
  the instant it is energised.  The second preload is there because the first
  one is sent to drives that were not yet listening.
  """
  arm = robot.PiperArm(can)
  arm.connect()
  start = _wait_feedback(arm)
  from .run import _start_pose_fault
  fault = _start_pose_fault(start)
  if fault is not None:
    arm.disconnect()
    raise RuntimeError(fault)
  preload = np.concatenate([start.q, [start.gripper]])
  arm.command(preload)
  arm.enable()
  arm.command(preload)
  return arm


def _wait_feedback(arm, timeout_s: float = 3.0):
  """The first read after connect is six exact zeros; ``run.py`` already has
  the wait that proves the drives have really reported, and a second
  implementation of it is a second thing to get wrong."""
  from .run import _wait_for_feedback
  return _wait_for_feedback(arm, timeout_s)


# ---------------------------------------------------------------------------
# probe -- no motion
# ---------------------------------------------------------------------------

def cmd_probe(a) -> int:
  """What the drives report, and how it relates to what the joints report.

  Three questions, none of which needs the arm to move:

  * is the high-speed motor feedback in joint units or motor units?  The ratio
    of ``motor_pos`` to the joint angle is the gear reduction, and it decides
    whether a MIT ``t_ref`` of 8 is 8 N.m at the joint or 8 at the rotor.
  * is ``motor_speed`` the joint's velocity?  ``robot.PiperArm.read`` refuses to
    use it because the unit was undocumented; the V2 SDK documents it as
    0.001 rad/s, so the only question left is which shaft.
  * do the drives already carry gravity?  ``effort`` at rest, against the
    model's ``qfrc_bias`` at the same pose, answers it.
  """
  arm = _connect_readonly(a.can)
  ff = mit.GravityFeedforward()
  rec = Recorder("probe", {"can": a.can, "seconds": a.seconds})
  t0 = time.time()
  try:
    while time.time() - t0 < a.seconds:
      st = arm.read()
      mf = mit.read_motors(arm.iface)
      rec.add(t=time.time() - t0, **{f"q{i+1}": st.q[i] for i in range(6)},
              **{f"dq{i+1}": st.dq[i] for i in range(6)},
              **{f"mpos{i+1}": mf.motor_pos[i] for i in range(6)},
              **{f"mspd{i+1}": mf.motor_speed[i] for i in range(6)},
              **{f"eff{i+1}": mf.effort[i] for i in range(6)},
              **{f"cur{i+1}": mf.current[i] for i in range(6)})
      time.sleep(1.0 / config.CONTROL_HZ)
  finally:
    arm_iface = arm.iface
    arm.disconnect()

  enabled = _drives_enabled(arm_iface)
  q = np.array([[r[f"q{i+1}"] for i in range(6)] for r in rec.rows])
  mp = np.array([[r[f"mpos{i+1}"] for i in range(6)] for r in rec.rows])
  eff = np.array([[r[f"eff{i+1}"] for i in range(6)] for r in rec.rows])
  tau_model = ff(q.mean(axis=0))

  qm = q.mean(axis=0)
  mpm = mp.mean(axis=0)
  effm = eff.mean(axis=0)

  print("\nresting pose (deg):       ", np.round(np.degrees(qm), 2))
  print("motor_pos register:       ", np.round(mpm, 1))
  # A ratio is the wrong statistic here: at a joint sitting near zero it is
  # dominated by any zero-point offset and reads as a different gear ratio per
  # joint.  Fit motor_pos = k * q + b across the six instead, which separates
  # the two.
  A = np.vstack([qm, np.ones(6)]).T
  (k, b), *_ = np.linalg.lstsq(A, mpm, rcond=None)
  resid = mpm - (k * qm + b)
  print("  fit  motor_pos = %.1f * q_rad %+.1f   residual %s"
        % (k, b, np.round(resid, 1)))
  if abs(k - 1000.0) / 1000.0 < 0.05:
    print("  -> the register is the JOINT angle in milliradians, not a motor")
    print("     shaft.  So the MIT fields are joint-referred: t_ref +-8 is +-8")
    print("     N.m at the joint, and joint2's 14 N.m worst case will not fit.")
  else:
    print("  -> not milliradians; %.1f is a reduction, and every MIT field is"
          % k)
    print("     rotor-referred.  Keep --torque-limit small.")

  print("\ndrives enabled:           ", enabled)
  print("reported effort (N.m):    ", np.round(effm, 3))
  print("effort noise (std, N.m):  ", np.round(eff.std(axis=0), 4))
  print("model qfrc_bias (N.m):    ", np.round(tau_model, 3))
  if not all(enabled):
    print("  -> the drives are DISABLED, so there is no holding current and the")
    print("     effort register is meaningless.  Whether they carry gravity of")
    print("     their own is answered by `hold`, which enables.")
    ratio = np.full(6, np.nan)
  else:
    ratio = np.where(np.abs(tau_model) > 0.3,
                     effm / np.where(np.abs(tau_model) > 0.3, tau_model, 1.0),
                     np.nan)
    print("  effort / model:         ", np.round(ratio, 3))
    print("  -> near 1: the drives hold gravity themselves and report it in")
    print("     joint N.m.  Near 0: it is carried upstream and never reaches")
    print("     this register.  Anything else is a scale for `gravity`.")

  d = rec.save(rest_deg=np.degrees(qm).tolist(), motor_pos=mpm.tolist(),
               motor_pos_per_rad=float(k), motor_pos_offset=float(b),
               enabled=list(enabled), effort=effm.tolist(),
               model_bias=tau_model.tolist(),
               effort_over_model=ratio.tolist())
  print(f"\nwrote {d}")
  return 0


# ---------------------------------------------------------------------------
# the shared MIT streaming loop
# ---------------------------------------------------------------------------

def _mit_loop(arm, drv, ff, ref, seconds: float, rate: float, rec: Recorder,
              gravity: float = 1.0, tag: str = "", abort_rad: float = ABORT_TRACKING_RAD):
  """Stream a scripted reference and record both sides of every frame.

  ``ref(t) -> (q_ref, dq_ref)``, both (6,).  The loop owns the abort: a joint
  that falls ``abort_rad`` behind its reference is colliding or unstable, and
  either way the drives want to be back in position mode inside one period.
  """
  period = 1.0 / float(rate)
  st = _wait_feedback(arm)
  drv.enter(st.q)
  stop = ""
  t0 = time.time()
  n = 0
  try:
    while True:
      t = time.time() - t0
      if t >= seconds:
        stop = "done"
        break
      st = arm.read()
      mf = mit.read_motors(arm.iface)
      q_ref, dq_ref = ref(t)
      tau = ff(st.q, st.dq) * gravity if gravity else np.zeros(6)
      err = q_ref - st.q
      if np.abs(err).max() > abort_rad:
        stop = "tracking error %.3f rad" % np.abs(err).max()
        break
      if np.abs(st.dq).max() > ABORT_SPEED_RAD_S:
        stop = "speed %.2f rad/s" % np.abs(st.dq).max()
        break
      sent = drv.send(q_ref, dq_ref, tau)
      rec.add(t=t, tag=n, **{f"q{i+1}": st.q[i] for i in range(6)},
              **{f"dq{i+1}": st.dq[i] for i in range(6)},
              **{f"qref{i+1}": q_ref[i] for i in range(6)},
              **{f"dqref{i+1}": dq_ref[i] for i in range(6)},
              **{f"tff{i+1}": sent["tau_ff"][i] for i in range(6)},
              **{f"kp{i+1}": sent["kp"][i] for i in range(6)},
              **{f"kd{i+1}": sent["kd"][i] for i in range(6)},
              **{f"eff{i+1}": mf.effort[i] for i in range(6)},
              **{f"mspd{i+1}": mf.motor_speed[i] for i in range(6)})
      n += 1
      slack = period - (time.time() - t0 - t)
      if slack > 0:
        time.sleep(slack)
  finally:
    last = arm.read()
    drv.leave(last.q)
  return stop


def _col(rec: Recorder, name: str) -> np.ndarray:
  return np.array([r[name] for r in rec.rows], dtype=np.float64)


def _open(a, name: str, meta: dict):
  arm = _connect(a.can)
  ff = mit.GravityFeedforward()
  gains = (mit.Gains.load(pathlib.Path(a.gains)) if a.gains
           else mit.Gains.uniform(a.kp, a.kd))
  drv = mit.MitDriver(arm.iface, gains, torque_limit=a.torque_limit)
  rec = Recorder(name, {**meta, "kp": gains.kp.tolist(), "kd": gains.kd.tolist(),
                        "torque_limit": a.torque_limit, "rate": a.rate,
                        "gravity": a.gravity})
  return arm, ff, drv, rec


# ---------------------------------------------------------------------------
# hold -- MIT, standing still
# ---------------------------------------------------------------------------

def cmd_hold(a) -> int:
  """Enter MIT at the measured pose and do nothing, for a few seconds.

  The cheapest thing that can go wrong is the most important one: if the
  written gains make the loop unstable, it shows here, standing still, instead
  of half way through a chirp.  It also measures the steady-state error against
  gravity, which is the static stiffness if the drives are not compensating.
  """
  if not _confirm(f"MIT hold on {a.can}: kp {a.kp}, kd {a.kd}, "
                  f"gravity x{a.gravity}, {a.seconds:.0f} s.", a.yes):
    return 1
  arm, ff, drv, rec = _open(a, "hold", {"experiment": "hold"})
  st = _wait_feedback(arm)
  q0 = st.q.copy()
  stop = _mit_loop(arm, drv, ff, lambda t: (q0, np.zeros(6)),
                   a.seconds, a.rate, rec, gravity=a.gravity)
  arm.close()

  q = np.stack([_col(rec, f"q{i+1}") for i in range(6)], axis=1)
  tau = np.stack([_col(rec, f"tff{i+1}") for i in range(6)], axis=1)
  settled = q[len(q) // 2:]
  err = q0 - settled.mean(axis=0)
  model = ff(q0)
  print("\nstop: %s   frames %d   clamps %s"
        % (stop, drv.frames_sent, drv.clamps.report() or "none"))
  print("\njoint   sag (deg)   sag (mrad)   model bias   t_ff sent   implied kp_eff")
  for i in range(6):
    residual = model[i] - tau[:, i].mean()
    kp_eff = (residual / err[i]) if abs(err[i]) > 2e-4 else float("nan")
    print("  %d    %+8.3f    %+9.2f   %+9.2f   %+9.2f   %s"
          % (i + 1, math.degrees(err[i]), err[i] * 1000, model[i],
             tau[:, i].mean(),
             "n/a" if not np.isfinite(kp_eff) else "%8.1f" % kp_eff))
  print("\nimplied kp_eff is (uncarried gravity) / (sag).  It is only a stiffness")
  print("if the drives carry no gravity of their own -- compare against `probe`.")
  print("Drift while holding: %.3f deg" % math.degrees(
    np.abs(q[-1] - q[len(q) // 2]).max()))
  d = rec.save(stop=stop, sag_rad=err.tolist(), model_bias=model.tolist(),
               clamps=drv.clamps.report())
  print(f"wrote {d}")
  return 0


# ---------------------------------------------------------------------------
# gravity -- the torque scale, from a staircase at a fixed reference
# ---------------------------------------------------------------------------

def cmd_gravity(a) -> int:
  """Hold one joint and push on it with known feedforward.

  ``t_ref`` is the field this whole design hangs on: a written 8 is either 8
  N.m at the joint, in which case joint2 cannot be gravity compensated at every
  posture, or 8 at the rotor, in which case it is a hundred times that and the
  first frame had better be small.  The slope of steady-state position against
  written torque is ``g_t / kp_eff`` -- one number that separates into two once
  ``step`` has measured the bandwidth.

  The staircase is symmetric and returns to zero between rungs, so a drift that
  is really creep does not read as compliance.
  """
  j = a.joint - 1
  rungs = [0.0] + [s * v for v in a.torques for s in (+1.0, -1.0)] + [0.0]
  if max(abs(v) for v in a.torques) > a.torque_limit:
    raise SystemExit("--torques exceeds --torque-limit")
  if not _confirm(
      f"MIT torque staircase on joint {a.joint}: {rungs} N.m written, "
      f"{a.dwell:.1f} s each, kp {a.kp} kd {a.kd}.", a.yes):
    return 1
  arm, ff, drv, rec = _open(a, f"gravity_j{a.joint}",
                            {"experiment": "gravity", "joint": a.joint,
                             "rungs": rungs, "dwell": a.dwell})
  st = _wait_feedback(arm)
  q0 = st.q.copy()
  base = ff(q0) * a.gravity

  def ref(t):
    return q0, np.zeros(6)

  # The staircase rides on top of whatever gravity term the run is using, so a
  # rung of zero is not "no torque", it is "the same torque as the hold".
  extra = np.zeros(6)

  def loop_ff(q, dq):
    return ff(q, dq) * a.gravity + extra

  seconds = a.dwell * len(rungs)
  period = 1.0 / a.rate
  drv.enter(q0)
  stop = "done"
  t0 = time.time()
  try:
    while True:
      t = time.time() - t0
      if t >= seconds:
        break
      k = min(int(t / a.dwell), len(rungs) - 1)
      extra[:] = 0.0
      extra[j] = rungs[k]
      st = arm.read()
      mf = mit.read_motors(arm.iface)
      if abs(q0[j] - st.q[j]) > ABORT_TRACKING_RAD:
        stop = "tracking error"
        break
      sent = drv.send(q0, np.zeros(6), loop_ff(st.q, st.dq))
      rec.add(t=t, rung=rungs[k], q=st.q[j], dq=st.dq[j],
              tff=sent["tau_ff"][j], eff=mf.effort[j], mspd=mf.motor_speed[j])
      slack = period - (time.time() - t0 - t)
      if slack > 0:
        time.sleep(slack)
  finally:
    drv.leave(arm.read().q)
    arm.close()

  t = _col(rec, "t"); rung = _col(rec, "rung")
  q = _col(rec, "q"); tff = _col(rec, "tff"); eff = _col(rec, "eff")
  # Only the settled second half of each dwell, so the transient is not fitted.
  keep = (t % a.dwell) > 0.6 * a.dwell
  levels, pos, torque, meas = [], [], [], []
  for r in sorted(set(rung.tolist())):
    m = keep & (rung == r)
    if m.sum() < 5:
      continue
    levels.append(r); pos.append(q[m].mean())
    torque.append(tff[m].mean()); meas.append(eff[m].mean())
  levels = np.array(levels); pos = np.array(pos)
  torque = np.array(torque); meas = np.array(meas)
  A = np.vstack([torque, np.ones_like(torque)]).T
  slope, _ = np.linalg.lstsq(A, pos, rcond=None)[0]
  eslope = np.linalg.lstsq(A, meas, rcond=None)[0][0] if len(torque) > 2 else np.nan

  print("\nstop: %s   clamps %s" % (stop, drv.clamps.report() or "none"))
  print("\n written t_ff   sent    settled pos (mrad from start)   reported effort")
  for r, tq, p, e in zip(levels, torque, pos, meas):
    print("   %+8.2f   %+7.2f          %+9.3f                   %+8.3f"
          % (r, tq, (p - q0[j]) * 1000, e))
  print("\nd(position)/d(written torque) = %+.5f rad/N.m" % slope)
  if abs(slope) > 1e-9:
    print("   -> kp_eff / g_t = %.1f  (N.m of joint torque per rad, per written unit)"
          % (1.0 / abs(slope)))
  print("d(reported effort)/d(written torque) = %+.3f" % eslope)
  print("   -> 1.0 means the effort register is in the same units as t_ref.")
  d = rec.save(stop=stop, levels=levels.tolist(), settled=pos.tolist(),
               pos_per_torque=float(slope), effort_per_torque=float(eslope),
               clamps=drv.clamps.report())
  print(f"wrote {d}")
  return 0


# ---------------------------------------------------------------------------
# step -- second-order fit, which is what pins kp_eff independently
# ---------------------------------------------------------------------------

def _fit_second_order(t, y, y0, y1):
  """(omega_n, zeta, delay) for a step from ``y0`` to ``y1``.

  Fitted rather than read off overshoot and period, because the small steps
  this is allowed to use are not always underdamped enough to have either.
  """
  span = y1 - y0
  if abs(span) < 1e-6:
    return float("nan"), float("nan"), float("nan")
  z = (y - y0) / span

  def model(p):
    wn, ze, td = p
    tt = np.clip(t - td, 0.0, None)
    if ze < 1.0:
      wd = wn * math.sqrt(max(1.0 - ze * ze, 1e-9))
      r = 1.0 - np.exp(-ze * wn * tt) * (
        np.cos(wd * tt) + (ze * wn / wd) * np.sin(wd * tt))
    else:
      r = 1.0 - np.exp(-wn * tt) * (1.0 + wn * tt)
    return np.where(t < td, 0.0, r)

  best, bp = np.inf, (np.nan, np.nan, np.nan)
  for wn in np.geomspace(1.0, 120.0, 60):
    for ze in np.linspace(0.10, 1.60, 40):
      for td in np.linspace(0.0, 0.06, 13):
        e = float(((model((wn, ze, td)) - z) ** 2).sum())
        if e < best:
          best, bp = e, (wn, ze, td)
  return bp


def cmd_step(a) -> int:
  """Step one joint, three amplitudes, both directions.

  ``omega_n`` with the model's effective inertia gives ``kp_eff`` without any
  assumption about gravity, and ``zeta`` gives ``kd_eff``.  That is the leg of
  the identification the static tests cannot supply on their own.
  """
  if abs(a.degrees) > MAX_DEG:
    raise SystemExit(f"{a.degrees} deg is past the {MAX_DEG} deg cap")
  amps = [a.degrees * f for f in (0.4, 0.7, 1.0)]
  if not _confirm(f"MIT steps on joint {a.joint}: +-{amps} deg, "
                  f"kp {a.kp} kd {a.kd}.", a.yes):
    return 1
  arm, ff, drv, rec = _open(a, f"step_j{a.joint}",
                            {"experiment": "step", "joint": a.joint,
                             "amplitudes_deg": amps, "dwell": a.dwell})
  j = a.joint - 1
  st = _wait_feedback(arm)
  q0 = st.q.copy()
  plan = [(0.0, 0.0)]
  for amp in amps:
    plan += [(math.radians(amp), 0.0), (0.0, 0.0),
             (-math.radians(amp), 0.0), (0.0, 0.0)]
  seconds = a.dwell * len(plan)

  def ref(t):
    k = min(int(t / a.dwell), len(plan) - 1)
    q = q0.copy()
    q[j] = q0[j] + plan[k][0]
    return q, np.zeros(6)

  stop = _mit_loop(arm, drv, ff, ref, seconds, a.rate, rec, gravity=a.gravity)
  arm.close()

  t = _col(rec, "t"); q = _col(rec, f"q{a.joint}"); qr = _col(rec, f"qref{a.joint}")
  print("\nstop: %s   clamps %s" % (stop, drv.clamps.report() or "none"))
  print("\n  step (deg)   omega_n (rad/s)   zeta    delay (ms)   settle 5%% (ms)")
  fits = []
  for k in range(1, len(plan)):
    m = (t >= k * a.dwell) & (t < (k + 1) * a.dwell)
    if m.sum() < 20:
      continue
    y0, y1 = qr[m][0] * 0 + q[m][0], qr[m][-1]
    if abs(y1 - y0) < math.radians(0.5):
      continue
    tt = t[m] - t[m][0]
    wn, ze, td = _fit_second_order(tt, q[m], y0, y1)
    band = 0.05 * abs(y1 - y0)
    late = np.where(np.abs(q[m] - y1) > band)[0]
    settle = tt[late[-1]] * 1000 if late.size else 0.0
    fits.append({"deg": math.degrees(y1 - y0), "wn": wn, "zeta": ze,
                 "delay_ms": td * 1000, "settle_ms": settle})
    print("   %+8.2f      %10.1f     %5.2f     %7.1f        %7.0f"
          % (fits[-1]["deg"], wn, ze, td * 1000, settle))

  inertia = _joint_inertia(ff, q0, j)
  wn = np.nanmedian([f["wn"] for f in fits]) if fits else float("nan")
  ze = np.nanmedian([f["zeta"] for f in fits]) if fits else float("nan")
  print("\neffective inertia at this posture: %.4f kg.m^2 (model)" % inertia)
  print("kp_eff = I * omega_n^2       = %8.1f N.m/rad" % (inertia * wn * wn))
  print("kd_eff = 2 zeta omega_n I    = %8.2f N.m.s/rad" % (2 * ze * wn * inertia))
  print("written kp %.1f  kd %.2f  ->  ratio kp_eff/kp %.2f   kd_eff/kd %.2f"
        % (a.kp, a.kd, inertia * wn * wn / max(a.kp, 1e-9),
           2 * ze * wn * inertia / max(a.kd, 1e-9)))
  print("\ntarget for this joint from robot.SYSID_GAINS: kp_eff %.0f  kd_eff %.1f"
        % _target_gains(a.joint))
  d = rec.save(stop=stop, fits=fits, inertia=inertia,
               kp_eff=float(inertia * wn * wn),
               kd_eff=float(2 * ze * wn * inertia),
               clamps=drv.clamps.report())
  print(f"wrote {d}")
  return 0


def _joint_inertia(ff: "mit.GravityFeedforward", q: np.ndarray, j: int) -> float:
  """Diagonal of the mass matrix for one joint, at one posture."""
  import mujoco
  ff(q)
  d = ff.data
  dof = int(ff._dadr[j])
  # M e_j, not inverse dynamics.  ``mj_inverse`` returns the torque including
  # constraint forces, and at a posture where any collision geom is touching
  # the table that term dwarfs the inertia: it read 1031 kg.m^2 for joint1 on
  # a 4.85 kg arm.  ``mj_mulM`` multiplies by the mass matrix and nothing else.
  # (mujoco 3.11 no longer exposes ``MjData.qM``, so it cannot be read out.)
  vec = np.zeros(ff.model.nv)
  vec[dof] = 1.0
  out = np.zeros(ff.model.nv)
  mujoco.mj_mulM(ff.model, d, out, vec)
  return float(out[dof])


def _target_gains(joint: int) -> tuple[float, float]:
  """What the simulator's actuator presents for this joint."""
  from piper_push import robot as sim_robot
  import re
  for expr, (kp, kd) in sim_robot.SYSID_GAINS.items():
    if re.fullmatch(expr.replace("[", "[").replace("]", "]"), f"joint{joint}"):
      return float(kp), float(kd)
  return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# sine -- stepped-sine frequency response
# ---------------------------------------------------------------------------

FREQS_HZ = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def _sine_fit(t, y, f):
  """Amplitude and phase of ``y`` at frequency ``f``, by least squares.

  A stepped sine rather than a chirp: each frequency gets its own settled
  window, so a fit cannot borrow energy from a neighbouring band, and a run
  that has to be aborted still leaves every completed frequency usable.
  """
  w = 2 * math.pi * f
  A = np.vstack([np.sin(w * t), np.cos(w * t), np.ones_like(t)]).T
  c, *_ = np.linalg.lstsq(A, y, rcond=None)
  return math.hypot(c[0], c[1]), math.atan2(c[1], c[0])


def cmd_sine(a) -> int:
  """Stepped sine on one joint: the Bode plot the policy's bandwidth lives in.

  The command stream this arm has to follow moves at its slew ceiling and
  reverses; how much of that survives is a magnitude and a phase at a few Hz,
  not a step response.  Phase lag here converts directly into the tracking lag
  measured on the bench.
  """
  amp = math.radians(a.degrees)
  peak = 2 * math.pi * max(FREQS_HZ) * amp
  if not _confirm(
      f"MIT stepped sine on joint {a.joint}: +-{a.degrees} deg over "
      f"{FREQS_HZ[0]}-{FREQS_HZ[-1]} Hz, peak speed {peak:.2f} rad/s, "
      f"kp {a.kp} kd {a.kd}.", a.yes):
    return 1
  arm, ff, drv, rec = _open(a, f"sine_j{a.joint}",
                            {"experiment": "sine", "joint": a.joint,
                             "degrees": a.degrees, "freqs": list(FREQS_HZ)})
  j = a.joint - 1
  st = _wait_feedback(arm)
  q0 = st.q.copy()
  # Enough cycles to fit, and at least a second so the low frequencies settle.
  blocks = [(f, max(1.0, a.cycles / f)) for f in FREQS_HZ]
  edges = np.cumsum([0.0] + [d for _, d in blocks])

  def ref(t):
    k = int(np.searchsorted(edges, t, side="right") - 1)
    k = min(max(k, 0), len(blocks) - 1)
    f = blocks[k][0]
    tau = t - edges[k]
    q = q0.copy()
    q[j] = q0[j] + amp * math.sin(2 * math.pi * f * tau)
    dq = np.zeros(6)
    return q, dq

  stop = _mit_loop(arm, drv, ff, ref, float(edges[-1]), a.rate, rec,
                   gravity=a.gravity)
  arm.close()

  t = _col(rec, "t"); q = _col(rec, f"q{a.joint}"); qr = _col(rec, f"qref{a.joint}")
  print("\nstop: %s   clamps %s" % (stop, drv.clamps.report() or "none"))
  print("\n  f (Hz)   |cmd| (deg)   |out| (deg)   gain (dB)   phase (deg)   lag (ms)")
  rows = []
  for k, (f, dur) in enumerate(blocks):
    m = (t >= edges[k] + 0.35 * dur) & (t < edges[k + 1])
    if m.sum() < 20:
      continue
    tt = t[m] - edges[k]
    ai, pi = _sine_fit(tt, qr[m] - qr[m].mean(), f)
    ao, po = _sine_fit(tt, q[m] - q[m].mean(), f)
    if ai < 1e-5:
      continue
    dphi = math.degrees(((po - pi) + math.pi) % (2 * math.pi) - math.pi)
    rows.append({"f": f, "gain_db": 20 * math.log10(ao / ai),
                 "phase_deg": dphi, "lag_ms": -dphi / 360.0 / f * 1000})
    print("   %5.2f      %8.3f      %8.3f     %+8.2f     %+8.1f     %7.1f"
          % (f, math.degrees(ai), math.degrees(ao), rows[-1]["gain_db"],
             dphi, rows[-1]["lag_ms"]))
  bw = next((r["f"] for r in rows if r["gain_db"] < -3.0), None)
  print("\n-3 dB bandwidth: %s" % ("%.2f Hz" % bw if bw else "above the sweep"))
  d = rec.save(stop=stop, rows=rows, bandwidth_hz=bw, clamps=drv.clamps.report())
  print(f"wrote {d}")
  return 0


# ---------------------------------------------------------------------------
# slew -- the acceptance metric, and the gain search
# ---------------------------------------------------------------------------

SLEW_MAX_DEG = 45.0


def _slew_stats(t, qr, q):
  lag = np.abs(qr - q)
  dq = np.abs(np.gradient(q, t))
  return {
    "lag_p50": float(np.percentile(lag, 50)),
    "lag_p95": float(np.percentile(lag, 95)),
    "lag_max": float(lag.max()),
    "dq_p50": float(np.percentile(dq, 50)),
    "dq_p95": float(np.percentile(dq, 95)),
  }


def cmd_slew(a) -> int:
  """Triangle at the command path's own slew ceiling -- what the policy does.

  The bench numbers this is meant to move: on 2026-08-30 the arm under MOVE J
  lagged 0.106 rad at p50 and 0.602 at p95 while the simulator, running the
  same policy on the same scene, sat at 0.101 and 0.281.  The simulator's
  achieved speed was 0.80 / 2.28 rad/s against the arm's 0.39 / 1.47.  Those
  four numbers are the acceptance test, and this is what produces them without
  a policy, a camera or an object in the loop.

  ``--kp-list``/``--kd-list`` run the same triangle at several impedances back
  to back, which is the search.
  """
  from piper_push import robot as sim_robot
  j = a.joint - 1
  name = ARM[j]
  ceiling = sim_robot.COMMAND_RATE_LIMIT_RAD_S[name]
  speed = a.rate_fraction * ceiling
  amp = math.radians(a.degrees)
  if a.degrees > SLEW_MAX_DEG:
    raise SystemExit(f"{a.degrees} deg is past the {SLEW_MAX_DEG} deg cap")
  half = 2 * amp / speed
  pairs = list(zip(a.kp_list or [a.kp], a.kd_list or [a.kd]))
  if a.kp_list and a.kd_list and len(a.kp_list) != len(a.kd_list):
    raise SystemExit("--kp-list and --kd-list must be the same length")
  if not _confirm(
      f"{a.mode.upper()} triangle on joint {a.joint}: +-{a.degrees} deg at "
      f"{speed:.2f} rad/s ({a.rate_fraction:.0%} of the {ceiling:.2f} ceiling), "
      f"reversing every {half:.2f} s, {len(pairs)} gain pair(s).", a.yes):
    return 1

  arm, ff, drv, rec = _open(a, f"slew_j{a.joint}_{a.mode}",
                            {"experiment": "slew", "joint": a.joint,
                             "mode": a.mode, "degrees": a.degrees,
                             "rate_fraction": a.rate_fraction,
                             "speed_rad_s": speed, "pairs": pairs})
  st = _wait_feedback(arm)
  q0 = st.q.copy()

  def ref(t):
    # A triangle, phase-shifted so it starts at the centre travelling up.
    u = (t / half) % 2.0
    x = u if u < 1.0 else 2.0 - u
    q = q0.copy()
    q[j] = q0[j] + amp * (2.0 * x - 1.0)
    dq = np.zeros(6)
    return q, dq

  results = []
  try:
    for kp, kd in pairs:
      drv.gains = mit.Gains.uniform(kp, kd)
      before = len(rec.rows)
      if a.mode == "impedance":
        stop = _mit_loop(arm, drv, ff, ref, a.seconds, a.rate, rec,
                         gravity=a.gravity)
      else:
        arm.response_mode = (robot.RESPONSE_MIT if a.mode == "mit"
                             else robot.RESPONSE_PLAIN)
        stop = _movej_loop(arm, ref, a.seconds, config.CONTROL_HZ, rec)
      seg = rec.rows[before:]
      t = np.array([r["t"] for r in seg])
      qq = np.array([r[f"q{a.joint}"] for r in seg])
      qr = np.array([r[f"qref{a.joint}"] for r in seg])
      s = _slew_stats(t, qr, qq)
      s.update(kp=kp, kd=kd, stop=stop, n=len(seg))
      results.append(s)
      print("  kp %6.1f  kd %5.2f   lag p50 %.3f  p95 %.3f  max %.3f   "
            "|dq| p50 %.2f  p95 %.2f  %s"
            % (kp, kd, s["lag_p50"], s["lag_p95"], s["lag_max"],
               s["dq_p50"], s["dq_p95"], "" if stop == "done" else stop))
      if stop != "done":
        break
  finally:
    arm.close()

  print("\ntarget, from the simulator running the same policy on the failed scene:")
  print("  lag p50 0.101   p95 0.281      |dq| p50 0.80   p95 2.28")
  print("measured under MOVE J on the bench, for comparison:")
  print("  lag p50 0.106   p95 0.602      |dq| p50 0.39   p95 1.47")
  d = rec.save(results=results, clamps=drv.clamps.report())
  print(f"wrote {d}")
  return 0


def _movej_loop(arm, ref, seconds, rate, rec):
  """The same triangle through the old command path, for the A/B."""
  period = 1.0 / rate
  t0 = time.time()
  stop = "done"
  while True:
    t = time.time() - t0
    if t >= seconds:
      break
    st = arm.read()
    q_ref, dq_ref = ref(t)
    if np.abs(q_ref - st.q).max() > ABORT_TRACKING_RAD:
      stop = "tracking error %.3f rad" % np.abs(q_ref - st.q).max()
      break
    target = np.concatenate([q_ref, [st.gripper]])
    arm.command(target, period)
    rec.add(t=t, **{f"q{i+1}": st.q[i] for i in range(6)},
            **{f"dq{i+1}": st.dq[i] for i in range(6)},
            **{f"qref{i+1}": q_ref[i] for i in range(6)})
    slack = period - (time.time() - t0 - t)
    if slack > 0:
      time.sleep(slack)
  return stop


# ---------------------------------------------------------------------------
# replay -- the same command stream that failed, through both command paths
# ---------------------------------------------------------------------------

def cmd_replay(a) -> int:
  """Replay a recorded policy command stream, open loop, MIT or MOVE J.

  The triangle is a controlled excitation; this is the real thing.  The targets
  come out of a session that failed, so the arm will retrace a path that once
  went for the table -- the grasp-height floor is not optional here, and the
  replay stops at it rather than holding, because a held pose in the middle of
  a replay is no longer the trajectory being measured.
  """
  from . import proprio
  meta = json.loads((a.session / "meta.json").read_text())
  cmds = [m for m in meta if m.get("event") == "command" and "target" in m]
  if not cmds:
    raise SystemExit(f"no command events in {a.session}")
  targets = np.array([m["target"][:7] for m in cmds], dtype=np.float64)
  kin = proprio.Kinematics()

  def height(q7):
    kin.update(np.concatenate([q7[:6], [q7[6], -q7[6]]]))
    return float(kin.site_pos[2])

  heights = np.array([height(t) for t in targets])
  low = int((heights < a.min_grasp_height).sum())
  print(f"{len(targets)} commands, grasp site {heights.min()*1000:.1f} to "
        f"{heights.max()*1000:.1f} mm; {low} below the "
        f"{a.min_grasp_height*1000:.0f} mm floor")
  if not _confirm(
      f"{a.mode.upper()} replay of {a.session.name} on {a.can}: "
      f"{len(targets)} commands at {config.CONTROL_HZ:.0f} Hz "
      f"({len(targets)/config.CONTROL_HZ:.1f} s).", a.yes):
    return 1

  arm, ff, drv, rec = _open(a, f"replay_{a.mode}",
                            {"experiment": "replay", "mode": a.mode,
                             "session": str(a.session)})
  arm.response_mode = (robot.RESPONSE_MIT if a.mode == "mit"
                       else robot.RESPONSE_PLAIN)
  st = _wait_feedback(arm)
  # Start from where the recording started, not from wherever the arm is: an
  # open-loop replay whose first command is 40 degrees away is a step, not a
  # trajectory.
  gap = float(np.abs(st.q - targets[0][:6]).max())
  if gap > math.radians(a.max_start_gap_deg):
    arm.close()
    raise SystemExit(
      f"the arm is {math.degrees(gap):.1f} deg from the recording's first "
      f"command; home it first (run.py --home-first) or raise "
      f"--max-start-gap-deg deliberately")

  period = 1.0 / config.CONTROL_HZ
  stop = "done"
  k = 0
  if a.mode == "impedance":
    drv.enter(st.q)
  t0 = time.time()
  try:
    while k < len(targets):
      t = time.time() - t0
      k = int(t / period)
      if k >= len(targets):
        break
      st = arm.read()
      q_ref = targets[k][:6]
      if heights[k] < a.min_grasp_height:
        stop = "grasp height %.1f mm below the floor" % (heights[k] * 1000)
        break
      if np.abs(q_ref - st.q).max() > ABORT_TRACKING_RAD:
        stop = "tracking error %.3f rad" % np.abs(q_ref - st.q).max()
        break
      if a.mode == "impedance":
        sent = drv.send(q_ref, np.zeros(6), ff(st.q, st.dq) * a.gravity)
      else:
        arm.command(targets[k], period)
        sent = {"tau_ff": np.zeros(6)}
      rec.add(t=t, k=k, **{f"q{i+1}": st.q[i] for i in range(6)},
              **{f"dq{i+1}": st.dq[i] for i in range(6)},
              **{f"qref{i+1}": q_ref[i] for i in range(6)},
              **{f"tff{i+1}": sent["tau_ff"][i] for i in range(6)})
      slack = period - (time.time() - t0 - t)
      if slack > 0:
        time.sleep(slack)
  finally:
    last = arm.read()
    if a.mode == "impedance":
      drv.leave(last.q)
    else:
      arm.hold()
    arm.close()

  t = _col(rec, "t")
  lag = np.stack([np.abs(_col(rec, f"qref{i+1}") - _col(rec, f"q{i+1}"))
                  for i in range(6)], axis=1)
  dq = np.stack([np.abs(_col(rec, f"dq{i+1}")) for i in range(6)], axis=1)
  print("\nstop: %s   %d of %d commands   clamps %s"
        % (stop, len(rec.rows), len(targets), drv.clamps.report() or "none"))
  print("\n  joint   lag p50   lag p95   lag max   |dq| p50   |dq| p95")
  for i in range(6):
    print("    %d    %8.3f  %8.3f  %8.3f   %8.2f   %8.2f"
          % (i + 1, np.percentile(lag[:, i], 50), np.percentile(lag[:, i], 95),
             lag[:, i].max(), np.percentile(dq[:, i], 50),
             np.percentile(dq[:, i], 95)))
  print("    all  %8.3f  %8.3f  %8.3f   %8.2f   %8.2f"
        % (np.percentile(lag, 50), np.percentile(lag, 95), lag.max(),
           np.percentile(dq, 50), np.percentile(dq, 95)))
  print("\nsimulator, same policy on the failed scene:  lag 0.101 / 0.281   "
        "|dq| 0.80 / 2.28")
  print("MOVE J on the bench, 2026-08-30:              lag 0.106 / 0.602   "
        "|dq| 0.39 / 1.47")
  d = rec.save(stop=stop, lag_p50=float(np.percentile(lag, 50)),
               lag_p95=float(np.percentile(lag, 95)),
               dq_p50=float(np.percentile(dq, 50)),
               dq_p95=float(np.percentile(dq, 95)),
               clamps=drv.clamps.report())
  print(f"wrote {d}")
  return 0


# ---------------------------------------------------------------------------
# gains -- what to write, given what was measured.  No hardware.
# ---------------------------------------------------------------------------

GAINS_FILE = mit.GAINS_FILE


def cmd_gains(a) -> int:
  """Turn measured written-to-effective scales into the frame the loop sends.

  The target is not "track as well as possible".  The policy was distilled and
  fine-tuned against ``robot.SYSID_GAINS`` with ``gravcomp = 1.0``; an arm
  tuned stiffer than that is as far out of distribution as one tuned softer,
  and the whole point of moving to MIT is that the law the simulator uses is
  now expressible.  So the written gain is the simulated one divided by the
  measured amplification, and the interesting output is which joints cannot
  get there.
  """
  scales = {}
  if a.from_runs:
    for d in a.from_runs:
      run = json.loads((pathlib.Path(d) / "run.json").read_text())
      j = run["meta"].get("joint")
      s = run.get("summary", {})
      if j is None or "kp_eff" not in s:
        print(f"skipping {d}: no per-joint step fit in it")
        continue
      kp_w = run["meta"]["kp"][j - 1]
      kd_w = run["meta"]["kd"][j - 1]
      scales[j] = (s["kp_eff"] / max(kp_w, 1e-9), s["kd_eff"] / max(kd_w, 1e-9))
      print("joint %d: measured kp_eff %7.1f / written %5.1f = %5.2f   "
            "kd_eff %6.2f / written %4.2f = %5.2f"
            % (j, s["kp_eff"], kp_w, scales[j][0],
               s["kd_eff"], kd_w, scales[j][1]))
  gkp = a.g_kp
  gkd = a.g_kd if a.g_kd is not None else a.g_kp

  kp = np.zeros(6)
  kd = np.zeros(6)
  notes = []
  print("\n joint   target kp_eff  kd_eff   scale kp   kd    write kp    kd")
  for i in range(6):
    tkp, tkd = _target_gains(i + 1)
    skp, skd = scales.get(i + 1, (gkp, gkd))
    kp[i], kd[i] = tkp / skp, tkd / skd
    lo_kp, hi_kp = mit.FIELD_RANGE["kp"]
    lo_kd, hi_kd = mit.FIELD_RANGE["kd"]
    flag = ""
    if not (lo_kp <= kp[i] <= hi_kp):
      notes.append(f"joint{i+1} kp {kp[i]:.1f} is outside {lo_kp}-{hi_kp}")
      flag = "  UNREACHABLE"
    if not (lo_kd <= kd[i] <= hi_kd):
      notes.append(f"joint{i+1} kd {kd[i]:.2f} is outside {lo_kd}-{hi_kd}")
      flag = "  UNREACHABLE"
    print("   %d      %7.1f     %5.2f     %5.2f   %5.2f    %7.2f  %5.2f%s"
          % (i + 1, tkp, tkd, skp, skd, kp[i], kd[i], flag))
  for n in notes:
    print("  !! " + n)
  if notes and not a.force:
    print("\nNot writing.  A gain the field cannot carry means the simulated")
    print("plant is not reachable on this arm, and the profile has to move")
    print("instead -- pass --force only to record the attempt.")
    return 2
  out = pathlib.Path(a.out) if a.out else GAINS_FILE
  mit.Gains(kp, kd).save(
    out, target="robot.SYSID_GAINS", scale_kp=gkp, scale_kd=gkd,
    per_joint_scales={str(k): list(v) for k, v in scales.items()},
    note="written gains; effective joint impedance is these times the scale")
  print(f"\nwrote {out}")
  return 0


def cmd_plan(a) -> int:
  """Print the session, in the order the answers depend on each other."""
  print(__doc__)
  print("""
Order and why
-------------
  probe      no motion.  Settles whether the drives report joint units or motor
             units, and whether they already hold gravity.  Everything after
             this reads its answer.
  hold       MIT standing still, vendor gains, gravity 0 first.  If the loop is
             unstable it is unstable here, at zero speed.
  gravity    one joint, torque staircase.  Gives kp_eff/g_t, and whether an 8
             N.m field can carry joint2's 14 N.m worst-case gravity.
  step       one joint, three amplitudes.  omega_n with the model's inertia is
             the only measurement that pins kp_eff without assuming anything
             about gravity; zeta gives kd_eff.  Run it on every joint.
  sine       bandwidth and phase lag, which is what the bang-bang command
             stream actually meets.
  gains      no hardware.  Divides the simulator's gains by the measured
             amplification and refuses to write a gain the CAN field cannot
             carry.
  slew       the acceptance test.  MOVE J first for the baseline, then MIT with
             the written gains, then --kp-list to search around them.
  replay     the same command stream that failed on 2026-08-30, both paths.

Acceptance
----------
  simulator, same policy on the failed scene   lag p50 0.101  p95 0.281
                                               |dq| p50 0.80  p95 2.28
  MOVE J, measured on the bench                lag p50 0.106  p95 0.602
                                               |dq| p50 0.39  p95 1.47

  MIT is worth switching to when its p95 lag lands near the simulator's 0.281
  rather than MOVE J's 0.602.  Matching, not beating: a stiffer arm than the
  one the policy trained against is a different arm.
""")
  return 0


# ---------------------------------------------------------------------------

def main() -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--yes", action="store_true", help="skip the confirmation")
  p.add_argument("--rate", type=float, default=200.0,
                 help="MIT streaming rate in Hz.  The drives run their law at "
                      "200 Hz and the simulator interpolates the command "
                      "across its substeps, so 200 is the faithful default "
                      "even though the policy is 50 Hz")
  p.add_argument("--kp", type=float, default=mit.VENDOR_KP)
  p.add_argument("--kd", type=float, default=mit.VENDOR_KD)
  p.add_argument("--gains", help="mit_gains.json instead of --kp/--kd")
  p.add_argument("--gravity", type=float, default=1.0,
                 help="scale on the model's qfrc_bias feedforward.  0 sends "
                      "none, which is what `probe` should be read before "
                      "changing")
  p.add_argument("--torque-limit", type=float, default=2.0,
                 help="N.m written, per joint, before the field's own 8.  The "
                      "small default survives t_ref turning out to be "
                      "motor-referred")
  sub = p.add_subparsers(dest="cmd", required=True)

  q = sub.add_parser("probe", help="read-only: units, gear ratio, resting torque")
  q.add_argument("--seconds", type=float, default=3.0)
  q.set_defaults(fn=cmd_probe)

  q = sub.add_parser("hold", help="MIT, standing still: stability and sag")
  q.add_argument("--seconds", type=float, default=4.0)
  q.set_defaults(fn=cmd_hold)

  q = sub.add_parser("gravity", help="torque staircase: the t_ref scale")
  q.add_argument("--joint", type=int, required=True, choices=range(1, 7))
  q.add_argument("--torques", type=float, nargs="+", default=[0.25, 0.5, 1.0])
  q.add_argument("--dwell", type=float, default=1.5)
  q.set_defaults(fn=cmd_gravity)

  q = sub.add_parser("step", help="step response: omega_n, zeta, delay")
  q.add_argument("--joint", type=int, required=True, choices=range(1, 7))
  q.add_argument("--degrees", type=float, default=DEFAULT_DEG)
  q.add_argument("--dwell", type=float, default=1.2)
  q.set_defaults(fn=cmd_step)

  q = sub.add_parser("sine", help="stepped sine: bandwidth and phase lag")
  q.add_argument("--joint", type=int, required=True, choices=range(1, 7))
  q.add_argument("--degrees", type=float, default=2.0)
  q.add_argument("--cycles", type=float, default=8.0)
  q.set_defaults(fn=cmd_sine)

  q = sub.add_parser("slew", help="triangle at the slew ceiling: the acceptance test")
  q.add_argument("--joint", type=int, required=True, choices=range(1, 7))
  q.add_argument("--degrees", type=float, default=20.0)
  q.add_argument("--rate-fraction", type=float, default=0.5)
  q.add_argument("--seconds", type=float, default=6.0)
  q.add_argument("--mode", choices=("mit", "movej", "impedance"), default="mit",
                 help="'mit' is the drives' fast-response law on the position "
                      "path; 'impedance' is per-joint JointMitCtrl")
  q.add_argument("--kp-list", type=float, nargs="+")
  q.add_argument("--kd-list", type=float, nargs="+")
  q.set_defaults(fn=cmd_slew)

  q = sub.add_parser("gains", help="written gains from measured scales (no arm)")
  q.add_argument("--g-kp", type=float, default=5.0,
                 help="measured effective-over-written stiffness scale")
  q.add_argument("--g-kd", type=float, default=None, help="defaults to --g-kp")
  q.add_argument("--from-runs", nargs="*", help="sysid step run directories")
  q.add_argument("--out")
  q.add_argument("--force", action="store_true")
  q.set_defaults(fn=cmd_gains)

  q = sub.add_parser("plan", help="print the session and the acceptance numbers")
  q.set_defaults(fn=cmd_plan)

  q = sub.add_parser("replay", help="a recorded policy command stream, either path")
  q.add_argument("session", type=pathlib.Path)
  q.add_argument("--mode", choices=("mit", "movej", "impedance"), default="mit")
  q.add_argument("--min-grasp-height", type=float, default=0.020)
  q.add_argument("--max-start-gap-deg", type=float, default=10.0)
  q.set_defaults(fn=cmd_replay)

  a = p.parse_args()
  LOG_ROOT.mkdir(parents=True, exist_ok=True)
  return a.fn(a)


if __name__ == "__main__":
  sys.exit(main())
