"""General plant identification for the PiPER-X, on the MIT command path.

Deliberately not task-shaped.  The excitations here are chosen to identify the
arm -- mass, inertia, friction, servo response -- and not to resemble anything
the pick policy does, so the result is a model rather than a fit to one
trajectory.

    python -m hardware.deploy.ident pose        # go to the sysid posture
    python -m hardware.deploy.ident static      # torque vs posture -> mass, com
    python -m hardware.deploy.ident sweep --joint 2
    python -m hardware.deploy.ident friction --joint 2
    python -m hardware.deploy.ident excite      # all joints -> inertia

Everything runs at 200 Hz, which is the rate the drives publish joint angles
at; above it the loop re-commands against a measurement it has already used.

**The table.**  The sysid posture stands the arm up: the grasp site sits at
814 mm and the nearest collision geometry to the table is ``link1_collision``
at 84 mm, which is upstream of joint2 and therefore cannot be moved toward the
table by any joint.  Every commanded waypoint is still checked against the
calibrated plane before it is sent, because a posture that is safe by argument
and a posture that is safe by measurement are not the same thing.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

from . import agx, config, proprio

LOG_ROOT = pathlib.Path("logs/ident")
RIG_FILE = pathlib.Path("hardware/deploy/rig_d455.json")

# Arm up, wrist high, and every joint free to swing +-25 deg without the
# nearest geometry to the table changing.  Searched over the J2/J3/J5 family.
SYSID_POSE_DEG = np.array([0.0, 60.0, -150.0, 0.0, -60.0, 0.0])
MIN_CLEARANCE_M = 0.050
GOTO_SPEED_RAD_S = 0.22
ABORT_TRACKING_RAD = math.radians(20.0)


class TableGuard:
  """Clearance of the moving collision geometry above the calibrated plane."""

  def __init__(self, rig_file: pathlib.Path = RIG_FILE) -> None:
    self.rig = config.Rig.load(rig_file)
    if self.rig.table_normal_base is None:
      raise SystemExit(f"{rig_file} has no table_normal_base; recalibrate")
    self.kin = proprio.Kinematics()

  def clearance(self, q) -> tuple[float, str]:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    self.kin.update(np.array([*q, 0.05, -0.05]))
    return self.kin.collision_plane_clearance(
      self.rig.table_normal_base, self.rig.table_z)

  def check(self, q, floor: float = MIN_CLEARANCE_M) -> None:
    c, g = self.clearance(q)
    if c < floor:
      raise RuntimeError(
        f"commanded pose puts {g} {c*1000:.1f} mm above the table, under the "
        f"{floor*1000:.0f} mm floor")


class Recorder:
  def __init__(self, name: str, meta: dict) -> None:
    self.dir = LOG_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    self.dir.mkdir(parents=True, exist_ok=True)
    self.meta = dict(meta)
    self.rows: list[dict] = []

  def add(self, **kw) -> None:
    self.rows.append(kw)

  def save(self, **summary) -> pathlib.Path:
    if self.rows:
      keys = sorted({k for r in self.rows for k in r})
      np.savez_compressed(self.dir / "trace.npz", **{
        k: np.array([r.get(k, np.nan) for r in self.rows]) for k in keys})
    (self.dir / "run.json").write_text(
      json.dumps({"meta": self.meta, "summary": summary, "n": len(self.rows)},
                 indent=1, default=float) + "\n")
    return self.dir


def _confirm(what: str, yes: bool) -> bool:
  if yes:
    return True
  print(f"\n{what}\nWorkspace clear?  estop in reach?")
  return input("type 'go' to continue: ").strip() == "go"


def open_arm(a) -> tuple[agx.AgxArm, TableGuard]:
  guard = TableGuard(pathlib.Path(a.rig_file))
  arm = agx.AgxArm(a.can)
  arm.connect()
  s = arm.read()
  guard.check(s.q, floor=0.0)          # only refuses a pose already in the table
  print("feedback %.0f Hz   start (deg) %s"
        % (arm.feedback_hz(), np.round(np.degrees(s.q), 2)))
  arm.enable()
  return arm, guard


def goto(arm: agx.AgxArm, guard: TableGuard, target_rad: np.ndarray,
         speed: float = GOTO_SPEED_RAD_S, rate: float = 50.0) -> None:
    """Slow, checked, position-mode move.

    Every waypoint is cleared against the table before it is sent, and the
    whole path is cleared before the first one is: a guard that only checks
    the pose it is about to command cannot refuse a path that is safe at both
    ends and not in the middle.
    """
    q0 = arm.read().q
    target = np.asarray(target_rad, dtype=np.float64).reshape(6)
    distance = float(np.abs(target - q0).max())
    if distance < 1e-4:
      return
    duration = max(1.0, 1.5 * distance / speed)   # 3u^2-2u^3 peaks at 1.5x mean
    steps = int(duration * rate)
    path = []
    for k in range(steps + 1):
      u = k / steps
      path.append(q0 + (target - q0) * (u * u * (3.0 - 2.0 * u)))
    start_clear = guard.clearance(q0)[0]
    worst = min(guard.clearance(q)[0] for q in path)
    # The floor cannot be stricter than where the arm already is.  A pose that
    # starts below it -- which is exactly the situation a recovery move exists
    # for -- would otherwise be un-leavable: the first waypoint is the start,
    # so an absolute floor rejects every path out of it, including the ones
    # that climb.  What must be refused is a path that goes LOWER than it began.
    floor = min(MIN_CLEARANCE_M, start_clear)
    if worst < floor - 1e-9:
      raise RuntimeError(
        f"the path drops to {worst*1000:.1f} mm above the table, below the "
        f"{floor*1000:.1f} mm it starts at; refusing")
    if start_clear < MIN_CLEARANCE_M:
      print("  note: starting at %.0f mm, under the %.0f mm floor; this move "
            "only climbs" % (start_clear * 1000, MIN_CLEARANCE_M * 1000))
    print("  moving %.1f deg over %.1f s, path clears the table by %.0f mm"
          % (math.degrees(distance), duration, worst * 1000))
    for q in path:
      arm.move_js(q)
      time.sleep(1.0 / rate)
    time.sleep(0.4)


def cmd_pose(a) -> int:
  """Stand the arm up into the identification posture."""
  target = np.radians(SYSID_POSE_DEG if a.target is None else np.array(a.target))
  guard = TableGuard(pathlib.Path(a.rig_file))
  c, g = guard.clearance(target)
  print("target (deg) %s\n  clearance %.0f mm (%s)"
        % (np.round(np.degrees(target), 1), c * 1000, g))
  if not _confirm(f"Move the arm to {np.round(np.degrees(target),1).tolist()} "
                  f"deg at {GOTO_SPEED_RAD_S} rad/s.", a.yes):
    return 1
  arm, guard = open_arm(a)
  try:
    goto(arm, guard, target)
    s = arm.read()
    print("arrived (deg) %s   error %.2f deg   clearance %.0f mm"
          % (np.round(np.degrees(s.q), 2),
             math.degrees(np.abs(s.q - target).max()),
             guard.clearance(s.q)[0] * 1000))
  finally:
    arm.release()
  return 0


# ---------------------------------------------------------------------------
# the 200 Hz MIT streaming loop every excitation runs through
# ---------------------------------------------------------------------------

def stream(arm, guard, rec, ref, seconds, rate, kp, kd, tau_ff=None,
           label="", abort_rad=ABORT_TRACKING_RAD):
  """Drive ``ref(t) -> (q_ref, dq_ref)`` and record both sides.

  The table is checked on the COMMANDED pose, before the frame is sent, not on
  the measured one: by the time a measurement is low the command that put it
  there has already been executed.
  """
  period = 1.0 / float(rate)
  stop = "done"
  t0 = time.monotonic()
  n = 0
  while True:
    t = time.monotonic() - t0
    if t >= seconds:
      break
    s = arm.read()
    q_ref, dq_ref = ref(t)
    c, geom = guard.clearance(q_ref)
    if c < MIN_CLEARANCE_M:
      stop = f"command would put {geom} at {c*1000:.0f} mm"
      break
    if np.abs(q_ref - s.q).max() > abort_rad:
      stop = "tracking error %.3f rad" % np.abs(q_ref - s.q).max()
      break
    tau = np.zeros(6) if tau_ff is None else np.asarray(tau_ff(s), dtype=float)
    sent = arm.move_mit(q_ref, dq_ref, kp, kd, tau)
    rec.add(t=t, label=label, stamp=s.stamp,
            **{f"q{i+1}": s.q[i] for i in range(6)},
            **{f"dq{i+1}": s.dq[i] for i in range(6)},
            **{f"tau{i+1}": s.torque[i] for i in range(6)},
            **{f"qref{i+1}": q_ref[i] for i in range(6)},
            **{f"dqref{i+1}": dq_ref[i] for i in range(6)},
            **{f"tff{i+1}": sent["tau_ff"][i] for i in range(6)})
    n += 1
    slack = period - (time.monotonic() - t0 - t)
    if slack > 0:
      time.sleep(slack)
  return stop, n


def _col(rec, name):
  return np.array([r[name] for r in rec.rows], dtype=np.float64)


# ---------------------------------------------------------------------------
# static -- torque against posture, which is the mass and centre of mass
# ---------------------------------------------------------------------------

def cmd_static(a) -> int:
  """Hold still at many postures and read the torque the drives supply.

  With the wrist bare and the arm at rest, the holding torque IS the gravity
  torque, so the residual against ``qfrc_bias`` is a direct measurement of the
  model's mass and centre-of-mass error -- one equation per joint per posture.
  Nothing moves fast and nothing is released: the arm only ever holds.
  """
  from piper_push import robot as R
  names = [f"joint{i}" for i in range(1, 7)]
  lo = np.array([R.SAFE_TARGET_CLIP[n][0] for n in names])
  hi = np.array([R.SAFE_TARGET_CLIP[n][1] for n in names])
  guard = TableGuard(pathlib.Path(a.rig_file))
  rng = np.random.default_rng(a.seed)

  poses, tries = [], 0
  while len(poses) < a.poses and tries < 20000:
    tries += 1
    q = lo + (hi - lo) * rng.random(6)
    if guard.clearance(q)[0] >= a.clearance:
      poses.append(q)
  print("sampled %d postures clearing %.0f mm (from %d draws)"
        % (len(poses), a.clearance * 1000, tries))
  if not _confirm(f"Visit {len(poses)} postures at {GOTO_SPEED_RAD_S} rad/s, "
                  f"holding {a.dwell:.1f} s at each "
                  f"(~{len(poses)*(a.dwell+4):.0f} s).", a.yes):
    return 1

  arm, guard = open_arm(a)
  rec = Recorder("static", {"experiment": "static", "poses": len(poses),
                            "kp": agx.START_KP.tolist(),
                            "kd": agx.START_KD.tolist()})
  ff = _gravity_model()
  ok = 0
  try:
    for k, q in enumerate(poses):
      try:
        goto(arm, guard, q)
      except RuntimeError as e:
        print("  %2d skipped: %s" % (k, e))
        continue
      ref = lambda t, q=q: (q, np.zeros(6))
      stop, _ = stream(arm, guard, rec, ref, a.dwell, a.rate,
                       agx.START_KP, agx.START_KD, label=k)
      s = arm.read()
      print("  %2d/%d  %s  |tau| %s  %s"
            % (k + 1, len(poses), np.round(np.degrees(s.q), 0).astype(int),
               np.round(s.torque, 2), "" if stop == "done" else stop))
      ok += 1
      if stop != "done":
        break
  finally:
    arm.release()
  d = rec.save(poses=len(poses), visited=ok)
  print(f"\nwrote {d}")
  return 0


def _gravity_model():
  from . import mit
  return mit.GravityFeedforward()


# ---------------------------------------------------------------------------
# sweep -- stepped sine, which is the servo response
# ---------------------------------------------------------------------------

SWEEP_HZ = (0.5, 0.8, 1.2, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0)


def cmd_sweep(a) -> int:
  """Stepped sine on one joint, amplitude tapered to bound peak velocity.

  A fixed amplitude cannot be swept to 15 Hz: 5 degrees at 15 Hz is 8 rad/s,
  well past the joint's trip.  So the amplitude is whichever of the angular cap
  and the velocity cap binds, which keeps every band excited as hard as it can
  safely be and no harder.
  """
  j = a.joint - 1
  arm, guard = open_arm(a)
  q0 = arm.read().q.copy()
  amp_cap = math.radians(a.degrees)
  band = a.stiction_deg and math.radians(a.stiction_deg)

  # Two regimes, split at the frequency where the two constraints cross.
  #
  # Below it, the amplitude can simply be held above the joint's stiction band
  # (Coulomb/kp; 2.6 deg on joint2) without exceeding a safe peak velocity, and
  # no carrier is needed.
  #
  # Above it, an amplitude that clears the band would need more peak velocity
  # than the joint may have, so the sine is instead ridden on a one-directional
  # carrier: the joint never reverses, stiction never re-engages, and a small
  # amplitude is tracked linearly.  One direction PER BLOCK, because a carrier
  # that turns inside a block leaves an alternating slope that a single linear
  # detrend cannot remove -- measured, and it produced positive phase lags.
  blocks = []
  for f in SWEEP_HZ:
    if f <= a.carrier_above:
      amp = min(amp_cap, a.low_speed / (2 * math.pi * f))
      if band:
        amp = max(amp, 1.6 * band)
      blocks.append((f, amp, max(1.5, min(8.0, a.cycles / f)), 0.0))
    else:
      amp = min(amp_cap, a.max_speed / (2 * math.pi * f))
      dur = max(a.carrier_seconds, 3.0 / f)
      blocks.append((f, amp, dur, a.carrier_speed))
  if any(c and c <= a.max_speed for _, _, _, c in blocks):
    raise SystemExit("--carrier-speed must exceed --max-speed")

  # Alternate the carrier direction so the joint walks back and forth about the
  # start rather than marching away from it.
  # Each carrier block returns to where it began.  Alternating the sign does
  # NOT do that when the blocks have different durations: measured, joint2 came
  # out of a sweep 35 degrees from where it went in, and release() then held it
  # there.  A per-block out-and-back keeps the posture the experiment was
  # designed around.
  starts = [0.0] * len(blocks)
  total = sum(d for _, _, d, _ in blocks)
  print("joint %d about %.1f deg" % (a.joint, math.degrees(q0[j])))
  for (f, amp, d, c), b0 in zip(blocks, starts):
    print("   %5.1f Hz  +-%5.2f deg  peak %.2f rad/s  %4.1f s  carrier %.2f rad/s"
          % (f, math.degrees(amp), 2 * math.pi * f * amp, d, c))
  if not _confirm(f"Sweep joint {a.joint}, {total:.0f} s.", a.yes):
    arm.release()
    return 1

  edges = np.cumsum([0.0] + [d for _, _, d, _ in blocks])
  sgn = [1.0 if k % 2 == 0 else -1.0 for k in range(len(blocks))]

  def ref(t):
    k = min(max(int(np.searchsorted(edges, t, side="right") - 1), 0),
            len(blocks) - 1)
    f, amp, _, c = blocks[k]
    tau = t - edges[k]
    q = q0.copy()
    # Out for half the block, back for the other half: one direction at a
    # time (so stiction stays disengaged within each half) and net zero.
    ramp = c * (tau if tau < 0.5 * blocks[k][2]
                else blocks[k][2] - tau)
    q[j] = q0[j] + ramp + amp * math.sin(2 * math.pi * f * tau)
    return q, np.zeros(6)

  rec = Recorder(f"sweep_j{a.joint}",
                 {"experiment": "sweep", "joint": a.joint,
                  "blocks": [[f, amp, d, c] for f, amp, d, c in blocks],
                  "kp": agx.START_KP.tolist(), "kd": agx.START_KD.tolist()})
  try:
    stop, n = stream(arm, guard, rec, ref, float(edges[-1]), a.rate,
                     agx.START_KP, agx.START_KD)
  finally:
    arm.release()

  t = _col(rec, "t"); q = _col(rec, f"q{a.joint}"); qr = _col(rec, f"qref{a.joint}")
  tau = _col(rec, f"tau{a.joint}")
  print("\nstop: %s   %d samples" % (stop, n))
  print("\n  f (Hz)  |cmd| deg  |out| deg   gain dB   phase deg   lag ms")
  rows = []
  for k, (f, amp, d, c) in enumerate(blocks):
    m = (t >= edges[k] + 0.30 * d) & (t < edges[k + 1])
    if m.sum() < 20:
      continue
    tt = t[m] - edges[k]
    ai, pi = _sine_fit(tt, qr[m] - qr[m].mean(), f)
    ao, po = _sine_fit(tt, q[m] - q[m].mean(), f)
    if ai < 1e-6:
      continue
    dphi = math.degrees(((po - pi) + math.pi) % (2 * math.pi) - math.pi)
    rows.append({"f": f, "gain_db": 20 * math.log10(ao / ai),
                 "phase_deg": dphi, "lag_ms": -dphi / 360.0 / f * 1000,
                 "amp_cmd": ai, "amp_out": ao,
                 "tau_rms": float(np.std(tau[m]))})
    print("   %5.1f   %8.3f   %8.3f  %+8.2f   %+9.1f   %7.1f"
          % (f, math.degrees(ai), math.degrees(ao), rows[-1]["gain_db"],
             dphi, rows[-1]["lag_ms"]))
  bw = next((r["f"] for r in rows if r["gain_db"] < -3.0), None)
  lags = [r["lag_ms"] for r in rows if r["f"] >= 2.0]
  print("\n-3 dB bandwidth: %s" % ("%.1f Hz" % bw if bw else "above the sweep"))
  if lags:
    print("phase lag above 2 Hz: median %.1f ms" % np.median(lags))
  d = rec.save(stop=stop, rows=rows, bandwidth_hz=bw)
  print(f"wrote {d}")
  return 0


def _sine_fit(t, y, f):
  """Amplitude and phase at ``f``, with a linear trend removed.

  The trend term is not cosmetic: the excitation rides on a constant-velocity
  carrier, and a ramp left in the residual biases both the amplitude and the
  phase of a short window.
  """
  w = 2 * math.pi * f
  A = np.vstack([np.sin(w * t), np.cos(w * t), t, np.ones_like(t)]).T
  c, *_ = np.linalg.lstsq(A, y, rcond=None)
  return math.hypot(c[0], c[1]), math.atan2(c[1], c[0])



# ---------------------------------------------------------------------------
# step -- the one measurement that separates delay from dynamics
# ---------------------------------------------------------------------------

def cmd_step(a) -> int:
  """Step one joint and time the two onsets separately.

  A sweep cannot tell a transport delay from a heavily damped servo: both bend
  the phase without bending the magnitude much.  A step can, because the two
  show up in different channels.

  * **torque onset** is transport: the drive produces current the moment the
    frame lands, whatever the mechanism then does.
  * **position onset** is transport plus inertia plus breaking stiction.

  If the arm's 75 ms of apparent lag is delay, the torque is flat for 75 ms.
  If it is servo dynamics, the torque moves within a tick or two and only the
  position is slow.
  """
  j = a.joint - 1
  arm, guard = open_arm(a)
  rec = Recorder(f"step_j{a.joint}",
                 {"experiment": "step", "joint": a.joint,
                  "degrees": a.degrees, "kp": agx.START_KP.tolist(),
                  "kd": agx.START_KD.tolist()})
  try:
    if a.home:
      goto(arm, guard, np.radians(SYSID_POSE_DEG))
    q0 = arm.read().q.copy()
    amp = math.radians(a.degrees)
    plan = []
    for k in range(a.repeats):
      plan += [0.0, +amp, 0.0, -amp]
    dwell = a.dwell
    if not _confirm(f"Step joint {a.joint} by +-{a.degrees} deg, "
                    f"{len(plan)} steps of {dwell:.1f} s.", a.yes):
      return 1

    def ref(t):
      k = min(int(t / dwell), len(plan) - 1)
      q = q0.copy()
      q[j] = q0[j] + plan[k]
      return q, np.zeros(6)

    stop, n = stream(arm, guard, rec, ref, dwell * len(plan), a.rate,
                     agx.START_KP, agx.START_KD)
  finally:
    arm.release()

  t = _col(rec, "t"); q = _col(rec, f"q{a.joint}")
  qr = _col(rec, f"qref{a.joint}"); tau = _col(rec, f"tau{a.joint}")
  print("\nstop: %s   %d samples at %.0f Hz" % (stop, n, n / max(t[-1], 1e-9)))
  print("\n  step   torque onset   position onset   10-90%% rise")
  rows = []
  edges = np.arange(1, len(plan)) * dwell
  for e in edges:
    pre = (t > e - 0.25) & (t < e - 0.01)
    post = (t >= e) & (t < e + 0.6)
    if pre.sum() < 20 or post.sum() < 40:
      continue
    dq_cmd = qr[post][-1] - qr[pre][-1]
    if abs(dq_cmd) < math.radians(0.5):
      continue
    def onset(sig, k=6.0):
      base, sd = sig[pre].mean(), max(sig[pre].std(), 1e-9)
      hit = np.where(np.abs(sig[post] - base) > k * sd)[0]
      return (t[post][hit[0]] - e) * 1000 if hit.size else float("nan")
    ot, oq = onset(tau), onset(q)
    y = q[post] - q[pre].mean()
    tgt = dq_cmd
    lo = np.where(np.abs(y) > 0.1 * abs(tgt))[0]
    hi = np.where(np.abs(y) > 0.9 * abs(tgt))[0]
    rise = ((t[post][hi[0]] - t[post][lo[0]]) * 1000
            if lo.size and hi.size else float("nan"))
    rows.append({"deg": math.degrees(dq_cmd), "torque_ms": ot,
                 "pos_ms": oq, "rise_ms": rise})
    print("   %+5.1f      %7.1f ms      %7.1f ms     %8.1f ms"
          % (rows[-1]["deg"], ot, oq, rise))
  if rows:
    tt = np.array([r["torque_ms"] for r in rows], dtype=float)
    pp = np.array([r["pos_ms"] for r in rows], dtype=float)
    print("\nmedian torque onset   %.1f ms   <- transport delay" % np.nanmedian(tt))
    print("median position onset %.1f ms" % np.nanmedian(pp))
    print("\nsample period is %.1f ms, so onsets are quantised to that."
          % (1000.0 / a.rate))
    if np.nanmedian(tt) < 15.0:
      print("VERDICT: transport is a tick or two.  The ~75 ms in the sweep is")
      print("         servo dynamics, not delay.")
    else:
      print("VERDICT: the command really does take %.0f ms to reach the drive."
            % np.nanmedian(tt))
  d = rec.save(stop=stop, rows=rows)
  print(f"wrote {d}")
  return 0



# ---------------------------------------------------------------------------
# chirp -- the excitation a trajectory-matching fit wants
# ---------------------------------------------------------------------------

# Staggered bands so the joints do not move in lockstep.  A synchronised
# excitation leaves the inertia matrix's off-diagonal terms unidentifiable:
# every pose is then a scalar multiple of one direction, and the fit can trade
# coupling against diagonal mass without changing the residual.
CHIRP_BANDS = ((0.20, 6.0), (0.15, 5.0), (0.25, 7.0),
               (0.35, 8.0), (0.30, 9.0), (0.40, 10.0))


def _chirp_plan(a, q0):
  """Per-joint (amplitude, f0, f1, phase), amplitude capped by speed."""
  amp_cap = math.radians(a.degrees)
  plan = []
  for j in range(6):
    f0, f1 = CHIRP_BANDS[j]
    plan.append((amp_cap, f0, f1, 0.6180339887 * j))
  return plan


def _chirp_ref(plan, q0, seconds, max_speed=1.5):
  """Logarithmic chirp per joint, phase-continuous."""
  def ref(t):
    q = q0.copy()
    dq = np.zeros(6)
    u = min(max(t / seconds, 0.0), 1.0)
    # Taper in and out so the trajectory starts and ends at rest at q0.
    w = 0.5 - 0.5 * math.cos(2 * math.pi * min(u, 0.5) if u < 0.5
                             else 2 * math.pi * (1.0 - u) * 0.5 + math.pi)
    w = 0.5 - 0.5 * math.cos(math.pi * min(u / 0.1, 1.0)) if u < 0.1 else w
    w = 1.0 if 0.1 <= u <= 0.9 else (
      0.5 - 0.5 * math.cos(math.pi * min(u / 0.1, 1.0)) if u < 0.1
      else 0.5 - 0.5 * math.cos(math.pi * min((1.0 - u) / 0.1, 1.0)))
    for j, (amp, f0, f1, ph) in enumerate(plan):
      k = math.log(f1 / f0) / max(seconds, 1e-9)
      phase = 2 * math.pi * f0 * (math.exp(k * t) - 1.0) / k
      # Amplitude tracks the INSTANTANEOUS frequency, so the peak velocity is
      # bounded everywhere while the low-frequency end still gets a large
      # excursion.  Sizing the whole chirp for its top frequency instead left
      # joint6 moving 1.8 degrees over the entire run.
      f_now = f0 * math.exp(k * t)
      a_now = min(amp, max_speed / (2 * math.pi * f_now))
      q[j] = q0[j] + w * a_now * math.sin(phase + 2 * math.pi * ph)
    return q, dq
  return ref


def cmd_chirp(a) -> int:
  """Simultaneous multi-joint chirp, table-checked before anything moves."""
  guard = TableGuard(pathlib.Path(a.rig_file))
  q0 = np.radians(SYSID_POSE_DEG)
  plan = _chirp_plan(a, q0)
  ref = _chirp_ref(plan, q0, a.seconds, a.max_speed)

  # Pre-flight: the guard checks each frame as it goes, but a trajectory that
  # is only discovered to be unsafe half way through is a trajectory that has
  # already put the arm somewhere it should not be.  Walk the whole thing here.
  # Sampled, not every control tick: the clearance query walks every collision
  # geom's vertices, and a 40 s trajectory at 200 Hz is 8000 of them -- two
  # minutes of pre-flight for a trajectory that is smooth enough to be checked
  # at 25 Hz.  The per-frame guard inside stream() is what catches the rest.
  worst, worst_t = 1e9, 0.0
  for t in np.linspace(0.0, a.seconds, min(1200, int(a.seconds * 25))):
    c, _ = guard.clearance(ref(t)[0])
    if c < worst:
      worst, worst_t = c, t
  print("chirp plan, centred on the sysid posture:")
  for j, (amp, f0, f1, _) in enumerate(plan):
    print("  joint%d  +-%5.2f deg   %.2f -> %.1f Hz   peak %.2f rad/s"
          % (j + 1, math.degrees(amp), f0, f1, 2 * math.pi * f1 * amp))
  print("worst clearance over the whole %.0f s trajectory: %.0f mm (at t=%.1f s)"
        % (a.seconds, worst * 1000, worst_t))
  if worst < MIN_CLEARANCE_M:
    raise SystemExit("that trajectory reaches the table; lower --degrees")
  if not _confirm(f"Chirp all six joints for {a.seconds:.0f} s.", a.yes):
    return 1

  arm, guard = open_arm(a)
  rec = Recorder("chirp", {"experiment": "chirp", "seconds": a.seconds,
                           "degrees": a.degrees, "max_speed": a.max_speed,
                           "bands": [list(b) for b in CHIRP_BANDS],
                           "plan": [[amp, f0, f1, ph] for amp, f0, f1, ph in plan],
                           "q0": q0.tolist(),
                           "kp": agx.START_KP.tolist(),
                           "kd": agx.START_KD.tolist()})
  try:
    goto(arm, guard, q0)
    # A large tracking error is what this excitation is FOR -- the fit needs to
    # see the servo fail to follow.  The table is still safe: the guard clears
    # every commanded pose, and the measured pose lags inside the hull of poses
    # already commanded, so it never leaves the checked envelope.
    stop, n = stream(arm, guard, rec, ref, a.seconds, a.rate,
                     agx.START_KP, agx.START_KD,
                     abort_rad=math.radians(a.abort_deg))
  finally:
    arm.release()
  t = _col(rec, "t")
  print("\nstop: %s   %d samples at %.0f Hz" % (stop, n, n / max(t[-1], 1e-9)))
  print("\n  joint   range (deg)   peak |dq| (rad/s)   peak |tau| (N.m)")
  for j in range(6):
    q = _col(rec, f"q{j+1}"); dq = _col(rec, f"dq{j+1}"); tq = _col(rec, f"tau{j+1}")
    print("    %d      %8.2f        %8.2f            %8.2f"
          % (j + 1, math.degrees(np.ptp(q)),
             np.percentile(np.abs(dq), 99), np.percentile(np.abs(tq), 99)))
  d = rec.save(stop=stop, samples=n, worst_clearance_m=worst)
  print(f"wrote {d}")
  return 0



# ---------------------------------------------------------------------------
# friction -- constant-velocity strokes, where the chirp barely goes
# ---------------------------------------------------------------------------

FRICTION_SPEEDS = (0.03, 0.10, 0.30)


def cmd_friction(a) -> int:
  """Sweep one joint back and forth at a few constant speeds.

  At constant velocity the acceleration term vanishes, so the measured torque
  minus the model's gravity term IS the friction, signed by the direction of
  travel.  Both directions at each speed, because Coulomb friction is
  antisymmetric and anything left over after averaging the two is a model bias
  rather than friction.

  Worth its own experiment because the chirp hardly visits these speeds, and
  the low-speed regime is the one that matters: the identified curve is
  Stribeck-falling, so a viscous coefficient fitted here has the wrong sign at
  working speeds.
  """
  joints = [a.joint] if a.joint else list(range(1, 7))
  amp = math.radians(a.degrees)
  guard = TableGuard(pathlib.Path(a.rig_file))
  q0 = np.radians(SYSID_POSE_DEG)
  plan = []
  for j in joints:
    for v in FRICTION_SPEEDS:
      period = 4.0 * amp / v
      plan.append((j, v, min(a.max_seconds, max(period, 6.0))))
  total = sum(d for _, _, d in plan)
  print("constant-velocity strokes, +-%.1f deg about the sysid posture" % a.degrees)
  for j, v, d in plan:
    print("  joint%d  %.2f rad/s   %4.1f s" % (j, v, d))
  print("total %.0f s" % total)
  for j in joints:
    for sgn in (-1.0, 1.0):
      q = q0.copy(); q[j - 1] += sgn * amp
      guard.check(q)
  if not _confirm(f"Stroke joints {joints} at {FRICTION_SPEEDS} rad/s, "
                  f"{total:.0f} s.", a.yes):
    return 1

  arm, guard = open_arm(a)
  rec = Recorder("friction", {"experiment": "friction", "joints": joints,
                              "speeds": list(FRICTION_SPEEDS),
                              "degrees": a.degrees, "q0": q0.tolist(),
                              "kp": agx.START_KP.tolist(),
                              "kd": agx.START_KD.tolist()})
  ff = _gravity_model()
  try:
    goto(arm, guard, q0)
    for idx, (j, v, dur) in enumerate(plan):
      period = 4.0 * amp / v

      def ref(t, j=j, v=v, period=period):
        # Triangle at constant speed v, starting at q0 travelling up.
        u = (t / (0.5 * period) + 0.5) % 2.0
        x = u if u < 1.0 else 2.0 - u
        q = q0.copy()
        q[j - 1] = q0[j - 1] + amp * (2.0 * x - 1.0)
        return q, np.zeros(6)

      stop, n = stream(arm, guard, rec, ref, dur, a.rate,
                       agx.START_KP, agx.START_KD,
                       label=idx, abort_rad=math.radians(a.abort_deg))
      seg = rec.rows[-n:] if n else []
      if seg:
        dq = np.array([r[f"dq{j}"] for r in seg])
        tq = np.array([r[f"tau{j}"] for r in seg])
        print("  joint%d %.2f rad/s: achieved |dq| median %.3f, |tau| %.2f  %s"
              % (j, v, np.median(np.abs(dq)), np.median(np.abs(tq)),
                 "" if stop == "done" else stop))
      if stop != "done":
        break
      goto(arm, guard, q0)
  finally:
    arm.release()

  # friction = measured torque - model gravity, split by direction of travel
  lab = _col(rec, "label")
  print("\n joint  speed   +dir tau-model   -dir tau-model   Coulomb   bias")
  out = []
  for idx, (j, v, _) in enumerate(plan):
    m = lab == idx
    if m.sum() < 50:
      continue
    dq = _col(rec, f"dq{j}")[m]
    tq = _col(rec, f"tau{j}")[m]
    qq = np.stack([_col(rec, f"q{i+1}")[m] for i in range(6)], axis=1)
    grav = np.array([ff(qq[i])[j - 1] for i in range(0, len(qq), 7)])
    tqs = tq[::7][:len(grav)]
    dqs = dq[::7][:len(grav)]
    resid = tqs - grav
    up = dqs > 0.3 * v
    dn = dqs < -0.3 * v
    if up.sum() < 5 or dn.sum() < 5:
      continue
    fu, fd = float(np.median(resid[up])), float(np.median(resid[dn]))
    coulomb, bias = 0.5 * (fu - fd), 0.5 * (fu + fd)
    out.append({"joint": j, "speed": v, "up": fu, "down": fd,
                "coulomb": coulomb, "bias": bias})
    print("   %d    %.2f     %+9.3f       %+9.3f      %6.3f   %+6.3f"
          % (j, v, fu, fd, coulomb, bias))
  print("\nCoulomb is half the up-down gap; bias is what survives averaging the")
  print("two directions, which is model error rather than friction.")
  print("\nsim SYSID_COULOMB_NM: 0.25 1.13 0.54 0.08 0.03 0.05  (measured WITH")
  print("the 4.1 N wrist payload, which is now off)")
  d = rec.save(rows=out)
  print(f"wrote {d}")
  return 0


def build_parser() -> argparse.ArgumentParser:
  # The global flags go on a parent parser so they are accepted on BOTH sides
  # of the subcommand.  argparse's default puts them only before it, which at
  # the bench reads as a broken tool rather than as an argument-order rule --
  # it cost a run on 2026-08-31.
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument("--can", default="can0")
  common.add_argument("--rig-file", default=str(RIG_FILE))
  common.add_argument("--yes", action="store_true")
  common.add_argument("--rate", type=float, default=agx.FEEDBACK_HZ)

  p = argparse.ArgumentParser(description=__doc__, parents=[common],
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="cmd", required=True)

  q = sub.add_parser("pose", parents=[common],
                     help="go to the identification posture")
  q.add_argument("--target", type=float, nargs=6, metavar="DEG")
  q.set_defaults(fn=cmd_pose)

  q = sub.add_parser("static", parents=[common],
                     help="torque vs posture -> mass and centre of mass")
  q.add_argument("--poses", type=int, default=24)
  q.add_argument("--dwell", type=float, default=0.8)
  q.add_argument("--clearance", type=float, default=0.08)
  q.add_argument("--seed", type=int, default=0)
  q.set_defaults(fn=cmd_static)

  q = sub.add_parser("friction", parents=[common],
                     help="constant-velocity strokes -> Coulomb and Stribeck")
  q.add_argument("--joint", type=int, choices=range(1, 7),
                 help="default is all six")
  q.add_argument("--degrees", type=float, default=9.0)
  q.add_argument("--max-seconds", type=float, default=22.0)
  q.add_argument("--abort-deg", type=float, default=25.0)
  q.set_defaults(fn=cmd_friction)

  q = sub.add_parser("chirp", parents=[common],
                     help="simultaneous multi-joint chirp for a trajectory fit")
  q.add_argument("--seconds", type=float, default=40.0)
  q.add_argument("--degrees", type=float, default=18.0)
  q.add_argument("--max-speed", type=float, default=1.5)
  q.add_argument("--abort-deg", type=float, default=35.0)
  q.set_defaults(fn=cmd_chirp)

  q = sub.add_parser("step", parents=[common],
                     help="step response -> transport delay vs servo dynamics")
  q.add_argument("--joint", type=int, required=True, choices=range(1, 7))
  q.add_argument("--degrees", type=float, default=5.0)
  q.add_argument("--dwell", type=float, default=0.8)
  q.add_argument("--repeats", type=int, default=3)
  q.add_argument("--home", action="store_true",
                 help="return to the sysid posture first")
  q.set_defaults(fn=cmd_step)

  q = sub.add_parser("sweep", parents=[common],
                     help="stepped sine -> bandwidth, delay, kp/kd")
  q.add_argument("--joint", type=int, required=True, choices=range(1, 7))
  q.add_argument("--degrees", type=float, default=5.0)
  q.add_argument("--max-speed", type=float, default=0.30,
                 help="peak sinusoid velocity ABOVE --carrier-above, rad/s.  "
                      "It must stay under --carrier-speed or the joint "
                      "reverses and the sweep measures stiction again")
  q.add_argument("--cycles", type=float, default=6.0)
  q.add_argument("--carrier-speed", type=float, default=0.45,
                 help="one-directional carrier above --carrier-above, rad/s; "
                      "must exceed --max-speed so the joint never reverses")
  q.add_argument("--carrier-above", type=float, default=2.0,
                 help="frequency (Hz) above which the carrier is used")
  q.add_argument("--carrier-seconds", type=float, default=1.2)
  q.add_argument("--low-speed", type=float, default=1.2,
                 help="peak sinusoid velocity below --carrier-above")
  q.add_argument("--stiction-deg", type=float, default=2.6,
                 help="the joint's Coulomb/kp band; the low-frequency "
                      "amplitude is held at 1.6x it")
  q.set_defaults(fn=cmd_sweep)
  return p


def main() -> int:
  a = build_parser().parse_args()
  LOG_ROOT.mkdir(parents=True, exist_ok=True)
  return a.fn(a)


if __name__ == "__main__":
  sys.exit(main())
