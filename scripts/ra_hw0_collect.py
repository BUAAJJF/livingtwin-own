"""RA-HW-0 data collection.  Dry-run by default; hardware needs four keys.

    python scripts/ra_hw0_collect.py --stage H0            # dry run
    python scripts/ra_hw0_collect.py --stage H1 --joint joint5
    python scripts/ra_hw0_collect.py --stage H0 --hardware --i-have-approval

**The four keys.**  Even with ``--hardware`` this refuses to open CAN unless
*all* of:

1. ``--hardware`` is given -- the default is a dry run and always will be;
2. the audit report says an arm was actually READ, so the model, firmware and
   telemetry rate are facts rather than assumptions;
3. the limits table has no ``UNKNOWN`` row and carries ``motion_authorised``;
4. ``--i-have-approval`` is given, which the operator types only after the
   human sentence ``APPROVE RA-HW-0 MOTION`` has been said.

Key 3 is the one that cannot be argued with: the table sets
``motion_authorised`` from a file a human edits, and no code in this
repository writes it.

**H0 never commands.**  On hardware, stage H0 opens CAN, records telemetry and
closes.  It does not enable, does not command, and the code path that would
transmit is not reached -- ``--stage H0`` and a command stream are mutually
exclusive, asserted rather than assumed.

**Stopping is one-way.**  Any stop condition ends the session: no fault is
cleared, nothing is retried, and the next trajectory does not start.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class DryRunBackend:
  """A plausible arm: first-order lag, encoder quantisation, dropped frames.

  Deliberately not perfect.  A dry run whose telemetry is exactly the command
  exercises the logger and nothing else; this one makes the tracking-error
  stop condition, the telemetry-gap detector and the velocity estimator all do
  real work before anything is plugged in.
  """

  QUANT_RAD = math.radians(0.001)          # the SDK's 0.001-degree feedback

  def __init__(self, q0, tau_s: float = 0.06, drop_rate: float = 0.002,
               seed: int = 0):
    self.q = list(q0)
    self.prev = list(q0)
    self.tau = tau_s
    self.rng = random.Random(seed)
    self.drop_rate = drop_rate
    self.t_last = None

  def read(self, dt: float) -> dict | None:
    if self.rng.random() < self.drop_rate:
      return None                          # a dropped frame is a real event
    q = [round(v / self.QUANT_RAD) * self.QUANT_RAD for v in self.q]
    dq = [(a - b) / dt for a, b in zip(q, self.prev)]
    self.prev = q
    return {"q": q, "qdot": dq,
            "current": [0.0] * 6, "effort": [0.0] * 6,
            "motor_speed": dq, "driver_temp": [30] * 6, "motor_temp": [30] * 6,
            "voltage": 24.0, "faults": {f"motor_{i}": {} for i in range(1, 7)}}

  def command(self, target, dt: float) -> None:
    a = dt / (self.tau + dt)
    self.q = [c + a * (t - c) for c, t in zip(self.q, target)]


def run(stage: str, joint: str, backend, table, recorder, stop, segments,
        control_hz: float, session: str, dry: bool) -> str:
  """H0 records; H1 commands.  The difference is one argument and it is the
  difference between a stage that can and cannot hurt anybody."""
  from hardware.ra_hw0 import limits as L

  dt = 1.0 / control_hz
  idx = 0
  end_reason = "completed"
  prev_qd = None
  t_prev = None
  for seg in segments:
    if stop.stopped:
      break
    if stage == "H0" and seg.q:
      raise AssertionError("stage H0 must not carry a command stream")
    n = seg.n if seg.q else int(seg.duration_s * control_hz)
    for k in range(n):
      t0 = time.monotonic()
      cmd_req = list(seg.q[k]) if seg.q else None
      cmd_filt = L.clip_to_envelope(cmd_req, table) if cmd_req else None
      if cmd_filt is not None:
        backend.command(cmd_filt, dt)
      tel = backend.read(dt)
      gap = None if t_prev is None else t0 - t_prev
      t_prev = t0
      if tel is None:                       # dropped frame
        recorder.append(_sample(session, seg, idx, control_hz, cmd_req,
                                cmd_filt, None, gap, stop, dry))
        idx += 1
        if gap is not None and gap > table["environment"][
                "telemetry_gap_stop_s"]["value"]:
          stop.stop([f"telemetry gap {gap * 1e3:.1f} ms"])
          end_reason = "stopped"
          break
        continue
      qdd = None
      if prev_qd is not None:
        qdd = [(a - b) / dt for a, b in zip(tel["qdot"], prev_qd)]
      prev_qd = tel["qdot"]
      reasons = L.check_state(
        tel["q"], tel["qdot"], cmd_filt, table, qdd=qdd,
        temps={f"foc_{i}": tel["driver_temp"][i] for i in range(6)},
        fault_bits=tel["faults"], gap_s=gap,
        require_known=(stage != "H0"))
      recorder.append(_sample(session, seg, idx, control_hz, cmd_req, cmd_filt,
                              tel, gap, stop, dry))
      idx += 1
      if reasons:
        stop.stop(reasons)
        end_reason = "stopped"
        break
      time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    if stop.stopped:
      break
  return end_reason


def _sample(session, seg, idx, hz, cmd_req, cmd_filt, tel, gap, stop, dry):
  from hardware.ra_hw0.record import blank_sample
  return blank_sample(
    host_monotonic_s=time.monotonic(), host_wall_s=time.time(),
    device_timestamp_s=None,
    session_id=session, trial_id=seg.name, trajectory_id=seg.name,
    sample_index=idx,
    controller_mode="dry-run" if dry else "CAN position (MOVE J)",
    control_rate_hz=hz,
    command_requested_rad=cmd_req, command_after_safety_filter_rad=cmd_filt,
    q_rad=None if tel is None else tel["q"],
    qdot_rad_s=None if tel is None else tel["qdot"],
    servo_error_rad=None if (tel is None or cmd_filt is None) else
    [c - q for c, q in zip(cmd_filt, tel["q"])],
    joint_current_a=None if tel is None else tel["current"],
    joint_effort=None if tel is None else tel["effort"],
    motor_speed_rad_s=None if tel is None else tel["motor_speed"],
    driver_temp_c=None if tel is None else tel["driver_temp"],
    motor_temp_c=None if tel is None else tel["motor_temp"],
    bus_voltage_v=None if tel is None else tel["voltage"],
    fault_flags=None if tel is None else tel["faults"],
    watchdog_state="stopped" if stop.stopped else "ok",
    gripper_state="not commanded in RA-HW-0",
    operator_event=None,
    telemetry_gap_s=gap, msg_hz=None)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--stage", choices=("H0", "H1"), default="H0")
  ap.add_argument("--joint", default=None)
  ap.add_argument("--hardware", action="store_true")
  ap.add_argument("--i-have-approval", action="store_true",
                  help="the operator asserts that a human said "
                       "'APPROVE RA-HW-0 MOTION'.  Necessary, not sufficient.")
  ap.add_argument("--can", default="can0")
  ap.add_argument("--limits", default="configs/ra_hw0_safety_limits.json")
  ap.add_argument("--audit", default="results/ra_hw0/audit.json")
  ap.add_argument("--preview", default="results/ra_hw0/preview.json")
  ap.add_argument("--duration", type=float, default=30.0,
                  help="H0 telemetry window, seconds")
  ap.add_argument("--out", default="results/ra_hw0/sessions")
  ap.add_argument("--session", default=None)
  a = ap.parse_args()
  sys.path.insert(0, str(ROOT))
  sys.path.insert(0, str(ROOT / "src"))
  from hardware.ra_hw0 import limits as L
  from hardware.ra_hw0 import trajectories as T
  from hardware.ra_hw0.record import SessionRecorder, StopMachine, config_hash

  table = L.load(a.limits)
  joint = a.joint or T.H1_JOINT_ORDER[0]
  segments = ([T.h0_hold(a.duration)] if a.stage == "H0"
              else T.h1_plan(joint))

  refusals = []
  for seg in segments:
    refusals += T.check(seg, table)
  if a.hardware:
    refusals += L.blocks_motion(table)
    audit = json.loads(Path(a.audit).read_text()) if Path(a.audit).exists() \
      else {"device": {"status": "MISSING"}}
    if audit.get("device", {}).get("status") != "READ":
      refusals.append("the audit has not READ an arm: model, firmware and "
                      "telemetry rate are still assumptions")
    if not a.i_have_approval:
      refusals.append("--i-have-approval was not given")
    prev = Path(a.preview)
    if not prev.exists() or not json.loads(prev.read_text()).get("preview_ok"):
      refusals.append("no passing trajectory preview exists; run "
                      "scripts/ra_hw0_replay.py and show it to the operator")

  if a.hardware and refusals:
    print("HARDWARE REFUSED.  Nothing was opened, enabled or commanded.")
    for r in refusals:
      print("  -", r)
    return 2
  if refusals:
    print("trajectory refused even for a dry run:")
    for r in refusals:
      print("  -", r)
    return 2

  session = a.session or f"{a.stage}_{joint}_{time.strftime('%Y%m%d_%H%M%S')}"
  dry = not a.hardware
  backend = DryRunBackend([table["start_pose_rad"][j] for j in L.ARM_JOINTS])
  if not dry:
    raise SystemExit(
      "the hardware backend is not implemented in this phase.  RA-HW-0's "
      "deliverable is the design, the limits and the dry-run stack; the "
      "backend lands in the same commit as the operator's approval, so that "
      "no version of this repository can move an arm before that sentence "
      "exists.")

  meta = {"stage": a.stage, "joint": joint, "mode": "dry-run",
          "robot_model": "UNKNOWN (no arm audited)",
          "serial": "UNKNOWN", "firmware": "UNKNOWN",
          "controller_mode": "dry-run",
          "control_rate_hz": T.CONTROL_HZ,
          "joint_order": list(L.ARM_JOINTS),
          "units": table["units"],
          "limits_config_sha256": config_hash(table),
          "segments": [s.describe() for s in segments],
          "start_pose_rad": table["start_pose_rad"]}
  rec = SessionRecorder(a.out, session, meta)
  stop = StopMachine()
  try:
    end = run(a.stage, joint, backend, table, rec, stop, segments,
              T.CONTROL_HZ, session, dry)
  except KeyboardInterrupt:
    stop.stop(["operator interrupt"], operator=True)
    end = "operator"
  finally:
    m = rec.close(stop, end if "end" in dir() else "aborted")
  print(json.dumps({"session": session, "samples": m["n_samples"],
                    "end_reason": m["end_reason"],
                    "stopped": m["stop"]["stopped"],
                    "reasons": m["stop"]["reasons"][:4],
                    "raw": str(rec.raw_path)}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
