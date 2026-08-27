"""Append-only session logging, and the stop machine that ends a session.

Two files per session.  ``<session>.raw.jsonl`` is written once per sample and
never rewritten, never smoothed and never re-ordered; ``<session>.meta.json``
carries the provenance, the thresholds in force and the reason the session
ended.  Derived quantities -- differentiated velocity, filtered signals, time
alignment -- belong in a third file written by ``scripts/ra_hw0_validate.py``,
because a raw log that has been improved is a raw log nobody can check.

The stop machine is deliberately one-way.  Once it has stopped, nothing in
this module can start it again: no automatic fault clearing, no retry, no
"continue to the next trajectory".  Resuming is a human running the tool
again, which is the only thing that makes the sentence "a stop was
acknowledged" mean anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

REQUIRED_SAMPLE_FIELDS = (
  "host_monotonic_s", "host_wall_s", "device_timestamp_s",
  "session_id", "trial_id", "trajectory_id", "sample_index",
  "controller_mode", "control_rate_hz",
  "command_requested_rad", "command_after_safety_filter_rad",
  "q_rad", "qdot_rad_s", "servo_error_rad",
  "joint_current_a", "joint_effort", "motor_speed_rad_s",
  "driver_temp_c", "motor_temp_c", "bus_voltage_v",
  "fault_flags", "watchdog_state", "gripper_state",
  "operator_event", "telemetry_gap_s", "msg_hz",
)
"""Every sample carries all of these.  A field the arm cannot supply is
``None`` and stays in the record: a log whose columns depend on what happened
cannot be compared with another one."""


def _git(*args: str) -> str:
  try:
    return subprocess.run(("git", *args), capture_output=True, text=True,
                          timeout=10,
                          cwd=Path(__file__).resolve().parents[2]).stdout.strip()
  except Exception:
    return ""


def config_hash(obj) -> str:
  return hashlib.sha256(
    json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class StopMachine:
  """One-way.  Records why, when, and who."""

  reasons: list[str] = field(default_factory=list)
  stopped_at_monotonic: float | None = None
  operator_initiated: bool = False

  @property
  def stopped(self) -> bool:
    return self.stopped_at_monotonic is not None

  def stop(self, reasons, *, operator: bool = False) -> None:
    if self.stopped:
      self.reasons.extend(r for r in reasons if r not in self.reasons)
      return
    self.reasons = list(reasons)
    self.operator_initiated = bool(operator)
    self.stopped_at_monotonic = time.monotonic()

  def to_json(self) -> dict:
    return {"stopped": self.stopped, "reasons": self.reasons,
            "operator_initiated": self.operator_initiated,
            "stopped_at_monotonic_s": self.stopped_at_monotonic,
            "auto_recovery": False,
            "auto_recovery_note": "RA-HW-0 never clears a fault, never "
                                  "retries and never continues to the next "
                                  "trajectory; a stop ends the session"}


class SessionRecorder:
  """Opens two files, appends to one of them, and closes both exactly once."""

  def __init__(self, out_dir: str | Path, session_id: str, meta: dict):
    self.dir = Path(out_dir)
    self.dir.mkdir(parents=True, exist_ok=True)
    self.session_id = session_id
    self.raw_path = self.dir / f"{session_id}.raw.jsonl"
    self.meta_path = self.dir / f"{session_id}.meta.json"
    if self.raw_path.exists():
      raise FileExistsError(
        f"{self.raw_path} exists.  RA-HW-0 logs are append-only and a session "
        "id is used once; pick a new one rather than adding to a record whose "
        "provenance would then describe two runs.")
    self.meta = {
      "schema_version": SCHEMA_VERSION,
      "session_id": session_id,
      "opened_host_wall_s": time.time(),
      "opened_host_monotonic_s": time.monotonic(),
      "git_commit": _git("rev-parse", "HEAD"),
      "git_dirty": bool(_git("status", "--porcelain")),
      "pid": os.getpid(),
      **meta,
    }
    self._fh = open(self.raw_path, "a", buffering=1)
    self.n = 0
    self.closed = False

  def append(self, sample: dict) -> None:
    if self.closed:
      raise RuntimeError("recorder is closed; a stopped session does not "
                         "acquire more samples")
    missing = [k for k in REQUIRED_SAMPLE_FIELDS if k not in sample]
    if missing:
      raise KeyError(f"sample is missing required fields: {missing}")
    self._fh.write(json.dumps(sample, default=str) + "\n")
    self.n += 1

  def close(self, stop: StopMachine, end_reason: str) -> dict:
    if self.closed:
      return self.meta
    self._fh.flush()
    os.fsync(self._fh.fileno())
    self._fh.close()
    self.closed = True
    self.meta.update({
      "closed_host_wall_s": time.time(),
      "n_samples": self.n,
      "end_reason": end_reason,
      "stop": stop.to_json(),
      "raw_sha256": hashlib.sha256(self.raw_path.read_bytes()).hexdigest(),
      "raw_bytes": self.raw_path.stat().st_size,
    })
    self.meta_path.write_text(json.dumps(self.meta, indent=2, default=str))
    return self.meta


def blank_sample(**kw) -> dict:
  """Every required field, defaulted to ``None``, then overridden.

  Used by the dry-run backend and by the hardware one, so a mock log and a
  real log have the same columns in the same order and the validator does not
  have to know which it is reading.
  """
  s = {k: None for k in REQUIRED_SAMPLE_FIELDS}
  s.update(kw)
  return s
