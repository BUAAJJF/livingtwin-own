"""RA-HW-0: the tests that stand between this repository and a moving arm.

Three kinds.  Static checks that no code path can reach a motion-capable SDK
call; behavioural checks that the stop machine, the watchdog and the
append-only log do what the checklist promises; and end-to-end dry runs of the
whole stack, because a safety argument that has never executed is a comment.
"""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hardware.ra_hw0 import limits as L          # noqa: E402
from hardware.ra_hw0 import trajectories as T    # noqa: E402
from hardware.ra_hw0.record import (SessionRecorder, StopMachine,  # noqa: E402
                                    REQUIRED_SAMPLE_FIELDS, blank_sample)

PY_FILES = [ROOT / "hardware" / "ra_hw0" / f for f in
            ("__init__.py", "limits.py", "record.py", "trajectories.py")] + [
           ROOT / "scripts" / f for f in
           ("ra_hw0_audit.py", "ra_hw0_collect.py", "ra_hw0_validate.py",
            "ra_hw0_replay.py", "ra_hw0_limits.py")]

MOTION_CALLS = ("EnableArm", "DisableArm", "JointCtrl", "GripperCtrl",
                "MotionCtrl_1", "MotionCtrl_2", "EmergencyStop",
                "JointConfig", "MotorAngleLimitMaxSpdSet",
                "MotorMaxAccLimitSet", "JointMitCtrl", "EndPoseCtrl")


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
def test_no_ra_hw0_file_calls_a_motion_capable_sdk_method(path):
  """The whole phase, checked at the syntax tree rather than by grep.

  The audit's docstring *names* these methods precisely because it is
  documenting that it does not call them, so a grep would flag its own
  warning.  This looks at call sites.
  """
  tree = ast.parse(path.read_text())
  called = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      f = node.func
      if isinstance(f, ast.Attribute):
        called.add(f.attr)
      elif isinstance(f, ast.Name):
        called.add(f.id)
  bad = sorted(called & set(MOTION_CALLS))
  assert not bad, f"{path.name} calls {bad}"


def test_the_only_sdk_calls_are_reads_or_the_connect_pair():
  """`ra_hw0_audit.py` is the one file that opens CAN.  Everything it asks the
  interface for is a Get, a ConnectPort or a DisconnectPort."""
  tree = ast.parse((ROOT / "scripts" / "ra_hw0_audit.py").read_text())
  iface_calls = set()
  for node in ast.walk(tree):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "iface"):
      iface_calls.add(node.func.attr)
  assert iface_calls
  for name in iface_calls:
    assert name.startswith("Get") or name in ("ConnectPort", "DisconnectPort"), \
      f"audit calls iface.{name}"


def test_the_hardware_backend_is_not_implemented_in_this_phase():
  """The strongest guarantee available: there is no code to move an arm.

  A gate that can be bypassed by a flag is a gate; a gate that cannot be
  bypassed because the thing it guards does not exist is a fact.
  """
  src = (ROOT / "scripts" / "ra_hw0_collect.py").read_text()
  assert "the hardware backend is not implemented in this phase" in src
  assert "PiperArm" not in src
  assert "piper_sdk" not in src


# -- the limit table --------------------------------------------------------

def test_the_absolute_bounds_are_the_intersection_not_either_source():
  import sys as _s
  _s.path.insert(0, str(ROOT / "src"))
  from piper_push import robot as sim
  for j in L.ARM_JOINTS:
    lo, hi, _ = L.intersect_absolute(j, sim.SAFE_TARGET_CLIP[j])
    vlo, vhi = L.VENDOR_JOINT_LIMIT_RAD[j]
    slo, shi = sim.SAFE_TARGET_CLIP[j]
    assert lo >= vlo - 1e-9 and lo >= slo - 1e-9
    assert hi <= vhi + 1e-9 and hi <= shi + 1e-9


def test_the_joint5_discrepancy_is_pinned():
  """The finding the checklist asks a reader to read twice.

  If a later commit widens the vendor table or narrows the simulator's clip so
  that these agree, this test says so rather than letting the note go stale.
  """
  import sys as _s
  _s.path.insert(0, str(ROOT / "src"))
  from piper_push import robot as sim
  slo, shi = sim.SAFE_TARGET_CLIP["joint5"]
  vlo, vhi = L.VENDOR_JOINT_LIMIT_RAD["joint5"]
  assert shi - vhi == pytest.approx(0.0890, abs=1e-4)
  assert vlo - slo == pytest.approx(0.0890, abs=1e-4)
  lo, hi, _ = L.intersect_absolute("joint5", (slo, shi))
  assert (lo, hi) == (vlo, vhi)


def test_an_unknown_row_blocks_motion_by_itself():
  table = L.build({j: L.VENDOR_JOINT_LIMIT_RAD[j] for j in L.ARM_JOINTS})
  assert table["motion_authorised"] is False
  assert table["unknowns_blocking_motion"]
  assert L.blocks_motion(table)
  # and filling the human field alone is not enough
  t2 = copy.deepcopy(table)
  t2["motion_authorised"] = True
  assert L.blocks_motion(t2)


def test_the_committed_config_blocks_motion_today():
  table = L.load(ROOT / "configs" / "ra_hw0_safety_limits.json")
  reasons = L.blocks_motion(table)
  assert reasons, "the checked-in limits table must not authorise motion"
  assert table["gripper"]["commanded_in_this_phase"] is False


def test_clipping_refuses_when_the_envelope_is_unknown():
  table = L.build({j: L.VENDOR_JOINT_LIMIT_RAD[j] for j in L.ARM_JOINTS})
  table["joints"][0]["envelope_min_rad"]["value"] = None
  with pytest.raises(ValueError):
    L.clip_to_envelope([0.0] * 6, table)


def complete_table():
  """A hypothetical table with every UNKNOWN filled.

  Used only to exercise the stages that a real UNKNOWN blocks.  It is built
  here, in a test, and never written to disk: a file like this next to the
  real one is an accident waiting to be pointed at an arm.
  """
  import sys as _s
  _s.path.insert(0, str(ROOT / "src"))
  from piper_push import robot as sim
  t = L.build(sim.SAFE_TARGET_CLIP,
              {j: {"max_speed_rad_s": 3.0, "max_accel_rad_s2": 10.0}
               for j in L.ARM_JOINTS})
  for k, v in (("driver_temp_stop_c", 60.0), ("motor_temp_stop_c", 60.0),
               ("bus_voltage_min_v", 20.0), ("joint_current_stop_a", 2.0)):
    t["environment"][k].update(value=v, source="hypothetical")
  t["unknowns_blocking_motion"] = []
  t["hypothetical"] = True
  return t


# -- stop conditions --------------------------------------------------------

def test_every_stop_condition_fires_and_all_of_them_are_reported():
  t = complete_table()
  q = [t["start_pose_rad"][j] for j in L.ARM_JOINTS]
  assert L.check_state(q, [0.0] * 6, q, t) == []
  bad_q = list(q)
  bad_q[0] = 5.0
  reasons = L.check_state(bad_q, [9.0] + [0.0] * 5, q, t,
                          qdd=[99.0] + [0.0] * 5,
                          temps={"foc_0": 99.0},
                          fault_bits={"motor_1": {"stall_status": True}},
                          gap_s=1.0)
  joined = " | ".join(reasons)
  for expect in ("outside envelope", "over", "tracking error",
                 "acceleration", "telemetry gap", "stall_status"):
    assert expect in joined, expect
  # all of them, not the first one
  assert len(reasons) >= 6


def test_an_enabled_driver_is_not_a_fault():
  t = complete_table()
  q = [t["start_pose_rad"][j] for j in L.ARM_JOINTS]
  r = L.check_state(q, [0.0] * 6, q, t,
                    fault_bits={"motor_1": {"driver_enable_status": True}})
  assert r == []


def test_an_unknown_threshold_stops_a_commanding_stage_and_not_an_observing_one():
  t = L.load(ROOT / "configs" / "ra_hw0_safety_limits.json")
  q = [t["start_pose_rad"][j] for j in L.ARM_JOINTS]
  strict = L.check_state(q, [0.0] * 6, q, t, temps={"foc_0": 30.0})
  lenient = L.check_state(q, [0.0] * 6, q, t, temps={"foc_0": 30.0},
                          require_known=False)
  assert any("UNKNOWN" in r for r in strict)
  assert lenient == []


def test_non_finite_telemetry_is_a_stop():
  t = complete_table()
  q = [t["start_pose_rad"][j] for j in L.ARM_JOINTS]
  bad = list(q)
  bad[2] = float("nan")
  assert any("non-finite" in r for r in L.check_state(bad, [0.0] * 6, q, t))


# -- the stop machine and the recorder --------------------------------------

def test_the_stop_machine_is_one_way():
  s = StopMachine()
  assert not s.stopped
  s.stop(["over speed"])
  first = s.stopped_at_monotonic
  s.stop(["something else"])
  assert s.stopped_at_monotonic == first          # not restarted
  assert "over speed" in s.reasons and "something else" in s.reasons
  assert s.to_json()["auto_recovery"] is False


def test_an_operator_stop_is_recorded_as_one():
  s = StopMachine()
  s.stop(["operator interrupt"], operator=True)
  assert s.to_json()["operator_initiated"] is True


def test_a_session_id_is_used_once(tmp_path):
  r = SessionRecorder(tmp_path, "s1", {})
  r.close(StopMachine(), "completed")
  with pytest.raises(FileExistsError):
    SessionRecorder(tmp_path, "s1", {})


def test_a_sample_missing_a_required_field_is_refused(tmp_path):
  r = SessionRecorder(tmp_path, "s2", {})
  with pytest.raises(KeyError):
    r.append({"host_monotonic_s": 0.0})
  r.append(blank_sample(host_monotonic_s=0.0))
  r.close(StopMachine(), "completed")


def test_a_closed_recorder_does_not_acquire_more_samples(tmp_path):
  r = SessionRecorder(tmp_path, "s3", {})
  r.close(StopMachine(), "completed")
  with pytest.raises(RuntimeError):
    r.append(blank_sample())


def test_the_meta_carries_the_hash_of_the_raw_log(tmp_path):
  import hashlib
  r = SessionRecorder(tmp_path, "s4", {})
  r.append(blank_sample(host_monotonic_s=1.0))
  m = r.close(StopMachine(), "completed")
  assert m["raw_sha256"] == hashlib.sha256(r.raw_path.read_bytes()).hexdigest()
  assert m["n_samples"] == 1


def test_every_required_field_is_present_and_defaults_to_none():
  s = blank_sample()
  assert set(s) == set(REQUIRED_SAMPLE_FIELDS)
  assert all(v is None for v in s.values())


# -- trajectories -----------------------------------------------------------

def test_h0_transmits_nothing():
  seg = T.h0_hold(30.0)
  assert seg.q == []
  assert "NO command transmitted" in seg.describe()


def test_no_h1_segment_exceeds_the_five_second_ceiling():
  for seg in T.h1_plan("joint5"):
    assert seg.duration_s <= L.H1_MAX_SEGMENT_S + 1e-9, seg.name


def test_every_h1_segment_starts_and_ends_at_the_start_pose():
  start = [L.H1_START_POSE_RAD[j] for j in L.ARM_JOINTS]
  for seg in T.h1_plan("joint5"):
    assert seg.q[0] == pytest.approx(start, abs=1e-9), seg.name
    assert seg.q[-1] == pytest.approx(start, abs=1e-9), seg.name


def test_the_reversal_actually_crosses_the_start():
  seg = T.h1_reversal("joint5")
  i = L.ARM_JOINTS.index("joint5")
  vals = [row[i] - L.H1_START_POSE_RAD["joint5"] for row in seg.q]
  assert max(vals) > 0.04 and min(vals) < -0.04
  # and the one-sided triangle does not, which is what makes them different
  tri = T.h1_triangle("joint5")
  tvals = [row[i] - L.H1_START_POSE_RAD["joint5"] for row in tri.q]
  assert min(tvals) >= -1e-9


def test_a_single_period_step_is_refused():
  """The design error the checker caught: 0.02 rad in one 20 ms period is
  1.0 rad/s, three times H1's own stop threshold."""
  t = complete_table()
  fast = T._build("fake-step", "H1", "joint5",
                  lambda x: 0.0 if x < 0.5 else 0.02, 1.0)
  assert fast.peak_speed_rad_s == pytest.approx(1.0, rel=1e-6)
  assert any("commanded speed" in r for r in T.check(fast, t))
  # while the ramped one this phase actually uses passes
  assert T.check(T.h1_step("joint5"), t) == []


def test_a_command_outside_the_envelope_is_refused():
  t = complete_table()
  seg = T.h1_triangle("joint5", amplitude_rad=0.9)
  assert any("outside envelope" in r for r in T.check(seg, t))


def test_the_joint_order_is_least_energy_first():
  assert T.H1_JOINT_ORDER[0] == "joint5"
  assert T.H1_JOINT_ORDER[-1] == "joint1"
  assert set(T.H1_JOINT_ORDER) == set(L.ARM_JOINTS)


# -- the collector, end to end ---------------------------------------------

def _collect(*args):
  return subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "ra_hw0_collect.py"), *args],
    capture_output=True, text=True, cwd=ROOT, timeout=300)


def test_the_default_is_a_dry_run_and_h0_completes(tmp_path):
  r = _collect("--stage", "H0", "--duration", "1",
               "--out", str(tmp_path), "--session", "t_h0")
  assert r.returncode == 0, r.stdout + r.stderr
  raw = tmp_path / "t_h0.raw.jsonl"
  rows = [json.loads(x) for x in raw.read_text().splitlines()]
  assert 40 <= len(rows) <= 60
  assert all(x["command_requested_rad"] is None for x in rows), \
    "H0 must not record a command, because it must not send one"
  meta = json.loads((tmp_path / "t_h0.meta.json").read_text())
  assert meta["mode"] == "dry-run"
  assert meta["stop"]["auto_recovery"] is False


def test_h1_refuses_while_a_threshold_is_unknown(tmp_path):
  r = _collect("--stage", "H1", "--joint", "joint5",
               "--out", str(tmp_path), "--session", "t_h1")
  meta = json.loads((tmp_path / "t_h1.meta.json").read_text())
  assert meta["stop"]["stopped"] is True
  assert any("UNKNOWN" in x for x in meta["stop"]["reasons"])


def test_hardware_is_refused_and_nothing_is_opened(tmp_path):
  r = _collect("--stage", "H0", "--hardware", "--i-have-approval",
               "--out", str(tmp_path))
  assert r.returncode == 2
  assert "HARDWARE REFUSED" in r.stdout
  assert "Nothing was opened, enabled or commanded" in r.stdout
  assert not list(tmp_path.iterdir())


def test_hardware_is_refused_even_with_every_flag_but_the_audit(tmp_path):
  r = _collect("--stage", "H0", "--hardware", "--i-have-approval",
               "--audit", str(tmp_path / "nope.json"), "--out", str(tmp_path))
  assert r.returncode == 2
  assert "has not READ an arm" in r.stdout


def test_the_watchdog_stops_on_a_telemetry_gap():
  """The dry-run backend drops frames; a gap over the threshold must stop."""
  t = complete_table()
  assert any("telemetry gap" in r for r in
             L.check_state([t["start_pose_rad"][j] for j in L.ARM_JOINTS],
                           [0.0] * 6, None, t, gap_s=0.5))
