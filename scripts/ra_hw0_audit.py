"""RA-HW-0 Stage 2: the read-only hardware capability audit.

    python scripts/ra_hw0_audit.py                 # offline, SDK facts only
    python scripts/ra_hw0_audit.py --hardware      # opens CAN, still read-only

**What this touches.**  With ``--hardware`` it calls ``ConnectPort()``, the
SDK's ``Get*`` accessors, and ``DisconnectPort()``.  ``ConnectPort`` transmits
three *enquiry* frames -- maximum joint angle/speed, maximum acceleration, and
firmware version -- because that is what populates the limits this phase needs
from the arm itself rather than from a guess.  It transmits nothing else.

**What this never touches**, and a test asserts the source code does not name:
``EnableArm``, ``DisableArm``, ``JointCtrl``, ``GripperCtrl``, ``MotionCtrl_1``,
``MotionCtrl_2``, ``EmergencyStop``, ``JointConfig``, ``MotorAngleLimitMaxSpdSet``,
``ArmParamEnquiryAndConfig`` with a *set* argument, or anything in the SDK
whose name begins with ``Set``.

Anything the arm does not answer is written as ``UNKNOWN``.  The audit's own
verdict is never ``motion authorised``; that sentence is a human's.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNKNOWN = "UNKNOWN"


def _run(cmd: list[str]) -> str:
  try:
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=10).stdout.strip()
  except Exception as exc:
    return f"{UNKNOWN}: {type(exc).__name__}"


def host_facts() -> dict:
  ifaces = _run(["ip", "-br", "link"])
  can = [ln.split()[0] for ln in ifaces.splitlines()
         if ln and ln.split()[0].startswith(("can", "vcan", "slcan"))]
  return {
    "hostname": platform.node(),
    "kernel": platform.release(),
    "python": sys.version.split()[0],
    "can_interfaces_present": can,
    "can_kernel_modules": [ln.split()[0] for ln in _run(["lsmod"]).splitlines()
                           if ln.split() and ln.split()[0] in
                           ("can", "can_raw", "can_dev", "gs_usb", "peak_usb",
                            "slcan", "vcan", "mcp251x")],
    "usb_devices": _run(["lsusb"]).splitlines() if shutil.which("lsusb") else UNKNOWN,
    "ip_link": ifaces.splitlines(),
  }


def sdk_facts() -> dict:
  out: dict = {}
  try:
    import piper_sdk
    from piper_sdk.piper_param import C_PiperParamManager
  except Exception as exc:
    return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
  try:
    out["version"] = md.version("piper_sdk")
  except Exception:
    out["version"] = UNKNOWN
  out["installed"] = True
  out["path"] = str(Path(piper_sdk.__file__).parent)
  p = C_PiperParamManager().GetPiperParamOrigin()
  out["vendor_joint_limit_rad"] = p["joint_limit"]
  out["vendor_gripper_range_m"] = p["gripper_range"]
  out["units"] = {
    "joint_feedback": "0.001 deg", "joint_command": "0.001 deg",
    "motor_speed": "0.001 rad/s", "motor_current": "0.001 A",
    "gripper_position": "0.001 mm", "gripper_effort": "0.001 N.m",
    "driver_voltage": "0.1 V", "temperatures": "1 degC",
    "device_angle_limit": "0.1 deg", "device_max_accel": "0.001 rad/s^2",
    "source": "piper_sdk docstrings, read out of the installed package",
  }
  out["telemetry_channels"] = {
    "position": "GetArmJointMsgs -> joint_1..6, 0.001 deg",
    "velocity": "GetArmHighSpdInfoMsgs -> motor_speed, 0.001 rad/s "
                "(NOTE: hardware/deploy/robot.py differentiates position "
                "instead, on the grounds that the unit was undocumented; in "
                "SDK 0.6.2 it is documented, so RA-HW-0 records BOTH and lets "
                "H0 decide)",
    "current": "GetArmHighSpdInfoMsgs -> current, 0.001 A; "
               "GetArmLowSpdInfoMsgs -> bus_current",
    "effort": "GetArmHighSpdInfoMsgs -> effort, 'fixed coefficient'",
    "temperature": "GetArmLowSpdInfoMsgs -> foc_temp, motor_temp, 1 degC",
    "voltage": "GetArmLowSpdInfoMsgs -> vol, 0.1 V",
    "servo_error": "NOT published as such; computed host-side as "
                   "command_after_safety_filter - q",
    "fault_flags": "GetArmLowSpdInfoMsgs -> foc_status bits "
                   "(voltage_too_low, motor_overheating, driver_overcurrent, "
                   "driver_overheating, collision_status, driver_error_status, "
                   "driver_enable_status, stall_status); "
                   "GetArmStatus -> err_code bits (per-joint angle limit and "
                   "per-joint communication status) and arm_status enum",
    "timestamps": "every accessor returns time_stamp and Hz.  time_stamp is "
                  "python-can's frame timestamp, i.e. the HOST kernel's "
                  "receive time.  No device-side clock is exposed: "
                  "device_timestamp_s is UNKNOWN",
  }
  out["control_modes"] = {
    "MotionCtrl_2.ctrl_mode": {"0x00": "standby", "0x01": "CAN command",
                               "0x03": "ethernet", "0x04": "wifi",
                               "0x07": "offline trajectory"},
    "MotionCtrl_2.move_mode": {"0x00": "MOVE P", "0x01": "MOVE J",
                               "0x02": "MOVE L", "0x03": "MOVE C",
                               "0x04": "MOVE M (>= V1.5-2)",
                               "0x05": "MOVE CPV (>= V1.8-1)"},
    "MotionCtrl_2.is_mit_mode": {"0x00": "position-velocity", "0xAD": "MIT",
                                 "0xFF": "invalid"},
    "repo_uses": "MotionCtrl_2(0x01, 0x01, 100, 0x00) then JointCtrl -- CAN "
                 "command mode, MOVE J, 100% speed rate, position-velocity",
    "torque_control_available": "yes, is_mit_mode=0xAD, NOT used by this repo "
                                "and NOT used in RA-HW-0",
  }
  out["motion_capable_entry_points"] = {
    "EnableArm": "energises the drives",
    "DisableArm": "removes holding torque immediately; observed on this rig as "
                  "the arm falling when calibgui exited",
    "JointCtrl": "position command, no range check in the message class",
    "GripperCtrl": "gripper position and torque",
    "MotionCtrl_1.emergency_stop": "0x01 stop, 0x02 RESUME -- resume is a "
                                   "motion-enabling command",
    "MotionCtrl_1.grag_teach_ctrl": "0x01 enters drag-teach; the arm becomes "
                                    "back-driveable and can fall",
    "MotionCtrl_1.track_ctrl": "0x02 continues a stored trajectory -- can "
                               "start motion with no new position command",
    "MotionCtrl_2.ctrl_mode=0x07": "offline trajectory mode",
    "ConnectPort(piper_init=True)": "sends three ENQUIRY frames only "
                                    "(max angle/speed, max acceleration, "
                                    "firmware version); no motion",
    "hardware/deploy/robot.py PiperArm.close(disable=False)":
      "calls hold(), which calls command() -- i.e. CLOSING THE CLIENT SENDS A "
      "JOINT COMMAND.  RA-HW-0 uses disconnect() instead, never close()",
  }
  return out


FORBIDDEN = ("EnableArm", "DisableArm", "JointCtrl", "GripperCtrl",
             "MotionCtrl_1", "MotionCtrl_2", "EmergencyStop", "JointConfig",
             "MotorAngleLimitMaxSpdSet", "MotorMaxAccLimitSet")


def device_facts(can: str, dwell_s: float) -> dict:
  """Read-only interrogation of an attached arm.  Raises if none is there."""
  from piper_sdk import C_PiperInterface_V2

  iface = C_PiperInterface_V2(can)
  out: dict = {"can_interface": can, "connected": False}
  iface.ConnectPort()
  out["connected"] = True
  try:
    time.sleep(min(dwell_s, 2.0))            # let the read thread fill buffers
    def safe(fn, *a):
      try:
        return fn(*a)
      except Exception as exc:
        return f"{UNKNOWN}: {type(exc).__name__}: {exc}"

    out["firmware"] = str(safe(iface.GetPiperFirmwareVersion))
    out["sdk_interface_version"] = str(safe(iface.GetCurrentInterfaceVersion))
    out["protocol_version"] = str(safe(iface.GetCurrentProtocolVersion))
    out["can_fps"] = safe(iface.GetCanFps)
    out["can_name"] = safe(iface.GetCanName)
    st = safe(iface.GetArmStatus)
    out["arm_status_repr"] = str(st)
    out["enable_status"] = str(safe(iface.GetArmEnableStatus))
    out["motor_states"] = str(safe(iface.GetMotorStates))
    out["driver_states"] = str(safe(iface.GetDriverStates))
    out["angle_limit_max_spd"] = str(safe(iface.GetAllMotorAngleLimitMaxSpd))
    out["max_acc_limit"] = str(safe(iface.GetAllMotorMaxAccLimit))
    out["joint_msgs"] = str(safe(iface.GetArmJointMsgs))
    out["high_spd"] = str(safe(iface.GetArmHighSpdInfoMsgs))
    out["low_spd"] = str(safe(iface.GetArmLowSpdInfoMsgs))
    out["gripper_msgs"] = str(safe(iface.GetArmGripperMsgs))
    out["crash_protection"] = str(safe(iface.GetCrashProtectionLevelFeedback))
    # Telemetry rate and jitter, measured rather than assumed.
    stamps = []
    t_end = time.time() + dwell_s
    last = None
    while time.time() < t_end:
      ts = iface.GetArmJointMsgs().time_stamp
      if ts != last:
        stamps.append(ts)
        last = ts
      time.sleep(0.001)
    if len(stamps) > 3:
      gaps = [b - a for a, b in zip(stamps, stamps[1:])]
      gaps.sort()
      out["telemetry"] = {
        "n_frames": len(stamps), "window_s": dwell_s,
        "rate_hz": (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-9),
        "gap_median_s": gaps[len(gaps) // 2],
        "gap_p99_s": gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))],
        "gap_max_s": gaps[-1],
      }
    else:
      out["telemetry"] = UNKNOWN
  finally:
    try:
      iface.DisconnectPort()
    except Exception:
      pass
    out["disconnected"] = True
  return out


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--hardware", action="store_true",
                  help="open the CAN interface.  Still read-only: this script "
                       "cannot enable, command or e-stop the arm.")
  ap.add_argument("--can", default="can0")
  ap.add_argument("--dwell", type=float, default=5.0,
                  help="seconds of telemetry to time")
  ap.add_argument("--out", default="results/ra_hw0/audit.json")
  a = ap.parse_args()

  report = {"phase": "RA-HW-0", "stage": "2 hardware capability audit",
            "generated_host_wall_s": time.time(),
            "mode": "hardware" if a.hardware else "offline (no CAN opened)",
            "host": host_facts(), "sdk": sdk_facts()}

  if a.hardware:
    ifaces = report["host"]["can_interfaces_present"]
    if a.can not in ifaces:
      report["device"] = {
        "status": "NOT_FOUND",
        "detail": f"{a.can} is not present; interfaces are {ifaces or 'none'}",
        "next_step": "plug the USB-CAN adapter in and run the SDK's "
                     "can_activate.sh (needs root).  RA-HW-0 does not bring "
                     "up a network interface and does not use sudo.",
      }
    else:
      try:
        report["device"] = device_facts(a.can, a.dwell)
        report["device"]["status"] = "READ"
      except Exception as exc:
        report["device"] = {"status": "FAILED",
                            "error": f"{type(exc).__name__}: {exc}"}
  else:
    report["device"] = {"status": "NOT_ATTEMPTED",
                        "detail": "run with --hardware once an arm is attached"}

  dev = report["device"].get("status") == "READ"
  unknown = []
  if not dev:
    unknown += ["robot_model", "serial_number", "firmware_version",
                "device_joint_angle_limits", "device_max_joint_speed",
                "device_max_joint_acceleration", "telemetry_rate_and_jitter",
                "still_arm_current_baseline", "temperature_baseline",
                "bus_voltage", "watchdog_behaviour_on_comms_loss",
                "estop_verified", "gripper_configured_jaw_gap",
                "power_on_and_enable_motion_behaviour"]
  unknown += ["driver_and_motor_temperature_maximum",
              "joint_current_maximum",
              "undervoltage_threshold",
              "joint_positive_direction_convention_on_the_physical_arm",
              "serial_number_readback_path"]
  report["unknown"] = sorted(set(unknown))
  report["motion_authorised"] = False
  report["gate_H_A"] = {
    "hardware_identity_confirmed": dev,
    "estop_and_watchdog_verified": False,
    "per_joint_limit_table_complete": False,
    "H0_static_telemetry_passed": False,
    "sim_envelope_check_passed": None,
    "dry_run_tests_passed": None,
    "human_approval": False,
    "verdict": "BLOCKED",
  }
  out = Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2, default=str))
  print(json.dumps({"mode": report["mode"],
                    "can_interfaces": report["host"]["can_interfaces_present"],
                    "sdk": report["sdk"].get("version"),
                    "device": report["device"].get("status"),
                    "unknown_count": len(report["unknown"]),
                    "gate_H_A": report["gate_H_A"]["verdict"]}, indent=2))
  print(f"wrote {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
