"""Plan in full simulation, then replay the audited motion on the real arm.

Unlike :mod:`hardware.deploy.hitl`, virtual contacts are allowed to shape the
trajectory before any hardware command is sent.  The real arm receives the
recorded simulated joint state at 50 Hz.  This is the useful experiment when
the question is actuator dynamics: perception, object motion and contact
planning come from simulation, while the physical servo follows the exact
same free-space joint path.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

from piper_push import robot as sim_robot

from . import config, proprio, robot
from .hitl import (
  TASK, Recorder, SafetyStop, _build_env, _feedback_fault, _grasp_height,
  _joint_ids, _refresh_observation, _sha256, _table_clearance, _target,
  _write_measured,
)


def _state_qg(env, ids) -> np.ndarray:
  ent = env.scene["robot"]
  arm_ids, grip_ids = ids
  q = ent.data.joint_pos[0, arm_ids].detach().cpu().numpy()
  g = float(ent.data.joint_pos[0, grip_ids[0]].detach().cpu())
  return np.r_[q, g].astype(np.float64)


def _object_state(env) -> np.ndarray:
  obj = env.scene["object"]
  return torch.cat(
    [obj.data.root_link_pose_w, obj.data.root_com_vel_w], dim=-1
  )[0].detach().cpu().numpy().astype(np.float32)


def _read_start(can: str) -> robot.ArmState:
  arm = robot.PiperArm(can)
  arm.connect()
  try:
    time.sleep(0.3)
    st = arm.read()
    why = _feedback_fault(st, 0.3)
    if why:
      raise RuntimeError("start feedback failed: " + why)
    return st
  finally:
    arm.disconnect()


@torch.no_grad()
def _plan(a, start: robot.ArmState, rig: config.Rig, recorder: Recorder):
  env = _build_env(
    TASK, a.device, a.seed, a.command_rate_scale, True,
    a.command_accel_limit, a.gripper_accel_limit,
    a.virtual_table_lift, a.wall_seconds + 5.0,
  )
  agent_cfg = load_rl_cfg(TASK)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    wrapped, asdict(agent_cfg), device=a.device
  )
  runner.load(
    str(a.checkpoint), load_cfg={"actor": True}, strict=True,
    map_location=a.device,
  )
  policy = runner.get_inference_policy(device=a.device)
  ids = _joint_ids(env)
  try:
    _write_measured(env, ids, start)
    env.action_manager.reset()
    policy.reset()
    obs = _refresh_observation(env, start, ids, True)
    pick = env.command_manager.get_term("pick")
    initial_obj_z = float(_object_state(env)[2])
    n = int(round(a.wall_seconds / env.step_dt))
    commands, sim_dq = [], []
    grasped_any = placed_any = False
    max_lift = 0.0
    min_gripper = float("inf")
    for step in range(n):
      action = policy(obs)
      if wrapped.clip_actions is not None:
        action = action.clamp(-wrapped.clip_actions, wrapped.clip_actions)
      if action.shape != (1, 7) or not bool(torch.isfinite(action).all()):
        raise RuntimeError(f"invalid policy action at plan step {step}")
      out = wrapped.step(action)
      obs, done = out[0], out[2]
      qg = _state_qg(env, ids)
      dq = (
        env.scene["robot"].data.joint_vel[0, ids[0]]
        .detach().cpu().numpy().astype(np.float64)
      )
      obj = _object_state(env)
      policy_target = _target(env)
      grasped = bool(pick.grasped[0])
      placed = bool(pick.placed[0])
      grasped_any |= grasped
      placed_any |= placed
      max_lift = max(max_lift, float(obj[2]) - initial_obj_z)
      min_gripper = min(min_gripper, float(qg[6]))
      commands.append(qg)
      sim_dq.append(dq)
      recorder.record({
        "step": step, "t_sim_s": (step + 1) * env.step_dt,
        "action": action[0].detach().cpu().numpy().tolist(),
        "policy_target_qg": policy_target.tolist(),
        "sim_qg": qg.tolist(), "sim_dq": dq.tolist(),
        "object_state": obj.tolist(), "grasped": grasped,
        "placed": placed,
      }, obs["camera"][0].detach().cpu().numpy(), obj)
      if bool(done[0]):
        raise RuntimeError(f"sim plan terminated at step {step}")
    return (
      np.asarray(commands), np.asarray(sim_dq),
      {
        "plan_steps": n, "plan_step_dt_s": float(env.step_dt),
        "plan_grasped": grasped_any, "plan_placed": placed_any,
        "plan_object_max_lift_m": max_lift,
        "plan_min_gripper_m": min_gripper,
      },
    )
  finally:
    wrapped.close()


def _audit(commands: np.ndarray, rig: config.Rig, a) -> dict:
  kin = proprio.Kinematics()
  normal = np.asarray(rig.table_normal_base, dtype=np.float64)
  min_clearance = float("inf")
  min_height = float("inf")
  closest = ""
  # Include intermediate configurations: endpoints alone can miss a curved
  # geometry-clearance minimum between two 50 Hz samples.
  previous = np.r_[a.start_q, a.start_gripper]
  for command in commands:
    for alpha in np.linspace(0.2, 1.0, 5):
      qg = previous + alpha * (command - previous)
      clearance, geom = _table_clearance(
        kin, qg, normal, rig.table_z
      )
      height = _grasp_height(kin, qg)
      if clearance < min_clearance:
        min_clearance, closest = clearance, geom
      min_height = min(min_height, height)
    previous = command

  dt = 1.0 / config.CONTROL_HZ
  q0 = np.asarray(a.start_q, dtype=np.float64)[None]
  dq_cmd = np.diff(np.vstack([q0, commands[:, :6]]), axis=0) / dt
  limits = np.array([
    sim_robot.JOINT_TRIP_RAD_S[f"joint{i}"] for i in range(1, 7)
  ])
  speed_fraction = np.max(np.abs(dq_cmd) / limits[None], axis=0)
  return {
    "plan_min_table_clearance_m": float(min_clearance),
    "plan_closest_geom": closest,
    "plan_min_grasp_height_m": float(min_height),
    "plan_max_command_speed_fraction_by_joint": speed_fraction.tolist(),
    "plan_min_command_gripper_m": float(commands[:, 6].min()),
    "plan_max_command_gripper_m": float(commands[:, 6].max()),
  }


def _execute(a, commands: np.ndarray, sim_dq: np.ndarray,
             rig: config.Rig, record_dir: pathlib.Path) -> dict:
  arm = robot.PiperArm(a.can)
  kin = proprio.Kinematics()
  normal = np.asarray(rig.table_normal_base, dtype=np.float64)
  real_q, real_dq, real_g = [], [], []
  stop_reason = "time_limit"
  out = (record_dir / "real.jsonl").open("w", buffering=1)
  arm.connect()
  try:
    time.sleep(0.3)
    start = arm.read()
    drift = float(np.max(np.abs(start.q - np.asarray(a.start_q))))
    if drift > np.deg2rad(2.0):
      raise SafetyStop(
        f"arm moved {np.degrees(drift):.2f} deg while planning"
      )
    preload = np.r_[start.q, start.gripper]
    arm.command(preload)
    arm.enable()
    arm.command(preload)
    begin = time.monotonic()
    overrun_streak = 0
    for step, command in enumerate(commands):
      target_clearance, target_geom = _table_clearance(
        kin, command, normal, rig.table_z
      )
      if target_clearance < a.min_table_clearance:
        raise SafetyStop(
          f"audited target {target_geom} clearance changed to "
          f"{target_clearance:.4f} m"
        )
      arm.command(command, 1.0 / config.CONTROL_HZ)
      deadline = begin + (step + 1) / config.CONTROL_HZ
      time.sleep(max(0.0, deadline - time.monotonic()))
      st = arm.read()
      why = _feedback_fault(st, a.max_joint_speed_fraction)
      if why:
        raise SafetyStop(why)
      qg = np.r_[st.q, st.gripper]
      clearance, geom = _table_clearance(kin, qg, normal, rig.table_z)
      if clearance < a.min_table_clearance:
        raise SafetyStop(
          f"measured {geom} clearance {clearance:.4f} m"
        )
      tracking = float(np.max(np.abs(st.q - command[:6])))
      if step >= 25 and tracking > np.deg2rad(a.max_tracking_error_deg):
        raise SafetyStop(
          f"tracking error {np.degrees(tracking):.1f} deg"
        )
      lateness = time.monotonic() - deadline
      overrun_streak = overrun_streak + 1 if lateness > 0.080 else 0
      if overrun_streak >= 3:
        raise SafetyStop("3 consecutive replay periods over 100 ms")
      real_q.append(st.q.copy())
      real_dq.append(st.dq.copy())
      real_g.append(float(st.gripper))
      out.write(json.dumps({
        "step": step, "t": time.time(),
        "command_qg": command.tolist(),
        "sim_dq": sim_dq[step].tolist(),
        "real_q": st.q.tolist(), "real_dq": st.dq.tolist(),
        "real_gripper": float(st.gripper),
        "tracking_error_rad": tracking,
        "measured_clearance_m": float(clearance),
        "measured_closest_geom": geom,
        "lateness_s": float(lateness),
      }, separators=(",", ":")) + "\n")
  except BaseException as exc:
    stop_reason = str(exc)
    arm.hold()
    if isinstance(exc, SafetyStop):
      print(f"REPLAY STOP: {exc}", flush=True)
    else:
      raise
  finally:
    try:
      arm.hold()
    finally:
      arm.close()
      out.close()

  q = np.asarray(real_q)
  dq = np.asarray(real_dq)
  g = np.asarray(real_g)
  n = len(q)
  if n:
    qe = q - commands[:n, :6]
    dqe = dq - sim_dq[:n]
    ge = g - commands[:n, 6]
    q_rmse = np.sqrt(np.mean(qe * qe, axis=0))
    dq_rmse = np.sqrt(np.mean(dqe * dqe, axis=0))
    g_rmse = float(np.sqrt(np.mean(ge * ge)))
  else:
    q_rmse = np.full(6, np.nan)
    dq_rmse = np.full(6, np.nan)
    g_rmse = float("nan")
  return {
    "real_stop_reason": stop_reason, "real_steps": n,
    "real_duration_s": n / config.CONTROL_HZ,
    "q_replay_rmse_deg": np.degrees(q_rmse).tolist(),
    "dq_replay_rmse_rad_s": dq_rmse.tolist(),
    "gripper_replay_rmse_mm": g_rmse * 1000.0,
  }


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", required=True, type=pathlib.Path)
  p.add_argument("--record", required=True, type=pathlib.Path)
  p.add_argument("--wall-seconds", type=float, default=60.0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--rig-file", type=pathlib.Path, default=pathlib.Path(
    __file__).with_name("rig_d455.json"))
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--command-rate-scale", type=float, default=0.15)
  p.add_argument("--command-accel-limit", type=float, default=0.3)
  p.add_argument("--gripper-accel-limit", type=float, default=0.03)
  p.add_argument("--virtual-table-lift", type=float, default=0.13)
  p.add_argument("--min-table-clearance", type=float, default=0.040)
  p.add_argument("--min-grasp-height", type=float, default=0.050)
  p.add_argument("--max-joint-speed-fraction", type=float, default=0.3)
  p.add_argument("--max-tracking-error-deg", type=float, default=20.0)
  p.add_argument("--plan-only", action="store_true")
  a = p.parse_args()
  a.checkpoint = a.checkpoint.resolve()
  a.record = a.record.resolve()
  a.rig_file = a.rig_file.resolve()
  if not a.checkpoint.exists():
    p.error(f"checkpoint not found: {a.checkpoint}")
  if a.wall_seconds <= 0:
    p.error("--wall-seconds must be positive")
  if not (0 < a.command_rate_scale <= 1):
    p.error("--command-rate-scale must be in (0, 1]")

  rig = config.Rig.load(a.rig_file)
  start = _read_start(a.can)
  a.start_q = start.q.copy()
  a.start_gripper = float(start.gripper)
  metadata = {
    "mode": "pure-sim-plan_then_real-50hz-replay",
    "checkpoint": str(a.checkpoint),
    "checkpoint_sha256": _sha256(a.checkpoint),
    "rig_file": str(a.rig_file),
    "wall_seconds": a.wall_seconds,
    "virtual_table_lift_m": a.virtual_table_lift,
    "start_q_rad": start.q.tolist(),
    "start_gripper_m": float(start.gripper),
  }
  recorder = Recorder(a.record, metadata, queue_size=256)
  plan_summary = {}
  try:
    commands, sim_dq, plan_summary = _plan(a, start, rig, recorder)
    audit = _audit(commands, rig, a)
    plan_summary.update(audit)
    recorder.close(plan_summary)
  except BaseException:
    recorder.close({"plan_failed": True, **plan_summary})
    raise

  failures = []
  if not plan_summary["plan_grasped"]:
    failures.append("simulation never established a grasp")
  if plan_summary["plan_object_max_lift_m"] < 0.010:
    failures.append("object never lifted 10 mm")
  if plan_summary["plan_min_table_clearance_m"] < a.min_table_clearance:
    failures.append(
      f"minimum physical clearance {plan_summary['plan_min_table_clearance_m']:.4f} m"
    )
  if plan_summary["plan_min_grasp_height_m"] < a.min_grasp_height:
    failures.append(
      f"minimum grasp height {plan_summary['plan_min_grasp_height_m']:.4f} m"
    )
  if max(plan_summary["plan_max_command_speed_fraction_by_joint"]) > 0.3:
    failures.append("planned command speed exceeds 0.3 of a joint trip limit")
  if plan_summary["plan_min_command_gripper_m"] < -1e-4:
    failures.append("planned gripper position is negative")
  if failures:
    print("PLAN REJECTED:", "; ".join(failures))
    return 2
  print(json.dumps(plan_summary, indent=2))
  if a.plan_only:
    return 0
  if not sys.stdin.isatty():
    raise SystemExit("refusing real replay without an interactive terminal")
  print("Plan passed: it includes a virtual grasp/lift and every real target "
        "passed the calibrated table sweep.")
  if input("clear the real workspace, hold the E-stop, then type 'replay': "
           ).strip() != "replay":
    return 1
  real_summary = _execute(a, commands, sim_dq, rig, a.record)
  summary_path = a.record / "summary.json"
  summary = json.loads(summary_path.read_text())
  summary.update(real_summary)
  summary_path.write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps(real_summary, indent=2))
  return 0 if real_summary["real_stop_reason"] == "time_limit" else 2


if __name__ == "__main__":
  sys.exit(main())
