"""Hardware-in-the-loop playback with simulated vision and objects.

The target object, bin, contacts and camera live in MuJoCo.  The policy's arm
state does not: every control period the PiPER feedback is written into the
simulation and the virtual arm is swept along the measured trajectory while
the virtual object advances.  The policy action is sent to the real arm.

This isolates the arm command/actuator dynamics from perception and object
appearance.  It does *not* reproduce physical contact load on the real arm;
the object and pad contacts are virtual by construction.

Every step is also run once in a teacher-forced shadow simulator, anchored at
the same measured state.  Its one-step prediction is logged beside the next
real feedback, which makes the dynamics gap a number rather than a video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer

from piper_push import robot as sim_robot

from . import config, proprio, robot
from .run import (_feedback_fault, _grasp_height, _start_pose_fault,
                  _table_clearance)


TASK = "Mjlab-Pick-Place-PiperX-Vision"
SHADOW_TASK = "Mjlab-Pick-Place-PiperX"
ARM_NAMES = tuple(f"joint{i}" for i in range(1, 7))
GRIP_NAMES = ("gripper_joint1", "gripper_joint2")


class SafetyStop(RuntimeError):
  """A controlled stop: hold, log, and leave the drives enabled."""


def _sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _git(*args: str) -> str:
  return subprocess.run(("git", *args), capture_output=True, text=True,
                        timeout=10).stdout.strip()


class Recorder:
  """Crash-readable JSONL control log plus asynchronous simulated frames."""

  def __init__(self, path: pathlib.Path, metadata: dict, queue_size: int = 128):
    if path.exists() and any(path.iterdir()):
      raise RuntimeError(f"refusing to overwrite non-empty HITL log {path}")
    path.mkdir(parents=True, exist_ok=True)
    self.path = path
    self.frames = path / "frames"
    self.frames.mkdir()
    (path / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    self._control = (path / "control.jsonl").open("w", buffering=1)
    self._q: queue.Queue = queue.Queue(maxsize=queue_size)
    self._error: BaseException | None = None
    self._closed = False
    self._thread = threading.Thread(target=self._worker, daemon=True,
                                    name="hitl-log-writer")
    self._thread.start()

  def record(self, rec: dict, camera: np.ndarray, obj: np.ndarray) -> None:
    if self._error is not None:
      raise RuntimeError(f"HITL log writer failed: {self._error}")
    self._control.write(json.dumps(rec, separators=(",", ":")) + "\n")
    item = (int(rec["step"]), np.asarray(camera, dtype=np.float16).copy(),
            np.asarray(obj, dtype=np.float32).copy())
    try:
      self._q.put_nowait(item)
    except queue.Full as e:
      raise RuntimeError("HITL log queue full; refusing unlogged motion") from e

  def event(self, event: str, **fields) -> None:
    rec = {"t": time.time(), "event": event, **fields}
    self._control.write(json.dumps(rec, separators=(",", ":")) + "\n")

  def _worker(self) -> None:
    while True:
      item = self._q.get()
      try:
        if item is None:
          return
        step, camera, obj = item
        try:
          np.savez_compressed(self.frames / f"{step:06d}.npz",
                              camera=camera, object_state=obj)
        except BaseException as e:
          self._error = e
      finally:
        self._q.task_done()

  def close(self, summary: dict) -> None:
    if self._closed:
      return
    self._closed = True
    self._q.put(None)
    self._q.join()
    self._thread.join(timeout=5)
    self._control.close()
    (self.path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if self._error is not None:
      raise RuntimeError(f"HITL log writer failed: {self._error}")


def _joint_ids(env):
  ent = env.scene["robot"]
  arm, _ = ent.find_joints(ARM_NAMES, preserve_order=True)
  grip, _ = ent.find_joints(GRIP_NAMES, preserve_order=True)
  return (torch.as_tensor(arm, device=env.device),
          torch.as_tensor(grip, device=env.device))


def _write_measured(env, ids, st) -> None:
  ent = env.scene["robot"]
  arm_ids, grip_ids = ids
  q = torch.as_tensor(st.q, device=env.device, dtype=torch.float32)[None]
  dq = torch.as_tensor(st.dq, device=env.device, dtype=torch.float32)[None]
  g = torch.tensor([[st.gripper, -st.gripper]], device=env.device,
                   dtype=torch.float32)
  gd = torch.tensor([[st.gripper_vel, -st.gripper_vel]], device=env.device,
                    dtype=torch.float32)
  ent.write_joint_state_to_sim(q, dq, joint_ids=arm_ids)
  ent.write_joint_state_to_sim(g, gd, joint_ids=grip_ids)


def _write_interpolated(env, ids, a, b, alpha: float,
                        interval_s: float | None = None) -> None:
  dt = float(env.step_dt if interval_s is None else interval_s)
  q = np.asarray(a.q) + alpha * (np.asarray(b.q) - np.asarray(a.q))
  g = float(a.gripper + alpha * (b.gripper - a.gripper))
  dq = (np.asarray(b.q) - np.asarray(a.q)) / dt
  gv = (b.gripper - a.gripper) / dt
  class S:
    pass
  st = S()
  st.q, st.dq, st.gripper, st.gripper_vel = q, dq, g, gv
  _write_measured(env, ids, st)


def _refresh_observation(env, st, ids, update_history: bool) -> TensorDict:
  _write_measured(env, ids, st)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.scene.update(dt=0.0)
  env.sim.sense()
  obs = env.observation_manager.compute(update_history=update_history)
  env.obs_buf = obs
  return TensorDict(obs, batch_size=[env.num_envs])


def _finish_hybrid_step(env, action: torch.Tensor, st0, st1, ids,
                        interval_s: float):
  """Advance virtual objects along the measured, not simulated, arm path."""
  env.extras["log"] = {}
  # process_action has already run so its exact command can be sent first.
  substeps = max(1, int(round(float(interval_s) / env.physics_dt)))
  for k in range(substeps):
    env._sim_step_counter += 1
    env.action_manager.apply_action()
    _write_interpolated(env, ids, st0, st1, k / substeps, interval_s)
    env.scene.write_data_to_sim()
    env.sim.step()
    env.scene.update(dt=env.physics_dt)
    env.metrics_manager.compute_substep()

  # The camera and proprioception must see the feedback at the end of the
  # physical period, not the simulator's integration residue from substep 10.
  _write_measured(env, ids, st1)
  env.episode_length_buf += 1
  env.common_step_counter += 1
  env.reset_buf = env.termination_manager.compute()
  env.reset_terminated = env.termination_manager.terminated
  env.reset_time_outs = env.termination_manager.time_outs
  env.reward_buf = env.reward_manager.compute(dt=interval_s)
  env.metrics_manager.compute()
  if "step" in env.event_manager.available_modes:
    env.event_manager.apply(mode="step", dt=interval_s)
  if "interval" in env.event_manager.available_modes:
    env.event_manager.apply(mode="interval", dt=interval_s)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.command_manager.compute(dt=interval_s)
  env.sim.sense()
  env.obs_buf = env.observation_manager.compute(update_history=True)
  env.recorder_manager.record_post_step()
  return TensorDict(env.obs_buf, batch_size=[env.num_envs]), (
    env.reset_terminated | env.reset_time_outs)


def _target(env) -> np.ndarray:
  arm = env.action_manager.get_term("arm")._previous_target[0]
  grip = env.action_manager.get_term("gripper")._previous_target[0]
  return np.concatenate([arm.detach().cpu().numpy(),
                         grip.detach().cpu().numpy()]).astype(np.float64)


class Controller:
  def __init__(self, hil, wrapped, policy, arm, rig, recorder,
               args, real_motion: bool):
    self.hil, self.wrapped = hil, wrapped
    self.policy, self.arm, self.rig, self.recorder = policy, arm, rig, recorder
    self.args, self.real_motion = args, real_motion
    self.hil_ids = _joint_ids(hil)
    self.hil_obj = hil.scene["object"]
    self.kin = proprio.Kinematics()
    self.normal = np.asarray(rig.table_normal_base, dtype=np.float64)
    self.step_count = 0
    self.last_action = np.zeros(7, np.float32)
    self.stop_reason = "time_limit"
    self._closed = False
    self._tracking: list[float] = []
    self._timings: list[float] = []
    self._compute_times: list[float] = []
    self._feedback_intervals: list[float] = []
    self._overrun_streak = 0
    self.st = arm.read()

    why = _start_pose_fault(self.st)
    if why:
      raise RuntimeError(why)
    self._guard_actual(self.st)
    _write_measured(hil, self.hil_ids, self.st)
    hil.action_manager.reset()
    self.policy.reset()
    self.obs = _refresh_observation(hil, self.st, self.hil_ids, True)
    self.recorder.event(
      "connected_enabled" if real_motion else "dry_run_started",
      q_rad=self.st.q.tolist(), gripper_m=float(self.st.gripper),
      table_clearance_m=self._clearance(self.st)[0])

  def _clearance(self, st):
    q = np.concatenate([st.q, [st.gripper]])
    return _table_clearance(self.kin, q, self.normal, self.rig.table_z)

  def _guard_actual(self, st) -> tuple[float, str]:
    why = _feedback_fault(st, self.args.max_joint_speed_fraction)
    if why:
      raise SafetyStop(why)
    clearance, geom = self._clearance(st)
    if clearance < self.args.min_table_clearance:
      raise SafetyStop(
        f"measured {geom} table clearance {clearance:.4f} m below "
        f"{self.args.min_table_clearance:.4f} m")
    return clearance, geom

  def _guard_target(self, target) -> tuple[float, str, float]:
    clearance, geom = _table_clearance(
      self.kin, target, self.normal, self.rig.table_z)
    height = _grasp_height(self.kin, target)
    if clearance < self.args.min_table_clearance:
      raise SafetyStop(
        f"target {geom} table clearance {clearance:.4f} m below "
        f"{self.args.min_table_clearance:.4f} m")
    if height < self.args.min_grasp_height:
      raise SafetyStop(f"target grasp height {height:.4f} m below "
                       f"{self.args.min_grasp_height:.4f} m")
    return clearance, geom, height

  @torch.no_grad()
  def step(self) -> None:
    t0 = time.monotonic()
    st0 = self.st
    actual_clearance, actual_geom = self._guard_actual(st0)
    action = self.policy(self.obs)
    if self.wrapped.clip_actions is not None:
      action = action.clamp(-self.wrapped.clip_actions, self.wrapped.clip_actions)
    if action.shape != (1, 7) or not bool(torch.isfinite(action).all()):
      raise SafetyStop(f"invalid policy action shape/value: {tuple(action.shape)}")

    # The offline shadow replay consumes this exact action and measured anchor
    # after the arm is holding.  Keeping it out of this loop preserves 50 Hz.
    self.hil.action_manager.process_action(action)
    target = _target(self.hil)
    target_clearance, target_geom, target_height = self._guard_target(target)

    obj0 = torch.cat([self.hil_obj.data.root_link_pose_w,
                      self.hil_obj.data.root_com_vel_w], dim=-1).clone()
    command_wall_before = time.time()
    command_mono = time.monotonic()
    self.arm.command(target, self.hil.step_dt)
    command_wall = time.time()
    feedback_deadline = command_mono + self.hil.step_dt
    if not self.args.no_realtime:
      time.sleep(max(0.0, feedback_deadline - time.monotonic()))
    feedback_read_mono = time.monotonic()
    st1 = self.arm.read()
    feedback_dt = float(st1.stamp - st0.stamp)
    if not np.isfinite(feedback_dt) or feedback_dt <= 0.0:
      raise SafetyStop(f"invalid feedback interval {feedback_dt!r}")
    next_clearance, next_geom = self._guard_actual(st1)
    tracking = float(np.max(np.abs(st1.q - target[:6])))
    if (self.step_count >= int(0.5 / self.hil.step_dt)
        and tracking > np.deg2rad(self.args.max_tracking_error_deg)):
      raise SafetyStop(
        f"tracking error {np.degrees(tracking):.1f} deg exceeds "
        f"{self.args.max_tracking_error_deg:.1f} deg")

    self.obs, done = _finish_hybrid_step(
      self.hil, action, st0, st1, self.hil_ids, feedback_dt)
    self.st = st1
    self.last_action = action[0].detach().cpu().numpy().copy()
    camera = self.obs["camera"][0].detach().cpu().numpy()
    obj1 = torch.cat([self.hil_obj.data.root_link_pose_w,
                      self.hil_obj.data.root_com_vel_w], dim=-1)[0] \
      .detach().cpu().numpy()
    elapsed = time.monotonic() - t0
    compute_s = ((command_mono - t0)
                 + (time.monotonic() - feedback_read_mono))
    self._tracking.append(tracking)
    self._timings.append(elapsed)
    self._compute_times.append(compute_s)
    self._feedback_intervals.append(feedback_dt)
    # This loop deliberately waits for real feedback and then advances the
    # virtual world over that measured interval.  Its post-feedback compute
    # therefore need not fit inside the policy's nominal 20 ms step.  What is
    # safety-relevant is a sustained long wall-clock command period: while it
    # runs late the hardware holds the preceding rate-limited target.
    self._overrun_streak = (
      self._overrun_streak + 1
      if elapsed > self.args.max_control_period else 0)

    self.recorder.record({
      "step": self.step_count, "t": time.time(),
      "action": self.last_action.tolist(), "target_qg": target.tolist(),
      "start_real_q": st0.q.tolist(), "start_real_dq": st0.dq.tolist(),
      "start_real_gripper": float(st0.gripper),
      "start_real_gripper_vel": float(st0.gripper_vel),
      "real_q": st1.q.tolist(), "real_dq": st1.dq.tolist(),
      "real_gripper": float(st1.gripper),
      "virtual_object_start": obj0[0].detach().cpu().numpy().tolist(),
      "tracking_error_rad": tracking,
      "measured_clearance_m": float(next_clearance),
      "measured_closest_geom": next_geom,
      "target_clearance_m": float(target_clearance),
      "target_closest_geom": target_geom,
      "target_grasp_height_m": float(target_height),
      "loop_s": elapsed,
      "compute_s": compute_s,
      "feedback_interval_s": feedback_dt,
      "pre_command_hold_s": max(0.0, command_wall_before - st0.stamp),
      "command_call_s": max(0.0, command_wall - command_wall_before),
      "new_command_active_s": max(0.0, st1.stamp - command_wall),
      "virtual_terminated": bool(done[0]),
      "start_clearance_m": float(actual_clearance),
      "start_closest_geom": actual_geom,
    }, camera, obj1)
    self.step_count += 1

    if bool(done[0]):
      raise SafetyStop("virtual episode terminated; reset is forbidden in HITL")
    if self._overrun_streak >= self.args.max_overrun_streak:
      raise SafetyStop(
        f"{self._overrun_streak} consecutive control periods above "
        f"{self.args.max_control_period * 1000:.0f} ms")

  def hold(self, reason: str) -> None:
    self.stop_reason = reason
    try:
      self.arm.hold()
    finally:
      self.recorder.event("hold", reason=reason)

  def summary(self) -> dict:
    timing = np.asarray(self._timings)
    compute = np.asarray(self._compute_times)
    feedback_dt = np.asarray(self._feedback_intervals)
    return {
      "stop_reason": self.stop_reason, "steps": self.step_count,
      "tracking_p95_deg": (float(np.degrees(np.percentile(self._tracking, 95)))
                           if self._tracking else None),
      "loop_median_ms": (float(np.median(timing) * 1000) if len(timing) else None),
      "loop_p95_ms": (float(np.percentile(timing, 95) * 1000)
                      if len(timing) else None),
      "compute_median_ms": (float(np.median(compute) * 1000)
                            if len(compute) else None),
      "compute_p95_ms": (float(np.percentile(compute, 95) * 1000)
                         if len(compute) else None),
      "feedback_interval_median_ms": (float(np.median(feedback_dt) * 1000)
                                      if len(feedback_dt) else None),
    }

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      self.arm.hold()
    finally:
      self.arm.close()
      self.recorder.close(self.summary())


class HitlViewer(NativeMujocoViewer):
  """Native viewer whose only stepping authority is the HITL controller."""

  def __init__(self, env, controller):
    # BaseViewer requires a policy but _execute_step below owns inference.
    super().__init__(env, lambda obs: obs, enable_perturbations=False)
    self.controller = controller

  def _execute_step(self) -> bool:
    try:
      self.controller.step()
      self._step_count += 1
      self._stats_steps += 1
      return True
    except SafetyStop as e:
      print(f"\nHITL STOP: {e}", flush=True)
      self.controller.hold(str(e))
      self._interrupted = True
      return False
    except BaseException as e:
      self.controller.hold("exception: " + repr(e))
      raise

  def reset_environment(self) -> None:
    self.controller.hold("viewer reset requested")
    self._interrupted = True

  def increase_speed(self) -> None:
    print("HITL: speed-up is disabled")

  def decrease_speed(self) -> None:
    print("HITL: slow-motion is disabled; control remains 50 Hz")


def _build_env(task: str, device: str, seed: int, rate_scale: float,
               vision: bool, arm_accel: float = 0.3,
               gripper_accel: float = 0.03,
               virtual_table_lift: float = 0.0,
               episode_length_s: float | None = None):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.seed = seed
  cfg.auto_reset = False
  if episode_length_s is not None:
    cfg.episode_length_s = float(episode_length_s)
  # HITL has no real object to stop the fingers at the virtual tabletop.  A
  # literal replay can therefore drive the empty real gripper through the
  # physical table even though MuJoCo reports a perfectly ordinary grasp.
  # Lowering only the simulated robot base makes the virtual work surface
  # higher in the robot/camera frame.  Joint targets remain unchanged between
  # sim and real (the dynamics experiment is still 1:1), while the same angles
  # execute ``virtual_table_lift`` above the real tabletop.
  if virtual_table_lift:
    robot_cfg = cfg.scene.entities["robot"]
    pos = list(robot_cfg.init_state.pos)
    pos[2] -= float(virtual_table_lift)
    robot_cfg.init_state.pos = tuple(pos)
  cfg.actions["arm"].slew_scale = rate_scale
  cfg.actions["gripper"].slew_scale = rate_scale
  cfg.actions["arm"].accel_limit = arm_accel
  cfg.actions["gripper"].accel_limit = gripper_accel
  if not vision:
    # The state task still has gripper contact sensors; only a rendered camera
    # would silently double the real-time workload of the shadow environment.
    camera_sensors = [s for s in (cfg.scene.sensors or ())
                      if "camera" in type(s).__name__.lower()]
    assert not camera_sensors
  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)


@torch.no_grad()
def _offline_shadow(record_dir: pathlib.Path, args) -> dict:
  """Teacher-force one nominal step from every measured state after motion."""
  records = []
  with (record_dir / "control.jsonl").open() as f:
    for line in f:
      rec = json.loads(line)
      if "step" in rec:
        records.append(rec)
  if not records:
    return {"shadow_steps": 0}

  env = _build_env(SHADOW_TASK, args.device, args.seed,
                   args.command_rate_scale, False,
                   args.command_accel_limit, args.gripper_accel_limit,
                   args.virtual_table_lift)
  env.reset()
  ids = _joint_ids(env)
  obj = env.scene["object"]
  first = records[0]
  st = robot.ArmState(
    q=np.asarray(first["start_real_q"], dtype=np.float64),
    dq=np.asarray(first["start_real_dq"], dtype=np.float64),
    gripper=float(first["start_real_gripper"]),
    gripper_vel=float(first["start_real_gripper_vel"]),
    gripper_effort=0.0, stamp=0.0)
  _write_measured(env, ids, st)
  env.action_manager.reset()

  q_errors, dq_errors, motions = [], [], []
  replay_durations = []
  out_path = record_dir / "shadow.jsonl"
  completed = 0

  def advance(seconds: float) -> tuple[int, float]:
    substeps = max(0, int(round(float(seconds) / env.physics_dt)))
    for _ in range(substeps):
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)
    return substeps, substeps * env.physics_dt

  try:
    with out_path.open("w", buffering=1) as out:
      for rec in records:
        st.q = np.asarray(rec["start_real_q"], dtype=np.float64)
        st.dq = np.asarray(rec["start_real_dq"], dtype=np.float64)
        st.gripper = float(rec["start_real_gripper"])
        st.gripper_vel = float(rec["start_real_gripper_vel"])
        _write_measured(env, ids, st)
        obj_state = torch.as_tensor(
          rec["virtual_object_start"], device=env.device,
          dtype=torch.float32)[None]
        obj.write_root_state_to_sim(obj_state)
        action = torch.as_tensor(
          rec["action"], device=env.device, dtype=torch.float32)[None]
        # The measured state interval straddles the command call: first the
        # servo continues holding the previous target, then the new target is
        # installed.  Replaying that split is what makes this a like-for-like
        # dynamics comparison rather than 20 ms of sim against 30-60 ms real.
        if "new_command_active_s" in rec:
          old_s = (float(rec.get("pre_command_hold_s", 0.0))
                   + float(rec.get("command_call_s", 0.0)))
          new_s = float(rec["new_command_active_s"])
          old_n, old_actual = advance(old_s)
          env.action_manager.process_action(action)
          new_n, new_actual = advance(new_s)
          replay_s = old_actual + new_actual
        else:
          # Legacy HITL logs did not capture the command boundary.  Keep their
          # old 20 ms interpretation explicit rather than pretending it is
          # timestamp-correct.
          old_n, old_actual = 0, 0.0
          env.action_manager.process_action(action)
          new_n, new_actual = advance(env.step_dt)
          replay_s = new_actual
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.sim.sense()
        q_pred = env.scene["robot"].data.joint_pos[0, ids[0]] \
          .detach().cpu().numpy().copy()
        dq_pred = env.scene["robot"].data.joint_vel[0, ids[0]] \
          .detach().cpu().numpy().copy()
        q_real = np.asarray(rec["real_q"], dtype=np.float64)
        dq_real = np.asarray(rec["real_dq"], dtype=np.float64)
        q0 = np.asarray(rec["start_real_q"], dtype=np.float64)
        qe, dqe = q_real - q_pred, dq_real - dq_pred
        q_errors.append(qe)
        dq_errors.append(dqe)
        motions.append(q_real - q0)
        replay_durations.append(replay_s)
        out.write(json.dumps({
          "step": int(rec["step"]), "sim_pred_q": q_pred.tolist(),
          "sim_pred_dq": dq_pred.tolist(), "q_error": qe.tolist(),
          "dq_error": dqe.tolist(), "old_target_substeps": old_n,
          "new_target_substeps": new_n, "replayed_s": replay_s,
          "measured_feedback_s": rec.get("feedback_interval_s"),
        }, separators=(",", ":")) + "\n")
        completed += 1
  finally:
    env.close()

  q = np.asarray(q_errors)
  dq = np.asarray(dq_errors)
  motion = np.asarray(motions)
  q_rmse = np.sqrt(np.mean(q * q, axis=0))
  motion_rms = np.sqrt(np.mean(motion * motion, axis=0))
  return {
    "shadow_steps": completed,
    "shadow_timing": ("timestamp_split" if records and
                      "new_command_active_s" in records[0] else "legacy_20ms"),
    "replayed_interval_median_ms": float(np.median(replay_durations) * 1000),
    "q_one_step_rmse_rad": q_rmse.tolist(),
    "q_one_step_rmse_deg": np.degrees(q_rmse).tolist(),
    "q_one_step_motion_rms_rad": motion_rms.tolist(),
    "q_one_step_nrms": (q_rmse / np.maximum(motion_rms, 1e-9)).tolist(),
    "dq_one_step_rmse_rad_s": np.sqrt(np.mean(dq * dq, axis=0)).tolist(),
  }


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--task", default=TASK)
  p.add_argument("--seconds", type=float, default=10.0)
  p.add_argument("--wall-seconds", type=float, default=None,
                 help="stop after this much real wall time (headless only)")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--rig-file", default=str(
    pathlib.Path(__file__).with_name("rig_d455.json")))
  p.add_argument("--record", required=True)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--dry-run", action="store_true")
  p.add_argument("--no-viewer", action="store_true")
  p.add_argument("--no-realtime", action="store_true",
                 help="dry-run tests only")
  p.add_argument("--command-rate-scale", type=float, default=0.15)
  p.add_argument("--command-accel-limit", type=float, default=0.3)
  p.add_argument("--gripper-accel-limit", type=float, default=0.03)
  p.add_argument("--max-joint-speed-fraction", type=float, default=0.3)
  p.add_argument("--min-table-clearance", type=float, default=0.040)
  p.add_argument("--min-grasp-height", type=float, default=0.050)
  p.add_argument("--virtual-table-lift", type=float, default=0.0,
                 help="raise virtual work surface relative to the real base")
  p.add_argument("--max-tracking-error-deg", type=float, default=20.0)
  p.add_argument("--max-control-period", type=float, default=0.080,
                 help="hold after sustained wall-clock periods above this")
  p.add_argument("--max-overrun-streak", type=int, default=10)
  a = p.parse_args()

  if a.no_realtime and not a.dry_run:
    p.error("--no-realtime is only allowed with --dry-run")
  if a.wall_seconds is not None and a.wall_seconds <= 0:
    p.error("--wall-seconds must be positive")
  if a.wall_seconds is not None and not a.no_viewer:
    p.error("--wall-seconds requires --no-viewer")
  if a.virtual_table_lift < 0:
    p.error("--virtual-table-lift must be non-negative")
  if not (0 < a.command_rate_scale <= 1):
    p.error("--command-rate-scale must be in (0, 1]")
  if a.max_control_period <= 0:
    p.error("--max-control-period must be positive")
  if a.max_overrun_streak < 1:
    p.error("--max-overrun-streak must be at least 1")
  checkpoint = pathlib.Path(a.checkpoint).resolve()
  rig_path = pathlib.Path(a.rig_file).resolve()
  if not checkpoint.exists():
    p.error(f"checkpoint not found: {checkpoint}")
  if not rig_path.exists():
    p.error(f"rig file not found: {rig_path}")
  rig = config.Rig.load(rig_path)
  if rig.table_normal_base is None:
    p.error("rig has no calibrated table normal")

  real_motion = not a.dry_run
  if real_motion:
    print("HITL REAL ARM MOTION: camera, target and contacts are virtual; "
          "the real arm must have an empty workspace.")
    print("Reset/speed-up are disabled. Exit holds position and does not power off.")
    if not sys.stdin.isatty():
      raise SystemExit("refusing HITL motion without an interactive terminal")
    if input("clear the real workspace, hold the E-stop, then type 'hitl': ").strip() != "hitl":
      return 1

  metadata = {
    "mode": "sim-camera-object_real-arm", "task": a.task,
    "shadow_task": SHADOW_TASK, "checkpoint": str(checkpoint),
    "checkpoint_sha256": _sha256(checkpoint), "rig_file": str(rig_path),
    "args": vars(a), "git_commit": _git("rev-parse", "HEAD"),
    "git_dirty": bool(_git("status", "--porcelain")),
    "limitations": [
      "real arm has no physical object load; virtual contacts act only on the virtual object",
      "pad contacts are virtual; joint position, velocity and servo error are real",
    ],
  }
  recorder = Recorder(pathlib.Path(a.record), metadata)
  hil = wrapped = arm = controller = None
  offline = None
  try:
    hil = _build_env(a.task, a.device, a.seed, a.command_rate_scale, True,
                     a.command_accel_limit, a.gripper_accel_limit,
                     a.virtual_table_lift)
    agent_cfg = load_rl_cfg(a.task)
    wrapped = RslRlVecEnvWrapper(hil, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(
      wrapped, asdict(agent_cfg), device=a.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)

    spec = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())
    arm = robot.DryRunArm(spec) if a.dry_run else robot.PiperArm(a.can)
    arm.connect()
    if isinstance(arm, robot.PiperArm):
      time.sleep(0.3)
      start = arm.read()
      why = _start_pose_fault(start)
      if why:
        arm.disconnect()
        raise RuntimeError(why)
      why = _feedback_fault(start, a.max_joint_speed_fraction)
      kin = proprio.Kinematics()
      start7 = np.concatenate([start.q, [start.gripper]])
      clearance, closest = _table_clearance(
        kin, start7, rig.table_normal_base, rig.table_z)
      if why or clearance < a.min_table_clearance:
        arm.disconnect()
        detail = why or (f"measured {closest} table clearance "
                         f"{clearance:.4f} m below "
                         f"{a.min_table_clearance:.4f} m")
        raise RuntimeError("pre-enable safety check failed: " + detail)
      preload = np.concatenate([start.q, [start.gripper]])
      arm.command(preload)
      arm.enable()
      arm.command(preload)

    controller = Controller(hil, wrapped, policy, arm, rig,
                            recorder, a, real_motion)
    stopping = False
    def _stop(*_):
      nonlocal stopping
      stopping = True
      if controller is not None:
        controller.hold("operator_interrupt")
    signal.signal(signal.SIGINT, _stop)
    if a.no_viewer:
      if a.wall_seconds is not None:
        deadline = time.monotonic() + a.wall_seconds
        while not stopping and time.monotonic() < deadline:
          controller.step()
      else:
        n = max(1, int(round(a.seconds / hil.step_dt)))
        for _ in range(n):
          if stopping:
            break
          controller.step()
    else:
      n = max(1, int(round(a.seconds / hil.step_dt)))
      HitlViewer(wrapped, controller).run(num_steps=n, catch_sigint=False)
    if controller.stop_reason == "time_limit":
      controller.hold("time_limit")
    return 0
  except SafetyStop as e:
    if controller is not None:
      controller.hold(str(e))
    print(f"HITL STOP: {e}")
    return 2
  finally:
    if controller is not None:
      controller.close()
      print(json.dumps(controller.summary(), indent=2))
    else:
      if arm is not None:
        arm.close()
      recorder.close({"stop_reason": "startup_failure", "steps": 0})
    if wrapped is not None:
      wrapped.close()
    elif hil is not None:
      hil.close()
    if controller is not None and controller.step_count:
      offline = _offline_shadow(pathlib.Path(a.record), a)
      summary_path = pathlib.Path(a.record) / "summary.json"
      summary = json.loads(summary_path.read_text())
      summary.update(offline)
      summary_path.write_text(json.dumps(summary, indent=2) + "\n")
      print("offline shadow:")
      print(json.dumps(offline, indent=2))


if __name__ == "__main__":
  sys.exit(main())
