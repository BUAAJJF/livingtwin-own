"""Audit whether the vision pick policy uses the table as a mechanical stop.

This is simulation-only.  It adds a contact sensor for every robot collision
geometry against the terrain and retains all ten 2 ms physics substeps in each
20 ms policy step, so a brief collision cannot disappear between observations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


TASK = "Mjlab-Pick-Place-PiperX-Vision"
TABLE_SENSOR = "robot_table_audit"


def first_or_none(values: torch.Tensor) -> list[int | None]:
  result = []
  for row in values.cpu().tolist():
    result.append(None if row < 0 else int(row))
  return result


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("checkpoint", type=Path)
  p.add_argument("--task", default=TASK)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=1800)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--rate-scale", type=float, default=1.0)
  p.add_argument("--arm-accel", type=float, default=None)
  p.add_argument("--gripper-accel", type=float, default=None)
  p.add_argument("--out", type=Path, default=None)
  a = p.parse_args()

  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = a.num_envs
  cfg.seed = a.seed
  cfg.actions["arm"].slew_scale = a.rate_scale
  cfg.actions["gripper"].slew_scale = a.rate_scale
  cfg.actions["arm"].accel_limit = a.arm_accel
  cfg.actions["gripper"].accel_limit = a.gripper_accel
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name=TABLE_SENSOR,
      primary=ContactMatch(
        mode="geom",
        pattern=(".*_collision", "[lr]f_pad"),
        entity="robot",
        exclude=("base_link_collision",),
      ),
      secondary=ContactMatch(mode="geom", pattern="terrain"),
      fields=("found", "force", "dist"),
      reduce="mindist",
      num_slots=2,
      history_length=cfg.decimation,
    ),
  )

  agent_cfg = load_rl_cfg(a.task)
  raw = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(
    env, asdict(agent_cfg), device=a.device
  )
  runner.load(
    str(a.checkpoint.resolve()), load_cfg={"actor": True}, strict=True,
    map_location=a.device,
  )
  policy = runner.get_inference_policy(device=a.device)
  policy.reset()

  sensor = raw.scene.sensors[TABLE_SENSOR]
  pad = raw.scene.sensors["pad_contact"]
  pick = raw.command_manager.get_term("pick")
  names = sensor.primary_names
  P = len(names)
  N = a.num_envs
  dev = torch.device(a.device)
  alive = torch.ones(N, dtype=torch.bool, device=dev)
  ever_table = torch.zeros(N, dtype=torch.bool, device=dev)
  ever_pad = torch.zeros(N, dtype=torch.bool, device=dev)
  ever_grasp = torch.zeros(N, dtype=torch.bool, device=dev)
  ever_place = torch.zeros(N, dtype=torch.bool, device=dev)
  ever_safety = torch.zeros(N, dtype=torch.bool, device=dev)
  first_table = torch.full((N,), -1, dtype=torch.long, device=dev)
  first_pad = torch.full((N,), -1, dtype=torch.long, device=dev)
  first_grasp = torch.full((N,), -1, dtype=torch.long, device=dev)
  first_place = torch.full((N,), -1, dtype=torch.long, device=dev)
  first_safety = torch.full((N,), -1, dtype=torch.long, device=dev)
  table_frames = torch.zeros(N, dtype=torch.long, device=dev)
  table_before_pad_frames = torch.zeros(N, dtype=torch.long, device=dev)
  table_before_grasp_frames = torch.zeros(N, dtype=torch.long, device=dev)
  geom_hit_frames = torch.zeros(P, dtype=torch.long, device=dev)
  geom_episode_hits = torch.zeros(N, P, dtype=torch.bool, device=dev)
  max_force = torch.zeros(P, device=dev)
  min_dist = torch.zeros(P, device=dev)
  min_dist[:] = float("inf")
  terminal = torch.full((N,), -1, dtype=torch.long, device=dev)

  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]
  with torch.inference_mode():
    for step in range(a.steps):
      out = env.step(policy(obs))
      obs, dones = out[0], out[2].bool()

      # [B, P*S, H, 3] -> [B, P, S, H].  History is essential:
      # a table collision may resolve within one 20 ms control period.
      fh = sensor.data.force_history
      dh = sensor.data.dist_history
      if fh is None or dh is None:
        raise RuntimeError("table sensor history was not allocated")
      slots = fh.shape[1] // P
      force = fh.norm(dim=-1).view(N, P, slots, fh.shape[2])
      dist = dh.view(N, P, slots, dh.shape[2])
      geom_hit = force.amax(dim=(2, 3)) > 1.0e-5
      table = geom_hit.any(dim=1) & alive
      pad_now = (pad.data.found > 0).view(N, -1).any(dim=1) & alive
      grasp_now = pick.grasped.bool() & alive
      place_now = pick.placed.bool() & alive
      safety_now = (
        raw.termination_manager.get_term("table_safety").bool() & alive
        if "table_safety" in raw.termination_manager.active_terms
        else torch.zeros_like(alive)
      )

      new_table = table & ~ever_table
      new_pad = pad_now & ~ever_pad
      new_grasp = grasp_now & ~ever_grasp
      new_place = place_now & ~ever_place
      new_safety = safety_now & ~ever_safety
      first_table[new_table] = step
      first_pad[new_pad] = step
      first_grasp[new_grasp] = step
      first_place[new_place] = step
      first_safety[new_safety] = step

      table_frames += table.long()
      table_before_pad_frames += (table & ~ever_pad & ~pad_now).long()
      table_before_grasp_frames += (table & ~ever_grasp & ~grasp_now).long()
      geom_hit_frames += (geom_hit & alive[:, None]).sum(dim=0)
      geom_episode_hits |= geom_hit & alive[:, None]
      max_force = torch.maximum(max_force, force.amax(dim=(0, 2, 3)))
      # Only contact slots have meaningful distance.  Positive/negative sign
      # depends on MuJoCo; retaining the minimum reports deepest proximity.
      valid_dist = torch.where(
        force > 1.0e-5, dist, torch.full_like(dist, float("inf"))
      )
      min_dist = torch.minimum(min_dist, valid_dist.amin(dim=(0, 2, 3)))

      ever_table |= table
      ever_pad |= pad_now
      ever_grasp |= grasp_now
      ever_place |= place_now
      ever_safety |= safety_now
      newly_done = dones & alive
      terminal[newly_done] = step
      alive &= ~dones
      if not bool(alive.any()):
        break

  completed_steps = int(step + 1)
  table_eps = int(ever_table.sum())
  grasp_eps = int(ever_grasp.sum())
  # Did table contact occur strictly before the first object-pad contact?
  table_first = (
    (first_table >= 0) & ((first_pad < 0) | (first_table < first_pad))
  )
  table_before_grasp = (
    (first_table >= 0) & ((first_grasp < 0) | (first_table < first_grasp))
  )
  grasp_with_table = ever_grasp & ever_table
  grasp_without_table = ever_grasp & ~ever_table
  grasp_before_table = (
    (first_grasp >= 0) & ((first_table < 0) | (first_grasp < first_table))
  )
  grasp_at_or_after_table = ever_grasp & ~grasp_before_table
  result = {
    "checkpoint": str(a.checkpoint.resolve()),
    "task": a.task,
    "seed": a.seed,
    "num_envs": N,
    "steps_requested": a.steps,
    "steps_run": completed_steps,
    "sim_seconds_per_episode_max": completed_steps * raw.step_dt,
    "rate_scale": a.rate_scale,
    "arm_accel": a.arm_accel,
    "gripper_accel": a.gripper_accel,
    "episodes_terminated": int((terminal >= 0).sum()),
    "episodes_with_pad_object_contact": int(ever_pad.sum()),
    "episodes_with_grasp": grasp_eps,
    "episodes_with_place": int(ever_place.sum()),
    "episodes_with_table_safety_termination": int(ever_safety.sum()),
    "episodes_with_robot_table_contact": table_eps,
    "robot_table_contact_episode_fraction": table_eps / N,
    "table_contact_before_first_pad_episode_count": int(table_first.sum()),
    "table_contact_before_first_grasp_episode_count": int(table_before_grasp.sum()),
    "grasp_episodes_with_any_table_contact": int(grasp_with_table.sum()),
    "grasp_episodes_without_table_contact": int(grasp_without_table.sum()),
    "grasp_episodes_before_first_table_contact": int(grasp_before_table.sum()),
    "grasp_episodes_at_or_after_first_table_contact": int(
      grasp_at_or_after_table.sum()
    ),
    "table_contact_frames_total": int(table_frames.sum()),
    "table_contact_before_pad_frames_total": int(table_before_pad_frames.sum()),
    "table_contact_before_grasp_frames_total": int(table_before_grasp_frames.sum()),
    "per_geom": {
      name: {
        "episode_hits": int(geom_episode_hits[:, i].sum()),
        "control_frames": int(geom_hit_frames[i]),
        "max_force_n": float(max_force[i]),
        "min_contact_dist_m": (
          float(min_dist[i]) if bool(torch.isfinite(min_dist[i])) else None
        ),
      }
      for i, name in enumerate(names)
    },
    "first_table_step": first_or_none(first_table),
    "first_pad_step": first_or_none(first_pad),
    "first_grasp_step": first_or_none(first_grasp),
    "first_place_step": first_or_none(first_place),
    "first_table_safety_step": first_or_none(first_safety),
    "terminal_step": first_or_none(terminal),
  }
  text = json.dumps(result, indent=2)
  print(text)
  if a.out is not None:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text + "\n")
  env.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
