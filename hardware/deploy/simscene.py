"""Rebuild a measured real scene in simulation and run the policy on it.

The rig answers "did it work"; it does not answer "why".  When a run fails on
the table the useful question is whether the same scene fails in the simulator
the policy was trained in -- because if it does, nothing about the camera, the
calibration or the arm is implicated, and the fault is in the scene itself.

So this pins the one thing the deployment measures and the simulator normally
randomises: where the object is and how big it is.  Everything else -- vision,
contacts, actuators -- stays exactly as it was in training.

    python -m hardware.deploy.simscene --policy <dir> \
        --object-xy -0.169 0.386 --object-size 0.065 0.065 0.082 --shape cylinder

``--sweep`` runs the same position against a list of sizes, which is the form
the question usually takes: is this object out of distribution, or is this
*place* out of reach?
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types
from dataclasses import asdict

import numpy as np
import torch

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push import objects as sim_objects
from piper_push import shapes

TASK = "Mjlab-Pick-Place-PiperX-Vision-Robust"
CLASS_INDEX = {name: i for i, name in enumerate(sim_objects.SHAPE_CLASSES)}



# Measured on the arm 2026-08-31 (logs/ident/*).  kp from the step's torque
# against its position error, kd from the step's 10-90% rise, friction from
# constant-velocity strokes, delay from the torque onset.  Held-out chirp RMSE
# 0.580 deg against the real arm; the model shipped in robot.SYSID_GAINS gives
# 1.032 deg on the same data.
IDENTIFIED_KP = (367.3, 379.4, 279.9, 116.1, 22.3, 50.6)
IDENTIFIED_KD = tuple(k * 0.056 for k in IDENTIFIED_KP)
IDENTIFIED_COULOMB = (0.297, 0.844, 0.312, 0.083, 0.089, 0.059)
IDENTIFIED_VISCOUS = (0.581, 0.500, 0.886, 0.034, 0.046, 0.060)


def use_identified_plant():
  """Swap the simulated arm for the one that was measured.

  Patches the module-level tables the task reads at build time, so the change
  reaches every environment the task constructs without threading a config
  through mjlab.  Returns a restore callable.
  """
  from piper_push import robot as R
  saved = (dict(R.SYSID_GAINS), dict(R.SYSID_COULOMB_NM), R.get_pick_spec)
  names = [f"joint{i}" for i in range(1, 7)]
  R.SYSID_GAINS = {n: (IDENTIFIED_KP[i], IDENTIFIED_KD[i])
                   for i, n in enumerate(names)}
  R.SYSID_COULOMB_NM = {n: IDENTIFIED_COULOMB[i] for i, n in enumerate(names)}
  base_spec = saved[2]

  def spec_with_damping(profile=R.BARE_GRIPPER):
    spec = base_spec(profile)
    # The shipped model has no joint damping at all; the strokes measured a
    # viscous term on every joint and a large one on J1-J3.
    for jt in spec.joints:
      if jt.name in names:
        jt.damping[:] = IDENTIFIED_VISCOUS[names.index(jt.name)]
    return spec

  R.get_pick_spec = spec_with_damping

  def restore():
    R.SYSID_GAINS, R.SYSID_COULOMB_NM, R.get_pick_spec = saved
  return restore


def pin_shape(half: np.ndarray, cls: int):
  """Make every shape draw return one fixed object.

  Patching the composer rather than writing the model arrays afterwards keeps
  mass, inertia and the bounding half-extents consistent with the geometry --
  they are all derived from the composer's output inside the same event.
  """
  original = shapes._compose

  def fixed(n, device, generator, variety, weights=None):
    size, pos, _half, _cls = original(n, device, generator, variety, weights)
    hx, hy, hz = (float(v) for v in half)
    size = torch.full_like(size, sim_objects.COLLAPSED_HALF)
    pos = torch.zeros_like(pos)
    out_cls = torch.full_like(_cls, int(cls))
    if cls == CLASS_INDEX["cylinder"]:
      size[:, 1, 0] = 0.5 * (hx + hy)   # radius
      size[:, 1, 1] = hz                # half-length
      size[:, 1, 2] = 0.0
      host = pos[:, 1]
    else:
      size[:, 0, 0], size[:, 0, 1], size[:, 0, 2] = hx, hy, hz
      host = pos[:, 0]
    tiny = size.max(dim=-1).values <= sim_objects.COLLAPSED_HALF * 1.5
    pos = torch.where(tiny.unsqueeze(-1), host.unsqueeze(1), pos)
    out_half = torch.zeros_like(_half)
    out_half[:, 0], out_half[:, 1], out_half[:, 2] = hx, hy, hz
    return size, pos, out_half, out_cls

  shapes._compose = fixed
  return original


def pin_place(cmd, xy: np.ndarray, yaw: float):
  """Drop the object at one spot every time it is (re)placed."""

  def placed(self, idx: int, env_ids: torch.Tensor) -> None:
    if self.redraw_on_place and not self._resetting:
      self._reshape(idx, env_ids)
    n = len(env_ids)
    half = self.all_half_sizes[env_ids, idx]
    pose = torch.zeros(n, 7, device=self.device)
    pose[:, 0] = float(xy[0])
    pose[:, 1] = float(xy[1])
    pose[:, 2] = half[:, 2]
    pose[:, 3] = float(np.cos(yaw / 2.0))
    pose[:, 6] = float(np.sin(yaw / 2.0))
    pose[:, :3] += self._env.scene.env_origins[env_ids]
    obj = self._objects[idx]
    obj.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    obj.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=self.device), env_ids=env_ids)

  cmd._place_one = types.MethodType(placed, cmd)


def _tilt_deg(quat: torch.Tensor) -> torch.Tensor:
  """Angle between the object's own +z and world +z, in degrees."""
  w, x, y, z = quat.unbind(-1)
  up_z = 1.0 - 2.0 * (x * x + y * y)
  return torch.rad2deg(torch.acos(up_z.clamp(-1.0, 1.0)))


def _erode(m: torch.Tensor, rounds: int) -> torch.Tensor:
  """Shrink the target mask by ``rounds`` 3x3 erosions.

  The rig's segmenter keeps the core of what it sees and loses the rim -- the
  height filter cuts the base, the component filter cuts the fringe, and the
  sensor drops the curved flanks.  Measured against a matched simulator frame
  the surviving mask is a bit over half the area.  Eroding the simulator's own
  mask reproduces that without inventing a second segmenter.
  """
  x = m
  for _ in range(rounds):
    x = -torch.nn.functional.max_pool2d(-x, 3, stride=1, padding=1)
  return x


def pin_plant(cfg, slow: float) -> None:
  """Pin the plant at the slow end of its randomisation, or past it.

  ``slow`` = 1.0 is the corner the profile already reaches: the lowest response,
  the lowest stiffness, the highest friction, always two steps of latency.
  Above 1.0 the response is pushed further down, which is how a rig that turns
  out to be slower than anything the training domain contained gets measured
  rather than argued about.
  """
  from piper_push.tasks.pick_place.robust_cfg import HEAVY_DR_PROFILE as P
  lo = P["timing"]["arm_response"][0] / max(slow, 1e-3)
  for name in ("arm", "gripper"):
    hooks = cfg.actions[name].command_hooks
    if not hooks:
      continue
    h = hooks[0]
    h.response_range = (lo, lo)
    h.latency_weights = (0.0, 0.0, 1.0)     # always the longest delay
    h.hold_weights = (0.0, 0.0, 1.0)
  if "robust_pd_gains" in cfg.events:
    kp = P["robot"]["kp_scale"][0] / max(slow, 1e-3)
    kd = P["robot"]["kd_scale"][1]
    cfg.events["robust_pd_gains"].params["kp_range"] = (kp, kp)
    cfg.events["robust_pd_gains"].params["kd_range"] = (kd, kd)
  if "robust_joint_friction" in cfg.events:
    f = P["robot"]["joint_friction_scale"][1] * max(slow, 1e-3)
    cfg.events["robust_joint_friction"].params["ranges"] = (f, f)


def rollout(policy_dir: pathlib.Path, half, cls, xy, yaw, envs, steps,
            device, seed, mask_erode: int = 0, slow: float = 0.0,
            identified: bool = False) -> dict:
  restore = pin_shape(half, cls)
  restore_plant = use_identified_plant() if identified else (lambda: None)
  try:
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = envs
    cfg.seed = seed
    if slow:
      pin_plant(cfg, slow)
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    agent = load_rl_cfg(TASK)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
      wrapped, asdict(agent), None, device)
    runner.load(str(policy_dir / "checkpoint.pt"),
                load_cfg={"actor": True}, strict=True, map_location=device)
    pol = runner.get_inference_policy(device=device)
    cmd = env.command_manager.get_term("pick")
    pin_place(cmd, xy, yaw)
    obj = env.scene["object"]
    origins = env.scene.env_origins

    # Reset once more so the pinned placement is what the episode starts from:
    # the environment was already built (and the object already dropped) by the
    # constructor, before the placement could be patched.
    env.reset()
    o = wrapped.get_observations()
    if isinstance(o, tuple):
      o = o[0]

    spot = torch.as_tensor(xy, dtype=torch.float32, device=device)
    # Fractions of TIME, not per-episode flags: the object is teleported back
    # to the pinned spot whenever it is cleared or the episode resets, so a
    # time fraction needs no episode bookkeeping to stay meaningful, and a
    # sticky per-env flag would report the first 20 steps of a 600-step run.
    on_table = torch.zeros(envs, device=device)
    displaced = torch.zeros(envs, device=device)
    toppled = torch.zeros(envs, device=device)
    carried = torch.zeros(envs, device=device)
    # How low the hand goes, and how low it is when it starts closing.  The rig
    # reports this number directly (run.json's stop_reason is the same site
    # height), so it is the one quantity that can be put side by side with the
    # hardware without a definitional conversion.
    open_max = float(cmd._gripper_opening().max())
    min_site = torch.full((envs,), 1e3, device=device)
    close_z, close_n = torch.zeros(envs, device=device), torch.zeros(envs, device=device)
    prev_open = cmd._gripper_opening().clone()
    area_before, area_after = 0.0, 0.0
    acts, lags, rates = [], [], []
    # How often the policy can actually SEE its target.  On the rig this fell
    # to 22% of frames once the arm started reaching over the object, against
    # 100% with the arm parked; the simulator has the same camera and the same
    # arm, so the number it produces here is the honest comparison.
    seen = 0.0
    seen_n = 0
    ent = env.scene["robot"]
    arm_ids = ent.find_joints([f"joint{i}" for i in range(1, 7)],
                              preserve_order=True)[0]
    # ``_previous_target`` is the COMMAND stream -- slew-limited, but upstream
    # of the latency, hold and response the plant model applies.  That is what
    # ``run.py`` puts on the CAN bus, so it is the only sim quantity the rig's
    # recorded target can be differenced against.  ``joint_pos_target`` is the
    # post-plant, substep-interpolated setpoint, and measuring the lag against
    # that hides exactly the delay the comparison is about.
    arm_term = env.action_manager.get_term("arm")
    for _ in range(steps):
      if mask_erode:
        c = o["camera"].clone()
        area_before += float((c[:, 1] > 0.5).float().sum())
        c[:, 1] = _erode(c[:, 1:2], mask_erode)[:, 0]
        area_after += float((c[:, 1] > 0.5).float().sum())
        c[:, 2] = c[:, 0] * c[:, 1]
        o = o.clone()
        o["camera"] = c
      with torch.inference_mode():
        a = pol(o)
      acts.append(a.abs().flatten())
      seen += float((o["camera"][:, 1] > 0.5).any(dim=-1).any(dim=-1).float().mean())
      seen_n += 1
      o, _, _, _ = wrapped.step(a)
      lags.append((ent.data.joint_pos[:, arm_ids]
                   - arm_term._previous_target).abs().flatten())
      rates.append(ent.data.joint_vel[:, arm_ids].abs().flatten())
      pose = obj.data.root_link_pose_w
      here = pose[:, :3] - origins
      half_z = cmd.all_half_sizes[:, 0, 2]
      down = here[:, 2] < half_z + 0.02
      slide = torch.linalg.norm(here[:, :2] - spot, dim=-1)
      on_table += down.float()
      displaced += (down & (slide > 0.03)).float()
      toppled += (down & (_tilt_deg(pose[:, 3:7]) > 45.0)).float()
      carried += (~down).float()
      site = cmd._site_pos_w() - origins
      over = torch.linalg.norm(site[:, :2] - here[:, :2], dim=-1) < 0.06
      min_site = torch.minimum(min_site, torch.where(
        over, site[:, 2], torch.full_like(site[:, 2], 1e3)))
      now_open = cmd._gripper_opening()
      closing = over & (now_open < prev_open - 1e-4) & (now_open < 0.9 * open_max)
      close_z += torch.where(closing, site[:, 2], torch.zeros_like(site[:, 2]))
      close_n += closing.float()
      prev_open = now_open.clone()
    table = on_table.clamp(min=1.0)
    return {
      "placed": float(cmd.objects_placed.sum()),
      "grasp_attempts": float(cmd.grasp_attempts.sum()),
      "placed_per_env": float(cmd.objects_placed.mean()),
      "carried_frac": float((carried / steps).mean()),
      "knocked_frac": float((displaced / table).mean()),
      "toppled_frac": float((toppled / table).mean()),
      "min_site_mm": float(min_site[min_site < 1e2].median()) * 1000.0
                     if bool((min_site < 1e2).any()) else float("nan"),
      "closing_site_mm": float((close_z / close_n.clamp(min=1.0))[close_n > 0]
                               .median()) * 1000.0
                          if bool((close_n > 0).any()) else float("nan"),
      "mask_seen_frac": seen / max(seen_n, 1),
      "rate_p50": float(torch.cat(rates).quantile(0.5)),
      "rate_p95": float(torch.cat(rates).quantile(0.95)),
      "lag_p50": float(torch.cat(lags).quantile(0.5)),
      "lag_p95": float(torch.cat(lags).quantile(0.95)),
      "action_p50": float(torch.cat(acts).quantile(0.5)),
      "action_p95": float(torch.cat(acts).quantile(0.95)),
      "action_max": float(torch.cat(acts).max()),
      "mask_erode": mask_erode,
      "mask_area_ratio": (area_after / area_before) if area_before else 1.0,
      "envs": envs,
      "steps": steps,
    }
  finally:
    restore_plant()
    shapes._compose = restore
    try:
      env.close()
    except Exception:
      pass


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--policy", type=pathlib.Path, required=True)
  p.add_argument("--object-xy", type=float, nargs=2, required=True,
                 metavar=("X", "Y"), help="object centre in the robot base frame")
  p.add_argument("--object-size", type=float, nargs=3, metavar=("W", "D", "H"),
                 help="full extents in metres; height is above the table")
  p.add_argument("--shape", default="cylinder", choices=sim_objects.SHAPE_CLASSES)
  p.add_argument("--yaw", type=float, default=0.0)
  p.add_argument("--sweep", nargs="*", metavar="W,D,H[,shape]",
                 help="run several geometries at the same place")
  p.add_argument("--mask-erode", type=int, default=0,
                 help="shrink the policy's target mask by N 3x3 erosions, to "
                      "match the area the rig's segmenter actually delivers")
  p.add_argument("--identified", action="store_true",
                 help="use the arm measured on 2026-08-31 instead of the one "
                      "the policy was trained against")
  p.add_argument("--slow", type=float, default=0.0,
                 help="pin the plant at the slow corner of the DR (1.0), or "
                      "past it (>1) to reach an arm the profile never covered")
  p.add_argument("--envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", type=pathlib.Path)
  a = p.parse_args()

  cases = []
  if a.sweep:
    for s in a.sweep:
      f = s.split(",")
      shape = f[3] if len(f) > 3 else a.shape
      cases.append((tuple(float(v) for v in f[:3]), shape))
  elif a.object_size:
    cases.append((tuple(a.object_size), a.shape))
  else:
    p.error("give --object-size or --sweep")

  xy = np.asarray(a.object_xy, dtype=np.float64)
  rows = []
  for size, shape in cases:
    half = np.asarray(size, dtype=np.float64) / 2.0
    over_w = 2.0 * max(half[0], half[1]) > 2.0 * sim_objects.OBJECT_MAX_HALF_WIDTH
    over_h = half[2] > sim_objects.OBJECT_MAX_HALF_HEIGHT
    r = rollout(a.policy, half, CLASS_INDEX[shape], xy, a.yaw,
                a.envs, a.steps, a.device, a.seed, a.mask_erode, a.slow,
                a.identified)
    r.update(size_mm=[round(v * 1000, 1) for v in size], shape=shape,
             wider_than_trained=bool(over_w), taller_than_trained=bool(over_h))
    rows.append(r)
    print("%-9s %3.0f x %3.0f x %3.0f mm   placed/env %5.2f  attempts %4.0f  "
          "carried %4.1f%%  knocked %4.1f%%  toppled %4.1f%%  "
          "hand low %5.1f mm  closes at %5.1f mm  "
          "|action| p50 %.2f p95 %.2f  lag p50 %.3f p95 %.3f rad  "
          "|dq| p50 %.2f p95 %.2f rad/s  target visible %.0f%%%s%s"
          % (shape, size[0] * 1000, size[1] * 1000, size[2] * 1000,
             r["placed_per_env"], r["grasp_attempts"], 100 * r["carried_frac"],
             100 * r["knocked_frac"], 100 * r["toppled_frac"],
             r["min_site_mm"], r["closing_site_mm"],
             r["action_p50"], r["action_p95"], r["lag_p50"], r["lag_p95"],
             r["rate_p50"], r["rate_p95"], 100 * r["mask_seen_frac"],
             ("   mask x%.2f" % r["mask_area_ratio"]) if a.mask_erode else "",
             "   OUT OF DISTRIBUTION" if over_w or over_h else ""), flush=True)
  if a.out:
    a.out.write_text(json.dumps(
      {"object_xy": xy.tolist(), "policy": str(a.policy), "cases": rows},
      indent=1))
  return 0


if __name__ == "__main__":
  sys.exit(main())
