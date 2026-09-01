"""How often does the policy put its own arm between the camera and the object?

This is the quantity the whole v7 reward change is aimed at, so it needs a
measurement that is not the reward.  ``sight_cylinder`` charges a soft radius
around the camera-to-object axis; this casts rays and counts what is actually
blocked, which can disagree with the penalty and is the point of measuring.

**Ray against a sphere cover, deliberately, rather than ``mj_ray``.**  MuJoCo
could answer this exactly, and exactness is not what is wanted here.  The
number this has to be compared against was measured on the robot on
2026-08-31 -- the arm blocked the camera-to-object line 52% of frames on a
fast run and 16% on a slow one -- and that measurement traced rays against
``proprio.Kinematics.link_spheres``, the same sphere cover the deployment's
segmenter subtracts.  A simulator number computed a different way would differ
from the rig number for reasons that have nothing to do with the policy.  So
this builds the identical cover, from the same geom AABBs, and traces the same
way.  When the two are compared, the method is not one of the variables.

Rays go to points sampled on the target's own box, not to its centre: a
centre-only test says "occluded" the moment a finger crosses the middle, while
what the segmenter needs is enough surviving pixels to clear its component
floor.  The visible FRACTION is the useful number and the headline is how often
that fraction falls below what the rig's detector needs.

    python scripts/eval_occlusion.py \
        --checkpoint logs/.../model_2999.pt \
        --task Mjlab-Pick-Place-PiperX-Vision-Robust \
        --out occlusion.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict

import numpy as np
import torch

import mjlab.tasks  # noqa: F401  -- registers the task ids
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push import camera as sim_camera

COLLISION_GROUP = 3
"""The group ``piper_push.robot`` puts the arm's collision geoms in.

Three, not one.  ``proprio._COLLISION_GROUP`` is the authority and this must
agree with it, because the whole point of copying the cover is that the two
tracers see the same arm.  Read from the model rather than assumed the first
time round and the cover came back empty, which is a silent way to measure
zero occlusion: the visual meshes are group 2, and covering those as well
would double the cost for the same shapes."""


def sphere_cover(model):
  """Local sphere centres, radii and owning geoms, exactly as the rig builds.

  One sphere per geom is the obvious implementation and it fails quietly:
  ``geom_rbound`` for the upper arm is 230 mm, and a single sphere that size
  erases a 460 mm disc of the scene.  Each geom is covered by a row of spheres
  along its longest axis instead, each only as wide as the link's cross
  section.  This is a copy of ``proprio.Kinematics.link_spheres`` on purpose --
  see the module docstring for why the two must agree.
  """
  local, radii, geoms = [], [], []
  for g in range(model.ngeom):
    if model.geom_group[g] != COLLISION_GROUP:
      continue
    aabb = np.asarray(model.geom_aabb[g]).reshape(2, 3)
    centre, half = aabb[0], aabb[1]
    axis = int(np.argmax(half))
    cross = float(np.linalg.norm(np.delete(half, axis)))
    span = float(half[axis])
    n = int(np.clip(np.ceil(span / max(cross, 1e-4)), 1, 8))
    reach = max(span - cross, 0.0)
    for t in (np.linspace(-reach, reach, n) if n > 1 else [0.0]):
      c = centre.copy()
      c[axis] += t
      local.append(c)
      radii.append(cross)
      geoms.append(g)
  return (np.asarray(local, np.float64), np.asarray(radii, np.float64),
          np.asarray(geoms, dtype=int))


def sample_box(half, n_side: int = 3):
  """Points on the target's box, in its own frame.

  The eight corners plus the six face centres plus the centre: fourteen rays
  that between them span the silhouette, so a finger across the middle costs
  some of them and not all.  Cheap enough to run every step and coarse enough
  that the fraction is a fraction and not a boolean wearing one.
  """
  s = np.array([-1.0, 0.0, 1.0])
  pts = []
  for x in s:
    for y in s:
      for z in s:
        if abs(x) + abs(y) + abs(z) <= 1 or (abs(x) == abs(y) == abs(z) == 1):
          pts.append((x, y, z))
  return np.asarray(sorted(set(pts)), np.float64)


def ray_hits_spheres(origin, targets, centres, radii):
  """Which of ``targets`` cannot be seen from ``origin``.

  ``origin`` (3,), ``targets`` (B, K, 3), ``centres`` (B, S, 3), ``radii``
  (S,).  Returns (B, K) bool.

  A segment-sphere test, not a ray-sphere one: a sphere BEHIND the target does
  not occlude it, and an infinite ray would say it does.  The projection is
  clamped to the segment before the distance is taken, which is the whole
  difference and the same clamp the rig's tracer uses.
  """
  d = targets - origin.view(1, 1, 3)                       # (B, K, 3)
  length = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-9)
  u = d / length                                           # unit, (B, K, 3)

  rel = centres.unsqueeze(1) - origin.view(1, 1, 1, 3)     # (B, 1, S, 3)
  t = (rel * u.unsqueeze(2)).sum(dim=-1)                   # (B, K, S) along
  t = t.clamp(min=0.0).minimum(length)                     # onto the segment
  closest = origin.view(1, 1, 1, 3) + t.unsqueeze(-1) * u.unsqueeze(2)
  gap = torch.linalg.norm(centres.unsqueeze(1) - closest, dim=-1)  # (B, K, S)
  return (gap < radii.view(1, 1, -1)).any(dim=-1)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Vision-Robust")
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=20260902)
  p.add_argument("--visible-floor", type=float, default=0.35,
                 help="visible fraction below which the rig's segmenter is "
                      "assumed to report nothing.  Not a fitted number: it is "
                      "the point at which the simulated mask falls under the "
                      "150 px component floor on this camera, and it is "
                      "reported alongside the raw fractions so a reader can "
                      "pick a different one")
  p.add_argument("--out", default=None)
  a = p.parse_args()

  torch.manual_seed(a.seed)
  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = a.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  agent = load_rl_cfg(a.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(
    wrapped, asdict(agent), None, a.device)
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=a.device)
  policy = runner.get_inference_policy(device=a.device)

  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("pick")
  model = env.sim.mj_model if hasattr(env.sim, "mj_model") else robot.spec.compile()
  local_np, radii_np, geoms_np = sphere_cover(model)
  local = torch.tensor(local_np, dtype=torch.float32, device=a.device)
  radii = torch.tensor(radii_np, dtype=torch.float32, device=a.device)
  geoms = torch.tensor(geoms_np, dtype=torch.long, device=a.device)
  offsets = torch.tensor(sample_box(None), dtype=torch.float32, device=a.device)
  cam = torch.tensor(sim_camera.CAMERA_POS, dtype=torch.float32, device=a.device)

  obs = env.reset()
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  visible, held_flags = [], []
  for _ in range(a.steps):
    with torch.inference_mode():
      action = policy(obs)
    obs, _, _, _ = wrapped.step(action)

    origins = env.scene.env_origins
    # The cover, in each environment's own frame.
    gx = robot.data.geom_pos_w[:, geoms] - origins.unsqueeze(1)
    gm = robot.data.geom_quat_w[:, geoms]
    centres = gx + _rotate(gm, local.unsqueeze(0).expand(gx.shape[0], -1, -1))

    obj = cmd._object_pos_local()                      # (B, 3)
    half = cmd.object_half_size                        # (B, 3)
    pts = obj.unsqueeze(1) + offsets.unsqueeze(0) * half.unsqueeze(1)
    blocked = ray_hits_spheres(cam, pts, centres, radii)
    visible.append((~blocked).float().mean(dim=-1).cpu().numpy())
    held_flags.append(cmd.grasped.cpu().numpy().copy())

  vis = np.stack(visible)                              # (T, B)
  held = np.stack(held_flags).astype(bool)
  out = {
    "checkpoint": a.checkpoint,
    "task": a.task,
    "num_envs": a.num_envs,
    "steps": a.steps,
    "rays_per_frame": int(offsets.shape[0]),
    "visible_fraction": {
      "mean": float(vis.mean()),
      "median": float(np.median(vis)),
      "p10": float(np.percentile(vis, 10)),
      "p90": float(np.percentile(vis, 90)),
    },
    "blocked_rate": float((vis < a.visible_floor).mean()),
    "visible_floor": a.visible_floor,
    "fully_blocked_rate": float((vis <= 0.0).mean()),
    "by_phase": {
      "approach": {
        "n": int((~held).sum()),
        "visible_median": float(np.median(vis[~held])) if (~held).any() else None,
        "blocked_rate": float((vis[~held] < a.visible_floor).mean())
        if (~held).any() else None,
      },
      "holding": {
        "n": int(held.sum()),
        "visible_median": float(np.median(vis[held])) if held.any() else None,
        "blocked_rate": float((vis[held] < a.visible_floor).mean())
        if held.any() else None,
      },
    },
  }
  print(json.dumps(out, indent=2))
  print()
  print("for comparison, measured on the rig 2026-08-31 by the same method:")
  print("  arm blocked the camera-to-object line 52% of frames moving fast,")
  print("  16% moving slowly")
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
  return 0


def _rotate(quat, vec):
  """Rotate ``vec`` (B, S, 3) by wxyz ``quat`` (B, S, 4)."""
  w, xyz = quat[..., :1], quat[..., 1:]
  t = 2.0 * torch.cross(xyz, vec, dim=-1)
  return vec + w * t + torch.cross(xyz, t, dim=-1)


if __name__ == "__main__":
  raise SystemExit(main())
