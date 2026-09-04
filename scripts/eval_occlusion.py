"""How often does the policy put its own gripper between the camera and the object?

This is the quantity the whole v7 reward change is aimed at, so it needs a
measurement that is not the reward.

**Measured from the rendered segmentation, not by tracing spheres.**  The
first version of this script traced rays against a sphere cover of the arm and
its numbers were wrong twice over, badly enough to invert a conclusion:

1.  The cover is far too fat to stand for the arm's silhouette.  Its radii come
    from each collision geom's AABB cross-section diagonal, which puts a
    **49.9 mm** sphere on ``gripper_base`` and **30.8 mm** spheres on the
    fingers.  An object held in the jaws is *inside* the arm as far as that
    tracer is concerned, so it reported 97% of the carrying phase blocked
    where the renderer sees 6%.
2.  It indexed ``robot.data.geom_pos_w`` with **global** model geom ids, and
    that array is in the robot's own geom order.  Every sphere was attached to
    the next link along, 78-187 mm from where it belonged.

Both are invisible in the output -- the number looks plausible either way --
and they were caught only because a rendered frame captioned "visible 0%"
showed the object plainly.  So the measurement now reads the same buffer the
picture comes from: each sample point on the target is projected into the
policy camera and the segmentation is read at that pixel.  Three outcomes, and
the third is what the tracer could never give: the point shows the target
(visible), or it shows a robot geom (occluded, **by that named body**), or
something else.  Attribution is exact rather than a bounding volume's opinion.

Rays go to points on the target's own box, not to its centre: a centre-only
test says "occluded" the moment a finger crosses the middle, while what the
segmenter needs is enough surviving pixels to clear its component floor.

    python scripts/eval_occlusion.py \
        --checkpoint logs/.../model_2999.pt \
        --task Mjlab-Pick-Place-PiperX-Robust \
        --out occlusion.json

``scripts/sight_viewer.py`` writes a draggable per-frame page from the same
functions, and can overlay the superseded sphere cover so the discrepancy
above can be seen rather than taken on trust.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import torch

import mjlab.tasks  # noqa: F401  -- registers the task ids
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push import camera as sim_camera
from piper_push import evalcfg

VISIBLE, FINGERS, GRIPPER_BASE, ARM, OTHER = range(5)
GROUP_NAMES = ("visible", "fingers", "gripper_base", "arm", "other")

INSET = 0.85
"""Pull the box samples inside the target's surface before reading a pixel.

Eight of the fifteen points are corners of the bounding box and land exactly
on the silhouette edge, where rounding to a pixel throws about half of them
onto the table -- they then read as blocked by scenery.  Measured on a
60-frame rollout that alone moved apparent visibility from 0.579 to 0.711 with
no change in the scene.  At 0.85 the points sit inside the shape and still
spread across it, so the fraction measures occlusion and not the sampler's own
edge.  Reported in the output so a reader can see which convention produced a
number."""

COLLISION_GROUP = 3
"""The group ``piper_push.robot`` puts the arm's collision geoms in.

Only used by the superseded sphere cover below.  ``proprio._COLLISION_GROUP``
is the authority and this agrees with it."""


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def sample_box(half=None, inset: float = INSET) -> np.ndarray:
  """Unit offsets on the target's box, in its own frame.

  The eight corners plus the six face centres plus the centre: fifteen points
  that between them span the silhouette, so a finger across the middle costs
  some and not all.  Scaled by ``inset``; see its docstring.
  """
  s = np.array([-1.0, 0.0, 1.0])
  pts = []
  for x in s:
    for y in s:
      for z in s:
        if abs(x) + abs(y) + abs(z) <= 1 or (abs(x) == abs(y) == abs(z) == 1):
          pts.append((x, y, z))
  return np.asarray(sorted(set(pts)), np.float64) * float(inset)


def geom_group_map(model, device) -> torch.Tensor:
  """Per-geom label: fingers / gripper_base / arm / other, by owning body."""
  g = np.full(model.ngeom, OTHER, dtype=np.int64)
  for i in range(model.ngeom):
    b = int(model.geom_bodyid[i])
    full = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    if full.split("/")[0] != "robot":
      continue
    nm = full.split("/")[-1]
    g[i] = (FINGERS if nm.startswith("gripper_link")
            else GRIPPER_BASE if nm.startswith("gripper_base") else ARM)
  return torch.tensor(g, device=device)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
  """wxyz (B, 4) -> (B, 3, 3); columns are the camera's x, y, z axes."""
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return torch.stack([
    torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
    torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
    torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
  ], dim=1)


def sample_status(sensor, cmd, sim_model, cam_idx, gmap, offsets):
  """(B, K) label per sample point, read out of the segmentation buffer.

  The camera pose comes from ``sim_model.cam_pos``/``cam_quat`` per
  environment rather than from ``camera.CAMERA_POS``, so that the projection
  follows the camera when domain randomisation moves it.  (It does not move in
  ``play=True``; it does during training, and a metric that silently assumed
  the nominal pose would be wrong in exactly the runs that matter.)
  """
  pts = (cmd._object_pos_local().unsqueeze(1)
         + offsets.unsqueeze(0) * cmd.object_half_size.unsqueeze(1))
  cpos = sim_model.cam_pos[:, cam_idx].to(torch.float32)
  cmat = quat_to_mat(sim_model.cam_quat[:, cam_idx].to(torch.float32))

  W, H = sim_camera.WIDTH, sim_camera.HEIGHT
  f = 0.5 * H / np.tan(np.deg2rad(sim_camera.FOVY_DEG) / 2.0)
  local = torch.einsum("bij,bkj->bki", cmat.transpose(1, 2),
                       pts - cpos.unsqueeze(1))
  depth = (-local[..., 2]).clamp_min(1e-6)
  u = (W / 2.0 + f * local[..., 0] / depth).round().long().clamp(0, W - 1)
  v = (H / 2.0 - f * local[..., 1] / depth).round().long().clamp(0, H - 1)

  seg = sensor.data.segmentation
  ids, types = seg[..., 0], seg[..., 1]
  flat = v * W + u
  gid = ids.flatten(1).gather(1, flat)
  typ = types.flatten(1).gather(1, flat)
  is_geom = typ == int(mujoco.mjtObj.mjOBJ_GEOM)
  is_target = ((gid.unsqueeze(-1) == cmd.target_geom_ids.unsqueeze(1)).any(-1)
               & is_geom)
  label = torch.where(is_geom, gmap[gid.clamp_min(0)],
                      torch.full_like(gid, OTHER))
  return torch.where(is_target, torch.full_like(label, VISIBLE), label)


# ---------------------------------------------------------------------------
# The superseded sphere tracer.  Kept only so ``sight_viewer.py`` can draw the
# disagreement; it is not what this script measures.  See the module docstring.
# ---------------------------------------------------------------------------

def sphere_cover(model):
  """Local sphere centres, radii and owning geoms, as ``proprio`` builds them.

  A copy of ``proprio.Kinematics.link_spheres``.  Correct for its own job --
  the deployment subtracts these to remove the arm from a point cloud, where
  over-covering is the safe direction -- and wrong as a model of the arm's
  silhouette, which is what it was misused for here.
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


def ray_hits_spheres(origin, targets, centres, radii):
  """Which of ``targets`` the sphere cover hides from ``origin``.

  ``origin`` (3,), ``targets`` (B, K, 3), ``centres`` (B, S, 3), ``radii``
  (S,).  Returns (B, K) bool.  A segment-sphere test: a sphere behind the
  target does not occlude it, and an infinite ray would say it does.
  """
  d = targets - origin.view(1, 1, 3)
  length = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-9)
  u = d / length
  rel = centres.unsqueeze(1) - origin.view(1, 1, 1, 3)
  t = (rel * u.unsqueeze(2)).sum(dim=-1).clamp(min=0.0).minimum(length)
  closest = origin.view(1, 1, 1, 3) + t.unsqueeze(-1) * u.unsqueeze(2)
  gap = torch.linalg.norm(centres.unsqueeze(1) - closest, dim=-1)
  return (gap < radii.view(1, 1, -1)).any(dim=-1)


def load_policy(runner, ckpt, device):
  """Teachers and distilled students are saved differently; take either.

  A PPO checkpoint carries ``actor_state_dict``; a distillation checkpoint
  carries ``student_state_dict``.  Whichever it is, the weights go straight
  into the network the runner evaluates and are read back and compared, so
  that a mismatch between checkpoint kind and runner kind is an error and
  not an untrained network.  The old branch -- ``runner.load(ckpt,
  load_cfg={"actor": True})`` -- loads NOTHING on a ``-Distill*`` task,
  because rsl_rl's ``Distillation.load`` does not know the key ``actor``;
  ``results/decay/v4_final.json`` (twelve windows of 0.0) is that branch.
  """
  return evalcfg.load_policy(runner, ckpt, device)


def reset_recurrent(policy, dones) -> None:
  """Clear the GRU of the environments that just terminated.

  ``get_inference_policy`` knows nothing about episode boundaries, so a
  recurrent policy left alone carries its hidden state straight through a
  reset and starts every episode remembering the last one -- a state it is
  never in on the robot, and never was during training, where the runner does
  this for you.

  ``scripts/accept_s1.py`` has always done it.  These scripts did not, and the
  gap is not cosmetic: the same v4 checkpoint reads 29.95 objects/min under
  accept_s1 and 0.20 grasp attempts per 1800 steps without this call.  A state
  teacher is an MLP and is unaffected; every distilled student is recurrent and
  is not.
  """
  if getattr(policy, "is_recurrent", False) and dones is not None:
    # Inside inference mode: the hidden state was created there, and rsl_rl
    # zeroes it in place.  Outside, torch refuses -- "Inplace update to
    # inference tensor outside InferenceMode" -- which is easy to mistake for
    # "this policy is not recurrent" and skip.
    with torch.inference_mode():
      policy.reset(dones)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Robust")
  p.add_argument("--num-envs", type=int, default=128)
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=20260902)
  p.add_argument("--inset", type=float, default=INSET)
  p.add_argument("--visible-floor", type=float, default=0.35,
                 help="visible fraction below which a frame is called blocked. "
                      "Not a fitted number, and reported alongside the raw "
                      "fractions so a reader can pick a different one")
  evalcfg.add_sensor_arg(p, default="measured")
  p.add_argument("--out", default=None)
  a = p.parse_args()

  torch.manual_seed(a.seed)
  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = a.num_envs
  sensor_prov = evalcfg.apply_sensor(cfg, a.task, a.sensor)
  # The state task has no camera, and occlusion cannot be seen without one.
  if not any(getattr(s, "name", "") == sim_camera.CAMERA_NAME
             for s in (cfg.scene.sensors or ())):
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (sim_camera.camera_cfg(),)
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  agent = load_rl_cfg(a.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(
    wrapped, asdict(agent), None, a.device)
  policy = load_policy(runner, a.checkpoint, a.device)

  cmd = env.command_manager.get_term("pick")
  sensor = env.scene[sim_camera.CAMERA_NAME]
  gmap = geom_group_map(env.sim.mj_model, a.device)
  offsets = torch.tensor(sample_box(inset=a.inset), dtype=torch.float32,
                         device=a.device)

  env.reset()
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  # Smoothness, alongside occlusion, because "best teacher" is not decided by
  # visibility alone.  The rig has a 3.93 rad/s joint rating and three real
  # runs were stopped by the speed guard at 4.6-7.8 rad/s, so a teacher whose
  # joints move faster than that is one the arm cannot follow however well it
  # sees.  ``action_rate`` is the step-to-step change in the commanded action,
  # which is what the slew limiter and the latency budget actually see.
  robot = env.scene["robot"]
  status_l, held_l, dq_l, rate_l, reach_l = [], [], [], [], []
  prev = None
  for _ in range(a.steps):
    with torch.inference_mode():
      action = policy(obs)
    if prev is not None:
      rate_l.append((action - prev).abs().mean(dim=-1).cpu().numpy())
    prev = action.clone()
    obs, _, dones, _ = wrapped.step(action)
    reset_recurrent(policy, dones)
    status_l.append(sample_status(sensor, cmd, env.sim.model, sensor.camera_idx,
                                  gmap, offsets).cpu().numpy())
    held_l.append(cmd.grasped.cpu().numpy().copy())
    dq_l.append(robot.data.joint_vel[:, :6].abs().max(dim=-1).values
                .cpu().numpy())
    # How far the hand is from the object.  A policy that stops reaching
    # scores well on occlusion for the wrong reason -- its gripper is not in
    # front of the object because its gripper is not anywhere near it -- and
    # the sight reward is exactly the term that could produce that.  Without
    # this the metric cannot tell "keeps its hand out of the way" from
    # "stopped working".
    reach_l.append((cmd._site_pos_w() - env.scene.env_origins
                    - cmd._object_pos_local()).norm(dim=-1).cpu().numpy())

  st = np.stack(status_l)                                  # (T, B, K)
  held = np.stack(held_l).astype(bool)                     # (T, B)
  dq = np.stack(dq_l)                                      # (T, B) fastest joint
  reach = np.stack(reach_l)                                # (T, B) metres
  rate = np.stack(rate_l) if rate_l else np.zeros((1, 1))
  vis = (st == VISIBLE).mean(axis=-1)                       # (T, B)
  placed = float(cmd.objects_placed.float().mean())
  dur = env.step_dt * a.steps

  def block(sel):
    if not sel.any():
      return None
    return {
      "n": int(sel.sum()),
      "visible_mean": float(vis[sel].mean()),
      "visible_median": float(np.median(vis[sel])),
      "blocked_rate": float((vis[sel] < a.visible_floor).mean()),
      "occluded_by": {GROUP_NAMES[g]: float((st[sel] == g).mean())
                      for g in (FINGERS, GRIPPER_BASE, ARM, OTHER)},
    }

  out = {
    "checkpoint": a.checkpoint,
    "task": a.task,
    "num_envs": a.num_envs,
    "steps": a.steps,
    "method": "rendered segmentation at projected sample points",
    "rays_per_frame": int(offsets.shape[0]),
    "inset": a.inset,
    "visible_floor": a.visible_floor,
    "placed_per_min": placed / dur * 60.0,
    "held_fraction": float(held.mean()),
    "joint_speed_rad_s": {"mean": float(dq.mean()),
                          "p95": float(np.percentile(dq, 95)),
                          "p99": float(np.percentile(dq, 99)),
                          "max": float(dq.max()),
                          "over_rating": float((dq > 3.93).mean())},
    "action_rate": float(rate.mean()),
    "all": block(np.ones_like(held)),
    "by_phase": {"approach": block(~held), "holding": block(held)},
    # Engaged = the hand is within 150 mm of the object, so it is close enough
    # to be in the way.  This is the occlusion number that cannot be improved
    # by giving up.
    "engaged": block(~held & (reach < 0.15)),
    "reach_m": {"mean": float(reach.mean()),
                "engaged_fraction": float((reach < 0.15).mean())},
    "seed": a.seed,
    "provenance": evalcfg.provenance(argv=sys.argv, sensor=sensor_prov),
  }
  print(json.dumps(out, indent=2))
  print()
  print("occlusion is attributed to the body the segmentation shows at that")
  print("pixel, so 'fingers' and 'gripper_base' are the gripper's own doing")
  print("and 'other' is scenery or the sampler landing off the silhouette.")
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
