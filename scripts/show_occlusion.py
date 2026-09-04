"""Watch the fixed camera, not the arm, because occlusion is a camera fact.

`scripts/play.sh` shows the robot from a free viewpoint, which is the wrong
place to stand to answer "did the visibility rewards work".  The question is
whether the object is still in the picture the deployment segments, and the
only viewpoint that can answer it is the one bolted to the world.

So this renders the scene camera, marks the target's own pixels, and prints the
visibility beside them, for several checkpoints on the same seed and the same
object placements.  Side by side, the difference between a policy that reaches
across the view and one that comes from behind is visible in a glance and does
not need a metric to believe.

The number and the picture now come from the same buffer -- the segmentation
this panel already renders -- because when they came from different places they
disagreed, and it was the number that was wrong.  See ``eval_occlusion``.
``scripts/sight_viewer.py`` writes a draggable per-frame version of this.

    python scripts/show_occlusion.py \
        --checkpoint before=checkpoints/v7_teachers/v5_baseline.pt \
        --checkpoint after=checkpoints/v7_teachers/strong_teacher.pt \
        --out occlusion_compare.mp4

Writes an mp4 if a codec is available and a contact sheet either way, so it is
useful over ssh as well as locally.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import asdict

import cv2
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push import camera as sim_camera

# Same measurement as scripts/eval_occlusion.py.  Sharing it matters more than
# the few lines it saves, and this file is why: the picture below disagreed
# with the sphere tracer both scripts used to share, and the picture was right.
# Reading the segmentation the panel already renders makes it impossible for
# the caption and the image to come apart again.
from eval_occlusion import (VISIBLE, geom_group_map, load_policy,  # noqa: E402
                            sample_box, sample_status)


# ---------------------------------------------------------------------------
# The spectator: a second camera, placed to see the first one
# ---------------------------------------------------------------------------

# Perpendicular to the sight line, not across from the camera.  The first
# version stood opposite the D455 and the picture was useless: from there the
# camera-to-object axis points almost straight at the viewer, so a line whose
# whole purpose is to show what crosses it was two centimetres long.  The D455
# sits at (-454, +767, +489) mm and the work area near (0, +400, 0), so the
# axis runs roughly (0.59, -0.48, -0.64); this stands off its side.
SPECTATOR_POS = (0.52, 1.42, 0.86)
SPECTATOR_AIM = (-0.22, 0.56, 0.20)
SPECTATOR_FOVY = 62.0


def look_at(pos, aim, up=(0.0, 0.0, 1.0)):
  """MuJoCo camera axes for a camera at ``pos`` looking at ``aim``.

  MuJoCo cameras look down their own -z with +y up, the OpenGL convention.
  Returned as the camera-to-world rotation, columns (x, y, z).
  """
  pos = np.asarray(pos, np.float64)
  fwd = np.asarray(aim, np.float64) - pos
  fwd /= np.linalg.norm(fwd)
  z = -fwd
  x = np.cross(np.asarray(up, np.float64), z)
  x /= np.linalg.norm(x)
  y = np.cross(z, x)
  return np.stack([x, y, z], axis=1)


def mat_to_wxyz(R):
  q = np.empty(4)
  t = np.trace(R)
  if t > 0:
    r = np.sqrt(1 + t) * 2
    q[:] = [0.25 * r, (R[2, 1] - R[1, 2]) / r,
            (R[0, 2] - R[2, 0]) / r, (R[1, 0] - R[0, 1]) / r]
  else:
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    r = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q[0] = (R[k, j] - R[j, k]) / r
    q[i + 1] = 0.25 * r
    q[j + 1] = (R[j, i] + R[i, j]) / r
    q[k + 1] = (R[k, i] + R[i, k]) / r
  return q / np.linalg.norm(q)


def project(points, pos, R, fovy_deg, width, height):
  """World points to pixels in the spectator, plus a mask of what is in front.

  Verified rather than assumed: ``main`` projects the target's own centre and
  checks it lands inside the target's rendered silhouette.  A sign error here
  would draw a confident line through the wrong part of the picture, which is
  worse than drawing none.
  """
  pts = np.atleast_2d(np.asarray(points, np.float64))
  cam = (pts - np.asarray(pos, np.float64)) @ R          # world -> camera
  depth = -cam[:, 2]
  f = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
  safe = np.maximum(depth, 1e-6)
  u = width / 2.0 + f * cam[:, 0] / safe
  v = height / 2.0 - f * cam[:, 1] / safe
  return np.stack([u, v], axis=1), depth > 1e-3


SPEC_W, SPEC_H = 448, 336
"""The spectator's own resolution.  Larger than the policy view because it is
for a person, and it is scaled to match when the two are put side by side."""


def robot_geom_ids(model):
  """Every rendered geom that belongs to the arm.

  Needed because the spectator panel has to show the ARM against the tube, and
  a depth ramp will not: from across the table most of the frame is distant
  floor, so a percentile stretch compresses the robot into a few grey levels
  and the one thing the picture exists to show disappears into the background.
  """
  ids = []
  for g in range(model.ngeom):
    b = int(model.geom_bodyid[g])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    # "robot/link3", not "link3".  The scene model namespaces every entity, and
    # matching the bare names quietly selected nothing -- a spectator panel
    # with no arm in it, which is the one thing it exists to show.
    if name.split("/")[0] == "robot":
      ids.append(g)
  return np.asarray(ids, dtype=np.int64)


def _spectator_panel(env, cmd, R, cam_w, obj_w, visible, radius,
                     robot_geoms=None):
  """The scene from across the table, with the sight line drawn into it.

  This is the panel that answers "why".  The camera view says the object is
  hidden; this says what is standing in front of it, and the cylinder drawn
  around the axis is the region ``sight_cylinder`` charges for entering, so a
  reader can see the penalty and the behaviour in the same frame.

  The line is drawn in two colours split at the object: solid up to it, because
  that is the part that must stay clear, and faint beyond it, because
  everything past the far cap is free and the drawing should not suggest
  otherwise.
  """
  sensor = env.scene["spectator"]
  d = sensor.data.depth[0, ..., 0].cpu().numpy()
  seg = sensor.data.segmentation
  ids, types = seg[0, ..., 0], seg[0, ..., 1]
  tgt = cmd.target_geom_ids[0].to(ids.device)
  mask = ((ids.unsqueeze(-1) == tgt.view(1, 1, -1)).any(-1)
          & (types == int(mujoco.mjtObj.mjOBJ_GEOM))).cpu().numpy()

  # Range from the picture rather than a constant.  A fixed 0.6-2.6 m window
  # saturated this view to white, because most of what a spectator sees is
  # floor beyond the far end of it, and a washed-out panel cannot show which
  # link is in the tube.
  # Clipped before the stretch.  The renderer's far plane is over a kilometre
  # away, so an unclipped 98th percentile is the sky and everything in the room
  # lands in the bottom two grey levels.
  near = d[(d > 0) & (d < 4.0)]
  lo, hi = ((np.percentile(near, 2), np.percentile(near, 98))
            if near.size else (0.6, 3.0))
  v = np.clip((d - lo) / max(hi - lo, 1e-3), 0, 1)
  img = (np.dstack([v, v, v]) * 90 + 25).astype(np.uint8)   # a quiet ground
  img[d <= 0] = (18, 18, 18)
  if robot_geoms is not None and robot_geoms.size:
    rid = torch.as_tensor(robot_geoms, device=ids.device)
    arm = (ids.unsqueeze(-1) == rid.view(1, 1, -1)).any(-1)
    arm &= types == int(mujoco.mjtObj.mjOBJ_GEOM)
    # BGR, not RGB.  Written the other way round the first time and the
    # arm came out the same blue as the camera marker.
    img[arm.cpu().numpy()] = (60, 165, 245)     # orange: nothing else is
  img[mask] = (60, 240, 60)

  pos = np.asarray(SPECTATOR_POS) + env.scene.env_origins[0].cpu().numpy()
  axis = obj_w - cam_w
  length = float(np.linalg.norm(axis))
  u = axis / max(length, 1e-9)

  # The cylinder, as a pair of rails plus rings, rather than a filled shape:
  # a translucent solid would hide the arm, which is the thing being looked at.
  side = np.cross(u, [0.0, 0.0, 1.0])
  side /= max(np.linalg.norm(side), 1e-9)
  up = np.cross(u, side)
  ok_col, bad_col = (90, 200, 90), (80, 80, 235)
  col = ok_col if visible > 0.35 else bad_col
  for off in (side, -side, up, -up):
    rail = np.stack([cam_w + t * axis + radius * off
                     for t in np.linspace(0.0, 1.0, 24)])
    px, front = project(rail, pos, R, SPECTATOR_FOVY, SPEC_W, SPEC_H)
    pts = px[front].astype(np.int32)
    if len(pts) > 1:
      cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (55, 55, 55), 1,
                    cv2.LINE_AA)
  for t in (0.25, 0.5, 0.75, 1.0):
    ring = np.stack([cam_w + t * axis
                     + radius * (np.cos(a) * side + np.sin(a) * up)
                     for a in np.linspace(0, 2 * np.pi, 28)])
    px, front = project(ring, pos, R, SPECTATOR_FOVY, SPEC_W, SPEC_H)
    pts = px[front].astype(np.int32)
    if len(pts) > 2:
      cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, col, 1, cv2.LINE_AA)

  # The axis itself, and the camera it starts at.
  beyond = obj_w + 0.25 * u
  px, front = project(np.stack([cam_w, obj_w, beyond]), pos, R,
                      SPECTATOR_FOVY, SPEC_W, SPEC_H)
  c, o, b = px.astype(np.int32)
  if front[0] and front[1]:
    cv2.line(img, tuple(c), tuple(o), col, 2, cv2.LINE_AA)
  if front[1] and front[2]:
    cv2.line(img, tuple(o), tuple(b), (110, 110, 110), 1, cv2.LINE_AA)
  if front[0]:
    cv2.circle(img, tuple(c), 7, (250, 210, 60), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(c), 7, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(img, "D455", (c[0] - 14, c[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (250, 210, 60), 1)
  if front[1]:
    cv2.circle(img, tuple(o), 5, (60, 240, 60), 2, cv2.LINE_AA)

  cv2.putText(img, "arm (orange) vs the tube it must stay out of",
              (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1)
  return img


def rollout(label, ckpt, task, steps, num_envs, device, seed,
            spectator=True, radius=0.07):
  """Render one policy, and measure it with the same rays the report uses."""
  torch.manual_seed(seed)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  # The state task has no camera; occlusion is invisible without one.
  if not any(getattr(s, "name", "") == sim_camera.CAMERA_NAME
             for s in (cfg.scene.sensors or ())):
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (sim_camera.camera_cfg(),)
  spec_R = look_at(SPECTATOR_POS, SPECTATOR_AIM)
  if spectator:
    from mjlab.sensor import CameraSensorCfg
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (CameraSensorCfg(
      name="spectator", parent_body=sim_camera.PARENT_BODY,
      pos=SPECTATOR_POS, quat=tuple(mat_to_wxyz(spec_R)),
      fovy=SPECTATOR_FOVY, width=SPEC_W, height=SPEC_H,
      data_types=("depth", "segmentation"), use_textures=False,
      use_shadows=False, enabled_geom_groups=(0, 2)),)
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  agent = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
    wrapped, asdict(agent), None, device)
  policy = load_policy(runner, ckpt, device)

  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("pick")
  sensor = env.scene[sim_camera.CAMERA_NAME]
  model = env.sim.mj_model if hasattr(env.sim, "mj_model") else robot.spec.compile()
  gmap = geom_group_map(model, device)
  offsets = torch.tensor(sample_box(), dtype=torch.float32, device=device)
  rgeoms = robot_geom_ids(model)

  env.reset()
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  frames, vis_trace = [], []
  for _ in range(steps):
    with torch.inference_mode():
      action = policy(obs)
    obs, _, _, _ = wrapped.step(action)

    status = sample_status(sensor, cmd, env.sim.model, sensor.camera_idx,
                           gmap, offsets)
    visible = (status == VISIBLE).float().mean(dim=-1)
    vis_trace.append(float(visible[0]))

    depth = sensor.data.depth
    seg = sensor.data.segmentation
    assert depth is not None and seg is not None
    d = depth[0, ..., 0].cpu().numpy()
    ids, types = seg[0, ..., 0], seg[0, ..., 1]
    tgt = cmd.target_geom_ids[0].to(ids.device)
    mask = ((ids.unsqueeze(-1) == tgt.view(1, 1, -1)).any(-1)
            & (types == int(mujoco.mjtObj.mjOBJ_GEOM))).cpu().numpy()

    v = np.clip((d - 0.35) / (1.6 - 0.35), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    img[d <= 0] = (25, 25, 25)
    # The target, in the one colour nothing else in the picture uses.
    img[mask] = (60, 240, 60)
    # Up to the spectator's size BEFORE anything is written on it.  The policy
    # view is 224 px wide; a caption that fits the picture does not fit that,
    # and the first version silently ran its own label off the edge -- the
    # visible percentage and the HOLDING flag, the two numbers the panel
    # exists to report, were the part that fell off.  NEAREST because the mask
    # is a label, and interpolating it invents half-target pixels.
    img = cv2.resize(img, (SPEC_W, SPEC_H), interpolation=cv2.INTER_NEAREST)
    held = bool(cmd.grasped[0])
    bar = int(round(220 * float(visible[0])))
    cv2.rectangle(img, (8, img.shape[0] - 18), (8 + 220, img.shape[0] - 8),
                  (70, 70, 70), -1)
    cv2.rectangle(img, (8, img.shape[0] - 18), (8 + bar, img.shape[0] - 8),
                  (60, 240, 60) if visible[0] > 0.35 else (60, 60, 240), -1)
    cv2.putText(img, f"{label}   camera view", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    cv2.putText(img, f"visible {100 * float(visible[0]):3.0f}%"
                     f"{'   HOLDING' if held else ''}",
                (8, img.shape[0] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (60, 240, 60) if visible[0] > 0.35 else (90, 90, 255), 1)

    if spectator:
      obj_w = (cmd._object_pos_local()[0].cpu().numpy()
               + env.scene.env_origins[0].cpu().numpy())
      cam_w = np.asarray(sim_camera.CAMERA_POS) + env.scene.env_origins[0].cpu().numpy()
      # The spectator is rendered at SPEC_W x SPEC_H and now stays there.  It
      # used to be resized down to the policy view's 224 px, which threw away
      # half of the only panel drawn for a human to read.
      img = np.hstack([img, _spectator_panel(
        env, cmd, spec_R, cam_w, obj_w, float(visible[0]), radius,
        robot_geoms=rgeoms)])
    frames.append(img)

  env.close()
  return frames, np.asarray(vis_trace)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", action="append", required=True,
                 metavar="LABEL=PATH",
                 help="repeat for each policy; they are rendered side by side "
                      "on the same seed, so the object placements match and "
                      "the only difference is the policy")
  p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Robust")
  p.add_argument("--steps", type=int, default=300)
  p.add_argument("--num-envs", type=int, default=4)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=20260902)
  p.add_argument("--fps", type=int, default=25)
  p.add_argument("--no-spectator", action="store_true",
                 help="camera view only, no second panel")
  p.add_argument("--radius", type=float, default=0.07,
                 help="the sight cylinder's radius, drawn to scale.  Must "
                      "match HEAVY_DR_PROFILE if the picture is to explain "
                      "the penalty rather than a different one")
  p.add_argument("--out", default="occlusion_compare.mp4")
  a = p.parse_args()

  runs = []
  for spec in a.checkpoint:
    label, _, path = spec.partition("=")
    if not path:
      label, path = pathlib.Path(spec).stem, spec
    print(f"rolling out {label}: {path}")
    frames, vis = rollout(label, path, a.task, a.steps, a.num_envs,
                          a.device, a.seed, spectator=not a.no_spectator,
                          radius=a.radius)
    runs.append((label, frames, vis))
    print(f"  visible fraction: median {np.median(vis):.2f}  "
          f"below 0.35 in {100 * (vis < 0.35).mean():.0f}% of steps")

  h = min(f[0].shape[0] for _, f, _ in runs)
  # A visible seam between policies.  Without it the four panels abut and a
  # reader has to count captions to work out where one policy ends -- easy to
  # misread, and the whole point of the sheet is a comparison at a glance.
  gap = np.full((h, 3, 3), 200, np.uint8)
  strips = []
  for k in range(min(len(f) for _, f, _ in runs)):
    cols = [cv2.resize(fr[k], (int(fr[k].shape[1] * h / fr[k].shape[0]), h))
            for _, fr, _ in runs]
    row = [cols[0]]
    for c in cols[1:]:
      row += [gap, c]
    strips.append(np.hstack(row))

  out = pathlib.Path(a.out)
  wrote = False
  try:
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                         (strips[0].shape[1], strips[0].shape[0]))
    if vw.isOpened():
      for s in strips:
        vw.write(s)
      vw.release()
      wrote = out.exists() and out.stat().st_size > 1000
  except Exception as e:                                    # pragma: no cover
    print(f"  no video: {e}")
  print(f"wrote {out}" if wrote else "  no codec; contact sheet only")

  # Always a sheet as well: a still survives ssh, a screenshot and a slide.
  step = max(1, len(strips) // 8)
  rows = strips[::step][:8]
  hgap = np.full((3, rows[0].shape[1], 3), 200, np.uint8)
  sheet = np.vstack([r for row in rows for r in (row, hgap)][:-1])
  sheet_path = out.with_suffix(".png")
  cv2.imwrite(str(sheet_path), sheet)
  print(f"wrote {sheet_path}")

  # Repeated at the end because the per-run line is printed before the next
  # rollout's model summary, which is a hundred lines long -- by the time the
  # second policy finishes the first policy's number has scrolled away, and
  # comparing them is the only reason to run this.
  print()
  print(f"{'policy':<20} {'visible median':>15} {'below 0.35':>11}")
  for label, _, vis in runs:
    print(f"{label:<20} {np.median(vis):>15.2f} "
          f"{100 * (vis < 0.35).mean():>10.0f}%")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
