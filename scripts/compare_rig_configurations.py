"""Render the policy's old nominal rig beside a calibrated physical rig.

This is a configuration check, not a policy rollout.  Both scenes use the
same seed and initial task state.  The upper panels are an external RGB camera
attached to the robot base; the lower panels are the depth and target mask
that the vision policy actually receives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg

from piper_push import camera


TASK = "Mjlab-Pick-Place-PiperX-Vision"
OLD_CAMERA_POS = (0.589, 0.535, 0.480)
OLD_CAMERA_AIM = (0.321, 0.071, 0.030)
VIEW_CAM = "configuration_view"
VIEW_POS = (0.95, 0.82, 0.72)
VIEW_AIM = (0.25, 0.00, 0.10)
VIEW_WIDTH, VIEW_HEIGHT = 640, 360


def _font(size: int) -> ImageFont.FreeTypeFont:
  return ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
  )


def _configure(manifest: dict, calibrated: bool, seed: int):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  cfg.seed = seed

  if calibrated:
    tr = manifest["training"]
    robot = cfg.scene.entities["robot"].init_state
    robot.pos = tuple(float(x) for x in tr["robot_base_pos_world_m"])
    robot.rot = tuple(float(x) for x in tr["robot_base_quat_world_wxyz"])

    pos = tuple(float(x) for x in tr["camera_pos_base_m"])
    aim = tuple(float(x) for x in tr["camera_aim_base_m"])
  else:
    pos, aim = OLD_CAMERA_POS, OLD_CAMERA_AIM
  policy_cam = next(s for s in cfg.scene.sensors if s.name == camera.CAMERA_NAME)
  policy_cam.pos = pos
  policy_cam.quat = camera.look_at_quat(pos, aim)

  # Keep the joint state fixed.  Random reset postures are useful for training
  # but would make a rig comparison show two different arms.
  cfg.events.pop("reset_arm", None)

  for sensor in cfg.scene.sensors or ():
    sensor.use_textures = True
    sensor.use_shadows = True
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
    CameraSensorCfg(
      name=VIEW_CAM,
      parent_body=camera.PARENT_BODY,
      pos=VIEW_POS,
      quat=camera.look_at_quat(VIEW_POS, VIEW_AIM),
      fovy=42.0,
      width=VIEW_WIDTH,
      height=VIEW_HEIGHT,
      data_types=("rgb",),
      use_textures=True,
      use_shadows=True,
      enabled_geom_groups=(0, 2),
    ),
  )
  return cfg


def _capture(manifest: dict, calibrated: bool, seed: int, device: str):
  cfg = _configure(manifest, calibrated, seed)
  old_pos, old_aim = camera.CAMERA_POS, camera.CAMERA_AIM
  if calibrated:
    tr = manifest["training"]
    # The startup event reads these module constants.  Override them only
    # while the environment is built, then restore them.  Keeping the event
    # means old and new consume the same random draws before reset.
    candidate_pos = tuple(float(x) for x in tr["camera_pos_base_m"])
    candidate_aim = tuple(float(x) for x in tr["camera_aim_base_m"])
  else:
    candidate_pos, candidate_aim = OLD_CAMERA_POS, OLD_CAMERA_AIM
  camera.CAMERA_POS, camera.CAMERA_AIM = candidate_pos, candidate_aim
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  finally:
    camera.CAMERA_POS, camera.CAMERA_AIM = old_pos, old_aim
  try:
    # Construction allocates the sensors but does not render them.  reset()
    # performs the first forward/render and returns the observation from that
    # same frame, so the RGB and policy panels are genuinely comparable.
    obs, _ = env.reset()
    if isinstance(obs, tuple):
      obs = obs[0]
    policy = obs["camera"][0].detach().float().cpu().numpy()
    scene = env.scene.sensors[VIEW_CAM].data.rgb[0].detach().cpu().numpy()
    return np.ascontiguousarray(scene, dtype=np.uint8), policy
  finally:
    env.close()


def _policy_panel(policy: np.ndarray, width: int = 620) -> Image.Image:
  depth = np.clip(policy[0], 0.0, 1.0)
  mask = policy[1] > 0.5
  rgb = np.repeat((255 * depth).astype(np.uint8)[..., None], 3, axis=-1)
  # Orange is the commanded object.  Keep its depth visible underneath.
  rgb[mask] = (0.35 * rgb[mask] + 0.65 * np.array([255, 112, 20])).astype(np.uint8)
  image = Image.fromarray(rgb).resize(
    (width, round(width * depth.shape[0] / depth.shape[1])), Image.Resampling.NEAREST
  )
  draw = ImageDraw.Draw(image)
  draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=(110, 110, 115))
  return image


def _column(
  title: str,
  subtitle: list[str],
  scene: np.ndarray,
  policy: np.ndarray,
) -> Image.Image:
  w, pad = 680, 20
  scene_img = Image.fromarray(scene).resize((640, 360), Image.Resampling.LANCZOS)
  policy_img = _policy_panel(policy)
  h = 92 + scene_img.height + 42 + policy_img.height + 38
  out = Image.new("RGB", (w, h), (17, 18, 21))
  draw = ImageDraw.Draw(out)
  draw.text((pad, 14), title, fill=(245, 245, 247), font=_font(25))
  y = 48
  for line in subtitle:
    draw.text((pad, y), line, fill=(185, 188, 195), font=_font(14))
    y += 19
  out.paste(scene_img, (pad, 92))
  draw.text((pad, 92 + scene_img.height + 10), "External RGB view",
            fill=(205, 207, 212), font=_font(15))
  py = 92 + scene_img.height + 42
  out.paste(policy_img, (pad, py))
  visible = int((policy[1] > 0.5).sum())
  draw.text(
    (pad, py + policy_img.height + 9),
    f"Policy depth (orange = target mask, {visible} pixels)",
    fill=(205, 207, 212),
    font=_font(15),
  )
  return out


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument(
    "--manifest",
    default="hardware/deploy/calibration_snapshots/20260825_185414/manifest.json",
  )
  p.add_argument(
    "--out", default="results/calibration/old_vs_calibrated_configuration.png"
  )
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()

  manifest = json.loads(Path(a.manifest).read_text())
  old_scene, old_policy = _capture(manifest, False, a.seed, a.device)
  new_scene, new_policy = _capture(manifest, True, a.seed, a.device)
  tr, table = manifest["training"], manifest["table"]

  old = _column(
    "OLD — nominal training rig",
    [
      f"camera pos {tuple(round(x, 3) for x in OLD_CAMERA_POS)} m",
      "table z +0.000 m, level",
    ],
    old_scene,
    old_policy,
  )
  new = _column(
    "NEW — 2026-08-25 calibrated rig",
    [
      f"camera pos {tuple(round(x, 3) for x in tr['camera_pos_base_m'])} m",
      f"table z {1000 * table['corrected_table_z_m']:+.1f} mm, "
      f"tilt {table['tilt_deg']:.3f} deg (3.5 mm board corrected)",
    ],
    new_scene,
    new_policy,
  )

  gap = 16
  canvas = Image.new("RGB", (old.width + new.width + gap, max(old.height, new.height)),
                     (8, 9, 11))
  canvas.paste(old, (0, 0))
  canvas.paste(new, (old.width + gap, 0))
  out = Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(out)
  print(f"wrote {out.resolve()} ({canvas.width}x{canvas.height})")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
