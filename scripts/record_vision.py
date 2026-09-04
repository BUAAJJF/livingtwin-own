"""Record a vision policy: the scene, and the three channels it actually sees.

The point of the layout is that the right-hand panels are not illustrations of
the observation -- they are the observation, read straight out of the camera
group on the same step the scene frame was rendered.  The three channels are
what ``camera_scene`` returns:

    depth   the whole scene, normalised by the far plane
    mask    the segmentation channel: which pixels are the target object
    masked  their product, which is what the task is actually solved from

The third-person view is a second *camera sensor*, not ``env.render()``.  The
training machine has no libEGL and no OSMesa -- which is what killed the first
five training runs of this project -- so MuJoCo's own offscreen renderer cannot
open a context there.  Camera sensors go through mujoco_warp's rasteriser
instead, which needs no GL at all, so this records anywhere the policy trains.

Written offscreen so it needs no viewer, and at the environment's own control
rate so the video's timebase is the policy's.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push import camera

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Vision")
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=900)
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--view-pos", default="1.02,0.78,0.66",
               help="third-person camera position in the robot base frame")
p.add_argument("--view-aim", default="0.22,-0.04,0.14")
p.add_argument("--view-fovy", type=float, default=36.0)
p.add_argument("--episode-s", type=float, default=None)
p.add_argument("--fps", type=int, default=50)
p.add_argument("--sensor", choices=("clean", "real", "measured", "task"),
               default="clean",
               help="'clean' is the rendered depth, which makes two recordings "
                    "comparable.  'real' (alias 'measured') puts the sensor the "
                    "task TRAINS with back on -- the fitted D455 model on a "
                    "nominal task, the robust profile on a -Robust one -- "
                    "which is what the robot will hand the policy.")
p.add_argument("--device", default="cuda:0")
a = p.parse_args()

env_cfg = load_env_cfg(a.task, play=True)
# ``play`` switches the sensor model off so that two recordings of the same
# policy differ only by the policy.  'real' puts it back, for the times the
# question is what the camera does rather than what the policy does.
from piper_push import evalcfg  # noqa: E402

_term = env_cfg.observations["camera"].terms["scene"]
_term.params = dict(_term.params)
evalcfg.apply_sensor(env_cfg, a.task, "measured" if a.sensor == "real" else a.sensor)
agent_cfg = load_rl_cfg(a.task)
env_cfg.scene.num_envs = 1
if a.episode_s is not None:
    env_cfg.episode_length_s = a.episode_s
# The camera that films this, alongside the one the policy looks through.
# Attached to base_link for the same reason the policy's is: parallel
# environments sit on a grid with real offsets, and a world-fixed camera frames
# environment zero and misses every other one.
VIEW_CAM = "view_cam"
# mjlab requires every camera sensor in a scene to agree on these, and the
# policy's is built with both off because they cost throughput and it reads
# depth and segmentation, neither of which they touch: depth is geometry and
# segmentation is geom identity.  Turning them on for the recording therefore
# changes how the video looks and not one number the policy sees.
for _s in env_cfg.scene.sensors or ():
    if hasattr(_s, "use_textures"):
        _s.use_textures = True
        _s.use_shadows = True
view_pos = tuple(float(x) for x in a.view_pos.split(","))
view_aim = tuple(float(x) for x in a.view_aim.split(","))
env_cfg.scene.sensors = tuple(env_cfg.scene.sensors or ()) + (
    CameraSensorCfg(
        name=VIEW_CAM,
        parent_body=camera.PARENT_BODY,
        pos=view_pos,
        quat=camera.look_at_quat(view_pos, view_aim),
        fovy=a.view_fovy,
        width=a.width,
        height=a.height,
        data_types=("rgb",),
        use_textures=True,
        use_shadows=True,
        enabled_geom_groups=(0, 2),
    ),
)

env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=a.device)
policy = evalcfg.load_policy(runner, a.checkpoint, a.device)
recurrent = bool(getattr(policy, "is_recurrent", False))
if recurrent:
    policy.reset()

LABELS = (
    "1.  depth        black = near,  white = far plane (1.5 m)",
    "2.  target mask   segmentation of the commanded object",
    "3.  masked depth  = 1 x 2",
)
PAD, BAR, SCALE = 16, 24, 2
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)


def _rgb(img: np.ndarray) -> np.ndarray:
    """A single channel to 8-bit RGB, with the far plane reading as black."""
    return np.repeat((255 * np.clip(img, 0.0, 1.0)).astype(np.uint8)[..., None], 3, -1)


def compose(scene: np.ndarray, cam: np.ndarray, step: int, dt: float) -> np.ndarray:
    """The scene above, the three channels the policy reads in a row below.

    Nearest-neighbour upscaling on purpose: the policy sees 168x224 and a
    smooth interpolation would show it a resolution it does not have.
    """
    ch, cw = cam.shape[1:]
    pw, ph = cw * SCALE, ch * SCALE
    strip_h = BAR + ph + PAD
    w = max(scene.shape[1], 3 * pw + 4 * PAD)
    h = scene.shape[0] + strip_h

    out = Image.new("RGB", (w, h), (16, 16, 18))
    out.paste(Image.fromarray(scene), ((w - scene.shape[1]) // 2, 0))
    draw = ImageDraw.Draw(out)

    x0 = (w - (3 * pw + 2 * PAD)) // 2
    for i in range(3):
        x = x0 + i * (pw + PAD)
        draw.text((x, scene.shape[0] + 4), LABELS[i], fill=(200, 200, 205), font=FONT)
        img = np.repeat(np.repeat(_rgb(cam[i]), SCALE, 0), SCALE, 1)
        out.paste(Image.fromarray(img), (x, scene.shape[0] + BAR))
        draw.rectangle([x - 1, scene.shape[0] + BAR - 1, x + pw, scene.shape[0] + BAR + ph],
                       outline=(70, 70, 76))
    draw.text((PAD, 8), f"t = {step * dt:5.2f} s", fill=(230, 230, 235), font=SMALL)
    return np.asarray(out)


frames = []
dt = env.unwrapped.step_dt
obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]
with torch.inference_mode():
    for _ in range(a.steps):
        cam = obs["camera"][0].detach().float().cpu().numpy()
        scene = env.unwrapped.scene.sensors[VIEW_CAM].data.rgb[0].cpu().numpy()
        frames.append(compose(np.ascontiguousarray(scene, dtype=np.uint8), cam, len(frames), dt))
        out = env.step(policy(obs))
        obs, dones = out[0], out[2]
        if recurrent:
            policy.reset(dones)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(a.out, frames, fps=a.fps, quality=8)
print(f"wrote {a.out}: {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}")
