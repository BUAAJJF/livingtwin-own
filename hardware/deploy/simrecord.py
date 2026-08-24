"""Write a session in ``run.py --record`` format, out of the simulator.

There is no robot on this machine and no recordings from one, and the labelling
and training path has to be executable before there are.  So the simulator
produces a session in exactly the format the robot would: synthetic D405 depth
with the measured sensor model applied, the rendered image beside it, and the
joint state that goes with each frame.

Be clear about what this is and is not for.

It **is** for the plumbing: that ``autolabel.py`` writes a dataset YOLO can
read, that the polygons land on the objects, that ``train_yolo.py`` runs to
completion and that the resulting weights load into ``mask.YoloSegmenter`` and
produce instances of the right shape.  All of that is real and all of it is
checked here.

It is **not** a training set for the robot.  The scene renders with
``use_textures=False`` -- flat-shaded geometry, no material, no shadows, room
lighting that does not exist -- so a model fitted to these images has learned
what a MuJoCo render looks like.  The transferable path is the one
``autolabel.py`` describes: record the real table, let the depth segmenter label
the frames it is sure about, and train on those.

    python -m hardware.deploy.simrecord --frames 300 --out recordings/sim
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

import mjlab.tasks  # noqa: F401  -- import mjlab before piper_push
import torch

from piper_push import camera as sim_camera
from piper_push import depth_noise

from . import config, selftest


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--out", default="recordings/sim")
  p.add_argument("--frames", type=int, default=300)
  p.add_argument("--scenes", type=int, default=20,
                 help="how many times to reset, so the objects move; a session "
                      "of one arrangement teaches one arrangement")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--clean", action="store_true",
                 help="skip the sensor model, for comparing against it")
  a = p.parse_args()

  out = pathlib.Path(a.out)
  out.mkdir(parents=True, exist_ok=True)

  # The session carries the calibration it was recorded under.  Labelling
  # happens later -- possibly much later -- and ``autolabel.py`` turns depth
  # into base-frame points to decide what is an object, so a session labelled
  # against whatever ``rig.json`` is on disk that day is labelled against the
  # wrong camera.  A synthetic session's rig is the nominal one, which is
  # exactly right: the stand-in camera is at the nominal pose by construction.
  rig = config.Rig.nominal()
  rig.K = selftest.rectify.mujoco_K(config.D405_WIDTH, config.D405_HEIGHT,
                                    selftest.d405_fovy())
  rig.save(out / "rig.json")

  env = selftest.build_env(a.device)
  cam = env.scene[selftest.D405_CAM]
  corr = None if a.clean else depth_noise.DepthCorruption(
    1, config.D405_HEIGHT, config.D405_WIDTH,
    float(selftest.rectify._default_d405_K()[1, 1]), a.device,
    sim_camera.DEPTH_NOISE)

  action = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
  per_scene = max(1, a.frames // max(a.scenes, 1))
  meta = []
  n = 0
  env.reset()
  while n < a.frames:
    if n % per_scene == 0 and n:
      env.reset()
      if corr is not None:
        corr.reset(None)
    # Random small actions so the arm is somewhere different in each frame and
    # the labels are not all of the same picture.
    action.uniform_(-0.4, 0.4)
    env.step(action)

    depth = cam.data.depth.permute(0, 3, 1, 2)
    if corr is not None:
      d, ok = corr(depth.clamp(config.MIN_DEPTH_M, config.CUTOFF_M))
      depth = torch.where(ok, d, torch.zeros_like(d))
    depth_np = depth[0, 0].cpu().numpy()

    rgb = cam.data.rgb
    if rgb is None:
      raise SystemExit(
        "the camera sensor is not rendering colour.  Add 'rgb' to its "
        "data_types in selftest.d405_camera_cfg()."
      )
    gray = rgb[0, ..., :3].float().mean(-1).clamp(0, 255).to(torch.uint8) \
      .cpu().numpy()

    np.savez_compressed(out / f"{n:06d}.npz",
                        depth=(depth_np * 10000).astype(np.uint16), gray=gray)
    q = env.scene["robot"].data.joint_pos[0].cpu().numpy()
    dq = env.scene["robot"].data.joint_vel[0].cpu().numpy()
    tgt = env.scene["robot"].data.joint_pos_target[0].cpu().numpy()
    meta.append({"i": n, "t": time.time(),
                 "joint_pos": q.tolist(), "joint_vel": dq.tolist(),
                 "target": tgt.tolist(),
                 "action": action[0].cpu().numpy().tolist(), "label": 0})
    n += 1

  (out / "meta.json").write_text(json.dumps(meta))
  print(f"wrote {n} frames to {out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
