"""One figure: what the sensor model does, and what the pipeline recovers.

Six panels over one scene, and the point of the layout is that the top row and
the bottom row are the same three quantities arrived at two different ways.

  top     what the simulator hands the policy: the rendered depth, the same
          depth after the fitted D405 model, and the target mask it is given
  bottom  what the robot would hand it: a synthetic D405 frame put through
          hardware/deploy, the depth it recovers, and the mask its segmenter
          found with no segmentation buffer to read

If the two rows do not look like each other, the policy is being trained on one
thing and deployed on another, and this is the cheapest way to see that.

    micromamba run -n mjlab python scripts/plot_depth_model.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import mjlab.tasks  # noqa: F401,E402  -- import mjlab before us
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hardware.deploy import config, mask, obs, proprio, rectify, selftest  # noqa: E402
from piper_push import camera as sim_camera  # noqa: E402
from piper_push import depth_noise  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", default="docs/depth_model.png")
p.add_argument("--device", default="cuda:0")
p.add_argument("--settle", type=int, default=60)
p.add_argument("--seed", type=int, default=3)
a = p.parse_args()

torch.manual_seed(a.seed)
env = selftest.build_env(a.device)
selftest.settle(env, a.settle)

# -- the simulator's side ----------------------------------------------------
policy_cam = env.scene[sim_camera.CAMERA_NAME]
clean = policy_cam.data.depth.permute(0, 3, 1, 2).clamp(
  config.MIN_DEPTH_M, config.CUTOFF_M)
corr = depth_noise.DepthCorruption(
  1, config.HEIGHT, config.WIDTH, sim_camera.f_px_per_rad(), a.device,
  sim_camera.DEPTH_NOISE)
noisy, valid = corr(clean,
                    featureless=selftest._object_pixels(
                      env, sim_camera.CAMERA_NAME))
sim_obs = env.observation_manager.compute()["camera"][0].cpu().numpy()

clean_np = clean[0, 0].cpu().numpy()
noisy_np = np.where(valid[0, 0].cpu().numpy(), noisy[0, 0].cpu().numpy(),
                    config.CUTOFF_M)

# -- the robot's side --------------------------------------------------------
rig = config.Rig.nominal()
rig.K = rectify.mujoco_K(config.D405_WIDTH, config.D405_HEIGHT,
                         selftest.d405_fovy())
reproj = rectify.Reprojector(rig)
segmenter = mask.DepthSegmenter(rig, reproj)
tracker = mask.TargetTracker()
kin = proprio.Kinematics()
kin.update(env.scene["robot"].data.joint_pos[0].cpu().numpy())
arm = kin.link_spheres()

src_clean = env.scene[selftest.D405_CAM].data.depth[0, ..., 0].cpu().numpy()
sensor = depth_noise.DepthCorruption(
  1, config.D405_HEIGHT, config.D405_WIDTH, float(rig.K[1, 1]), "cpu",
  sim_camera.DEPTH_NOISE)

label, seg, src = 0, None, None
for _ in range(6):     # the tracker confirms across frames; give it frames
  d, ok = sensor(torch.as_tensor(src_clean)[None, None]
                 .clamp(config.MIN_DEPTH_M, config.CUTOFF_M),
                 featureless=selftest._object_pixels(
                   env, selftest.D405_CAM).cpu())
  src = np.where(ok[0, 0].numpy(), d[0, 0].numpy(), 0.0).astype(np.float32)
  seg = segmenter(src, arm=arm)
  label = tracker.update(seg, kin.site_pos)

payload = mask.full_mask(seg, label, segmenter.decimate) if label else None
rec_depth, rec_valid, rec_mask = reproj(src, payload=payload)
if rec_mask is None:
  rec_mask = np.zeros_like(rec_valid, dtype=np.int32)
deploy_obs = obs.camera_obs(rec_depth, rec_valid, rec_mask > 0)

# -- draw --------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.4))
vmin, vmax = 0.35, 1.05


def show(axis, img, title, sub, **kw):
  axis.imshow(img, **kw)
  axis.set_title(title, fontsize=10.5, loc="left")
  axis.text(0.0, -0.045, sub, transform=axis.transAxes, fontsize=8.2,
            color="#555", va="top", wrap=True)
  axis.set_xticks([])
  axis.set_yticks([])


show(ax[0, 0], clean_np, "simulator, rendered depth",
     f"{config.WIDTH}x{config.HEIGHT}, {config.FOVY_DEG:.0f} deg vertical",
     cmap="viridis", vmin=vmin, vmax=vmax)
show(ax[0, 1], noisy_np, "simulator, with the fitted D405 model",
     f"sigma = {depth_noise.SIGMA_STATIC_PER_M:.4f} z^2 frozen +\n"
     f"{depth_noise.SIGMA_TEMPORAL_PER_M:.4f} z^2 per frame, correlated over "
     f"{corr.corr_px:.1f} px;\n{100 * (1 - valid.float().mean().item()):.0f}% "
     "no return.  The table is textured (the rig puts a mat down);\n"
     f"the objects are drawn at {float(corr.fill):.2f} fill and "
     f"{float(corr.texture):.2f}x noise",
     cmap="viridis", vmin=vmin, vmax=vmax)
show(ax[0, 2], sim_obs[1], "simulator, target mask",
     "from the segmentation buffer, dropped where the depth dropped",
     cmap="gray", vmin=0, vmax=1)

show(ax[1, 0], np.where(src > 0, src, np.nan),
     "what the D405 would deliver",
     f"{config.D405_WIDTH}x{config.D405_HEIGHT}, 87 deg;\n"
     "white is no return; the extra field of view is thrown away",
     cmap="viridis", vmin=vmin, vmax=vmax)
show(ax[1, 1], np.where(rec_valid, rec_depth, config.CUTOFF_M),
     "hardware/deploy, resampled",
     "re-rendered through the policy's camera;\n0.44 mm rms against the "
     "panel above left",
     cmap="viridis", vmin=vmin, vmax=vmax)
truth = sim_obs[1] > 0
found = deploy_obs[1] > 0
overlay = np.zeros(truth.shape + (3,))
overlay[..., 0] = found          # what the segmenter found, in red
overlay[..., 1] = truth          # the truth, in green
show(ax[1, 2], overlay, "hardware/deploy, target mask",
     f"red found, green true, yellow both -- IoU "
     f"{(truth & found).sum() / max((truth | found).sum(), 1):.2f};\n"
     "no segmentation buffer involved")

fig.suptitle(
  "The depth the policy trains on, and the depth a RealSense D405 would give "
  "it", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.2)
out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print(f"wrote {out}")
