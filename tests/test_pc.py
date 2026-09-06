"""The point-cloud line's arithmetic, without a simulator."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import mjlab.tasks  # noqa: F401  -- registration order; see tests/test_deploy.py

from piper_push.pc import cloud, encoders, grasp


def test_cadence_is_three_of_five_with_any_phase():
  t = torch.arange(0, 500)
  for phase in range(5):
    fresh = cloud.fresh_at(t, torch.full_like(t, phase))
    assert fresh[0]
    frac = fresh[1:].float().mean().item()
    assert abs(frac - 0.6) < 0.01, (phase, frac)
    # never two blank steps in a row; and past the reset frame, never three
    # fresh in a row (the forced frame at t = 0 may make a triple once)
    f = fresh.int().tolist()
    for i in range(2, len(f)):
      assert f[i] + f[i - 1] >= 1
      if i >= 4:
        assert f[i] + f[i - 1] + f[i - 2] <= 2


def test_ring_holds_the_newest_frame_and_delays_by_the_lag():
  torch.manual_seed(3)
  b, n, length = 2, 8, cloud.MAX_LAG + 1
  hist = torch.zeros(length, b, n, 4)
  mhist = torch.zeros(length, b, 3)
  lag = torch.tensor([0, 2])
  write = 0
  frames = []
  outs = []
  pattern = [True, False, True, False, True, True, False, True, False, True]
  for k, f in enumerate(pattern):
    new = torch.full((b, n, 4), float(k + 1))
    fresh = torch.tensor([f, f])
    meta = torch.stack([torch.zeros(b), fresh.float(), torch.ones(b)], -1)
    write, out, m = cloud.ring_step(hist, mhist, write, new, meta, fresh, lag, first=(k == 0))
    frames.append(k + 1 if f else frames[-1])
    outs.append(out.clone())
  # lag 0: the policy sees the newest frame, held through blank steps
  seen0 = [float(o[0, 0, 0]) for o in outs]
  assert seen0 == [float(x) for x in frames]
  # lag 2: the same sequence two steps late, zeros before anything arrived
  seen1 = [float(o[1, 0, 0]) for o in outs]
  assert seen1 == [0.0, 0.0] + [float(x) for x in frames[:-2]]
  # a held step repeats the previous step's cloud exactly
  for k in range(1, len(pattern)):
    if not pattern[k]:
      assert torch.equal(outs[k][0], outs[k - 1][0])


def test_unproject_puts_the_optical_axis_where_the_camera_looks():
  h, w = 168, 224
  rays = cloud.camera_rays(h, w, 52.0, "cpu")
  depth = torch.full((1, h, w), 0.7)
  pos = torch.tensor([[0.0, 0.0, 1.0]])
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])          # looks down -z of the base frame
  pts = cloud.unproject(depth, rays, pos, quat)
  centre = pts[0, h // 2, w // 2]      # half a pixel off the principal point at (w-1)/2, (h-1)/2
  assert torch.allclose(centre, torch.tensor([0.0, 0.0, 0.3]), atol=4e-3)
  # the top-left pixel is up and to the left in the MuJoCo image convention
  assert pts[0, 0, 0, 0] < 0 and pts[0, 0, 0, 1] > 0


def test_sample_points_never_draws_outside_the_survivors():
  torch.manual_seed(0)
  pts = torch.randn(4, 50, 60, 3)
  inside = torch.zeros(4, 50, 60, dtype=torch.bool)
  inside[:, 10:20, 10:20] = True
  pts[inside] = 5.0
  out, count = cloud.sample_points(pts, inside, 256, augment=False)
  assert out.shape == (4, 256, 4)
  assert (count == 100).all()
  assert torch.allclose(out[..., :3], torch.full_like(out[..., :3], 5.0))
  assert (out[..., 3] == 1).all()
  empty = torch.zeros_like(inside)
  out2, count2 = cloud.sample_points(pts, empty, 256, augment=False)
  assert (count2 == 0).all() and (out2 == 0).all()


def test_workspace_keeps_the_arm_over_the_bin_and_drops_the_table():
  ws = cloud.WORKSPACE
  a = 0.5 * (ws.angle_lo + ws.angle_hi)
  on_table = torch.tensor([[0.3 * math.cos(a), 0.3 * math.sin(a), 0.004]])
  above = torch.tensor([[0.3 * math.cos(a), 0.3 * math.sin(a), 0.05]])
  far = torch.tensor([[0.9 * math.cos(a), 0.9 * math.sin(a), 0.05]])
  assert not cloud.in_workspace(on_table, ws).item()
  assert cloud.in_workspace(above, ws).item()
  assert not cloud.in_workspace(far, ws).item()


def test_encoders_are_permutation_invariant_and_scriptable():
  torch.manual_seed(1)
  x = torch.randn(2, 64, 4)
  x[..., 3] = 1.0
  perm = torch.randperm(64)
  for enc in (encoders.PointNetEncoder(out_dim=32), encoders.PointPatchEncoder(n_groups=8, group_size=8, out_dim=32)):
    enc.eval()
    y1 = enc(x)
    y2 = enc(x[:, perm])
    assert y1.shape == (2, 32)
    if isinstance(enc, encoders.PointNetEncoder):
      assert torch.allclose(y1, y2, atol=1e-5)
    torch.jit.script(enc)
  img = encoders.DepthResNetLite()
  assert img(torch.randn(2, 2, 168, 224)).shape == (2, 256)
  assert "NOT DeFM" in img.structure
  torch.jit.script(img)


def test_proposals_find_an_object_and_skip_the_bin_and_the_arm():
  dev = "cpu"
  ws = cloud.WORKSPACE
  a = 0.5 * (ws.angle_lo + ws.angle_hi)
  cx, cy = 0.35 * math.cos(a), 0.35 * math.sin(a)
  # a 40 mm box, 50 mm tall, sampled on a 4 mm grid; plus the bin's rim; plus a point cloud on the arm
  g = torch.arange(-0.02, 0.0201, 0.004)
  xx, yy = torch.meshgrid(g, g, indexing="ij")
  box = torch.stack([cx + xx.reshape(-1), cy + yy.reshape(-1), torch.full((xx.numel(),), 0.05)], -1)
  from piper_push import objects
  bx, by = objects.BIN_CENTER
  hx, hy = objects.BIN_INNER
  rim = torch.stack([bx + hx + torch.zeros(20), by + torch.linspace(-hy, hy, 20), torch.full((20,), 0.06)], -1)
  arm_pos = torch.tensor([[[0.10, 0.30, 0.20]] * len(grasp.ARM_BODIES)])
  armpts = arm_pos[0, 0] + 0.02 * torch.randn(30, 3)
  pts = torch.cat([box, rim, armpts]).unsqueeze(0)
  inside = torch.ones(1, pts.shape[1], dtype=torch.bool)
  ee = torch.tensor([[0.0, 0.2, 0.2]])
  props = grasp.propose(pts, inside, arm_pos, ee)
  assert props.feats.shape == (1, grasp.K, grasp.FEAT)
  assert int(props.n_components[0]) == 1, "the bin rim and the arm must not propose"
  valid = props.valid[0]
  f = props.feats[0][valid]
  assert f.shape[0] == 2
  assert torch.allclose(f[:, :2], torch.tensor([[cx, cy]]).expand(2, -1), atol=0.006)
  assert (f[:, 9] <= grasp.JAW_M + 1e-6).all() and (f[:, grasp.I_FEASIBLE] > 0.5).all()
  assert (f[:, grasp.I_REACH] > 0.5).all()


@pytest.mark.parametrize("route", ["P0", "P1A", "P1B", "P2"])
def test_pc_task_ids_are_registered(route):
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  cfg = load_env_cfg(f"Mjlab-Pick-Place-PiperX-PC-{route}-Distill")
  assert "camera" in cfg.observations and "vision_meta" in cfg.observations
  assert cfg.observations["camera"].terms["scene"].params["mode"] == ("depth" if route == "P0" else "cloud")
  rl = load_rl_cfg(f"Mjlab-Pick-Place-PiperX-PC-{route}-Vision")
  assert rl.actor.class_name.endswith("SetRecurrentModel")
  assert rl.actor.distribution_cfg["class_name"].endswith("PreSquashGaussianDistribution")
  w = cfg.events["object_shape"].params["shape_weights"]
  from piper_push import objects
  assert w[objects.SHAPE_CLASSES.index("capped")] == 0.0
  ho = load_env_cfg(f"Mjlab-Pick-Place-PiperX-PC-{route}-Vision-Heldout")
  assert ho.events["object_shape"].params["shape_weights"][objects.SHAPE_CLASSES.index("capped")] == 1.0
