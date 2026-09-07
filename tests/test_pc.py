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


# --- second generation: the target channel, the routes, the guards -----------


def test_oracle_flag_labels_only_sampled_target_survivors():
  torch.manual_seed(5)
  b, h, w = 3, 20, 30
  pts = torch.randn(b, h, w, 3)
  inside = torch.zeros(b, h, w, dtype=torch.bool)
  inside[:, 5:15, 5:25] = True
  target = torch.zeros(b, h, w, dtype=torch.bool)
  target[:, 8:12, 8:12] = True          # 16 target pixels, all inside
  target[:, 0:3, 0:3] = True            # 9 target pixels OUTSIDE the workspace: must never be labelled
  tmask = target & inside
  idx, count = cloud.draw_indices(inside, 256)
  out, count2 = cloud.sample_points(pts, inside, 256, augment=False, idx=idx)
  assert torch.equal(count, count2) and (count == 200).all()
  flag = torch.gather(tmask.reshape(b, -1).float(), 1, idx) * out[..., 3]
  # every flagged point is a target pixel that survived the crop, and the draw is shared
  sel_target = torch.gather(target.reshape(b, -1).float(), 1, idx)
  sel_inside = torch.gather(inside.reshape(b, -1).float(), 1, idx)
  assert torch.equal(flag, sel_target * sel_inside)
  assert (sel_inside == 1).all()
  # roughly 16/200 of the draws land on the target
  frac = flag.mean().item()
  assert 0.03 < frac < 0.14, frac
  # the zero channel is exactly zero and does not touch the points
  zero = torch.cat([out, torch.zeros_like(flag).unsqueeze(-1)], -1)
  assert zero.shape == (b, 256, 5) and (zero[..., 4] == 0).all()
  assert torch.equal(zero[..., :4], out)


def test_ring_carries_the_fifth_column_with_the_points():
  torch.manual_seed(4)
  b, n, length = 1, 6, cloud.MAX_LAG + 1
  hist = torch.zeros(length, b, n, 5)
  mhist = torch.zeros(length, b, 3)
  lag = torch.tensor([2])
  write = 0
  seen = []
  for k, f in enumerate([True, False, True, True, False, True, False]):
    new = torch.zeros(b, n, 5)
    new[..., 0] = k + 1
    new[..., 4] = float(k % 2)         # the label changes with the frame
    fresh = torch.tensor([f])
    meta = torch.stack([torch.zeros(b), fresh.float(), torch.ones(b)], -1)
    write, out, _ = cloud.ring_step(hist, mhist, write, new, meta, fresh, lag, first=(k == 0))
    seen.append((float(out[0, 0, 0]), float(out[0, 0, 4])))
  # the label the policy sees always belongs to the frame it sees
  for frame, label in seen:
    if frame > 0:
      assert label == float((int(frame) - 1) % 2), (frame, label)


def test_pointpatch_encoder_reads_per_point_features_and_keeps_old_shapes():
  torch.manual_seed(2)
  old = encoders.PointPatchEncoder(in_dim=4, n_groups=8, group_size=8, out_dim=32)
  assert old.patch[0].weight.shape == (64, 3) and old.feat_dim == 0
  new = encoders.PointPatchEncoder(in_dim=5, n_groups=8, group_size=8, out_dim=32)
  assert new.patch[0].weight.shape == (64, 4) and new.feat_dim == 1
  new.eval()
  x = torch.randn(2, 64, 5)
  x[..., 3] = 1.0
  x[..., 4] = 0.0
  y0 = new(x)
  x2 = x.clone()
  x2[:, :10, 4] = 1.0
  y1 = new(x2)
  assert y0.shape == (2, 32) and not torch.allclose(y0, y1), "the fifth column must reach the output"
  torch.jit.script(new)
  # a 4-column checkpoint loads into a 4-column module built through build_encoder
  spec = {"type": "pointpatch", "out_dim": 32, "n_groups": 8, "group_size": 8, "dim": 128, "n_layers": 2}
  again = encoders.build_encoder(spec, (64, 4))
  again.load_state_dict(old.state_dict())
  wide = encoders.build_encoder(spec, (64, 5))
  assert wide.feat_dim == 1


def test_routes_module_names_the_variants_and_refuses_the_oracle():
  from piper_push.pc import routes
  assert routes.base_route("P1BZ") == "P1B" and routes.base_route("P1BT") == "P1B" and routes.base_route("P2") == "P2"
  assert routes.target_channel("P1B") == "none" and routes.target_channel("P1BZ") == "zero" and routes.target_channel("P1BT") == "oracle"
  routes.check_deployable("P1BZ")
  with pytest.raises(ValueError, match="oracle-only"):
    routes.check_deployable("P1BT")
  with pytest.raises(ValueError):
    routes.check_deployable("P9")


@pytest.mark.parametrize("route,channel", [("P1BZ", "zero"), ("P1BT", "oracle")])
def test_gen2_routes_are_p1b_with_a_target_channel(route, channel):
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  cfg = load_env_cfg(f"Mjlab-Pick-Place-PiperX-PC-{route}-Distill")
  params = cfg.observations["camera"].terms["scene"].params
  assert params["mode"] == "cloud" and params["target_channel"] == channel
  base = load_env_cfg("Mjlab-Pick-Place-PiperX-PC-P1B-Distill").observations["camera"].terms["scene"].params
  for k in ("num_points", "latency_probs", "scenery_dr", "cutoff_distance", "mask_jitter_px", "augment"):
    assert params[k] == base[k], k
  assert params["noise_cfg"] == base["noise_cfg"]
  rl = load_rl_cfg(f"Mjlab-Pick-Place-PiperX-PC-{route}-Vision")
  assert rl.actor.cnn_cfg == load_rl_cfg("Mjlab-Pick-Place-PiperX-PC-P1B-Vision").actor.cnn_cfg
  assert rl.obs_groups["actor"] == ("proprio", "camera", "vision_meta")
  load_env_cfg(f"Mjlab-Pick-Place-PiperX-PC-{route}-Vision-Heldout")


def test_bundle_refuses_an_oracle_route(tmp_path, monkeypatch):
  import json, sys, runpy
  rd = tmp_path / "route"
  rd.mkdir()
  (rd / "manifest.json").write_text(json.dumps({"route": "P1BT", "oracle_only": True}))
  monkeypatch.setattr(sys, "argv", ["bundle.py", str(rd), "--spec", "x.json", "--out", str(tmp_path / "out")])
  with pytest.raises(SystemExit, match="oracle"):
    runpy.run_path("scripts/pc/bundle.py", run_name="__main__")
  assert not (tmp_path / "out").exists()


def test_deployment_pads_the_zero_channel_and_refuses_the_oracle():
  from hardware.deploy.pc_perception import pad_target_channel
  x = np.zeros((512, 4), np.float32)
  assert pad_target_channel(x, "none").shape == (512, 4)
  y = pad_target_channel(x, "zero")
  assert y.shape == (512, 5) and (y[:, 4] == 0).all()
  with pytest.raises(RuntimeError, match="oracle"):
    pad_target_channel(x, "oracle")


# --- the object-astray termination (2026-09-06) ------------------------------


def test_astray_dwell_counts_only_consecutive_violations():
  from piper_push.tasks.pick_place.mdp import astray_step
  c = torch.zeros(3, dtype=torch.long)
  seq = [[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]]     # env 0 stays out 4 steps, env 1 blips, env 2 never
  fired = []
  for v in seq:
    c, done = astray_step(c, torch.tensor(v, dtype=torch.bool), dwell_steps=3)
    fired.append(done.tolist())
  assert fired[2] == [True, False, False] and fired[3] == [True, False, False]
  assert fired[1] == [False, False, False]
  assert c.tolist() == [4, 2, 0]


def test_object_astray_is_a_default_termination_on_the_spawn_sector():
  from mjlab.tasks.registry import load_env_cfg
  from piper_push.tasks.pick_place import env_cfg as T, mdp as M
  for task in ("Mjlab-Pick-Place-PiperX-Robust", "Mjlab-Pick-Place-PiperX-PC-P1BZ-Vision"):
    cfg = load_env_cfg(task, play=True)
    term = cfg.terminations["object_astray"]
    assert term.func is M.ObjectAstray
    assert term.params["radius_range"] == T.SPAWN_RADIUS and term.params["angle_range"] == T.SPAWN_ANGLE
    assert term.params["margin_m"] == T.OBJECT_ASTRAY_MARGIN_M and term.params["dwell_s"] == T.OBJECT_ASTRAY_DWELL_S
    assert not term.time_out
  assert "OBJECT_ASTRAY_TERMINATE" in __import__("piper_push.evalcfg", fromlist=["ENV_KNOBS"]).ENV_KNOBS


def test_p1bz6_is_p1bz_with_a_six_millimetre_cut():
  from mjlab.tasks.registry import load_env_cfg
  from piper_push.pc import routes
  assert routes.crop_z_min("P1BZ6") == 0.006 and routes.crop_z_min("P1BZ") == 0.010 and routes.crop_z_min("P1B") == 0.010
  assert routes.target_channel("P1BZ6") == "zero" and routes.base_route("P1BZ6") == "P1B"
  routes.check_deployable("P1BZ6")
  a = load_env_cfg("Mjlab-Pick-Place-PiperX-PC-P1BZ6-Distill").observations["camera"].terms["scene"].params
  b = load_env_cfg("Mjlab-Pick-Place-PiperX-PC-P1BZ-Distill").observations["camera"].terms["scene"].params
  assert a["workspace"].z_min == 0.006 and b["workspace"].z_min == 0.010
  assert a["workspace"].r_max == b["workspace"].r_max and a["target_channel"] == b["target_channel"] == "zero"


def test_round3_shaping_knobs_are_off_by_default_and_recorded():
  from mjlab.tasks.registry import load_env_cfg
  from piper_push import evalcfg
  cfg = load_env_cfg("Mjlab-Pick-Place-PiperX-PC-P1BZ6-Vision")
  assert "object_pushed" not in cfg.rewards and "approach_speed" not in cfg.rewards
  for k in ("PUSH_PENALTY_W", "APPROACH_SPEED_W", "GRIPPER_KP_SCALE"):
    assert k in evalcfg.ENV_KNOBS


def test_round3_shaping_knobs_wire_the_v10d_terms(monkeypatch):
  import importlib, subprocess, sys, json
  code = """
import os, json
os.environ['PUSH_PENALTY_W'] = '2'; os.environ['APPROACH_SPEED_W'] = '1'; os.environ['GRIPPER_KP_SCALE'] = '1.5'
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
c = load_env_cfg('Mjlab-Pick-Place-PiperX-PC-P1BZ6-Vision')
print(json.dumps({'pushed': c.rewards['object_disturbed'].weight, 'approach': c.rewards['approach_speed'].weight,
                  'kp': list(c.events['robust_gripper_gains'].params['kp_range'])}))
"""
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__("os").environ, "MUJOCO_GL": "disable"})
  line = [l for l in out.stdout.splitlines() if l.startswith("{")][-1]
  d = json.loads(line)
  assert d["pushed"] == -2.0 and d["approach"] == -1.0 and abs(d["kp"][0] - 0.9) < 1e-6 and abs(d["kp"][1] - 2.1) < 1e-6
