"""v10d approach terms: slow arrival, undisturbed object, top-down wrist."""

from __future__ import annotations

import math
from types import SimpleNamespace

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest
import torch

from piper_push.tasks.pick_place import cold_curriculum as cc
from piper_push.tasks.pick_place import mdp


def _env(site_pos, site_vel, site_quat, obj_pos, obj_vel, grasped):
  B = site_pos.shape[0]
  robot = SimpleNamespace(data=SimpleNamespace(
    site_pos_w=site_pos.unsqueeze(1), site_lin_vel_w=site_vel.unsqueeze(1), site_quat_w=site_quat.unsqueeze(1)))
  cmd = SimpleNamespace(target_pos_w=lambda: obj_pos, target_lin_vel_w=lambda: obj_vel, grasped=grasped)
  return SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_term=lambda n: cmd))


SITE = SimpleNamespace(name="robot", site_ids=[0])


def test_approach_speed_charges_only_the_excess_near_the_object_and_not_while_held():
  obj = torch.zeros(4, 3)
  d = torch.tensor([0.30, 0.09, 0.03, 0.03])            # far, half way, at the stop, at the stop (held)
  site = torch.stack([torch.zeros(4), torch.zeros(4), d], dim=-1)
  vel = torch.tensor([[0.8, 0, 0], [0.5, 0, 0], [0.4, 0, 0], [0.4, 0, 0]])
  q = torch.tensor([[1.0, 0, 0, 0]] * 4)
  held = torch.tensor([False, False, False, True])
  r = mdp.approach_speed(_env(site, vel, q, obj, torch.zeros(4, 3), held), "pick", SITE)
  assert r[0].item() == pytest.approx(0.2)              # 0.8 - 0.6 allowance far away
  assert r[1].item() == pytest.approx(0.5 - (0.1 + 0.5 * 0.5), abs=1e-6)  # allowance 0.35 half way
  assert r[2].item() == pytest.approx(0.3)              # 0.4 - 0.1 at the stop distance
  assert r[3].item() == 0.0                             # held: no charge


def test_object_disturbed_is_the_free_object_speed_above_a_floor():
  v = torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [0.01, 0, 0], [0.5, 0, 0]])
  held = torch.tensor([False, False, False, True])
  env = _env(torch.zeros(4, 3), torch.zeros(4, 3), torch.tensor([[1.0, 0, 0, 0]] * 4), torch.zeros(4, 3), v, held)
  r = mdp.object_disturbed(env, "pick")
  assert r.tolist() == pytest.approx([0.0, 0.48, 0.0, 0.0])


def test_top_down_grasp_is_one_when_the_approach_axis_points_at_the_table():
  # site local +z = world -z: rotation of pi about x
  down = torch.tensor([math.cos(math.pi / 2), math.sin(math.pi / 2), 0.0, 0.0])
  up = torch.tensor([1.0, 0.0, 0.0, 0.0])
  horiz = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])  # pi/2 about y: z -> x
  q = torch.stack([down, up, horiz, down])
  obj = torch.zeros(4, 3)
  site = torch.tensor([[0, 0, 0.1], [0, 0, 0.1], [0, 0, 0.1], [0, 0, 0.5]])
  held = torch.tensor([False, False, False, False])
  r = mdp.top_down_grasp(_env(site, torch.zeros(4, 3), q, obj, torch.zeros(4, 3), held), "pick", SITE, near_m=0.2)
  assert r.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)   # far away: not paid
  held[3] = True
  r = mdp.top_down_grasp(_env(site, torch.zeros(4, 3), q, obj, torch.zeros(4, 3), held), "pick", SITE, near_m=0.2)
  assert r[3].item() == pytest.approx(1.0)                              # held: paid wherever


def test_cold2_schedule_and_tasks_carry_the_approach_terms():
  s = cc.schedule(True, approach=True)
  assert s["version"] == "v10d-1"
  assert s["stages"][0]["weights"]["approach_speed"] == -0.3 and s["stages"][-1]["weights"]["object_disturbed"] == -2.0
  assert all(st["weights"]["top_down_grasp"] == 0.3 for st in s["stages"])
  assert "approach_speed" not in cc.schedule(True)["stages"][0]["weights"]
  from mjlab.tasks.registry import load_env_cfg
  for tid in ("Mjlab-Pick-Place-PiperX-Robust-Cold2", "Mjlab-Pick-Place-PiperX-Robust-Cold2-NoSight"):
    cfg = load_env_cfg(tid)
    assert cfg.curriculum["cold_start"].params["approach"] is True
    assert cfg.rewards["approach_speed"].weight == -0.3 and cfg.rewards["top_down_grasp"].weight == 0.3
    play = load_env_cfg(tid, play=True)
    assert "approach_speed" in play.rewards and play.rewards["approach_speed"].weight == 0.0
  assert "approach_speed" not in load_env_cfg("Mjlab-Pick-Place-PiperX-Robust-Cold").rewards
