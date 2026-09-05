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


# --- v10d: realised-command smoothness, sigma floor, plant, guard -----------------

def _term_env(targets):
  """A fake env whose arm term reports the given realised targets one call at a time."""
  state = {"i": 0}
  term = SimpleNamespace(_previous_target=targets[0].clone())
  env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda n: term), step_dt=0.02)
  def advance():
    state["i"] += 1
    term._previous_target = targets[state["i"]].clone()
  return env, advance


def test_command_acc_and_reversal_charge_changes_of_speed_and_sign_not_speed():
  # joint 0: constant max-speed move (zero acc, no reversal); joint 1: flip-flop
  t = [torch.tensor([[0.0, 0.0]]), torch.tensor([[0.03, 0.03]]), torch.tensor([[0.06, 0.0]]), torch.tensor([[0.09, 0.03]])]
  env, adv = _term_env(t)
  acc = mdp.command_acc(None, env); rev = mdp.command_reversal(None, env)
  assert acc(env).item() == 0.0 and rev(env).item() == 0.0     # first call primes
  adv(); a1 = acc(env).item(); r1 = rev(env).item()             # d = (0.03, 0.03): first delta, prev_d = 0
  adv(); a2 = acc(env).item(); r2 = rev(env).item()             # d = (0.03, -0.03): joint 1 flipped
  adv(); a3 = acc(env).item(); r3 = rev(env).item()             # d = (0.03, +0.03): flipped again
  assert r2 == pytest.approx(0.5) and r3 == pytest.approx(0.5) and r1 == 0.0
  # joint 0 contributes nothing to the acceleration once at constant speed
  acc0 = mdp.command_acc(None, env)
  env0, adv0 = _term_env([torch.tensor([[0.0]]), torch.tensor([[0.03]]), torch.tensor([[0.06]]), torch.tensor([[0.09]])])
  acc0(env0); adv0(); acc0(env0); adv0(); assert acc0(env0).item() == pytest.approx(0.0, abs=1e-9)
  assert a2 > 0 and a3 > 0


def test_sigma_floor_is_per_dimension():
  from piper_push.squashed import PreSquashGaussianDistribution
  d = PreSquashGaussianDistribution(3, init_std=[0.01, 0.5, 0.01], std_range=([0.05, 0.02, 0.2], 2.0))
  d.update(torch.zeros(1, 3))
  assert d.std.squeeze().tolist() == pytest.approx([0.05, 0.5, 0.2])


def test_cold2_plant_exploration_and_terms():
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from piper_push import robot as piper
  from piper_push.tasks.pick_place.rl_cfg import V10D_ENTROPY_COEF, V10D_STD_MIN, BOUNDED_INIT_STD
  for tid in ("Mjlab-Pick-Place-PiperX-Robust-Cold2", "Mjlab-Pick-Place-PiperX-Robust-Cold2-NoSight"):
    cfg = load_env_cfg(tid); rl = load_rl_cfg(tid)
    for j, v in cfg.actions["arm"].velocity_limit.items():
      assert v == pytest.approx(0.35 * piper.JOINT_TRIP_RAD_S[j])
    assert rl.algorithm.entropy_coef == V10D_ENTROPY_COEF == 0.004
    assert rl.actor.distribution_cfg["std_range"][0] == V10D_STD_MIN
    assert all(abs(m - s / 6.0) < 1e-9 for m, s in zip(V10D_STD_MIN[:6], BOUNDED_INIT_STD[:6])) and V10D_STD_MIN[6] == 0.1
    for k in ("command_acc", "command_reversal"):
      assert k in cfg.rewards and cfg.rewards[k].weight == cc.APPROACH_TERMS["reach"][k]
    assert cfg.rewards["joint_acc"].weight == -2e-5 and cfg.rewards["action_rate"].weight == -0.06
    play = load_env_cfg(tid, play=True)
    assert play.actions["arm"].velocity_limit["joint1"] == pytest.approx(0.35 * piper.JOINT_TRIP_RAD_S["joint1"])
  # the v10c tasks and the base task are untouched
  base = load_env_cfg("Mjlab-Pick-Place-PiperX-Robust-Cold"); rl0 = load_rl_cfg("Mjlab-Pick-Place-PiperX-Robust-Cold")
  assert base.actions["arm"].velocity_limit["joint1"] == pytest.approx(0.5 * piper.JOINT_TRIP_RAD_S["joint1"])
  assert rl0.algorithm.entropy_coef == 0.012 and "std_range" not in rl0.actor.distribution_cfg
  assert "command_acc" not in base.rewards and base.rewards["joint_acc"].weight == -2e-7


def test_throughput_guard_holds_penalty_ramp_but_not_dr_or_guidance(monkeypatch, tmp_path):
  monkeypatch.setenv("PIPER_CURRICULUM_LOG", str(tmp_path / "c.jsonl"))
  import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
  from test_cold_curriculum import _fake_env, _step_iterations
  env, rewards, events, terms, cmd = _fake_env()
  for k in ("approach_speed", "object_disturbed", "top_down_grasp", "command_acc", "command_reversal", "joint_acc"):
    rewards[k] = SimpleNamespace(weight=0.0)
  env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: rewards[name], active_terms=list(rewards))
  term = cc.cold_start_curriculum(SimpleNamespace(params={"sight": False, "approach": True}), env)
  assert term.sched["version"] == "v10d-1"
  # to the place stage with 2.0 placed/episode at entry
  _step_iterations(term, env, cmd, 130, attempts=3.0, placed=2.0)   # -> grasp
  _step_iterations(term, env, cmd, 130, attempts=3.0, placed=2.0)   # -> place
  assert term.stage == 2 and term._placed_at_entry == pytest.approx(2.0)
  # throughput collapses right after entry: the penalty ramp freezes where it is
  _step_iterations(term, env, cmd, 30, attempts=3.0, placed=1.0)    # 50% of entry: guard engages
  frozen = rewards["approach_speed"].weight
  _step_iterations(term, env, cmd, 150, attempts=3.0, placed=1.0)
  hook = terms["arm"]._hooks[0]
  assert rewards["approach_speed"].weight == pytest.approx(frozen, abs=1e-9)          # held
  assert abs(frozen) < abs(cc.APPROACH_TERMS["place"]["approach_speed"])              # never reached the place value
  assert rewards["reach"].weight < 1.0                                                # guidance decay continued
  assert hook.cfg.response_range[0] < 1.0                                             # actuator DR continued
  # throughput recovers: the penalty ramp resumes (and the run may go on to the robust stage)
  _step_iterations(term, env, cmd, 250, attempts=3.0, placed=2.0)
  assert abs(rewards["approach_speed"].weight) >= abs(cc.APPROACH_TERMS["place"]["approach_speed"]) - 1e-6
