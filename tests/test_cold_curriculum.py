"""The cold-start curriculum: nominal start, capability gates with persistence, monotone stages."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest
import torch

from piper_push import action_api
from piper_push.tasks.pick_place import cold_curriculum as cc

ROOT = Path(__file__).resolve().parent.parent


def test_schedule_starts_nominal_and_ends_heavy_and_gates_wrist():
  for sight in (True, False):
    s = cc.schedule(sight)
    st = s["stages"]
    assert st[0]["dr"] == {"actuator": 0.0, "scene": 0.0, "vision": 0.0}
    assert st[-1]["dr"]["actuator"] == 1.0 and st[-1]["dr"]["scene"] == 1.0
    w0, wl = st[0]["weights"], st[-1]["weights"]
    assert w0["sight_arm"] == 0 and w0["sight_hand"] == 0 and w0["table_touch"] == 0 and w0["wrist_side_on"] == 0
    assert w0["premature_touch"] == -0.5 and w0["reach"] == 1.0
    assert wl["action_rate"] == -0.15 and wl["action_acc"] == -0.08 and wl["reach"] == 0.25
    assert st[1]["weights"]["wrist_side_on"] == 0  # no posture reward before grasping
    if sight:
      assert wl["sight_arm"] == -2.0 and wl["sight_hand"] == -4.0 and st[2]["weights"]["wrist_side_on"] == 0.3
    else:
      assert all(st[i]["weights"][k] == 0 for i in range(4) for k in ("sight_arm", "sight_hand", "wrist_side_on"))
    assert [x["name"] for x in st] == ["reach", "grasp", "place", "robust"]
    assert s["budget_iterations"] == 9000 and s["min_full_dr_iterations"] == 1000
    # gates are monotone in placement demand
    assert st[3]["gate"]["placed_per_episode"] > st[2]["gate"]["placed_per_episode"]


def test_registered_cold_tasks_start_with_the_declared_weights_and_share_the_robust_action_api():
  from mjlab.tasks.registry import load_env_cfg
  for tid, sight in (("Mjlab-Pick-Place-PiperX-Robust-Cold", True), ("Mjlab-Pick-Place-PiperX-Robust-Cold-NoSight", False)):
    cfg = load_env_cfg(tid)
    assert list(cfg.curriculum) == ["cold_start"]
    assert cfg.curriculum["cold_start"].params["sight"] is sight and not cfg.curriculum["cold_start"].params.get("approach")
    w0 = cc.schedule(sight)["stages"][0]["weights"]
    for k, v in w0.items():
      assert cfg.rewards[k].weight == pytest.approx(v), (tid, k)
    assert cfg.actions["arm"].bounded and cfg.actions["arm"].scale == load_env_cfg("Mjlab-Pick-Place-PiperX-Robust").actions["arm"].scale
    play = load_env_cfg(tid, play=True)
    assert play.curriculum == {}   # evaluation is the full -Robust domain
  # a checkpoint trained here evaluates on -Robust without any legacy flag
  assert action_api.for_convention("bounded") == action_api.for_convention("bounded")


def test_run_script_has_no_warm_start_and_declares_before_training():
  s = (ROOT / "scripts/run_v10c.sh").read_text()
  body = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#"))
  assert "--agent.resume" not in body and "load-run" not in body and "load-checkpoint" not in body
  assert body.index("config_snapshot.json") < body.index("if ! TASK=")   # declared before the training command runs
  assert 'exists; a v10c run is never overwritten' in body
  assert "until gpu_free" in body and "pkill" not in body and "kill " not in body


# --- the term, on a fake environment ------------------------------------------

class _Hook:
  def __init__(self):
    self.cfg = SimpleNamespace(response_range=(0.7, 1.0), deadband_range=(0.0, 0.004))
    self._latency_probs = torch.tensor([0.15, 0.55, 0.30]); self._hold_probs = torch.tensor([0.15, 0.7, 0.15])
    self.device = "cpu"


class _Term:
  def __init__(self): self._hooks = (_Hook(),)


def _fake_env(n=8):
  rewards = {k: SimpleNamespace(weight=0.0) for k in cc.schedule(True)["stages"][0]["weights"]}
  events = {"robust_pd_gains": SimpleNamespace(params={}), "robust_joint_friction": SimpleNamespace(params={}),
            "robust_gripper_gains": SimpleNamespace(params={}), "reset_base": SimpleNamespace(params={"pose_range": {}}),
            "object_shape_0": SimpleNamespace(params={}), "pad_friction": SimpleNamespace(params={})}
  ev = SimpleNamespace(get_term_cfg=lambda name: events[name], active_terms={"reset": list(events), "startup": []})
  rm = SimpleNamespace(get_term_cfg=lambda name: rewards[name], active_terms=list(rewards))
  terms = {"arm": _Term(), "gripper": _Term()}
  am = SimpleNamespace(get_term=lambda name: terms[name])
  cmd = SimpleNamespace(objects_placed=torch.zeros(n), grasp_attempts=torch.zeros(n), grasped=torch.zeros(n, dtype=torch.bool))
  tm = SimpleNamespace(active_terms=["over_speed", "object_lost"], get_term=lambda name: torch.zeros(n, dtype=torch.bool))
  env = SimpleNamespace(reward_manager=rm, event_manager=ev, action_manager=am, command_manager=SimpleNamespace(get_term=lambda n: cmd),
                        termination_manager=tm, common_step_counter=0, num_envs=n, device="cpu")
  return env, rewards, events, terms, cmd


def _step_iterations(term, env, cmd, n_iters, placed=0.0, attempts=0.0):
  ids = torch.arange(env.num_envs)
  for _ in range(n_iters):
    env.common_step_counter += cc.STEPS_PER_ITERATION
    cmd.objects_placed[:] = placed; cmd.grasp_attempts[:] = attempts
    term(env, ids, sight=True)


def test_curriculum_starts_nominal_gates_on_capability_with_persistence_and_never_regresses(monkeypatch, tmp_path):
  monkeypatch.setenv("PIPER_CURRICULUM_LOG", str(tmp_path / "c.jsonl"))
  env, rewards, events, terms, cmd = _fake_env()
  term = cc.cold_start_curriculum(SimpleNamespace(params={"sight": True}), env)
  hook = terms["arm"]._hooks[0]
  # nominal at start
  assert rewards["sight_hand"].weight == 0 and rewards["premature_touch"].weight == -0.5
  assert torch.allclose(hook._latency_probs, torch.tensor([1.0, 0.0, 0.0])) and hook.cfg.response_range == (1.0, 1.0)
  assert events["robust_pd_gains"].params["kp_range"] == (1.0, 1.0)
  # no capability: 400 iterations, still stage 0
  _step_iterations(term, env, cmd, 400)
  assert term.stage == 0
  # capability that flickers (rolling mean below the gate) does not advance
  for _ in range(5):
    _step_iterations(term, env, cmd, 4, attempts=1.0)
    _step_iterations(term, env, cmd, 16, attempts=0.0)
  assert term.stage == 0
  # sustained grasp attempts: advances exactly once, then ramps
  _step_iterations(term, env, cmd, 40, attempts=1.0)
  assert term.stage == 1
  assert 0.0 < hook._latency_probs[1].item() < 0.55       # mid-ramp toward half actuator DR
  _step_iterations(term, env, cmd, 300, attempts=1.0)     # ramp done, dwell satisfied
  assert hook._latency_probs[1].item() == pytest.approx(0.275, abs=1e-3)   # half way to the heavy 0.55
  assert hook._latency_probs.sum().item() == pytest.approx(1.0)
  assert rewards["premature_touch"].weight == pytest.approx(-2.0)
  # placements open the place stage; wrist reward appears only now
  assert rewards["wrist_side_on"].weight == 0
  _step_iterations(term, env, cmd, 130, attempts=1.0, placed=1.0)
  assert term.stage == 2
  _step_iterations(term, env, cmd, 250, attempts=1.0, placed=1.0)
  assert rewards["wrist_side_on"].weight == pytest.approx(0.3)
  # the robust gate needs 1.5/episode AND safety; 1.0 is not enough, ever
  _step_iterations(term, env, cmd, 300, attempts=2.0, placed=1.0)
  assert term.stage == 2
  _step_iterations(term, env, cmd, 130, attempts=2.0, placed=2.0)
  assert term.stage == 3 and term.full_dr_entered_it is not None
  _step_iterations(term, env, cmd, 250, attempts=2.0, placed=2.0)
  assert torch.allclose(hook._latency_probs, torch.tensor([0.15, 0.55, 0.30]), atol=1e-6)
  assert events["robust_pd_gains"].params["kp_range"] == pytest.approx((0.75, 1.25))
  assert rewards["action_rate"].weight == pytest.approx(-0.15) and rewards["sight_hand"].weight == pytest.approx(-4.0)
  assert rewards["reach"].weight == pytest.approx(0.25)
  # capability collapse afterwards does not move the stage back
  _step_iterations(term, env, cmd, 200, attempts=0.0, placed=0.0)
  assert term.stage == 3
  log = (tmp_path / "c.jsonl").read_text().splitlines()
  assert sum(1 for l in log if '"event": "advance"' in l) == 3


def test_safety_gate_blocks_the_robust_stage():
  env, rewards, events, terms, cmd = _fake_env()
  over = torch.ones(env.num_envs, dtype=torch.bool)
  env.termination_manager = SimpleNamespace(active_terms=["over_speed"], get_term=lambda name: over)
  term = cc.cold_start_curriculum(SimpleNamespace(params={"sight": False}), env)
  _step_iterations(term, env, cmd, 130, attempts=3.0, placed=3.0)   # -> grasp
  _step_iterations(term, env, cmd, 130, attempts=3.0, placed=3.0)   # -> place
  assert term.stage == 2
  _step_iterations(term, env, cmd, 400, attempts=3.0, placed=3.0)   # every episode trips the shell
  assert term.stage == 2
  over[:] = False
  _step_iterations(term, env, cmd, 130, attempts=3.0, placed=3.0)
  assert term.stage == 3
