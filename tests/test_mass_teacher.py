"""The mass-conditioned state teacher keeps the oracle actor-only."""

from __future__ import annotations

from types import SimpleNamespace

import mjlab.tasks  # noqa: F401
import pytest
import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from piper_push.tasks.pick_place import mdp
from piper_push.tasks.pick_place.rl_cfg import (
  pick_place_mass_ppo_runner_cfg,
  pick_place_ppo_runner_cfg,
)


def test_mass_teacher_requests_mass_but_baseline_does_not():
  baseline = pick_place_ppo_runner_cfg()
  mass = pick_place_mass_ppo_runner_cfg()
  assert baseline.obs_groups["actor"] == ("proprio", "object")
  assert mass.obs_groups["actor"] == ("proprio", "object", "mass")
  assert mass.obs_groups["critic"] == baseline.obs_groups["critic"]


@pytest.mark.parametrize("task_id", [
  "Mjlab-Pick-Place-PiperX-Robust-Mass",
  "Mjlab-Pick-Place-PiperX-Robust-Cold-Mass",
  "Mjlab-Pick-Place-PiperX-Robust-Cold-Mass-NoSight",
  "Mjlab-Pick-Place-PiperX-Robust-Cold2-Mass",
  "Mjlab-Pick-Place-PiperX-Robust-Cold2-Mass-NoSight",
])
def test_registered_mass_tasks_keep_mass_clean_and_critic_privileged(task_id):
  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)
  assert "mass" in env_cfg.observations
  assert not env_cfg.observations["mass"].enable_corruption
  assert rl_cfg.obs_groups["actor"] == ("proprio", "object", "mass")
  assert rl_cfg.obs_groups["critic"] == ("proprio", "object", "privileged")


def test_object_mass_reads_the_selected_target_body():
  objects = [
    SimpleNamespace(indexing=SimpleNamespace(body_ids=[1])),
    SimpleNamespace(indexing=SimpleNamespace(body_ids=[2])),
  ]
  command = SimpleNamespace(
    _objects=objects,
    target=torch.tensor([0, 1, 1]),
    num_envs=3,
    device=torch.device("cpu"),
  )
  model = SimpleNamespace(body_mass=torch.tensor([
    [0.0, 0.11, 0.22],
    [0.0, 0.33, 0.44],
    [0.0, 0.55, 0.66],
  ]))
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: command),
    sim=SimpleNamespace(model=model),
  )
  assert torch.equal(
    mdp.object_mass(env, "pick"),
    torch.tensor([[0.11], [0.44], [0.66]]),
  )
