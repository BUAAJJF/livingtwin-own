from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from livingtwin_mujoco_rl.config import load_training_config
from livingtwin_mujoco_rl.env import PlanarPushEnv, compute_reward
from livingtwin_mujoco_rl.networks import ActorCritic


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_training_config(ROOT / "configs" / "smoke.yaml")


def test_reset_observation_and_target_contract() -> None:
    env = PlanarPushEnv(CONFIG["asset_path"], CONFIG["environment"], seed=1)
    observation, info = env.reset(seed=2)
    assert observation.shape == (13,)
    assert np.all(np.isfinite(observation))
    assert info["distance_m"] > CONFIG["environment"]["task"]["success_distance_m"]


def test_reward_is_monotone_with_distance() -> None:
    reward_config = CONFIG["environment"]["reward"]
    far, _ = compute_reward(0.10, [0, 0], success_event=False, reward_config=reward_config)
    near, _ = compute_reward(0.03, [0, 0], success_event=False, reward_config=reward_config)
    success, _ = compute_reward(0.02, [0, 0], success_event=True, reward_config=reward_config)
    assert far < near < success


def test_squashed_action_and_log_probability_are_finite() -> None:
    torch.manual_seed(3)
    model = ActorCritic(13, 2, [128, 128])
    action, pre_tanh, log_probability, value = model.sample(torch.randn(512, 13))
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert torch.all(torch.isfinite(log_probability))
    recomputed, _, recomputed_value = model.evaluate_pre_tanh(torch.randn(512, 13), pre_tanh)
    assert torch.all(torch.isfinite(recomputed))
    assert value.shape == recomputed_value.shape == (512,)
