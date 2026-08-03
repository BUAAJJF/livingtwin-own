from __future__ import annotations

from pathlib import Path

import numpy as np

from livingtwin_mujoco_rl.config import load_training_config
from livingtwin_mujoco_rl.yaw_env import (
    YAW_OBSERVATION_NAMES,
    YawPlanarPushEnv,
    compute_yaw_reward,
    wrap_to_pi,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_training_config(ROOT / "configs" / "yaw_baseline.yaml")


def test_yaw_observation_contract() -> None:
    env = YawPlanarPushEnv(CONFIG["asset_path"], CONFIG["environment"], seed=10)
    observation, info = env.reset(seed=11)
    assert len(YAW_OBSERVATION_NAMES) == 15
    assert observation.shape == (15,)
    assert np.all(np.isfinite(observation))
    assert np.all(np.abs(observation[[6, 7, 13, 14]]) <= 1.0 + 1.0e-7)
    assert -np.pi <= info["delta_yaw_rad"] < np.pi


def test_angle_wrap_boundary_is_continuous() -> None:
    assert np.isclose(wrap_to_pi(np.deg2rad(-179) - np.deg2rad(179)), np.deg2rad(2))
    assert np.isclose(wrap_to_pi(np.deg2rad(179) - np.deg2rad(-179)), np.deg2rad(-2))
    left = np.asarray([np.sin(np.pi - 1e-7), np.cos(np.pi - 1e-7)])
    right = np.asarray([np.sin(-np.pi + 1e-7), np.cos(-np.pi + 1e-7)])
    assert np.linalg.norm(left - right) < 1e-5


def test_yaw_reward_components_are_monotone() -> None:
    reward = CONFIG["environment"]["reward"]
    reference, _ = compute_yaw_reward(0.10, 1.0, [0, 0], success_event=False, reward_config=reward)
    closer, _ = compute_yaw_reward(0.05, 1.0, [0, 0], success_event=False, reward_config=reward)
    aligned, _ = compute_yaw_reward(0.10, 0.2, [0, 0], success_event=False, reward_config=reward)
    both, _ = compute_yaw_reward(0.05, 0.2, [0, 0], success_event=False, reward_config=reward)
    assert closer > reference
    assert aligned > reference
    assert both > closer and both > aligned
