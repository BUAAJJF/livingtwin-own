from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from livingtwin_mujoco_rl.config import load_training_config
from livingtwin_mujoco_rl.yaw_absorbing_env import AbsorbingYawPlanarPushEnv
from livingtwin_mujoco_rl.yaw_env import YawPlanarPushEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_training_config(ROOT / "configs/yaw_absorbing_smoke.yaml")


def force_out_of_bounds(env: YawPlanarPushEnv) -> None:
    env.data.qpos[env.cube_qpos_adr] = float(env.config["task"]["cube_x_bounds_m"][1]) + 0.001
    mujoco.mj_forward(env.model, env.data)


def test_original_environment_still_terminates_immediately() -> None:
    env = YawPlanarPushEnv(CONFIG["asset_path"], CONFIG["environment"], seed=1)
    env.reset(seed=2)
    force_out_of_bounds(env)
    _, _, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated and not truncated
    assert info["terminated_reason"] == "out_of_bounds"


def test_absorbing_failure_is_fixed_until_horizon() -> None:
    env = AbsorbingYawPlanarPushEnv(CONFIG["asset_path"], CONFIG["environment"], seed=1)
    env.reset(seed=2)
    force_out_of_bounds(env)
    observation, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert not terminated and not truncated
    assert info["failure"] and info["absorption_step"] == 1
    assert reward == -1.262
    assert np.array_equal(observation, np.zeros(15, dtype=np.float32))
    qpos, mocap = env.data.qpos.copy(), env.data.mocap_pos.copy()
    for _ in range(198):
        observation, reward, terminated, truncated, info = env.step([1.0, -1.0])
        assert not terminated and not truncated
        assert reward == -1.262 and info["action_ignored"]
        assert np.array_equal(observation, np.zeros(15, dtype=np.float32))
    _, reward, terminated, truncated, info = env.step([-1.0, 1.0])
    assert not terminated and truncated
    assert info["step_count"] == 200
    assert np.array_equal(env.data.qpos, qpos)
    assert np.array_equal(env.data.mocap_pos, mocap)


def test_failure_reward_is_below_legal_ordinary_reward_bound() -> None:
    task = CONFIG["environment"]["task"]
    max_dx = max(abs(task["target_x_range_m"][1] - task["cube_x_bounds_m"][0]), abs(task["cube_x_bounds_m"][1] - task["target_x_range_m"][0]))
    max_dy = max(abs(task["target_y_range_m"][1] - task["cube_y_bounds_m"][0]), abs(task["cube_y_bounds_m"][1] - task["target_y_range_m"][0]))
    lower_bound = -2.0 * np.hypot(max_dx, max_dy) - 0.15 * np.pi - 0.002
    assert CONFIG["environment"]["absorbing_failure"]["reward_per_step"] < lower_bound
