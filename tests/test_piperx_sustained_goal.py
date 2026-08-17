from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from livingtwin_mujoco_rl.piperx_goal_push_env import PiperGoalPushEnv


ROOT = Path(__file__).resolve().parents[1]


def test_sustained_goal_action_decoding() -> None:
    config = yaml.safe_load((ROOT / "configs" / "piperx_goal_push_smoke.yaml").read_text())["environment"]
    env = PiperGoalPushEnv(config, seed=3)
    env.reset(seed=3)
    no_op = env._decode_action([0.0, 0.0])
    small = env._decode_action([1.0e-4, 0.0])
    forward = env._decode_action([1.0, 0.0])
    assert no_op["no_op"]
    assert not small["no_op"]
    assert np.isclose(small["travel_m"], 3.3e-6)
    assert not forward["no_op"]
    assert np.isclose(forward["travel_m"], 0.033)
    assert np.isclose(forward["speed_mps"], 0.350)


def test_operational_target_does_not_terminate_at_25_mm() -> None:
    config = yaml.safe_load((ROOT / "configs" / "piperx_goal_push_smoke.yaml").read_text())["environment"]
    env = PiperGoalPushEnv(config, seed=4)
    env.reset(seed=4)
    env.goal[:] = env._object_xy() + np.asarray([0.010, 0.0])
    _, _, terminated, truncated, info = env.step([0.0, 0.0])
    assert not terminated and not truncated
    assert not info["success"]
    assert np.isclose(info["pre_settled_goal_distance_m"], 0.010)
    assert np.isclose(info["post_settled_goal_distance_m"], 0.010)
    assert info["commanded_after_touch_travel_m"] == 0.0
