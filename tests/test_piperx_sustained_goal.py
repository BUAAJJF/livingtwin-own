from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from livingtwin_mujoco_rl.piperx_goal_push_env import PiperGoalPushEnv


ROOT = Path(__file__).resolve().parents[1]


def test_policy_v2_action_decoding() -> None:
    config = yaml.safe_load((ROOT / "configs" / "piperx_goal_push_smoke.yaml").read_text())["environment"]
    env = PiperGoalPushEnv(config, seed=3)
    env.reset(seed=3)
    forward_40 = env._decode_action([0.0, 0.0])
    left_40 = env._decode_action([1.0, 0.0])
    right_40 = env._decode_action([-1.0, 0.0])
    zero = env._decode_action([0.0, -1.0])
    expected = [( -0.875, 0.005), (-0.750, 0.010), (0.0, 0.040), (0.5, 0.060), (1.0, 0.080)]
    forward = env.goal - env._object_xy()
    forward /= np.linalg.norm(forward)
    left = np.asarray([-forward[1], forward[0]])
    assert not forward_40["no_op"]
    assert np.allclose(forward_40["direction"], forward)
    assert np.allclose(left_40["direction"], left, atol=1.0e-7)
    assert np.allclose(right_40["direction"], -left, atol=1.0e-7)
    assert zero["no_op"] and np.isclose(zero["travel_m"], 0.0)
    for action_magnitude, travel in expected:
        decoded = env._decode_action([0.0, action_magnitude])
        assert np.isclose(decoded["travel_m"], travel)
        assert np.allclose(decoded["direction"], forward_40["direction"])
        assert np.isclose(decoded["speed_mps"], 0.350)


def test_policy_v2_reward_ordering() -> None:
    reward = {"log_distance_epsilon_m": 0.005, "step_cost": 0.01, "success_bonus": 1.0, "execution_failure_penalty": 1.0, "unsafe_termination_penalty": 2.0}
    useful = PiperGoalPushEnv._reward_terms(reward, distance_before=0.100, distance_after=0.040, success=False, execution_failure=False, unsafe_termination=False)
    precise = PiperGoalPushEnv._reward_terms(reward, distance_before=0.008, distance_after=0.003, success=True, execution_failure=False, unsafe_termination=False)
    overshoot = PiperGoalPushEnv._reward_terms(reward, distance_before=0.008, distance_after=0.052, success=False, execution_failure=False, unsafe_termination=False)
    failure = PiperGoalPushEnv._reward_terms(reward, distance_before=0.008, distance_after=0.008, success=False, execution_failure=True, unsafe_termination=False)
    unsafe = PiperGoalPushEnv._reward_terms(reward, distance_before=0.008, distance_after=0.008, success=False, execution_failure=False, unsafe_termination=True)
    assert useful["total"] > 0.0
    assert precise["total"] > useful["total"]
    assert overshoot["total"] < 0.0
    assert failure["total"] < -1.0
    assert unsafe["total"] < failure["total"]


def test_operational_target_does_not_terminate_at_25_mm() -> None:
    config = yaml.safe_load((ROOT / "configs" / "piperx_goal_push_smoke.yaml").read_text())["environment"]
    env = PiperGoalPushEnv(config, seed=4)
    env.reset(seed=4)
    env.goal[:] = env._object_xy() + np.asarray([0.010, 0.0])
    _, _, terminated, truncated, info = env.step([0.0, -1.0])
    assert not terminated and not truncated
    assert not info["success"]
    assert np.isclose(info["pre_settled_goal_distance_m"], 0.010)
    assert np.isclose(info["post_settled_goal_distance_m"], 0.010)
    assert info["commanded_after_touch_travel_m"] == 0.0
