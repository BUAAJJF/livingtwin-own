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
    forward = env._decode_action([1.0, 0.0])
    assert no_op["no_op"]
    assert not forward["no_op"]
    assert np.isclose(forward["travel_m"], 0.033)
    assert np.isclose(forward["speed_mps"], 0.350)
