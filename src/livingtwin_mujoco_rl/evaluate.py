from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from livingtwin_mujoco_rl.env import PlanarPushEnv
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd


def evaluate_policy(
    model: ActorCritic,
    normalizer: RunningMeanStd,
    *,
    asset_path: str,
    env_config: Mapping[str, Any],
    seeds: Sequence[int],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    env = PlanarPushEnv(asset_path, env_config, seed=int(seeds[0]))
    rows: list[dict[str, Any]] = []
    model.eval()
    for seed in seeds:
        observation, reset_info = env.reset(seed=int(seed))
        episode_return = 0.0
        action_norms: list[float] = []
        final_info: dict[str, Any] = reset_info
        while True:
            normalized = normalizer.normalize(observation)
            with torch.no_grad():
                action = model.deterministic(
                    torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0)
                )[0].cpu().numpy()
            observation, reward, terminated, truncated, final_info = env.step(action)
            episode_return += float(reward)
            action_norms.append(float(np.linalg.norm(action)))
            if terminated or truncated:
                break
        rows.append(
            {
                "eval_seed": int(seed),
                "success": bool(final_info["success"]),
                "final_position_error_m": float(final_info["distance_m"]),
                "episode_return": float(episode_return),
                "episode_length": int(final_info["step_count"]),
                "contact_count": int(final_info["contact_count"]),
                "mean_action_norm": float(np.mean(action_norms)),
                "terminated_reason": str(final_info["terminated_reason"]),
            }
        )
    metrics = {
        "episode_count": float(len(rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_final_position_error_m": float(
            np.mean([row["final_position_error_m"] for row in rows])
        ),
        "mean_episode_return": float(
            np.mean([row["episode_return"] for row in rows])
        ),
        "mean_episode_length": float(
            np.mean([row["episode_length"] for row in rows])
        ),
    }
    return metrics, rows
