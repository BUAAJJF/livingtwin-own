from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.yaw_absorbing_env import AbsorbingYawPlanarPushEnv


def summarize_absorbing_episodes(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    position = np.asarray([float(row["final_position_error_m"]) for row in rows])
    yaw = np.asarray([float(row["final_yaw_error_rad"]) for row in rows])
    return {
        "episode_count": float(len(rows)),
        "joint_success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "position_success_rate": float(np.mean([bool(row["position_success"]) for row in rows])),
        "yaw_success_rate": float(np.mean([bool(row["yaw_success"]) for row in rows])),
        "failure_rate": float(np.mean([bool(row["failure"]) for row in rows])),
        "mean_absorption_step": float(np.mean([float(row["absorption_step"]) for row in rows if row["failure"]])) if any(row["failure"] for row in rows) else 0.0,
        "mean_final_position_error_m": float(np.mean(position)),
        "median_final_position_error_m": float(np.median(position)),
        "p90_final_position_error_m": float(np.percentile(position, 90)),
        "mean_final_yaw_error_rad": float(np.mean(yaw)),
        "median_final_yaw_error_rad": float(np.median(yaw)),
        "p90_final_yaw_error_rad": float(np.percentile(yaw, 90)),
        "mean_episode_return": float(np.mean([float(row["episode_return"]) for row in rows])),
        "mean_discounted_return_gamma_0_99": float(np.mean([float(row["discounted_return_gamma_0_99"]) for row in rows])),
        "mean_episode_length": float(np.mean([float(row["episode_length"]) for row in rows])),
        "contact_episode_rate": float(np.mean([bool(row["contact"]) for row in rows])),
        "mean_contact_control_step_fraction": float(np.mean([float(row["contact_control_step_fraction"]) for row in rows])),
        "action_saturation_rate": float(np.mean([float(row["action_saturation_rate"]) for row in rows])),
    }


def evaluate_absorbing_policy(
    model: ActorCritic,
    normalizer: RunningMeanStd,
    *,
    asset_path: str,
    env_config: Mapping[str, Any],
    seeds: Sequence[int],
    gamma: float = 0.99,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    env = AbsorbingYawPlanarPushEnv(asset_path, env_config, seed=int(seeds[0]))
    model.eval()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        episode_return = 0.0
        discounted_return = 0.0
        saturated = 0
        action_count = 0
        reward_sums = {"position": 0.0, "yaw": 0.0, "success": 0.0, "action": 0.0, "failure": 0.0}
        final_info: dict[str, Any] = {}
        while True:
            with torch.no_grad():
                action = model.deterministic(
                    torch.as_tensor(normalizer.normalize(observation), dtype=torch.float32).unsqueeze(0)
                )[0].cpu().numpy()
            saturated += int(np.any(np.abs(action) >= 0.95))
            observation, reward, terminated, truncated, final_info = env.step(action)
            discounted_return += (float(gamma) ** action_count) * float(reward)
            episode_return += float(reward)
            action_count += 1
            for name, value in final_info["reward_components"].items():
                reward_sums[name] += float(value)
            if terminated or truncated:
                break
        rows.append(
            {
                "eval_seed": int(seed),
                "success": bool(final_info["success"]),
                "failure": bool(final_info["failure"]),
                "absorption_step": final_info["absorption_step"],
                "position_success": bool(final_info["position_success"]),
                "yaw_success": bool(final_info["yaw_success"]),
                "final_position_error_m": float(final_info["position_error_m"]),
                "final_yaw_error_rad": float(final_info["yaw_error_rad"]),
                "episode_return": episode_return,
                "discounted_return_gamma_0_99": discounted_return,
                "episode_length": int(final_info["step_count"]),
                "contact": bool(final_info["contact_control_steps"] > 0),
                "contact_control_step_fraction": float(final_info["contact_control_steps"] / max(1, final_info["step_count"])),
                "action_saturation_rate": float(saturated / max(1, action_count)),
                "terminated_reason": str(final_info["terminated_reason"]),
                "target_x_m": float(final_info["target_xy"][0]),
                "target_y_m": float(final_info["target_xy"][1]),
                "target_yaw_rad": float(final_info["target_yaw_rad"]),
                "final_cube_yaw_rad": float(final_info["cube_yaw_rad"]),
                **{f"reward_{name}": value for name, value in reward_sums.items()},
            }
        )
    return summarize_absorbing_episodes(rows), rows
