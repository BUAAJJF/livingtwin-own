from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from livingtwin_mujoco_rl.yaw_env import YawPlanarPushEnv


class AbsorbingYawPlanarPushEnv(YawPlanarPushEnv):
    """Yaw environment whose safety failure occupies the rest of the fixed horizon."""

    def __init__(
        self,
        asset_path: str | Path,
        config: Mapping[str, Any],
        seed: int,
        physics_override: Mapping[str, float] | None = None,
    ) -> None:
        self.absorbing_failure = False
        self.absorption_step: int | None = None
        super().__init__(asset_path, config, seed, physics_override)

    @property
    def failure_reward(self) -> float:
        return float(self.config["absorbing_failure"]["reward_per_step"])

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.absorbing_failure = False
        self.absorption_step = None
        return super().reset(seed)

    def observation(self) -> np.ndarray:
        if self.absorbing_failure:
            return np.zeros(self.observation_size, dtype=np.float32)
        return super().observation()

    def info(self, reason: str) -> dict[str, Any]:
        result = super().info(reason)
        result.update(
            {
                "failure": bool(self.absorbing_failure),
                "absorbing_failure": bool(self.absorbing_failure),
                "absorption_step": self.absorption_step,
            }
        )
        return result

    def _absorbing_step(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        horizon = self.step_count >= int(self.config["task"]["episode_steps"])
        reason = "absorbing_failure_horizon" if horizon else "absorbing_failure"
        info = self.info(reason)
        info["reward_components"] = {
            "position": 0.0,
            "yaw": 0.0,
            "success": 0.0,
            "action": 0.0,
            "failure": self.failure_reward,
        }
        info["contact_this_step"] = False
        info["executed_normalized_action"] = [0.0, 0.0]
        info["action_ignored"] = True
        return self.observation(), self.failure_reward, False, bool(horizon), info

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.absorbing_failure:
            return self._absorbing_step()
        observation, reward, terminated, truncated, info = super().step(action)
        if info["terminated_reason"] != "out_of_bounds":
            info["failure"] = False
            info["absorbing_failure"] = False
            info["absorption_step"] = None
            info["reward_components"]["failure"] = 0.0
            info["action_ignored"] = False
            return observation, reward, terminated, truncated, info

        self.absorbing_failure = True
        self.absorption_step = int(self.step_count)
        horizon = self.step_count >= int(self.config["task"]["episode_steps"])
        info = self.info("absorbing_failure_horizon" if horizon else "absorbing_failure")
        info["reward_components"] = {
            "position": 0.0,
            "yaw": 0.0,
            "success": 0.0,
            "action": 0.0,
            "failure": self.failure_reward,
        }
        info["contact_this_step"] = False
        info["executed_normalized_action"] = [0.0, 0.0]
        info["action_ignored"] = True
        return self.observation(), self.failure_reward, False, bool(horizon), info


class VectorAbsorbingYawPlanarPushEnv:
    def __init__(self, num_envs: int, asset_path: str | Path, config: Mapping[str, Any], seed: int):
        self.num_envs = int(num_envs)
        self.envs = [
            AbsorbingYawPlanarPushEnv(asset_path, config, seed + 1009 * index)
            for index in range(self.num_envs)
        ]
        self.returns = np.zeros(self.num_envs, dtype=np.float64)
        self.discounted_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.lengths = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        rows = [env.reset(seed + index) for index, env in enumerate(self.envs)]
        self.returns[:] = 0.0
        self.discounted_returns[:] = 0.0
        self.lengths[:] = 0
        return np.stack([row[0] for row in rows]), [row[1] for row in rows]

    def step(
        self, actions: np.ndarray, reset_seed_base: int, gamma: float = 0.99
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, infos = [], []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        for index, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            self.returns[index] += reward
            self.discounted_returns[index] += (float(gamma) ** int(self.lengths[index])) * reward
            self.lengths[index] += 1
            if done:
                info = dict(info)
                info["episode_return"] = float(self.returns[index])
                info["discounted_episode_return_gamma_0_99"] = float(self.discounted_returns[index])
                info["episode_length"] = int(self.lengths[index])
                info["terminal_observation"] = observation.copy()
                observation, info["reset_info"] = env.reset(reset_seed_base + index)
                self.returns[index] = 0.0
                self.discounted_returns[index] = 0.0
                self.lengths[index] = 0
            observations.append(observation)
            rewards[index] = reward
            dones[index] = done
            infos.append(info)
        return np.stack(observations), rewards, dones, infos
