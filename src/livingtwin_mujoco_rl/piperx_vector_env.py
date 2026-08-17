from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from livingtwin_mujoco_rl.piperx_goal_push_env import PiperGoalPushEnv


class VectorPiperGoalPushEnv:
    def __init__(self, num_envs: int, config: Mapping[str, Any], seed: int):
        self.num_envs = int(num_envs)
        self.envs = [PiperGoalPushEnv(config, seed + 1009 * i) for i in range(self.num_envs)]
        self.episode_returns = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        rows = [env.reset(seed + i) for i, env in enumerate(self.envs)]
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
        return np.stack([r[0] for r in rows]), [r[1] for r in rows]

    def step(self, actions: np.ndarray, reset_seed_base: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, rewards, dones, infos = [], np.zeros(self.num_envs, dtype=np.float32), np.zeros(self.num_envs, dtype=bool), []
        for i, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            self.episode_returns[i] += reward; self.episode_lengths[i] += 1
            if done:
                info = dict(info); info["episode_return"] = float(self.episode_returns[i]); info["episode_length"] = int(self.episode_lengths[i]); info["terminal_observation"] = obs.copy()
                obs, reset_info = env.reset(reset_seed_base + i); info["reset_info"] = reset_info
                self.episode_returns[i] = 0.0; self.episode_lengths[i] = 0
            observations.append(obs); rewards[i] = reward; dones[i] = done; infos.append(info)
        return np.stack(observations), rewards, dones, infos
