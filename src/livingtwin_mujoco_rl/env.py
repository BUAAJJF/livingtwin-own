from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np


OBSERVATION_NAMES = (
    "pusher_x",
    "pusher_y",
    "pusher_vx",
    "pusher_vy",
    "cube_x",
    "cube_y",
    "sin_cube_yaw",
    "cos_cube_yaw",
    "cube_vx",
    "cube_vy",
    "cube_omega_z",
    "target_minus_cube_x",
    "target_minus_cube_y",
)


@dataclass(frozen=True)
class EpisodeState:
    seed: int
    initial_cube_xy: tuple[float, float]
    target_xy: tuple[float, float]
    initial_distance_m: float


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    return np.asarray(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        dtype=np.float64,
    )


def quaternion_to_yaw(quaternion_wxyz: Sequence[float]) -> float:
    w, x, y, z = map(float, quaternion_wxyz)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def compute_reward(
    distance_m: float,
    normalized_action: Sequence[float],
    *,
    success_event: bool,
    reward_config: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    action = np.clip(np.asarray(normalized_action, dtype=np.float64), -1.0, 1.0)
    components = {
        "distance": -float(reward_config["distance_scale"]) * float(distance_m),
        "success": float(reward_config["success_bonus"]) * float(success_event),
        "action": -float(reward_config["action_penalty"]) * float(np.sum(action**2)),
    }
    return float(sum(components.values())), components


class PlanarPushEnv:
    observation_size = len(OBSERVATION_NAMES)
    action_size = 2

    def __init__(self, asset_path: str | Path, config: Mapping[str, Any], seed: int):
        self.asset_path = str(Path(asset_path).resolve())
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(self.asset_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(int(seed))
        self.cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
        )
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_dof_adr = int(self.model.jnt_dofadr[self.cube_joint_id])
        self.pusher_mocap_id = int(
            self.model.body_mocapid[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher")
            ]
        )
        self.target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )
        self.cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        self.pusher_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "pusher_geom"
        )
        self.target = np.zeros(2, dtype=np.float64)
        self.pusher_velocity = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self.success_count = 0
        self.contact_count = 0
        self.last_episode_state: EpisodeState | None = None

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep) * int(self.config["physics"]["frame_skip"])

    def _cube_xy(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 2].copy()

    def _distance(self) -> float:
        return float(np.linalg.norm(self.target - self._cube_xy()))

    def _has_cube_pusher_contact(self) -> bool:
        pair = {self.cube_geom_id, self.pusher_geom_id}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if {int(contact.geom1), int(contact.geom2)} == pair:
                return True
        return False

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        task = self.config["task"]
        mujoco.mj_resetData(self.model, self.data)
        cube_x = float(self.rng.uniform(*task["cube_initial_x_range_m"]))
        cube_y = float(self.rng.uniform(*task["cube_initial_y_range_m"]))
        cube_yaw = float(self.rng.uniform(*task["cube_initial_yaw_range_rad"]))
        target_x = float(self.rng.uniform(*task["target_x_range_m"]))
        target_y = float(self.rng.uniform(*task["target_y_range_m"]))
        self.target[:] = (target_x, target_y)

        qpos = self.data.qpos
        qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = (cube_x, cube_y, 0.025)
        qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = yaw_to_quaternion(
            cube_yaw
        )
        self.data.qvel[self.cube_dof_adr : self.cube_dof_adr + 6] = 0.0

        pusher_y = cube_y + float(
            self.rng.uniform(
                -float(task["pusher_initial_y_jitter_m"]),
                float(task["pusher_initial_y_jitter_m"]),
            )
        )
        self.data.mocap_pos[self.pusher_mocap_id] = (
            cube_x + float(task["pusher_offset_x_m"]),
            pusher_y,
            0.026,
        )
        self.data.mocap_quat[self.pusher_mocap_id] = (1.0, 0.0, 0.0, 0.0)
        self.model.site_pos[self.target_site_id, :2] = self.target
        self.pusher_velocity[:] = 0.0
        self.step_count = 0
        self.success_count = 0
        self.contact_count = 0
        mujoco.mj_forward(self.model, self.data)
        initial_distance = self._distance()
        state_seed = int(seed if seed is not None else -1)
        self.last_episode_state = EpisodeState(
            seed=state_seed,
            initial_cube_xy=(cube_x, cube_y),
            target_xy=(target_x, target_y),
            initial_distance_m=initial_distance,
        )
        return self.observation(), self.info(success=False, terminated_reason="reset")

    def observation(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        cube_xy = qpos[self.cube_qpos_adr : self.cube_qpos_adr + 2]
        cube_quat = qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7]
        cube_velocity = qvel[self.cube_dof_adr : self.cube_dof_adr + 2]
        cube_omega_z = qvel[self.cube_dof_adr + 5]
        pusher_xy = self.data.mocap_pos[self.pusher_mocap_id, :2]
        yaw = quaternion_to_yaw(cube_quat)
        return np.asarray(
            [
                pusher_xy[0],
                pusher_xy[1],
                self.pusher_velocity[0],
                self.pusher_velocity[1],
                cube_xy[0],
                cube_xy[1],
                math.sin(yaw),
                math.cos(yaw),
                cube_velocity[0],
                cube_velocity[1],
                cube_omega_z,
                self.target[0] - cube_xy[0],
                self.target[1] - cube_xy[1],
            ],
            dtype=np.float32,
        )

    def info(self, *, success: bool, terminated_reason: str) -> dict[str, Any]:
        return {
            "success": bool(success),
            "distance_m": self._distance(),
            "cube_xy": self._cube_xy().tolist(),
            "target_xy": self.target.tolist(),
            "step_count": int(self.step_count),
            "success_hold_count": int(self.success_count),
            "contact_count": int(self.contact_count),
            "terminated_reason": str(terminated_reason),
        }

    def step(
        self, action: Sequence[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        task = self.config["task"]
        normalized_action = np.clip(
            np.asarray(action, dtype=np.float64), -1.0, 1.0
        )
        command_velocity = normalized_action * float(
            self.config["action"]["maximum_pusher_speed_mps"]
        )
        frame_skip = int(self.config["physics"]["frame_skip"])
        substep_dt = float(self.model.opt.timestep)
        start_xy = self.data.mocap_pos[self.pusher_mocap_id, :2].copy()
        bounds_x = task["pusher_x_bounds_m"]
        bounds_y = task["pusher_y_bounds_m"]
        for _ in range(frame_skip):
            next_xy = self.data.mocap_pos[self.pusher_mocap_id, :2] + (
                command_velocity * substep_dt
            )
            next_xy[0] = np.clip(next_xy[0], *bounds_x)
            next_xy[1] = np.clip(next_xy[1], *bounds_y)
            self.data.mocap_pos[self.pusher_mocap_id, :2] = next_xy
            mujoco.mj_step(self.model, self.data)
            if self._has_cube_pusher_contact():
                self.contact_count += 1
        self.pusher_velocity[:] = (
            self.data.mocap_pos[self.pusher_mocap_id, :2] - start_xy
        ) / self.control_dt
        self.step_count += 1

        distance = self._distance()
        inside = distance <= float(task["success_distance_m"])
        self.success_count = self.success_count + 1 if inside else 0
        success = self.success_count >= int(task["success_hold_steps"])
        cube_xy = self._cube_xy()
        out_of_bounds = bool(
            not (float(task["cube_x_bounds_m"][0]) <= cube_xy[0] <= float(task["cube_x_bounds_m"][1]))
            or not (float(task["cube_y_bounds_m"][0]) <= cube_xy[1] <= float(task["cube_y_bounds_m"][1]))
        )
        timeout = self.step_count >= int(task["episode_steps"])
        terminated = bool(success or out_of_bounds)
        truncated = bool(timeout and not terminated)
        reason = "success" if success else "out_of_bounds" if out_of_bounds else "timeout" if timeout else "running"
        reward, components = compute_reward(
            distance,
            normalized_action,
            success_event=success,
            reward_config=self.config["reward"],
        )
        info = self.info(success=success, terminated_reason=reason)
        info["reward_components"] = components
        info["executed_normalized_action"] = normalized_action.tolist()
        info["executed_velocity_mps"] = self.pusher_velocity.tolist()
        return self.observation(), reward, terminated, truncated, info


class VectorPlanarPushEnv:
    def __init__(
        self,
        num_envs: int,
        asset_path: str | Path,
        config: Mapping[str, Any],
        seed: int,
    ):
        self.num_envs = int(num_envs)
        self.envs = [
            PlanarPushEnv(asset_path, config, seed=seed + 1009 * index)
            for index in range(self.num_envs)
        ]
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        observations, infos = [], []
        for index, env in enumerate(self.envs):
            observation, info = env.reset(seed=int(seed) + index)
            observations.append(observation)
            infos.append(info)
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        return np.stack(observations), infos

    def step(
        self, actions: np.ndarray, reset_seed_base: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations: list[np.ndarray] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        for index, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            self._episode_returns[index] += reward
            self._episode_lengths[index] += 1
            if done:
                info = dict(info)
                info["episode_return"] = float(self._episode_returns[index])
                info["episode_length"] = int(self._episode_lengths[index])
                info["terminal_observation"] = obs.copy()
                obs, reset_info = env.reset(
                    seed=int(reset_seed_base) + index
                )
                info["reset_info"] = reset_info
                self._episode_returns[index] = 0.0
                self._episode_lengths[index] = 0
            observations.append(obs)
            rewards[index] = reward
            dones[index] = done
            infos.append(info)
        return np.stack(observations), rewards, dones, infos
