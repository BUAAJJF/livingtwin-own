from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from livingtwin_mujoco_rl.env import quaternion_to_yaw, yaw_to_quaternion


YAW_OBSERVATION_NAMES = (
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
    "sin_delta_yaw",
    "cos_delta_yaw",
)


def wrap_to_pi(value: float | np.ndarray) -> float | np.ndarray:
    result = (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi
    return float(result) if result.ndim == 0 else result


def compute_yaw_reward(
    position_error_m: float,
    yaw_error_rad: float,
    normalized_action: Sequence[float],
    *,
    success_event: bool,
    reward_config: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    action = np.clip(np.asarray(normalized_action, dtype=np.float64), -1.0, 1.0)
    components = {
        "position": -float(reward_config["position_scale"]) * float(position_error_m),
        "yaw": -float(reward_config["yaw_scale"]) * float(yaw_error_rad),
        "success": float(reward_config["success_bonus"]) * float(success_event),
        "action": -float(reward_config["action_penalty"]) * float(np.sum(action**2)),
    }
    return float(sum(components.values())), components


class YawPlanarPushEnv:
    observation_size = len(YAW_OBSERVATION_NAMES)
    action_size = 2

    def __init__(
        self,
        asset_path: str | Path,
        config: Mapping[str, Any],
        seed: int,
        physics_override: Mapping[str, float] | None = None,
    ) -> None:
        self.asset_path = str(Path(asset_path).resolve())
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(self.asset_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(int(seed))
        self.cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_dof_adr = int(self.model.jnt_dofadr[self.cube_joint_id])
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.pusher_mocap_id = int(
            self.model.body_mocapid[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher")
            ]
        )
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.pusher_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pusher_geom")
        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.target = np.zeros(3, dtype=np.float64)
        self.pusher_velocity = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self.success_hold_count = 0
        self.episode_success = False
        self.contact_substeps = 0
        self.contact_control_steps = 0
        self.physics_parameters = self._apply_physics(physics_override or {})

    def _apply_physics(self, override: Mapping[str, float]) -> dict[str, float]:
        nominal = dict(self.config["nominal_physics"])
        parameters = {**nominal, **{key: float(value) for key, value in override.items()}}
        mass = float(parameters["mass_kg"])
        nominal_mass = float(nominal["mass_kg"])
        self.model.body_mass[self.cube_body_id] = mass
        self.model.body_inertia[self.cube_body_id] *= mass / nominal_mass
        self.model.body_ipos[self.cube_body_id, :3] = (
            float(parameters["com_x_m"]),
            float(parameters["com_y_m"]),
            0.0,
        )
        friction = np.asarray(
            [
                parameters["sliding_friction"],
                parameters["torsional_friction"],
                parameters["rolling_friction"],
            ],
            dtype=np.float64,
        )
        for geom_id in (self.table_geom_id, self.cube_geom_id, self.pusher_geom_id):
            self.model.geom_friction[geom_id] = friction
        mujoco.mj_setConst(self.model, self.data)
        return {key: float(value) for key, value in parameters.items()}

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep) * int(self.config["physics"]["frame_skip"])

    def cube_xy(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 2].copy()

    def cube_yaw(self) -> float:
        return quaternion_to_yaw(self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7])

    def errors(self) -> tuple[float, float, float]:
        position_error = float(np.linalg.norm(self.target[:2] - self.cube_xy()))
        delta_yaw = float(wrap_to_pi(float(self.target[2]) - self.cube_yaw()))
        return position_error, abs(delta_yaw), delta_yaw

    def _has_contact(self) -> bool:
        pair = {self.cube_geom_id, self.pusher_geom_id}
        return any(
            {int(self.data.contact[index].geom1), int(self.data.contact[index].geom2)} == pair
            for index in range(self.data.ncon)
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        task = self.config["task"]
        mujoco.mj_resetData(self.model, self.data)
        cube_x = float(self.rng.uniform(*task["cube_initial_x_range_m"]))
        cube_y = float(self.rng.uniform(*task["cube_initial_y_range_m"]))
        cube_yaw = float(self.rng.uniform(*task["cube_initial_yaw_range_rad"]))
        self.target[:] = (
            float(self.rng.uniform(*task["target_x_range_m"])),
            float(self.rng.uniform(*task["target_y_range_m"])),
            float(self.rng.uniform(*task["target_yaw_range_rad"])),
        )
        self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = (cube_x, cube_y, 0.025)
        self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = yaw_to_quaternion(cube_yaw)
        self.data.qvel[self.cube_dof_adr : self.cube_dof_adr + 6] = 0.0
        pusher_y = cube_y + float(
            self.rng.uniform(-float(task["pusher_initial_y_jitter_m"]), float(task["pusher_initial_y_jitter_m"]))
        )
        self.data.mocap_pos[self.pusher_mocap_id] = (
            cube_x + float(task["pusher_offset_x_m"]), pusher_y, 0.026
        )
        self.data.mocap_quat[self.pusher_mocap_id] = (1.0, 0.0, 0.0, 0.0)
        self.model.site_pos[self.target_site_id, :2] = self.target[:2]
        self.model.site_quat[self.target_site_id] = yaw_to_quaternion(float(self.target[2]))
        self.pusher_velocity[:] = 0.0
        self.step_count = 0
        self.success_hold_count = 0
        self.episode_success = False
        self.contact_substeps = 0
        self.contact_control_steps = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observation(), self.info("reset")

    def observation(self) -> np.ndarray:
        cube_xy = self.cube_xy()
        yaw = self.cube_yaw()
        delta = float(wrap_to_pi(float(self.target[2]) - yaw))
        qvel = self.data.qvel
        pusher_xy = self.data.mocap_pos[self.pusher_mocap_id, :2]
        return np.asarray(
            [
                pusher_xy[0], pusher_xy[1], self.pusher_velocity[0], self.pusher_velocity[1],
                cube_xy[0], cube_xy[1], math.sin(yaw), math.cos(yaw),
                qvel[self.cube_dof_adr], qvel[self.cube_dof_adr + 1], qvel[self.cube_dof_adr + 5],
                self.target[0] - cube_xy[0], self.target[1] - cube_xy[1],
                math.sin(delta), math.cos(delta),
            ],
            dtype=np.float32,
        )

    def info(self, reason: str) -> dict[str, Any]:
        position_error, yaw_error, delta_yaw = self.errors()
        instantaneous = bool(
            position_error <= float(self.config["task"]["success_position_m"])
            and yaw_error <= float(self.config["task"]["success_yaw_rad"])
        )
        return {
            "success": bool(self.episode_success),
            "instantaneous_success": instantaneous,
            "sustained_success": bool(self.episode_success),
            "position_success": position_error <= float(self.config["task"]["success_position_m"]),
            "yaw_success": yaw_error <= float(self.config["task"]["success_yaw_rad"]),
            "position_error_m": position_error,
            "yaw_error_rad": yaw_error,
            "delta_yaw_rad": delta_yaw,
            "cube_xy": self.cube_xy().tolist(),
            "cube_yaw_rad": self.cube_yaw(),
            "target_xy": self.target[:2].tolist(),
            "target_yaw_rad": float(self.target[2]),
            "step_count": int(self.step_count),
            "success_hold_count": int(self.success_hold_count),
            "contact_substeps": int(self.contact_substeps),
            "contact_control_steps": int(self.contact_control_steps),
            "terminated_reason": reason,
            "physics_parameters": dict(self.physics_parameters),
        }

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        task = self.config["task"]
        normalized_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        command_velocity = normalized_action * float(self.config["action"]["maximum_pusher_speed_mps"])
        start_xy = self.data.mocap_pos[self.pusher_mocap_id, :2].copy()
        contacted = False
        for _ in range(int(self.config["physics"]["frame_skip"])):
            next_xy = self.data.mocap_pos[self.pusher_mocap_id, :2] + command_velocity * float(self.model.opt.timestep)
            next_xy[0] = np.clip(next_xy[0], *task["pusher_x_bounds_m"])
            next_xy[1] = np.clip(next_xy[1], *task["pusher_y_bounds_m"])
            self.data.mocap_pos[self.pusher_mocap_id, :2] = next_xy
            mujoco.mj_step(self.model, self.data)
            if self._has_contact():
                self.contact_substeps += 1
                contacted = True
        if contacted:
            self.contact_control_steps += 1
        self.pusher_velocity[:] = (
            self.data.mocap_pos[self.pusher_mocap_id, :2] - start_xy
        ) / self.control_dt
        self.step_count += 1
        position_error, yaw_error, _ = self.errors()
        instantaneous = bool(
            position_error <= float(task["success_position_m"])
            and yaw_error <= float(task["success_yaw_rad"])
        )
        self.success_hold_count = self.success_hold_count + 1 if instantaneous else 0
        first_success = bool(
            not self.episode_success
            and self.success_hold_count >= int(task["success_hold_steps"])
        )
        if first_success:
            self.episode_success = True
        cube_xy = self.cube_xy()
        out_of_bounds = bool(
            not (float(task["cube_x_bounds_m"][0]) <= cube_xy[0] <= float(task["cube_x_bounds_m"][1]))
            or not (float(task["cube_y_bounds_m"][0]) <= cube_xy[1] <= float(task["cube_y_bounds_m"][1]))
        )
        timeout = self.step_count >= int(task["episode_steps"])
        terminated = bool(self.episode_success or out_of_bounds)
        truncated = bool(timeout and not terminated)
        reason = "success" if self.episode_success else "out_of_bounds" if out_of_bounds else "timeout" if timeout else "running"
        reward, components = compute_yaw_reward(
            position_error, yaw_error, normalized_action,
            success_event=first_success, reward_config=self.config["reward"],
        )
        info = self.info(reason)
        info["contact_this_step"] = bool(contacted)
        info["reward_components"] = components
        info["executed_normalized_action"] = normalized_action.tolist()
        return self.observation(), reward, terminated, truncated, info


class VectorYawPlanarPushEnv:
    def __init__(self, num_envs: int, asset_path: str | Path, config: Mapping[str, Any], seed: int):
        self.num_envs = int(num_envs)
        self.envs = [YawPlanarPushEnv(asset_path, config, seed + 1009 * index) for index in range(self.num_envs)]
        self.returns = np.zeros(self.num_envs, dtype=np.float64)
        self.lengths = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        rows = [env.reset(seed + index) for index, env in enumerate(self.envs)]
        self.returns[:] = 0.0
        self.lengths[:] = 0
        return np.stack([row[0] for row in rows]), [row[1] for row in rows]

    def step(self, actions: np.ndarray, reset_seed_base: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, infos = [], []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        for index, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            self.returns[index] += reward
            self.lengths[index] += 1
            if done:
                info = dict(info)
                info["episode_return"] = float(self.returns[index])
                info["episode_length"] = int(self.lengths[index])
                info["terminal_observation"] = observation.copy()
                observation, info["reset_info"] = env.reset(reset_seed_base + index)
                self.returns[index] = 0.0
                self.lengths[index] = 0
            observations.append(observation)
            rewards[index] = reward
            dones[index] = done
            infos.append(info)
        return np.stack(observations), rewards, dones, infos
