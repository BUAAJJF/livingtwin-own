from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np


STRIKE_PUCK_OBSERVATION_NAMES = (
    "initial_puck_x",
    "initial_puck_y",
    "target_direction_cos",
    "target_direction_sin",
    "target_distance",
)
STRIKE_PUCK_ACTION_NAMES = (
    "direction_offset",
    "strike_speed",
    "tangential_contact_offset",
    "contact_duration",
)


@dataclass(frozen=True)
class StrikePuckContext:
    context_id: int
    seed: int
    direction_index: int
    distance_index: int
    initial_puck_x: float
    initial_puck_y: float
    target_x: float
    target_y: float

    @property
    def cell_index(self) -> int:
        return self.distance_index * 8 + self.direction_index

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["cell_index"] = self.cell_index
        return result


def _linear_map(value: float, limits: Sequence[float]) -> float:
    clipped = float(np.clip(value, -1.0, 1.0))
    low, high = float(limits[0]), float(limits[1])
    return low + 0.5 * (clipped + 1.0) * (high - low)


def decode_strike_action(
    normalized_action: Sequence[float], config: Mapping[str, Any]
) -> dict[str, float]:
    action = np.clip(np.asarray(normalized_action, dtype=np.float64), -1.0, 1.0)
    if action.shape != (4,):
        raise ValueError(f"expected action shape (4,), got {action.shape}")
    action_config = config["action"]
    return {
        "direction_offset_rad": _linear_map(action[0], action_config["direction_offset_rad"]),
        "strike_speed_mps": _linear_map(action[1], action_config["strike_speed_mps"]),
        "tangential_contact_offset_m": _linear_map(
            action[2], action_config["tangential_contact_offset_m"]
        ),
        "contact_duration_s": _linear_map(action[3], action_config["contact_duration_s"]),
    }


def context_from_seed(
    seed: int,
    config: Mapping[str, Any],
    context_id: int = -1,
    *,
    cell_index: int | None = None,
) -> StrikePuckContext:
    seed = int(seed)
    rng = np.random.default_rng(seed)
    directions = config["target"]["direction_degrees"]
    distances = config["target"]["distances_m"]
    if cell_index is None:
        cell_index = seed % (len(directions) * len(distances))
    direction_index = cell_index % len(directions)
    distance_index = cell_index // len(directions)
    initial_x = float(rng.uniform(*config["target"]["initial_puck_x_range_m"]))
    initial_y = float(rng.uniform(*config["target"]["initial_puck_y_range_m"]))
    angle = math.radians(float(directions[direction_index]))
    distance = float(distances[distance_index])
    return StrikePuckContext(
        context_id=int(context_id),
        seed=seed,
        direction_index=direction_index,
        distance_index=distance_index,
        initial_puck_x=initial_x,
        initial_puck_y=initial_y,
        target_x=initial_x + distance * math.cos(angle),
        target_y=initial_y + distance * math.sin(angle),
    )


def make_balanced_benchmark(config: Mapping[str, Any]) -> list[StrikePuckContext]:
    count = int(config["benchmark"]["contexts_per_cell"])
    start = int(config["benchmark"]["context_seed_start"])
    directions = len(config["target"]["direction_degrees"])
    distances = len(config["target"]["distances_m"])
    contexts: list[StrikePuckContext] = []
    context_id = 0
    for distance_index in range(distances):
        for direction_index in range(directions):
            cell = distance_index * directions + direction_index
            for replicate in range(count):
                seed = start + replicate * directions * distances + cell
                context = context_from_seed(
                    seed, config, context_id=context_id, cell_index=cell
                )
                if context.direction_index != direction_index or context.distance_index != distance_index:
                    raise RuntimeError("benchmark seed-to-cell mapping is not balanced")
                contexts.append(context)
                context_id += 1
    return contexts


def release_index_from_history(
    contacts: Sequence[bool],
    retracting: Sequence[bool],
    *,
    consecutive_absent_frames: int,
    future_no_contact_frames: int,
) -> int | None:
    absent = int(consecutive_absent_frames)
    future = int(future_no_contact_frames)
    if len(contacts) != len(retracting):
        raise ValueError("contact and retract histories must have equal length")
    for index in range(1, len(contacts) - absent - future + 1):
        if not contacts[index - 1] or not retracting[index]:
            continue
        if any(contacts[index : index + absent + future]):
            continue
        return index
    return None


class StrikePuck2DEnv:
    observation_size = len(STRIKE_PUCK_OBSERVATION_NAMES)
    action_size = len(STRIKE_PUCK_ACTION_NAMES)

    def __init__(self, asset_path: str | Path, config: Mapping[str, Any], seed: int) -> None:
        self.asset_path = str(Path(asset_path).resolve())
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(self.asset_path)
        self.data = mujoco.MjData(self.model)
        self.puck_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "puck")
        self.puck_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "puck_free")
        self.puck_qpos_adr = int(self.model.jnt_qposadr[self.puck_joint_id])
        self.puck_dof_adr = int(self.model.jnt_dofadr[self.puck_joint_id])
        self.puck_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "puck_geom")
        self.striker_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "striker")
        self.striker_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "striker_geom")
        self.striker_mocap_id = int(self.model.body_mocapid[self.striker_body_id])
        self.target_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "target")
        if self.striker_mocap_id < 0:
            raise ValueError("striker must be a mocap body")
        self.rng = np.random.default_rng(int(seed))
        self.context = context_from_seed(seed, config)
        self._last_observation = np.zeros(self.observation_size, dtype=np.float32)
        self.reset(seed=seed)

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        value = int(mujoco.mj_name2id(self.model, object_type, name))
        if value < 0:
            raise ValueError(f"missing MuJoCo object: {name}")
        return value

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def maximum_steps(self) -> int:
        return int(round(float(self.config["physics"]["maximum_episode_time_s"]) / self.timestep))

    def _puck_xy(self) -> np.ndarray:
        return self.data.qpos[self.puck_qpos_adr : self.puck_qpos_adr + 2].copy()

    def _puck_velocity(self) -> np.ndarray:
        return self.data.qvel[self.puck_dof_adr : self.puck_dof_adr + 2].copy()

    def _puck_omega_z(self) -> float:
        return float(self.data.qvel[self.puck_dof_adr + 5])

    def _has_contact(self) -> bool:
        target = {self.puck_geom_id, self.striker_geom_id}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if {int(contact.geom1), int(contact.geom2)} == target:
                return True
        return False

    def _set_context_state(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.puck_qpos_adr : self.puck_qpos_adr + 3] = (
            self.context.initial_puck_x,
            self.context.initial_puck_y,
            float(self.config["puck"]["half_height_m"]),
        )
        self.data.qpos[self.puck_qpos_adr + 3 : self.puck_qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[self.puck_dof_adr : self.puck_dof_adr + 6] = 0.0
        self.model.site_pos[self.target_site_id, :2] = (self.context.target_x, self.context.target_y)
        self.data.mocap_pos[self.striker_mocap_id] = (
            self.context.initial_puck_x - 0.08,
            self.context.initial_puck_y,
            float(self.config["striker"]["half_height_m"]),
        )
        self.data.mocap_quat[self.striker_mocap_id] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def observation(self) -> np.ndarray:
        delta = np.asarray(
            [self.context.target_x - self.context.initial_puck_x, self.context.target_y - self.context.initial_puck_y],
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(delta))
        unit = delta / distance
        return np.asarray(
            [
                self.context.initial_puck_x,
                self.context.initial_puck_y,
                unit[0],
                unit[1],
                distance,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        seed: int | None = None,
        *,
        context: StrikePuckContext | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if context is not None:
            self.context = context
        elif seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self.context = context_from_seed(int(seed), self.config)
        self._set_context_state()
        self._last_observation = self.observation()
        return self._last_observation.copy(), {
            **self.context.to_dict(),
            "task_name": self.config["task_name"],
        }

    def step(
        self, normalized_action: Sequence[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._set_context_state()
        action = np.clip(np.asarray(normalized_action, dtype=np.float64), -1.0, 1.0)
        decoded = decode_strike_action(action, self.config)
        puck_initial = np.asarray(
            [self.context.initial_puck_x, self.context.initial_puck_y], dtype=np.float64
        )
        target = np.asarray([self.context.target_x, self.context.target_y], dtype=np.float64)
        target_delta = target - puck_initial
        target_angle = math.atan2(target_delta[1], target_delta[0])
        strike_angle = target_angle + decoded["direction_offset_rad"]
        strike_unit = np.asarray([math.cos(strike_angle), math.sin(strike_angle)])
        tangent_unit = np.asarray([-strike_unit[1], strike_unit[0]])
        start_distance = (
            float(self.config["puck"]["radius_m"])
            + float(self.config["striker"]["radius_m"])
            + float(self.config["striker"]["approach_gap_m"])
        )
        striker_xy = (
            puck_initial
            - strike_unit * start_distance
            + tangent_unit * decoded["tangential_contact_offset_m"]
        )
        striker_start = striker_xy.copy()
        self.data.mocap_pos[self.striker_mocap_id, :2] = striker_xy
        mujoco.mj_forward(self.model, self.data)

        contacts: list[bool] = []
        retracting: list[bool] = []
        states: list[tuple[np.ndarray, np.ndarray, float]] = []
        contact_start_index: int | None = None
        retract_start_index: int | None = None
        phase = "approach"
        path_length = 0.0
        numerical_anomaly = False
        out_of_bounds = False
        bounds_x = self.config["target"]["workspace_x_bounds_m"]
        bounds_y = self.config["target"]["workspace_y_bounds_m"]
        maximum_approach_steps = int(
            round(float(self.config["striker"]["maximum_approach_time_s"]) / self.timestep)
        )
        retract_steps = int(round(float(self.config["striker"]["retract_time_s"]) / self.timestep))
        contact_duration_steps = max(1, int(round(decoded["contact_duration_s"] / self.timestep)))

        for step_index in range(self.maximum_steps):
            previous_striker = striker_xy.copy()
            if phase in {"approach", "contact"}:
                striker_xy = striker_xy + strike_unit * decoded["strike_speed_mps"] * self.timestep
            elif phase == "retract":
                striker_xy = striker_xy - strike_unit * float(self.config["striker"]["retract_speed_mps"]) * self.timestep
            else:
                striker_xy = (
                    puck_initial
                    - strike_unit * float(self.config["striker"]["parking_distance_m"])
                    + tangent_unit * decoded["tangential_contact_offset_m"]
                )
            path_length += float(np.linalg.norm(striker_xy - previous_striker))
            self.data.mocap_pos[self.striker_mocap_id, :2] = striker_xy
            mujoco.mj_step(self.model, self.data)
            contact = self._has_contact()
            if contact and contact_start_index is None:
                contact_start_index = step_index
                phase = "contact"
            if phase == "contact" and contact_start_index is not None:
                if step_index - contact_start_index + 1 >= contact_duration_steps:
                    phase = "retract"
                    retract_start_index = step_index
            elif phase == "approach" and step_index + 1 >= maximum_approach_steps:
                phase = "retract"
                retract_start_index = step_index
            elif phase == "retract" and retract_start_index is not None:
                if step_index - retract_start_index + 1 >= retract_steps:
                    phase = "park"
            puck_xy = self._puck_xy()
            puck_velocity = self._puck_velocity()
            omega_z = self._puck_omega_z()
            finite = bool(
                np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))
                and np.all(np.isfinite(self.data.mocap_pos))
            )
            numerical_anomaly = numerical_anomaly or not finite
            out_of_bounds = out_of_bounds or not (
                float(bounds_x[0]) <= puck_xy[0] <= float(bounds_x[1])
                and float(bounds_y[0]) <= puck_xy[1] <= float(bounds_y[1])
            )
            contacts.append(bool(contact))
            retracting.append(phase in {"retract", "park"})
            states.append((puck_xy, puck_velocity, omega_z))
            if numerical_anomaly:
                break

        release_config = self.config["release"]
        release_index = release_index_from_history(
            contacts,
            retracting,
            consecutive_absent_frames=int(release_config["consecutive_absent_frames"]),
            future_no_contact_frames=int(release_config["future_no_contact_frames"]),
        )
        if release_index is None:
            release_xy = np.full(2, np.nan)
            release_velocity = np.full(2, np.nan)
            release_omega = float("nan")
            release_time = float("nan")
            recontact = False
        else:
            release_xy, release_velocity, release_omega = states[release_index]
            release_time = release_index * self.timestep
            check_from = release_index + int(release_config["consecutive_absent_frames"])
            recontact = any(contacts[check_from:])
        stop_time = float("nan")
        if release_index is not None:
            required = int(self.config["stop"]["consecutive_frames"])
            stable = 0
            for index in range(release_index, len(states)):
                _, velocity, omega = states[index]
                stopped = bool(
                    np.linalg.norm(velocity) <= float(self.config["stop"]["linear_speed_mps"])
                    and abs(omega) <= float(self.config["stop"]["angular_speed_radps"])
                )
                stable = stable + 1 if stopped else 0
                if stable >= required:
                    stop_time = (index - required + 1) * self.timestep
                    break
        final_xy = self._puck_xy()
        final_velocity = self._puck_velocity()
        final_omega = self._puck_omega_z()
        final_error = float(np.linalg.norm(final_xy - target))
        final_speed = float(np.linalg.norm(final_velocity))
        success = bool(
            release_index is not None
            and final_error <= float(self.config["target"]["success_radius_m"])
            and final_speed <= float(self.config["target"]["success_speed_mps"])
            and not recontact
            and not out_of_bounds
            and not numerical_anomaly
        )
        reward_config = self.config["reward"]
        reward = (
            float(reward_config["success_bonus"]) * float(success)
            - float(reward_config["final_distance_scale"]) * final_error
            - float(reward_config["final_speed_scale"]) * final_speed
            - float(reward_config["action_penalty"]) * float(np.sum(action**2))
            - float(reward_config["no_contact_penalty"]) * float(contact_start_index is None)
            - float(reward_config["recontact_penalty"]) * float(recontact)
            - float(reward_config["out_of_bounds_penalty"]) * float(out_of_bounds)
            - float(reward_config["numerical_anomaly_penalty"]) * float(numerical_anomaly)
        )
        release_speed = float(np.linalg.norm(release_velocity)) if np.all(np.isfinite(release_velocity)) else float("nan")
        release_direction = (
            math.degrees(math.atan2(release_velocity[1], release_velocity[0])) % 360.0
            if release_speed > 1.0e-9
            else float("nan")
        )
        info: dict[str, Any] = {
            **self.context.to_dict(),
            "success": success,
            "normalized_action": action.tolist(),
            **decoded,
            "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.95)),
            "initial_puck_xy": puck_initial.tolist(),
            "target_xy": target.tolist(),
            "striker_start_xy": striker_start.tolist(),
            "striker_final_xy": striker_xy.tolist(),
            "commanded_striker_path_length_m": path_length,
            "contact_start_time_s": (
                contact_start_index * self.timestep if contact_start_index is not None else float("nan")
            ),
            "contact_end_time_s": (
                max(index for index, value in enumerate(contacts) if value) * self.timestep
                if any(contacts)
                else float("nan")
            ),
            "release_time_s": release_time,
            "release_position_x": float(release_xy[0]),
            "release_position_y": float(release_xy[1]),
            "release_velocity_x": float(release_velocity[0]),
            "release_velocity_y": float(release_velocity[1]),
            "release_speed_mps": release_speed,
            "release_direction_deg": release_direction,
            "release_angular_velocity_radps": float(release_omega),
            "final_position_x": float(final_xy[0]),
            "final_position_y": float(final_xy[1]),
            "final_velocity_x": float(final_velocity[0]),
            "final_velocity_y": float(final_velocity[1]),
            "final_speed_mps": final_speed,
            "final_angular_velocity_radps": float(final_omega),
            "final_error_m": final_error,
            "stop_time_s": stop_time,
            "contact_detected": bool(contact_start_index is not None),
            "recontact": bool(recontact),
            "out_of_bounds": bool(out_of_bounds),
            "numerical_anomaly": bool(numerical_anomaly),
            "episode_steps": len(states),
            "reward": float(reward),
        }
        return np.zeros(self.observation_size, dtype=np.float32), float(reward), True, False, info


class VectorStrikePuck2DEnv:
    def __init__(self, num_envs: int, asset_path: str | Path, config: Mapping[str, Any], seed: int) -> None:
        self.num_envs = int(num_envs)
        self.envs = [
            StrikePuck2DEnv(asset_path, config, seed=seed + index)
            for index in range(self.num_envs)
        ]

    def reset(self, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        pairs = [env.reset(seed=int(seed) + index) for index, env in enumerate(self.envs)]
        return np.stack([pair[0] for pair in pairs]), [pair[1] for pair in pairs]

    def step(
        self, actions: np.ndarray, reset_seed_base: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations: list[np.ndarray] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.ones(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        for index, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            _, reward, _, _, info = env.step(action)
            observation, reset_info = env.reset(seed=int(reset_seed_base) + index)
            info["reset_info"] = reset_info
            observations.append(observation)
            rewards[index] = reward
            infos.append(info)
        return np.stack(observations), rewards, dones, infos
