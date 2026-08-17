from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from livingtwin_mujoco_rl.piperx_continuous_selfplay import DirectPushEnv
from livingtwin_mujoco_rl.piperx_engineering import _joint_addresses


class PiperGoalPushEnv:
    """Goal-conditioned wrapper around the proven short sustained-push primitive."""

    action_size = 2

    def __init__(self, config: Mapping[str, Any], seed: int):
        self.config = copy.deepcopy(dict(config))
        self.rng = np.random.default_rng(seed)
        execution_config = copy.deepcopy(self.config)
        execution_config["sustained_push"] = True
        execution_config["direct_gripper"] = {
            "marker_above_puck_center_m": 0.035,
            "exploration_bounds_xy_m": list(execution_config["task"]["workspace_xy_bounds_m"]),
        }
        self.executor = DirectPushEnv(execution_config, friction=float(self.config["task"]["table_friction"][0]))
        self.model, self.data = self.executor.model, self.executor.data
        self.object_joint = self.model.joint("strike_puck_free")
        self.object_qpos = int(self.object_joint.qposadr[0])
        self.object_dof = int(self.object_joint.dofadr[0])
        self.joint_qpos, _, self.joint_ranges = _joint_addresses(self.model, self.config["robot"]["arm_joint_names"])
        self.goal = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self.last_info: dict[str, Any] = {}
        joint_count = len(self.config["robot"]["arm_joint_names"])
        self.observation_size = 4 + joint_count + (2 if self.config["observation"].get("include_object_velocity", True) else 0)

    def _object_xy(self) -> np.ndarray:
        return self.data.qpos[self.object_qpos : self.object_qpos + 2].copy()

    def _object_velocity(self) -> np.ndarray:
        return self.data.qvel[self.object_dof : self.object_dof + 2].copy()

    def _distance(self) -> float:
        return float(np.linalg.norm(self.goal - self._object_xy()))

    def _legal_xy(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.config["task"]["workspace_xy_bounds_m"])

    def _normalized_joint_position(self) -> np.ndarray:
        q = self.data.qpos[self.joint_qpos]
        return (2.0 * (q - self.joint_ranges[:, 0]) / (self.joint_ranges[:, 1] - self.joint_ranges[:, 0]) - 1.0).astype(np.float32)

    def _decode_action(self, action: Sequence[float]) -> dict[str, Any]:
        """Map a bounded local vector to one short sustained push or a no-op.

        ``action[0]`` is goal-forward and ``action[1]`` is goal-left.  The
        vector angle selects the approach/push direction; its clipped Euclidean
        norm maps continuously to the historically validated 11--33 mm travel.
        """
        local = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        norm = min(float(np.linalg.norm(local)), 1.0)
        if norm < 0.05:
            return {"no_op": True, "local_action": local.tolist()}
        forward = self.goal - self._object_xy()
        forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
        lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        direction = local[0] * forward + local[1] * lateral
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        travel_lo, travel_hi = (float(value) for value in self.config["action"]["sustained_push_travel_range_m"])
        return {
            "no_op": False,
            "local_action": local.tolist(),
            "direction": direction,
            "direction_rad": float(math.atan2(direction[1], direction[0])),
            "travel_m": float(travel_lo + norm * (travel_hi - travel_lo)),
            "speed_mps": float(self.config["action"]["sustained_push_speed_mps"]),
        }

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        task = self.config["task"]
        x0 = self.rng.uniform(*task["cube_initial_x_range_m"])
        y0 = self.rng.uniform(*task["cube_initial_y_range_m"])
        lo_x, hi_x, lo_y, hi_y = self._legal_xy()
        radius = self.rng.uniform(*task["goal_distance_range_m"])
        angle = self.rng.uniform(*task.get("goal_angle_range_rad", [-math.pi, math.pi]))
        self.goal[:] = np.clip([x0 + radius * math.cos(angle), y0 + radius * math.sin(angle)], [lo_x, lo_y], [hi_x, hi_y])
        self.executor.reset([x0, y0])
        self.model.site_pos[self.model.site("goal_site").id, :2] = self.goal
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.last_info = {"success": False, "reset": True}
        return self.observation(), self.info(False, "reset")

    def observation(self) -> np.ndarray:
        object_xy = self._object_xy()
        ee_xy = self.data.site_xpos[self.model.site("strike_site").id, :2]
        values = [*(self.goal - object_xy), *(ee_xy - object_xy), *self._normalized_joint_position()]
        if self.config["observation"].get("include_object_velocity", True):
            values.extend(self._object_velocity())
        return np.asarray(values, dtype=np.float32)

    def info(self, success: bool, reason: str) -> dict[str, Any]:
        return {
            "success": bool(success), "distance_m": self._distance(),
            "object_xy": self._object_xy().tolist(), "goal_xy": self.goal.tolist(),
            "step_count": self.step_count, "terminated_reason": reason, **self.last_info,
        }

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        before = self._distance()
        decoded = self._decode_action(action)
        if decoded["no_op"]:
            result: dict[str, Any] = {"unrecoverable": False, "gripper_puck_contact": False, "gripper_table_contact": False, "oob": False}
        else:
            result, _ = self.executor.push(decoded["direction_rad"], decoded["travel_m"], decoded["speed_mps"])
        self.step_count += 1
        after = self._distance()
        lo_x, hi_x, lo_y, hi_y = self._legal_xy()
        obj = self._object_xy()
        oob = bool(result.get("oob", False) or not (lo_x <= obj[0] <= hi_x and lo_y <= obj[1] <= hi_y))
        ik_failure = bool(result.get("unrecoverable", False))
        table_collision = bool(result.get("gripper_table_contact", False))
        failure = bool(ik_failure or table_collision or result.get("numerical_anomaly", False))
        success = after <= float(self.config["task"]["success_radius_m"])
        timeout = self.step_count >= int(self.config["task"]["episode_steps"])
        terminated = bool(success or oob or failure)
        truncated = bool(timeout and not terminated)
        reward = float(self.config["reward"]["progress_scale"]) * (before - after) + float(self.config["reward"]["success_bonus"]) * success
        if oob:
            reward -= float(self.config["reward"]["oob_penalty"])
        if failure:
            reward -= float(self.config["reward"]["safety_penalty"])
        reward -= float(self.config["reward"]["step_cost"])
        reason = "success" if success else "oob" if oob else "safety_failure" if failure else "timeout" if timeout else "running"
        self.last_info = {
            "ik_failure": ik_failure, "push_ik_failure": ik_failure,
            "contact_failure": bool(not decoded["no_op"] and not result.get("gripper_puck_contact", False)),
            "contact_established": bool(result.get("gripper_puck_contact", False)),
            "table_collision": table_collision, "oob": oob,
            "safety_termination": failure, "action_no_op": bool(decoded["no_op"]),
            "executed_push": {key: value for key, value in decoded.items() if key != "direction"},
        }
        return self.observation(), reward, terminated, truncated, self.info(success, reason)
