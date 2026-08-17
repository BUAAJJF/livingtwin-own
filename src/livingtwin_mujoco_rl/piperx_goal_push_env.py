from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from livingtwin_mujoco_rl.piperx_cartesian_controller import PiperCartesianController
from livingtwin_mujoco_rl.piperx_engineering import build_engineering_spec


class PiperGoalPushEnv:
    action_size = 2

    def __init__(self, config: Mapping[str, Any], seed: int):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.model = build_engineering_spec(config).compile()
        self.data = mujoco.MjData(self.model)
        self.controller = PiperCartesianController(self.model, self.data, config)
        joint_count = len(config["robot"]["arm_joint_names"])
        self.observation_size = 4 + joint_count + (2 if config["observation"].get("include_object_velocity", True) else 0)
        self.object_joint = self.model.joint("strike_puck_free")
        self.object_qpos = int(self.object_joint.qposadr[0])
        self.object_dof = int(self.object_joint.dofadr[0])
        self.goal = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self.last_info: dict[str, Any] = {}

    def _object_xy(self) -> np.ndarray:
        return self.data.qpos[self.object_qpos : self.object_qpos + 2].copy()

    def _object_velocity(self) -> np.ndarray:
        return self.data.qvel[self.object_dof : self.object_dof + 2].copy()

    def _distance(self) -> float:
        return float(np.linalg.norm(self.goal - self._object_xy()))

    def _legal_xy(self) -> tuple[float, float, float, float]:
        task = self.config["task"]
        return tuple(float(v) for v in task["workspace_xy_bounds_m"])

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        task = self.config["task"]
        mujoco.mj_resetData(self.model, self.data)
        x0 = self.rng.uniform(*task["cube_initial_x_range_m"])
        y0 = self.rng.uniform(*task["cube_initial_y_range_m"])
        lo_x, hi_x, lo_y, hi_y = self._legal_xy()
        radius = self.rng.uniform(*task["goal_distance_range_m"])
        angle = self.rng.uniform(*task.get("goal_angle_range_rad", [-math.pi, math.pi]))
        self.goal[:] = np.clip([x0 + radius * math.cos(angle), y0 + radius * math.sin(angle)], [lo_x, lo_y], [hi_x, hi_y])
        table_z = float(task["table_center_xyz_m"][2]) + float(task["table_half_size_xyz_m"][2])
        half_z = float(task["object_halfsize_xyz_m"][2])
        self.data.qpos[self.object_qpos : self.object_qpos + 3] = (x0, y0, table_z + half_z)
        self.data.qpos[self.object_qpos + 3 : self.object_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[self.object_dof : self.object_dof + 6] = 0.0
        self.controller.current_q = np.asarray(task.get("robot_home_override", self.config["robot"]["home_joint_positions_rad"]), dtype=np.float64).copy()
        self.controller.reset()
        goal_direction = self.goal - np.asarray([x0, y0], dtype=np.float64)
        norm = float(np.linalg.norm(goal_direction))
        if norm < 1.0e-8:
            goal_direction = np.asarray([1.0, 0.0])
        else:
            goal_direction /= norm
        pre_xy = np.asarray([x0, y0], dtype=np.float64) - goal_direction * (float(max(task["object_halfsize_xyz_m"][:2])) + float(task["precontact_gap_m"]))
        pre_target = np.asarray([np.clip(pre_xy[0], lo_x, hi_x), np.clip(pre_xy[1], lo_y, hi_y), float(task["ee_contact_z_m"])], dtype=np.float64)
        self.controller.move_to_target(pre_target)
        self.model.site_pos[self.model.site("goal_site").id, :2] = self.goal
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.last_info = {"success": False, "reset": True}
        return self.observation(), self.info(False, "reset")

    def observation(self) -> np.ndarray:
        object_xy = self._object_xy()
        ee_xy = self.controller.ee_position()[:2]
        values = [*(self.goal - object_xy), *(ee_xy - object_xy), *self.controller.normalized_joint_position()]
        if self.config["observation"].get("include_object_velocity", True):
            values.extend(self._object_velocity())
        return np.asarray(values, dtype=np.float32)

    def info(self, success: bool, reason: str) -> dict[str, Any]:
        return {"success": bool(success), "distance_m": self._distance(), "object_xy": self._object_xy().tolist(), "goal_xy": self.goal.tolist(), "step_count": self.step_count, "terminated_reason": reason, **self.last_info}

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        before = self._distance()
        result = self.controller.command_delta(action * float(self.config["action"]["delta_xy_scale_m"]))
        self.step_count += 1
        after = self._distance()
        task = self.config["task"]
        lo_x, hi_x, lo_y, hi_y = self._legal_xy()
        obj = self._object_xy()
        oob = not (lo_x <= obj[0] <= hi_x and lo_y <= obj[1] <= hi_y)
        success = after <= float(task["success_radius_m"])
        failure = bool(result["ik_failure"] or result["table_collision"] or result.get("unexpected_contacts"))
        timeout = self.step_count >= int(task["episode_steps"])
        terminated = bool(success or oob or failure)
        truncated = bool(timeout and not terminated)
        reward = float(self.config["reward"]["progress_scale"]) * (before - after) + float(self.config["reward"]["success_bonus"]) * success
        if oob:
            reward -= float(self.config["reward"]["oob_penalty"])
        if failure:
            reward -= float(self.config["reward"]["safety_penalty"])
        reward -= float(self.config["reward"]["step_cost"])
        reason = "success" if success else "oob" if oob else "safety_failure" if failure else "timeout" if timeout else "running"
        self.last_info = {"ik_failure": bool(result["ik_failure"]), "table_collision": bool(result["table_collision"]), "oob": oob, "action_saturated": bool(np.any(np.abs(action) >= 0.98)), "executed_delta_xy_m": (action * float(self.config["action"]["delta_xy_scale_m"])).tolist()}
        return self.observation(), reward, terminated, truncated, self.info(success, reason)
