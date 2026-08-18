from __future__ import annotations

from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from livingtwin_mujoco_rl.piperx_engineering import (
    _joint_addresses,
    set_robot_state,
    solve_strike_ik,
    unexpected_contacts,
)


class PiperCartesianController:
    """Simulation controller with the same Cartesian target boundary intended for real Piper."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: Mapping[str, Any]):
        self.model, self.data, self.config = model, data, config
        self.names = tuple(config["robot"]["arm_joint_names"])
        self.qpos_addr, _, self.ranges = _joint_addresses(model, self.names)
        self.site_id = model.site("strike_site").id
        self.table_id = model.geom("strike_table").id
        self.object_id = model.geom("strike_puck_geom").id
        self.current_q = np.asarray(config["robot"]["home_joint_positions_rad"], dtype=np.float64).copy()

    def reset(self) -> None:
        set_robot_state(self.model, self.data, self.config, self.current_q)
        self.data.ctrl[:] = self.current_q
        mujoco.mj_forward(self.model, self.data)

    def ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def normalized_joint_position(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_addr]
        return (2.0 * (q - self.ranges[:, 0]) / (self.ranges[:, 1] - self.ranges[:, 0]) - 1.0).astype(np.float32)

    def command_delta(self, delta_xy: Sequence[float]) -> dict[str, Any]:
        start = self.ee_position()
        target = start.copy()
        target[:2] += np.asarray(delta_xy, dtype=np.float64)
        return self.move_to_target(target)

    def move_to_target(self, target: Sequence[float]) -> dict[str, Any]:
        target = np.asarray(target, dtype=np.float64)
        result = solve_strike_ik(self.model, self.config, target, self.current_q)
        if not result["converged"]:
            return {"ik_failure": True, "table_collision": False, "target": target, "position_error_m": float(result["position_error_m"])}
        q_target = np.asarray(result["q"], dtype=np.float64)
        start_q = self.data.qpos[self.qpos_addr].copy()
        steps = max(1, int(round(float(self.config["control"]["duration_s"]) / self.model.opt.timestep)))
        table_collision = False
        for index in range(steps):
            alpha = (index + 1) / steps
            self.data.ctrl[:] = (1.0 - alpha) * start_q + alpha * q_target
            mujoco.mj_step(self.model, self.data)
            names = {self.model.geom(int(c.geom1)).name for c in self.data.contact[: self.data.ncon]}
            names |= {self.model.geom(int(c.geom2)).name for c in self.data.contact[: self.data.ncon]}
            if "strike_table" in names and "strike_puck_geom" not in names:
                table_collision = True
        settled = False
        tolerance = float(self.config["ik"]["position_tolerance_m"])
        for _ in range(4 * steps):
            self.data.ctrl[:] = q_target
            mujoco.mj_step(self.model, self.data)
            if float(np.linalg.norm(self.ee_position() - target)) <= tolerance:
                settled = True
                break
        self.current_q = self.data.qpos[self.qpos_addr].copy()
        return {
            "ik_failure": not settled,
            "tracking_failure": not settled,
            "table_collision": table_collision,
            "target": target,
            "position_error_m": float(result["position_error_m"]),
            "achieved_position_error_m": float(np.linalg.norm(self.ee_position() - target)),
            "joint_residual_rad": float(np.max(np.abs(self.current_q - q_target))),
            "unexpected_contacts": unexpected_contacts(self.model, self.data),
        }
