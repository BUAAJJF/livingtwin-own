from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np
import yaml

from livingtwin_mujoco_rl.piperx_engineering import (
    _joint_addresses,
    _smoothstep,
    build_engineering_spec,
    set_robot_state,
    solve_position_ik,
    solve_strike_ik,
    unexpected_contacts,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_direct_model(config: Mapping[str, Any]) -> mujoco.MjModel:
    spec = build_engineering_spec(config)
    # The old synthetic cylinder remains as a non-colliding marker/site only.
    proxy = spec.geom("strike_tip")
    proxy.contype = 0
    proxy.conaffinity = 0
    proxy.group = 5
    # Non-colliding reference for the measured paired distal mesh center.
    gripper = spec.body("gripper_base")
    strike = config["strike_interface"]["grasp_site_position_in_gripper_base_m"]
    gripper.add_site(name="closed_tip_site", pos=[strike[0], strike[1], strike[2] + 0.0115], size=[0.002, 0.002, 0.002], group=5)
    for body_name in ("gripper_link1", "gripper_link2"):
        body = spec.body(body_name)
        for geom in body.geoms:
            if "collision" in geom.name:
                geom.contype = 1
                geom.conaffinity = 1
                geom.group = 3
    # Direct pushing uses a fixed passive aperture: there is no close/grasp command.
    opening = float(config["robot"]["gripper_joint_positions_m"]["gripper_joint1"])
    aperture = spec.add_equality()
    aperture.type = mujoco.mjtEq.mjEQ_JOINT
    aperture.objtype = mujoco.mjtObj.mjOBJ_JOINT
    aperture.name1 = "gripper_joint1"
    aperture.data[:5] = [opening, 0.0, 0.0, 0.0, 0.0]
    aperture.solref = [0.005, 1.0]
    return spec.compile()


def pair_contact(model: mujoco.MjModel, data: mujoco.MjData, ids_a: set[int], ids_b: set[int]) -> bool:
    for index in range(data.ncon):
        a, b = int(data.contact[index].geom1), int(data.contact[index].geom2)
        if (a in ids_a and b in ids_b) or (b in ids_a and a in ids_b):
            return True
    return False


class DirectPushEnv:
    def __init__(self, config: Mapping[str, Any], friction: float = 0.20):
        self.config = json.loads(json.dumps(config))
        self.config["task"]["table_friction"][0] = float(friction)
        self.model = build_direct_model(self.config)
        self.data = mujoco.MjData(self.model)
        self.names = self.config["robot"]["arm_joint_names"]
        self.qpos_addr, _, _ = _joint_addresses(self.model, self.names)
        self.puck_joint = self.model.joint("strike_puck_free")
        self.puck_qpos = int(self.puck_joint.qposadr[0])
        self.puck_dof = int(self.puck_joint.dofadr[0])
        self.site_id = self.model.site("strike_site").id
        self.closed_tip_site_id = self.model.site("closed_tip_site").id
        self.table_id = self.model.geom("strike_table").id
        self.puck_id = self.model.geom("strike_puck_geom").id
        self.finger_ids = {
            name: self.model.geom(name).id
            for name in ("gripper_link1_collision_1", "gripper_link2_collision_1")
        }
        self.gripper_qpos = {
            name: int(self.model.joint(name).qposadr[0])
            for name in ("gripper_joint1", "gripper_joint2")
        }
        self.native_gripper_ids = {
            i for i in range(self.model.ngeom)
            if self.model.body(int(self.model.geom_bodyid[i])).name in {"link6", "gripper_base", "gripper_link1", "gripper_link2"}
            and int(self.model.geom_contype[i]) != 0
        }
        self.time_step_index = 0
        self.safety_resets = 0
        # Strictly optional diagnostic sink.  Production callers leave this
        # unset, so it cannot affect target construction or execution.
        self.diagnostic_hook: Any | None = None
        self.reset([*self.config["task"]["puck_center_xy_m"]])

    def _diagnostic(self, event: str, **payload: Any) -> None:
        if self.diagnostic_hook is not None:
            self.diagnostic_hook(event, payload)

    def reset(self, puck_xy: Sequence[float]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q = np.asarray(self.config["robot"]["home_joint_positions_rad"], dtype=float)
        set_robot_state(self.model, self.data, self.config, q)
        self.data.ctrl[:] = q
        self.data.qpos[self.puck_qpos:self.puck_qpos + 2] = puck_xy
        mujoco.mj_forward(self.model, self.data)
        self.current_q = q.copy()
        self.safety_resets += 1

    def puck_xy(self) -> np.ndarray:
        return self.data.qpos[self.puck_qpos:self.puck_qpos + 2].copy()

    def puck_v(self) -> np.ndarray:
        return self.data.qvel[self.puck_dof:self.puck_dof + 2].copy()

    def contact_surface(self, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
        matrix = np.zeros(9); mujoco.mju_quat2Mat(matrix, self.data.qpos[self.puck_qpos + 3:self.puck_qpos + 7])
        rotation = matrix.reshape(3, 3)[:2, :2]
        local = rotation.T @ direction
        half = np.asarray(self.config["task"]["object_halfsize_xyz_m"][:2], dtype=float)
        return center - rotation @ (local * float(np.min(half / np.maximum(np.abs(local), 1e-12))))

    @staticmethod
    def opening_axis(direction: np.ndarray) -> np.ndarray:
        # local-z remains down; local-y is the side-wall tangent for the -x face.
        return np.asarray([-direction[1], direction[0], 0.0])

    def goal(self, center: np.ndarray, direction: np.ndarray, phase: str, magnitude: float) -> np.ndarray:
        task = self.config["task"]
        strike = self.config["strike_interface"]
        table_z = float(task["table_center_xyz_m"][2]) + float(task["table_half_size_xyz_m"][2])
        # The kinematic marker is 11.5 mm above the finger distal face in the source asset.
        # This height was selected from native mesh geometry, not target performance.
        marker_offset = float(self.config.get("direct_gripper", {}).get("marker_above_puck_center_m", 0.0115))
        z = table_z + float(task["puck_half_height_m"]) + marker_offset
        surface = self.contact_surface(center, direction)
        gap = float(strike["approach_gap_m"])
        if phase == "pre":
            xy = surface - direction * gap
            return np.r_[xy, z]
        if phase == "post":
            xy = surface + direction * magnitude
            return np.r_[xy, z]
        if phase == "high":
            xy = surface - direction * (gap + 0.02)
            return np.r_[xy, z + 0.05]
        if phase == "retract":
            xy = surface - direction * (gap + 0.02)
            return np.r_[xy, z + 0.05]
        raise ValueError(phase)

    def solve(
        self,
        goal: np.ndarray,
        seed: np.ndarray,
        *,
        orientation: bool = True,
        direction: np.ndarray | None = None,
        desired_tip_axis_world: np.ndarray | None = None,
        desired_opening_axis_world: np.ndarray | None = None,
        phase: str = "unspecified",
    ) -> np.ndarray | None:
        solver = solve_strike_ik if orientation else solve_position_ik
        kwargs: dict[str, np.ndarray] = {}
        if orientation:
            if desired_tip_axis_world is not None:
                kwargs["desired_tip_axis_world"] = np.asarray(desired_tip_axis_world, dtype=float)
            if desired_opening_axis_world is not None:
                kwargs["desired_opening_axis_world"] = np.asarray(desired_opening_axis_world, dtype=float)
            elif direction is not None:
                kwargs["desired_opening_axis_world"] = self.opening_axis(direction)

        def query(branch: str, query_seed: np.ndarray, query_kwargs: Mapping[str, Any]) -> dict[str, Any]:
            result = solver(self.model, self.config, goal, query_seed, **query_kwargs)
            ranges = _joint_addresses(self.model, self.names)[2]
            margin = np.minimum(np.asarray(result["q"]) - ranges[:, 0], ranges[:, 1] - np.asarray(result["q"]))
            self._diagnostic(
                "ik_query",
                phase=phase,
                branch=branch,
                orientation=orientation,
                desired_pose=np.asarray(goal).copy(),
                warm_start_q=np.asarray(query_seed).copy(),
                desired_tip_axis_world=(None if "desired_tip_axis_world" not in query_kwargs else np.asarray(query_kwargs["desired_tip_axis_world"]).copy()),
                desired_opening_axis_world=(None if "desired_opening_axis_world" not in query_kwargs else np.asarray(query_kwargs["desired_opening_axis_world"]).copy()),
                converged=bool(result["converged"]),
                returned_q=np.asarray(result["q"]).copy(),
                position=np.asarray(result["position"]).copy(),
                position_error_m=float(result["position_error_m"]),
                orientation_axis_error_rad=float(result.get("orientation_axis_error_rad", 0.0)),
                opening_axis_error_rad=float(result.get("opening_axis_error_rad", 0.0)),
                iterations=int(result["iterations"]),
                minimum_joint_margin_rad=float(result["minimum_joint_margin_rad"]),
                closest_limit_joint=self.names[int(np.argmin(margin))],
                unexpected_contacts=list(result["unexpected_contacts"]),
            )
            return result

        result = query("production_warm_start", seed, kwargs)
        # A closed symmetric pair admits the equivalent +x-side branch: flipping
        # local x/y by pi about downward local-z keeps the selected distal side
        # normal toward the command while swapping identical fingers.
        if not result["converged"] and orientation and direction is not None:
            kwargs = dict(kwargs)
            kwargs["desired_opening_axis_world"] = -np.asarray(
                kwargs.get("desired_opening_axis_world", self.opening_axis(direction)), dtype=float,
            )
            result = query("production_symmetric_branch", seed, kwargs)
        if not result["converged"]:
            home = np.asarray(self.config["robot"]["home_joint_positions_rad"], dtype=float)
            result = query("production_home_seed", home, kwargs)
        return np.asarray(result["q"]) if result["converged"] else None

    def _actual_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_addr].copy()

    def _target_site(self, target: np.ndarray) -> np.ndarray:
        probe = mujoco.MjData(self.model)
        set_robot_state(self.model, probe, self.config, target)
        mujoco.mj_forward(self.model, probe)
        return probe.site_xpos[self.site_id].copy()

    def _target_rotation(self, target: np.ndarray) -> np.ndarray:
        probe = mujoco.MjData(self.model)
        set_robot_state(self.model, probe, self.config, target)
        mujoco.mj_forward(self.model, probe)
        return probe.site_xmat[self.site_id].reshape(3, 3).copy()

    def commanded_pushing_rotation(self, direction: np.ndarray) -> np.ndarray:
        """Exact frozen task-space pushing frame, independent of endpoint FK error."""
        z_axis = np.asarray(self.config["strike_interface"]["strike_tip_axis_world"], dtype=float)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = self.opening_axis(direction)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    @staticmethod
    def _rotation_slerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
        """Shortest-path quaternion interpolation for a Cartesian pose segment."""
        first_quat = np.empty(4); second_quat = np.empty(4); interpolated = np.empty(4)
        mujoco.mju_mat2Quat(first_quat, np.asarray(first, dtype=float).reshape(-1))
        mujoco.mju_mat2Quat(second_quat, np.asarray(second, dtype=float).reshape(-1))
        if float(np.dot(first_quat, second_quat)) < 0.0:
            second_quat *= -1.0
        dot = float(np.clip(np.dot(first_quat, second_quat), -1.0, 1.0))
        if dot > 0.9995:
            interpolated[:] = first_quat + alpha * (second_quat - first_quat)
            interpolated /= np.linalg.norm(interpolated)
        else:
            angle = math.acos(dot)
            sine = math.sin(angle)
            interpolated[:] = (
                math.sin((1.0 - alpha) * angle) / sine * first_quat
                + math.sin(alpha * angle) / sine * second_quat
            )
        matrix = np.empty(9); mujoco.mju_quat2Mat(matrix, interpolated)
        return matrix.reshape(3, 3)

    def move_cartesian_incremental(
        self,
        target_q: np.ndarray,
        duration: float,
        trace: list[dict[str, Any]],
        phase: str,
        *,
        orientation: bool,
        direction: np.ndarray | None,
    ) -> dict[str, Any]:
        """Execute one existing pre-contact phase as short Cartesian pose steps.

        This deliberately reuses the existing constrained IK and joint position
        controller.  The only change is that each low-level joint segment comes
        from the current *measured* TCP pose to the next Cartesian pose, rather
        than from a single endpoint-to-endpoint joint interpolation.
        """
        start_site = self.data.site_xpos[self.site_id].copy()
        start_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        target_site = self._target_site(target_q)
        # A directed pre-contact target has one physical source of truth: the
        # original downward tip axis and commanded opening/yaw axis.  Do not
        # promote an endpoint IK solution's tolerated FK residual into a new
        # orientation command for the Cartesian trajectory.
        target_rotation = (
            self.commanded_pushing_rotation(direction)
            if orientation and direction is not None
            else self._target_rotation(target_q)
        )
        position_distance = float(np.linalg.norm(target_site - start_site))
        rotation_angle = float(math.acos(np.clip((np.trace(start_rotation.T @ target_rotation) - 1.0) / 2.0, -1.0, 1.0)))
        position_increment = 0.003
        angular_increment = math.radians(5.0)
        segments = max(1, int(math.ceil(max(position_distance / position_increment, rotation_angle / angular_increment))))
        segment_duration = max(0.03, float(duration) / segments)
        minimum_margin = float("inf")
        first_unintended_contact_step: int | None = None
        first_table_contact_step: int | None = None
        max_tracking_residual = 0.0
        for index in range(segments):
            alpha = float(index + 1) / segments
            desired_site = (1.0 - alpha) * start_site + alpha * target_site
            desired_rotation = self._rotation_slerp(start_rotation, target_rotation, alpha)
            intermediate_q = self.solve(
                desired_site,
                self._actual_q(),
                orientation=orientation,
                direction=direction,
                desired_tip_axis_world=(desired_rotation[:, 2] if orientation else None),
                desired_opening_axis_world=(desired_rotation[:, 1] if orientation else None),
                phase=f"{phase}_cartesian_ik",
            )
            if intermediate_q is None:
                return {
                    "settled": False, "cartesian": True, "cartesian_segments": segments,
                    "completed_segments": index, "failure": "cartesian_ik",
                    "minimum_joint_margin_rad": minimum_margin,
                    "first_unintended_contact_step": first_unintended_contact_step,
                    "first_table_contact_step": first_table_contact_step,
                    "maximum_tracking_residual_m": max_tracking_residual,
                }
            trace_start = len(trace)
            tracking = self.move(intermediate_q, segment_duration, trace, phase)
            max_tracking_residual = max(max_tracking_residual, float(tracking["position_residual_m"]))
            ranges = _joint_addresses(self.model, self.names)[2]
            actual_q = self._actual_q()
            minimum_margin = min(minimum_margin, float(np.min(np.minimum(actual_q - ranges[:, 0], ranges[:, 1] - actual_q))))
            for sample_index, sample in enumerate(trace[trace_start:], start=trace_start):
                if sample["finger_puck"] and first_unintended_contact_step is None:
                    first_unintended_contact_step = sample_index
                if sample["gripper_table"] and first_table_contact_step is None:
                    first_table_contact_step = sample_index
            if not tracking["settled"] or first_unintended_contact_step is not None or first_table_contact_step is not None:
                return {
                    "settled": bool(tracking["settled"]), "cartesian": True, "cartesian_segments": segments,
                    "completed_segments": index + 1,
                    "failure": "tracking" if not tracking["settled"] else "premature_contact" if first_unintended_contact_step is not None else "table_contact",
                    "minimum_joint_margin_rad": minimum_margin,
                    "first_unintended_contact_step": first_unintended_contact_step,
                    "first_table_contact_step": first_table_contact_step,
                    "maximum_tracking_residual_m": max_tracking_residual,
                    "target_site": target_site, "actual_site": self.data.site_xpos[self.site_id].copy(),
                }
        return {
            "settled": True, "cartesian": True, "cartesian_segments": segments,
            "completed_segments": segments, "minimum_joint_margin_rad": minimum_margin,
            "first_unintended_contact_step": first_unintended_contact_step,
            "first_table_contact_step": first_table_contact_step,
            "maximum_tracking_residual_m": max_tracking_residual,
            "target_site": target_site, "actual_site": self.data.site_xpos[self.site_id].copy(),
        }

    # ---- Frozen unified staged pre-contact controller -----------------

    def _closed_tip_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.site_xpos[self.closed_tip_site_id].copy(),
            self.data.site_xmat[self.closed_tip_site_id].reshape(3, 3).copy(),
        )

    def _strike_to_closed_tip_local(self) -> np.ndarray:
        """Read the rigid strike-site -> closed-tip transform from the model."""
        strike = self.data.site_xpos[self.site_id]
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        tip = self.data.site_xpos[self.closed_tip_site_id]
        return rotation.T @ (tip - strike)

    @staticmethod
    def _branch_rotation(rotation: np.ndarray, branch: int) -> np.ndarray:
        if branch not in (-1, 1):
            raise ValueError(f"invalid closed-gripper branch {branch}")
        return np.asarray(rotation, dtype=float) @ np.diag([float(branch), float(branch), 1.0])

    def _tip_to_strike(self, tip: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        return np.asarray(tip, dtype=float) - np.asarray(rotation, dtype=float) @ self._strike_to_closed_tip_local()

    def _mesh_vertices_tip_local(self) -> np.ndarray:
        """Return real closed-finger collision vertices in closed-tip coordinates."""
        tip, rotation = self._closed_tip_pose()
        points: list[np.ndarray] = []
        for geom_id in self.finger_ids.values():
            mesh = int(self.model.geom_dataid[geom_id])
            first = int(self.model.mesh_vertadr[mesh])
            count = int(self.model.mesh_vertnum[mesh])
            vertices = self.model.mesh_vert[first:first + count]
            world = vertices @ self.data.geom_xmat[geom_id].reshape(3, 3).T + self.data.geom_xpos[geom_id]
            points.append((world - tip) @ rotation)
        return np.concatenate(points)

    @staticmethod
    def _mesh_world(tip: np.ndarray, rotation: np.ndarray, local_vertices: np.ndarray) -> np.ndarray:
        return np.asarray(local_vertices, dtype=float) @ np.asarray(rotation, dtype=float).T + np.asarray(tip, dtype=float)

    @staticmethod
    def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
        return float(math.acos(np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0)))

    def _rotation_samples(self, first: np.ndarray, second: np.ndarray) -> list[np.ndarray]:
        angular_increment = math.radians(5.0)
        count = max(1, int(math.ceil(self._rotation_angle(first, second) / angular_increment)))
        return [self._rotation_slerp(first, second, float(i) / count) for i in range(count + 1)]

    def _transport_rotation(self, measured_rotation: np.ndarray) -> np.ndarray:
        """Closest downward-tip frame retaining the measured planar heading."""
        z_axis = np.asarray(self.config["strike_interface"]["strike_tip_axis_world"], dtype=float)
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.asarray(measured_rotation, dtype=float)[:, 0].copy()
        x_axis -= z_axis * float(np.dot(x_axis, z_axis))
        if float(np.linalg.norm(x_axis)) < 1.0e-8:
            x_axis = np.asarray(measured_rotation, dtype=float)[:, 1].copy()
            x_axis -= z_axis * float(np.dot(x_axis, z_axis))
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def _cube_top_z(self) -> float:
        position = self.data.geom_xpos[self.puck_id]
        rotation = self.data.geom_xmat[self.puck_id].reshape(3, 3)
        half = np.asarray(self.model.geom_size[self.puck_id][:3], dtype=float)
        corners = np.asarray([
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
        ])
        return float(np.max(corners @ rotation[2, :] + position[2]))

    def _transport_height(
        self, local_vertices: np.ndarray, start_rotation: np.ndarray, transport_rotation: np.ndarray,
    ) -> tuple[float, float]:
        """Minimum closed-tip z that clears cube/table for the measured->transport rotation."""
        clearance = float(self.config["strike_interface"]["approach_gap_m"])
        min_relative_z = min(
            float(np.min(local_vertices @ rotation[2, :]))
            for rotation in self._rotation_samples(start_rotation, transport_rotation)
        )
        table_z = float(self.config["task"]["table_center_xyz_m"][2]) + float(self.config["task"]["table_half_size_xyz_m"][2])
        return max(table_z, self._cube_top_z()) + clearance - min_relative_z, clearance

    def _current_transport_clearance(self, local_vertices: np.ndarray) -> tuple[bool, float, float, float]:
        """Check the actual current real-finger mesh against the clearance lower bound."""
        tip, rotation = self._closed_tip_pose()
        world_vertices = self._mesh_world(tip, rotation, local_vertices)
        minimum_mesh_z = float(np.min(world_vertices[:, 2]))
        cube_top = self._cube_top_z()
        table_top = float(self.config["task"]["table_center_xyz_m"][2]) + float(self.config["task"]["table_half_size_xyz_m"][2])
        clearance = float(self.config["strike_interface"]["approach_gap_m"])
        required_z = max(cube_top, table_top) + clearance
        return minimum_mesh_z >= required_z, minimum_mesh_z, required_z, clearance

    def _yaw_acquisition_standoff(
        self,
        surface_xy: np.ndarray,
        tip_z: float,
        direction: np.ndarray,
        local_vertices: np.ndarray,
        transport_rotation: np.ndarray,
        push_rotation: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Real-mesh swept support gives the one collision-certified yaw standoff."""
        d3 = np.r_[np.asarray(direction, dtype=float), 0.0]
        maximum_support = max(
            float(np.max((local_vertices @ rotation.T) @ d3))
            for rotation in self._rotation_samples(transport_rotation, push_rotation)
        )
        clearance = float(self.config["strike_interface"]["approach_gap_m"])
        return np.r_[np.asarray(surface_xy, dtype=float) - np.asarray(direction, dtype=float) * (maximum_support + clearance), float(tip_z)], maximum_support, clearance

    def _probe_arm_state(self, arm_q: np.ndarray) -> dict[str, Any]:
        """Static real-geometry candidate check without modifying the live state."""
        probe = mujoco.MjData(self.model)
        probe.qpos[:] = self.data.qpos
        probe.qvel[:] = self.data.qvel
        probe.ctrl[:] = self.data.ctrl
        set_robot_state(self.model, probe, self.config, arm_q)
        mujoco.mj_forward(self.model, probe)
        finger_puck = pair_contact(self.model, probe, set(self.finger_ids.values()), {self.puck_id})
        gripper_table = pair_contact(self.model, probe, self.native_gripper_ids, {self.table_id})
        return {
            "finger_puck": bool(finger_puck),
            "gripper_table": bool(gripper_table),
            "unexpected_contacts": unexpected_contacts(self.model, probe),
            "finite": bool(np.all(np.isfinite(probe.qpos)) and np.all(np.isfinite(probe.qvel))),
        }

    def _solve_tip_pose(
        self, tip: np.ndarray, rotation: np.ndarray, seed: np.ndarray, phase: str,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """One fixed-branch constrained IK query; no mid-path fallback/branch flip."""
        strike = self._tip_to_strike(tip, rotation)
        result = solve_strike_ik(
            self.model, self.config, strike, seed,
            desired_tip_axis_world=rotation[:, 2],
            desired_opening_axis_world=rotation[:, 1],
        )
        self._diagnostic(
            "ik_query", phase=phase, branch="staged_fixed", orientation=True,
            desired_pose=strike.copy(), warm_start_q=np.asarray(seed).copy(),
            desired_tip_axis_world=rotation[:, 2].copy(), desired_opening_axis_world=rotation[:, 1].copy(),
            converged=bool(result["converged"]), returned_q=np.asarray(result["q"]).copy(),
            position=np.asarray(result["position"]).copy(), position_error_m=float(result["position_error_m"]),
            orientation_axis_error_rad=float(result["orientation_axis_error_rad"]),
            opening_axis_error_rad=float(result["opening_axis_error_rad"]), iterations=int(result["iterations"]),
            minimum_joint_margin_rad=float(result["minimum_joint_margin_rad"]),
            unexpected_contacts=list(result["unexpected_contacts"]),
        )
        return (np.asarray(result["q"]) if result["converged"] else None), result

    def _shadow_segment(self, target_q: np.ndarray, duration: float, phase: str) -> dict[str, Any]:
        """Replay the exact low-level joint interpolation on an isolated MjData state."""
        shadow = DirectPushEnv(self.config, friction=float(self.config["task"]["table_friction"][0]))
        shadow.data.qpos[:] = self.data.qpos
        shadow.data.qvel[:] = self.data.qvel
        shadow.data.ctrl[:] = self.data.ctrl
        shadow.data.time = float(self.data.time)
        shadow.current_q = self.current_q.copy()
        shadow.time_step_index = self.time_step_index
        mujoco.mj_forward(shadow.model, shadow.data)
        trace: list[dict[str, Any]] = []
        tracking = shadow.move(target_q, duration, trace, f"{phase}_shadow")
        premature = any(sample["finger_puck"] for sample in trace)
        table = any(sample["gripper_table"] for sample in trace)
        unexpected = [contact for sample in trace for contact in sample["unexpected_contacts"]]
        finite = all(sample["finite"] for sample in trace)
        return {
            "settled": bool(tracking["settled"]), "premature_finger_puck": premature,
            "gripper_table": table, "unexpected_contacts": sorted(set(unexpected)), "finite": finite,
            "position_residual_m": float(tracking["position_residual_m"]),
        }

    def _execute_tip_segment(
        self,
        tip: np.ndarray,
        rotation: np.ndarray,
        duration: float,
        trace: list[dict[str, Any]],
        phase: str,
        *,
        allow_subdivision: bool,
    ) -> dict[str, Any]:
        """One certified task-space segment, with at most one same-path bisection."""
        target_q, ik = self._solve_tip_pose(tip, rotation, self._actual_q(), f"{phase}_ik")
        continuation_ik: dict[str, Any] | None = None
        if target_q is None:
            # One bounded numerical continuation of the exact same constrained
            # pose.  The returned iterate is not a new target, branch, route,
            # waypoint, tolerance, or home-seed fallback.
            target_q, continuation_ik = self._solve_tip_pose(
                tip, rotation, np.asarray(ik["q"]), f"{phase}_ik_continuation",
            )
        if target_q is None:
            if allow_subdivision:
                current_tip, current_rotation = self._closed_tip_pose()
                midpoint_tip = 0.5 * (current_tip + np.asarray(tip, dtype=float))
                midpoint_rotation = self._rotation_slerp(current_rotation, rotation, 0.5)
                first = self._execute_tip_segment(midpoint_tip, midpoint_rotation, duration * 0.5, trace, phase, allow_subdivision=False)
                if not first["ok"]:
                    return first
                return self._execute_tip_segment(tip, rotation, duration * 0.5, trace, phase, allow_subdivision=False)
            return {
                "ok": False, "cause": "INTERMEDIATE_POSE", "stage": phase,
                "ik": ik, "continuation_ik": continuation_ik,
            }
        candidate = self._probe_arm_state(target_q)
        if (not candidate["finite"] or candidate["finger_puck"] or candidate["gripper_table"] or candidate["unexpected_contacts"]):
            return {"ok": False, "cause": "PATH_COLLISION", "stage": phase, "candidate": candidate, "ik": ik, "continuation_ik": continuation_ik}
        shadow = self._shadow_segment(target_q, duration, phase)
        if (not shadow["settled"] or not shadow["finite"] or shadow["premature_finger_puck"] or shadow["gripper_table"] or shadow["unexpected_contacts"]):
            return {"ok": False, "cause": "PATH_COLLISION" if (shadow["premature_finger_puck"] or shadow["gripper_table"] or shadow["unexpected_contacts"]) else "TRACKING", "stage": phase, "shadow": shadow, "candidate": candidate, "ik": ik, "continuation_ik": continuation_ik}
        trace_start = len(trace)
        tracking = self.move(target_q, duration, trace, phase)
        live = trace[trace_start:]
        live_finger = any(sample["finger_puck"] for sample in live)
        live_table = any(sample["gripper_table"] for sample in live)
        live_unexpected = sorted({contact for sample in live for contact in sample["unexpected_contacts"]})
        live_finite = all(sample["finite"] for sample in live)
        actual_tip, actual_rotation = self._closed_tip_pose()
        position_residual = float(np.linalg.norm(actual_tip - tip))
        orientation_residual = self._rotation_angle(actual_rotation, rotation)
        ranges = _joint_addresses(self.model, self.names)[2]
        actual_q = self._actual_q()
        margin = float(np.min(np.minimum(actual_q - ranges[:, 0], ranges[:, 1] - actual_q)))
        result = {
            "ok": bool(tracking["settled"] and not live_finger and not live_table and not live_unexpected and live_finite),
            "cause": "" if tracking["settled"] and not live_finger and not live_table and not live_unexpected and live_finite else ("PATH_COLLISION" if (live_finger or live_table or live_unexpected) else "TRACKING"),
            "stage": phase, "candidate": candidate, "shadow": shadow,
            "live": {"finger_puck": live_finger, "gripper_table": live_table, "unexpected_contacts": live_unexpected, "finite": live_finite},
            "position_residual_m": position_residual, "orientation_residual_rad": orientation_residual,
            "minimum_joint_margin_rad": margin, "ik": ik, "continuation_ik": continuation_ik,
        }
        return result

    def _execute_tip_path(
        self,
        target_tip: np.ndarray,
        target_rotation: np.ndarray,
        duration: float,
        trace: list[dict[str, Any]],
        phase: str,
    ) -> dict[str, Any]:
        start_tip, start_rotation = self._closed_tip_pose()
        distance = float(np.linalg.norm(np.asarray(target_tip) - start_tip))
        angle = self._rotation_angle(start_rotation, target_rotation)
        segments = max(1, int(math.ceil(max(distance / 0.003, angle / math.radians(5.0)))))
        records: list[dict[str, Any]] = []
        for index in range(segments):
            alpha = float(index + 1) / segments
            desired_tip = (1.0 - alpha) * start_tip + alpha * np.asarray(target_tip, dtype=float)
            desired_rotation = self._rotation_slerp(start_rotation, target_rotation, alpha)
            record = self._execute_tip_segment(desired_tip, desired_rotation, max(0.03, duration / segments), trace, phase, allow_subdivision=True)
            records.append(record)
            if not record["ok"]:
                return {"ok": False, "stage": phase, "cause": record["cause"], "segments": records}
        return {"ok": True, "stage": phase, "segments": records}

    def _select_final_push_pose(
        self, center: np.ndarray, direction: np.ndarray, magnitude: float, local_vertices: np.ndarray,
    ) -> dict[str, Any] | None:
        """Endpoint-only bounded branch selection for the frozen final pushing frame."""
        surface = self.contact_surface(center, direction)
        nominal_strike = self.goal(center, direction, "pre", magnitude)
        push_rotation = self.commanded_pushing_rotation(direction)
        home = np.asarray(self.config["robot"]["home_joint_positions_rad"], dtype=float)
        queries = ((1, self._actual_q(), "endpoint_warm"), (-1, self._actual_q(), "endpoint_symmetric"), (-1, home, "endpoint_home"))
        for branch, seed, label in queries:
            rotation = self._branch_rotation(push_rotation, branch)
            nominal_tip = nominal_strike + rotation @ self._strike_to_closed_tip_local()
            d3 = np.r_[direction, 0.0]
            support = float(np.max((local_vertices @ rotation.T) @ d3))
            mesh_pre_tip = np.r_[surface - direction * (support + float(self.config["strike_interface"]["approach_gap_m"])), nominal_tip[2]]
            contact_tip = np.r_[surface - direction * support + direction * float(self.config["ik"]["position_tolerance_m"]), nominal_tip[2]]
            pre_q, pre_ik = self._solve_tip_pose(mesh_pre_tip, rotation, seed, f"mesh_pre_{label}")
            if pre_q is None:
                continue
            contact_q, contact_ik = self._solve_tip_pose(contact_tip, rotation, pre_q, f"contact_{label}")
            if contact_q is None:
                continue
            candidate = self._probe_arm_state(pre_q)
            if candidate["finger_puck"] or candidate["gripper_table"] or candidate["unexpected_contacts"] or not candidate["finite"]:
                continue
            return {
                "branch": branch, "push_rotation": rotation, "surface_xy": surface,
                "mesh_pre_tip": mesh_pre_tip, "contact_tip": contact_tip,
                "mesh_pre_q": pre_q, "contact_q": contact_q, "support_m": support,
                "mesh_pre_ik": pre_ik, "contact_ik": contact_ik,
            }
        return None

    def execute_staged_precontact(
        self, center: np.ndarray, direction: np.ndarray, magnitude: float, trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """The sole production pre-contact path: lift, transport, yaw, normal approach, contact."""
        start_tip, start_rotation = self._closed_tip_pose()
        local_vertices = self._mesh_vertices_tip_local()
        final = self._select_final_push_pose(center, direction, magnitude, local_vertices)
        if final is None:
            return {"ok": False, "cause": "FINAL_CONSTRAINED_TARGET", "stage": "MEASURE_CONSTRUCT"}
        transport_rotation = self._transport_rotation(start_rotation)
        transport_height, clearance = self._transport_height(local_vertices, start_rotation, transport_rotation)
        already_safe, minimum_mesh_z, required_mesh_z, _ = self._current_transport_clearance(local_vertices)
        # The geometry-derived transport height is a lower bound, not a target
        # that should force a safe high start downward. High transport keeps
        # the measured reachable frame; only an actually-below-bound start is
        # lifted, and never lowered, to the conservative bound.
        transport_z = max(float(start_tip[2]), float(transport_height))
        standoff_tip, swept_support, standoff_clearance = self._yaw_acquisition_standoff(
            final["surface_xy"], final["mesh_pre_tip"][2], direction, local_vertices,
            transport_rotation, final["push_rotation"],
        )
        stages = (
            ("CLEARANCE_LIFT", np.r_[start_tip[:2], transport_z], start_rotation, 0.30),
            ("SAFE_TRANSPORT", np.r_[standoff_tip[:2], transport_z], start_rotation, 0.30),
            # Acquire the conservative transport frame only at the existing
            # standoff, after high lateral transport has remained in measured
            # R0.  This preserves the original set-down semantics without
            # forcing R_transport through the distant/high path.
            ("SET_DOWN", standoff_tip, transport_rotation, 0.30),
            ("PUSH_YAW_ACQUISITION", standoff_tip, final["push_rotation"], 0.30),
            ("FINAL_NORMAL_APPROACH", final["mesh_pre_tip"], final["push_rotation"], 0.18),
        )
        stage_records: list[dict[str, Any]] = []
        for name, tip, rotation, duration in stages:
            outcome = self._execute_tip_path(tip, rotation, duration, trace, name)
            stage_records.append(outcome)
            if not outcome["ok"]:
                return {
                    "ok": False, "cause": outcome["cause"], "stage": name,
                    "branch": final["branch"], "transport_height_m": transport_height,
                    "transport_target_height_m": transport_z, "already_transport_safe": already_safe,
                    "minimum_mesh_z_m": minimum_mesh_z, "required_mesh_z_m": required_mesh_z,
                    "acquisition_standoff_tip": standoff_tip.tolist(), "clearance_m": clearance,
                    "swept_support_m": swept_support, "stage_records": stage_records,
                }
        contact_q, contact_ik = self._solve_tip_pose(final["contact_tip"], final["push_rotation"], self._actual_q(), "FIRST_CONTACT_ik")
        if contact_q is None:
            return {"ok": False, "cause": "INTERMEDIATE_POSE", "stage": "FIRST_CONTACT", "branch": final["branch"], "stage_records": stage_records, "contact_ik": contact_ik}
        if not self.move_until_contact(contact_q, 0.18, trace):
            return {"ok": False, "cause": "TRACKING", "stage": "FIRST_CONTACT", "branch": final["branch"], "stage_records": stage_records}
        return {
            "ok": True, "branch": final["branch"], "transport_height_m": transport_height,
            "transport_target_height_m": transport_z, "already_transport_safe": already_safe,
            "minimum_mesh_z_m": minimum_mesh_z, "required_mesh_z_m": required_mesh_z,
            "acquisition_standoff_tip": standoff_tip.tolist(), "clearance_m": clearance,
            "standoff_clearance_m": standoff_clearance, "swept_support_m": swept_support,
            "stage_records": stage_records,
        }

    def closed_side_normal(self, q: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Select the symmetric distal +/-x side whose outward normal faces d."""
        probe = mujoco.MjData(self.model)
        set_robot_state(self.model, probe, self.config, q)
        mujoco.mj_forward(self.model, probe)
        axis = probe.site_xmat[self.site_id].reshape(3, 3)[:, 0]
        return -axis if float(np.dot(-axis[:2], direction)) >= float(np.dot(axis[:2], direction)) else axis

    def distal_mesh_support(self, q: np.ndarray, direction: np.ndarray) -> tuple[float, np.ndarray]:
        """Actual paired distal-end mesh support along the planned side-wall normal."""
        probe = mujoco.MjData(self.model)
        set_robot_state(self.model, probe, self.config, q)
        mujoco.mj_forward(self.model, probe)
        site = probe.site_xpos[self.site_id]
        rotation = probe.site_xmat[self.site_id].reshape(3, 3)
        points: list[np.ndarray] = []
        for geom_id in self.finger_ids.values():
            mesh = int(self.model.geom_dataid[geom_id]); first = int(self.model.mesh_vertadr[mesh]); count = int(self.model.mesh_vertnum[mesh])
            vertices = self.model.mesh_vert[first:first + count]
            world = vertices @ probe.geom_xmat[geom_id].reshape(3, 3).T + probe.geom_xpos[geom_id]
            local = (world - site) @ rotation
            distal = world[np.isclose(local[:, 2], np.max(local[:, 2]), atol=1.0e-6)]
            points.append(distal)
        distal_points = np.concatenate(points)
        support = float(np.max((distal_points - site) @ np.r_[direction, 0.0]))
        return support, np.mean(distal_points, axis=0)

    def move(self, target: np.ndarray, duration: float, trace: list[dict[str, Any]], phase: str) -> dict[str, Any]:
        start = self._actual_q()
        self._diagnostic(
            "move_start", phase=phase, start_q=start.copy(), start_site=self.data.site_xpos[self.site_id].copy(),
            target_q=np.asarray(target).copy(), target_site=self._target_site(target), duration_s=float(duration),
        )
        steps = max(1, round(duration / self.model.opt.timestep))
        for step in range(steps):
            alpha = _smoothstep((step + 1) / steps)
            self.data.ctrl[:] = (1.0 - alpha) * start + alpha * target
            mujoco.mj_step(self.model, self.data)
            self.time_step_index += 1
            trace.append(self.sample(phase))
        expected = self._target_site(target)
        tolerance = float(self.config["ik"]["position_tolerance_m"])
        # Hold until measured convergence, bounded by four existing controller
        # windows.  The budget is a safety timeout; it is never a success signal.
        settled = False
        for _ in range(max(1, round(4.0 * float(self.config["control"]["duration_s"]) / self.model.opt.timestep))):
            self.data.ctrl[:] = target
            mujoco.mj_step(self.model, self.data)
            self.time_step_index += 1
            trace.append(self.sample(phase))
            if float(np.linalg.norm(self.data.site_xpos[self.site_id] - expected)) <= tolerance:
                settled = True
                break
        actual = self.data.site_xpos[self.site_id].copy()
        self.current_q = self._actual_q()
        result = {"settled": settled, "target_site": expected, "actual_site": actual, "position_residual_m": float(np.linalg.norm(actual - expected)), "joint_residual_rad": float(np.max(np.abs(self.current_q - target)))}
        self._diagnostic("move_end", phase=phase, end_q=self.current_q.copy(), end_site=actual.copy(), **result)
        return result

    def move_push_after_touch(
        self,
        target: np.ndarray,
        duration: float,
        trace: list[dict[str, Any]],
        direction: np.ndarray,
        after_touch_displacement: float,
    ) -> dict[str, Any]:
        """Stop the existing push path after a fixed EE displacement from first contact."""
        start = self._actual_q()
        self._diagnostic("push_motion_start", phase="push", start_q=start.copy(), start_site=self.data.site_xpos[self.site_id].copy(), target_q=np.asarray(target).copy(), target_site=self._target_site(target), direction=np.asarray(direction).copy(), requested_after_touch_m=float(after_touch_displacement), duration_s=float(duration))
        steps = max(1, round(duration / self.model.opt.timestep))
        first_contact_site: np.ndarray | None = None
        achieved = 0.0
        reached = False
        for step in range(steps):
            alpha = _smoothstep((step + 1) / steps)
            command = (1.0 - alpha) * start + alpha * target
            self.data.ctrl[:] = command
            mujoco.mj_step(self.model, self.data)
            self.time_step_index += 1
            sample = self.sample("push")
            trace.append(sample)
            site_xy = np.asarray([sample["site_x"], sample["site_y"]], dtype=float)
            if first_contact_site is None and sample["finger_puck"]:
                first_contact_site = site_xy.copy()
            if first_contact_site is not None:
                achieved = float(np.dot(site_xy - first_contact_site, direction))
                if achieved >= after_touch_displacement:
                    self.current_q = self._actual_q()
                    reached = True
                    break
        if not reached:
            # The commanded Cartesian endpoint is unchanged; continue holding it
            # until the measured after-touch displacement is reached or the same
            # bounded execution budget used by the other phases expires.
            for _ in range(max(1, round(4.0 * float(self.config["control"]["duration_s"]) / self.model.opt.timestep))):
                self.data.ctrl[:] = target
                mujoco.mj_step(self.model, self.data)
                self.time_step_index += 1
                sample = self.sample("push")
                trace.append(sample)
                site_xy = np.asarray([sample["site_x"], sample["site_y"]], dtype=float)
                if first_contact_site is None and sample["finger_puck"]:
                    first_contact_site = site_xy.copy()
                if first_contact_site is not None:
                    achieved = float(np.dot(site_xy - first_contact_site, direction))
                    if achieved >= after_touch_displacement:
                        reached = True
                        break
        if not reached:
            self.current_q = self._actual_q()
        result = {
            "first_contact_detected": first_contact_site is not None,
            "after_touch_displacement_achieved_m": achieved,
            "after_touch_displacement_reached": reached,
        }
        self._diagnostic("push_motion_end", phase="push", end_q=self.current_q.copy(), end_site=self.data.site_xpos[self.site_id].copy(), first_contact_site=first_contact_site, **result)
        return result

    def move_until_contact(self, target: np.ndarray, duration: float, trace: list[dict[str, Any]]) -> bool:
        """Bounded acquisition stroke; first physical finger contact defines push start."""
        start = self._actual_q()
        self._diagnostic("contact_motion_start", phase="contact", start_q=start.copy(), start_site=self.data.site_xpos[self.site_id].copy(), target_q=np.asarray(target).copy(), target_site=self._target_site(target), duration_s=float(duration))
        steps = max(1, round(duration / self.model.opt.timestep))
        for step in range(steps):
            alpha = _smoothstep((step + 1) / steps)
            self.data.ctrl[:] = (1.0 - alpha) * start + alpha * target
            mujoco.mj_step(self.model, self.data); self.time_step_index += 1
            trace.append(self.sample("contact"))
            if trace[-1]["finger_puck"]:
                self.current_q = self._actual_q()
                self._diagnostic("contact_motion_end", phase="contact", established=True, end_q=self.current_q.copy(), end_site=self.data.site_xpos[self.site_id].copy())
                return True
        for _ in range(max(1, round(4.0 * float(self.config["control"]["duration_s"]) / self.model.opt.timestep))):
            self.data.ctrl[:] = target
            mujoco.mj_step(self.model, self.data); self.time_step_index += 1
            trace.append(self.sample("contact"))
            if trace[-1]["finger_puck"]:
                self.current_q = self._actual_q()
                self._diagnostic("contact_motion_end", phase="contact", established=True, end_q=self.current_q.copy(), end_site=self.data.site_xpos[self.site_id].copy())
                return True
        self.current_q = self._actual_q()
        self._diagnostic("contact_motion_end", phase="contact", established=False, end_q=self.current_q.copy(), end_site=self.data.site_xpos[self.site_id].copy())
        return False

    def move_safe_precontact(
        self, target: np.ndarray, duration: float, trace: list[dict[str, Any]], phase: str,
    ) -> dict[str, Any]:
        """Track one acquisition segment and reject any premature mesh contact."""
        start = len(trace)
        result = self.move(target, duration, trace, phase)
        result["premature_finger_puck_contact"] = any(sample["finger_puck"] for sample in trace[start:])
        return result

    def _finger_puck_contact_records(self) -> list[dict[str, Any]]:
        """Return every native finger--cube contact in the current model state."""
        records: list[dict[str, Any]] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            finger_id = first if first in self.finger_ids.values() else second if second in self.finger_ids.values() else None
            if finger_id is None or (second if finger_id == first else first) != self.puck_id:
                continue
            records.append({
                "geom": self.model.geom(finger_id).name,
                "position": np.asarray(contact.pos, dtype=float).tolist(),
                "normal": np.asarray(contact.frame[:3], dtype=float).tolist(),
                "distance_m": float(contact.dist),
            })
        return records

    def original_preapproach_preflight(self, high_q: np.ndarray, pre_q: np.ndarray) -> dict[str, Any]:
        """Non-destructively replay original high->pre motion and reject early contact.

        This deliberately uses a separate MuJoCo data/model instance.  It is a
        geometry/path predicate, not a second live route or a policy heuristic.
        The intended contact stroke is excluded: any native finger--cube contact
        during high repositioning or planar pre-approach is premature.
        """
        live_qpos = self.data.qpos.copy()
        live_qvel = self.data.qvel.copy()
        live_ctrl = self.data.ctrl.copy()
        live_time = float(self.data.time)
        live_current_q = self.current_q.copy()
        shadow = DirectPushEnv(self.config, friction=float(self.config["task"]["table_friction"][0]))
        shadow.data.qpos[:] = live_qpos
        shadow.data.qvel[:] = live_qvel
        shadow.data.ctrl[:] = live_ctrl
        shadow.data.time = live_time
        shadow.current_q = live_current_q.copy()
        shadow.time_step_index = self.time_step_index
        mujoco.mj_forward(shadow.model, shadow.data)
        trace: list[dict[str, Any]] = []
        contacts: list[dict[str, Any]] = []
        for target, duration, phase in ((high_q, 0.30, "reposition_high"), (pre_q, 0.18, "approach")):
            start = len(trace)
            tracking = shadow.move(target, duration, trace, phase)
            for sample_index in range(start, len(trace)):
                if not trace[sample_index]["finger_puck"]:
                    continue
                # The post-step shadow state holds the contact set for this
                # sample; keep a compact geometry record for audit artifacts.
                records = trace[sample_index]["finger_puck_contacts"]
                contacts.append({"phase": phase, "sample_index": sample_index, "records": records})
                break
            if contacts:
                break
            if not tracking["settled"]:
                # A path that cannot reach its target is not certified safe.
                contacts.append({"phase": phase, "sample_index": -1, "records": [], "tracking_unsettled": True})
                break
        live_unchanged = (
            np.array_equal(self.data.qpos, live_qpos)
            and np.array_equal(self.data.qvel, live_qvel)
            and np.array_equal(self.data.ctrl, live_ctrl)
            and float(self.data.time) == live_time
            and np.array_equal(self.current_q, live_current_q)
        )
        if not live_unchanged:
            raise RuntimeError("original pre-approach shadow preflight mutated live state")
        return {
            "unsafe": bool(contacts),
            "reason": "premature_finger_puck_contact" if contacts and "tracking_unsettled" not in contacts[0] else ("shadow_tracking_unsettled" if contacts else "clear"),
            "contacts": contacts,
            "live_state_unchanged": live_unchanged,
        }

    def sample(self, phase: str) -> dict[str, Any]:
        finger1_puck = pair_contact(self.model, self.data, {self.finger_ids["gripper_link1_collision_1"]}, {self.puck_id})
        finger2_puck = pair_contact(self.model, self.data, {self.finger_ids["gripper_link2_collision_1"]}, {self.puck_id})
        finger_puck = finger1_puck or finger2_puck
        finger_puck_contacts = self._finger_puck_contact_records()
        gripper_table = pair_contact(self.model, self.data, self.native_gripper_ids, {self.table_id})
        return {
            "step": self.time_step_index,
            "time_s": float(self.data.time),
            "phase": phase,
            "puck_x": float(self.data.qpos[self.puck_qpos]),
            "puck_y": float(self.data.qpos[self.puck_qpos + 1]),
            "puck_vx": float(self.data.qvel[self.puck_dof]),
            "puck_vy": float(self.data.qvel[self.puck_dof + 1]),
            "site_x": float(self.data.site_xpos[self.site_id, 0]),
            "site_y": float(self.data.site_xpos[self.site_id, 1]),
            "site_z": float(self.data.site_xpos[self.site_id, 2]),
            "finger_puck": finger_puck,
            "finger_puck_contacts": finger_puck_contacts,
            "finger1_puck": finger1_puck,
            "finger2_puck": finger2_puck,
            "dual_finger_puck": finger1_puck and finger2_puck,
            "gripper_joint1_m": float(self.data.qpos[self.gripper_qpos["gripper_joint1"]]),
            "gripper_joint2_m": float(self.data.qpos[self.gripper_qpos["gripper_joint2"]]),
            "gripper_aperture_m": float(
                self.data.qpos[self.gripper_qpos["gripper_joint1"]]
                - self.data.qpos[self.gripper_qpos["gripper_joint2"]]
            ),
            "gripper_table": gripper_table,
            "unexpected_contacts": unexpected_contacts(self.model, self.data),
            "finite": bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))),
        }

    def push(self, direction_rad: float, magnitude: float, speed: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        center = self.puck_xy()
        before_v = self.puck_v()
        unit = np.array([math.cos(direction_rad), math.sin(direction_rad)])
        self._diagnostic(
            "macro_start",
            puck_xy=center.copy(),
            ee_pose=self.data.site_xpos[self.site_id].copy(),
            current_q=self._actual_q(),
            direction=unit.copy(),
            direction_rad=float(direction_rad),
            magnitude_m=float(magnitude),
            speed_mps=float(speed),
        )
        trace: list[dict[str, Any]] = []
        precontact = self.execute_staged_precontact(center, unit, magnitude, trace)
        route_info = {
            "preapproach_route": "staged_task_space",
            "precontact_stage": precontact.get("stage", "FIRST_CONTACT" if precontact.get("ok") else "MEASURE_CONSTRUCT"),
            "precontact_failure_cause": precontact.get("cause", ""),
            "precontact_selected_branch": precontact.get("branch"),
            "precontact_transport_height_m": precontact.get("transport_height_m", float("nan")),
            "precontact_acquisition_standoff_tip": precontact.get("acquisition_standoff_tip"),
            "precontact_clearance_m": precontact.get("clearance_m", float("nan")),
            "precontact_swept_support_m": precontact.get("swept_support_m", float("nan")),
            "precontact_stage_records": precontact.get("stage_records", []),
        }
        if not precontact["ok"]:
            return {
                "unrecoverable": True, "reason": "precontact_execution_infeasible",
                "precontact_failure_cause": precontact["cause"],
                "puck_before_x": float(center[0]), "puck_before_y": float(center[1]),
                "command_direction_rad": float(direction_rad), "command_magnitude_m": float(magnitude),
                "command_speed_mps": float(speed), **route_info,
            }, trace
        requested_after_touch = float(magnitude)
        executed_after_touch = requested_after_touch
        endpoint = self.data.site_xpos[self.site_id].copy() + np.r_[unit * requested_after_touch, 0.0]
        post_q = self.solve(endpoint, self.current_q, direction=unit, phase="after_touch")
        # A measured-contact endpoint may lie just beyond a local orientation
        # reachability boundary.  Preserve an exact feasible request; only on
        # failure try the same direction in bounded 1-mm decrements (<=8 mm).
        if post_q is None:
            for trim_mm in range(1, 9):
                candidate = requested_after_touch - 0.001 * trim_mm
                if candidate <= 0.0:
                    break
                candidate_endpoint = self.data.site_xpos[self.site_id].copy() + np.r_[unit * candidate, 0.0]
                candidate_q = self.solve(candidate_endpoint, self.current_q, direction=unit, phase="after_touch_trim")
                if candidate_q is not None:
                    executed_after_touch, endpoint, post_q = candidate, candidate_endpoint, candidate_q
                    break
        if post_q is None:
            return {"unrecoverable": True, "reason": "after_touch_ik", **route_info}, trace
        path = executed_after_touch
        after_touch_displacement = (
            executed_after_touch
            if self.config.get("sustained_push", False)
            else self.config.get("direct_gripper", {}).get("contact_after_touch_displacement_m")
        )
        if after_touch_displacement is None:
            push_tracking = self.move(post_q, max(0.06, path / speed), trace, "push")
            if not push_tracking["settled"]:
                return {"unrecoverable": True, "reason": "push_tracking", "tracking": push_tracking, **route_info}, trace
            contact_standardization = {
                "first_contact_detected": False,
                "after_touch_displacement_achieved_m": float("nan"),
                "after_touch_displacement_reached": False,
            }
        else:
            contact_standardization = self.move_push_after_touch(
                post_q,
                max(0.06, path / speed),
                trace,
                unit,
                float(after_touch_displacement),
            )
        retract_goal = self.goal(center, unit, "retract", magnitude)
        retract_q = self.solve(retract_goal, self.current_q, orientation=False, phase="retract")
        if retract_q is None:
            return {"unrecoverable": True, "reason": "retract_ik", **route_info}, trace
        retract_tracking = self.move(retract_q, 0.12, trace, "retract")
        if not retract_tracking["settled"]:
            return {"unrecoverable": True, "reason": "retract_tracking", "tracking": retract_tracking, **route_info}, trace
        settle_steps = 0
        while settle_steps < round(2.0 / self.model.opt.timestep):
            self.data.ctrl[:] = retract_q
            mujoco.mj_step(self.model, self.data)
            self.time_step_index += 1
            trace.append(self.sample("settle"))
            settle_steps += 1
            if settle_steps > 25 and np.linalg.norm(self.puck_v()) < 0.005:
                break
        after = self.puck_xy()
        after_v = self.puck_v()
        contact_indices = [i for i, s in enumerate(trace) if s["finger_puck"]]
        finger1_indices = [i for i, s in enumerate(trace) if s["finger1_puck"]]
        finger2_indices = [i for i, s in enumerate(trace) if s["finger2_puck"]]
        dual_indices = [i for i, s in enumerate(trace) if s["dual_finger_puck"]]
        recontact = False
        if contact_indices:
            last_first_run = contact_indices[0]
            while last_first_run + 1 in contact_indices:
                last_first_run += 1
            recontact = any(i > last_first_run + 2 for i in contact_indices)
        table_contacts = sum(bool(s["gripper_table"]) for s in trace)
        actual_contact = trace[contact_indices[0]] if contact_indices else None
        bounds = self.workspace_bounds()
        oob = not (bounds[0] <= after[0] <= bounds[1] and bounds[2] <= after[1] <= bounds[3])
        return {
            "unrecoverable": False,
            **route_info,
            "simulation_step_before": trace[0]["step"] if trace else self.time_step_index,
            "simulation_step_after": self.time_step_index,
            "puck_before_x": float(center[0]), "puck_before_y": float(center[1]),
            "puck_after_x": float(after[0]), "puck_after_y": float(after[1]),
            "puck_delta_x": float(after[0] - center[0]), "puck_delta_y": float(after[1] - center[1]),
            "puck_velocity_before_x": float(before_v[0]), "puck_velocity_before_y": float(before_v[1]),
            "puck_velocity_after_x": float(after_v[0]), "puck_velocity_after_y": float(after_v[1]),
            "gripper_before_q": self.current_q.tolist(),
            "command_direction_rad": float(direction_rad), "command_magnitude_m": float(magnitude), "command_speed_mps": float(speed),
            "requested_after_touch_travel_m": requested_after_touch,
            "executed_after_touch_travel_m": executed_after_touch,
            "after_touch_endpoint_trim_m": requested_after_touch - executed_after_touch,
            "actual_contact_x": float(actual_contact["site_x"]) if actual_contact else float("nan"),
            "actual_contact_y": float(actual_contact["site_y"]) if actual_contact else float("nan"),
            "actual_contact_velocity_mps": float(speed) if actual_contact else float("nan"),
            "contact_standardization_mode": (
                "after_touch_displacement" if after_touch_displacement is not None else "legacy_full_stroke"
            ),
            "contact_after_touch_displacement_command_m": (
                float(after_touch_displacement) if after_touch_displacement is not None else float("nan")
            ),
            "contact_after_touch_displacement_achieved_m": float(
                contact_standardization["after_touch_displacement_achieved_m"]
            ),
            "contact_after_touch_displacement_reached": bool(
                contact_standardization["after_touch_displacement_reached"]
            ),
            "contact_duration_s": len(contact_indices) * float(self.model.opt.timestep),
            "finger1_puck_contact": bool(finger1_indices), "finger2_puck_contact": bool(finger2_indices),
            "dual_finger_contact": bool(dual_indices),
            "dual_finger_contact_duration_s": len(dual_indices) * float(self.model.opt.timestep),
            "gripper_puck_contact": bool(contact_indices), "gripper_table_contact": bool(table_contacts),
            "gripper_table_contact_frames": table_contacts, "recontact": recontact,
            "oob": oob, "safety_reset": False,
            "numerical_anomaly": not all(s["finite"] for s in trace),
            "dynamics_mass_kg": float(self.config["task"]["puck_mass_kg"]),
            "dynamics_friction": float(self.config["task"]["table_friction"][0]),
            "object_shape": "cylinder",
            "object_radius_m": float(self.config["task"]["puck_radius_m"]),
            "object_half_height_m": float(self.config["task"]["puck_half_height_m"]),
            "gripper_joint1_command_m": float(self.config["robot"]["gripper_joint_positions_m"]["gripper_joint1"]),
            "gripper_joint2_command_m": float(self.config["robot"]["gripper_joint_positions_m"]["gripper_joint2"]),
            "gripper_aperture_command_m": float(
                self.config["robot"]["gripper_joint_positions_m"]["gripper_joint1"]
                - self.config["robot"]["gripper_joint_positions_m"]["gripper_joint2"]
            ),
            "gripper_aperture_before_m": float(trace[0]["gripper_aperture_m"]),
            "gripper_aperture_contact_m": float(actual_contact["gripper_aperture_m"]) if actual_contact else float("nan"),
            "gripper_aperture_after_m": float(trace[-1]["gripper_aperture_m"]),
        }, trace

    def workspace_bounds(self) -> tuple[float, float, float, float]:
        task = self.config["task"]
        cx, cy, _ = task["table_center_xyz_m"]
        hx, hy, _ = task["table_half_size_xyz_m"]
        margin = float(task["puck_radius_m"]) + 0.025
        return cx - hx + margin, cx + hx - margin, cy - hy + margin, cy + hy - margin

    def exploration_bounds(self) -> tuple[float, float, float, float]:
        configured = self.config.get("direct_gripper", {}).get("exploration_bounds_xy_m")
        return tuple(float(v) for v in configured) if configured is not None else self.workspace_bounds()


def sample_action(rng: np.random.Generator, puck: np.ndarray, bounds: Sequence[float]) -> tuple[float, float, float]:
    angle = rng.uniform(-math.pi, math.pi)
    center = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2])
    half = np.array([(bounds[1] - bounds[0]) / 2, (bounds[3] - bounds[2]) / 2])
    normalized = (puck - center) / half
    inward = center - puck
    edge = float(np.max(np.abs(normalized)))
    boundary_strength = 0.0
    if edge > 0.70 and np.linalg.norm(inward) > 1e-9:
        inward_angle = math.atan2(inward[1], inward[0])
        strength = np.clip((edge - 0.70) / 0.30, 0.0, 1.0)
        boundary_strength = float(strength)
        candidate = np.array([math.cos(angle), math.sin(angle)])
        vector = (1.0 - strength) * candidate + strength * inward / np.linalg.norm(inward)
        angle = math.atan2(vector[1], vector[0])
    magnitude_high = 0.028 - 0.010 * boundary_strength
    speed_high = 0.50 - 0.15 * boundary_strength
    return float(angle), float(rng.uniform(0.008, magnitude_high)), float(rng.uniform(0.20, speed_high))


def run_collection(config: Mapping[str, Any], count: int, seed: int, output: Path, friction: float) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    env = DirectPushEnv(config, friction=friction)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    longest = 0
    current_run = 0
    for interaction in range(count):
        action_attempts = 0
        while True:
            angle, magnitude, speed = sample_action(rng, env.puck_xy(), env.exploration_bounds())
            row, trace = env.push(angle, magnitude, speed)
            action_attempts += 1
            if not row.get("unrecoverable") or action_attempts >= 16:
                break
        row["interaction_id"] = interaction
        row["action_sampling_attempts"] = action_attempts
        current_xy = env.puck_xy()
        current_v = env.puck_v()
        aperture_observed = float(
            env.data.qpos[env.gripper_qpos["gripper_joint1"]]
            - env.data.qpos[env.gripper_qpos["gripper_joint2"]]
        )
        row.setdefault("object_shape", "cylinder")
        row.setdefault("object_radius_m", float(config["task"]["puck_radius_m"]))
        row.setdefault("object_half_height_m", float(config["task"]["puck_half_height_m"]))
        row.setdefault("gripper_joint1_command_m", float(config["robot"]["gripper_joint_positions_m"]["gripper_joint1"]))
        row.setdefault("gripper_joint2_command_m", float(config["robot"]["gripper_joint_positions_m"]["gripper_joint2"]))
        row.setdefault("gripper_aperture_command_m", float(
            config["robot"]["gripper_joint_positions_m"]["gripper_joint1"]
            - config["robot"]["gripper_joint_positions_m"]["gripper_joint2"]
        ))
        row.setdefault("gripper_aperture_before_m", aperture_observed)
        row.setdefault("gripper_aperture_contact_m", float("nan"))
        row.setdefault("gripper_aperture_after_m", aperture_observed)
        row.setdefault("puck_before_x", float(current_xy[0]))
        row.setdefault("puck_before_y", float(current_xy[1]))
        row.setdefault("puck_after_x", float(current_xy[0]))
        row.setdefault("puck_after_y", float(current_xy[1]))
        row.setdefault("puck_delta_x", 0.0)
        row.setdefault("puck_delta_y", 0.0)
        row.setdefault("puck_velocity_before_x", float(current_v[0]))
        row.setdefault("puck_velocity_before_y", float(current_v[1]))
        row.setdefault("puck_velocity_after_x", float(current_v[0]))
        row.setdefault("puck_velocity_after_y", float(current_v[1]))
        contact_geoms = []
        if row.get("finger1_puck_contact"):
            contact_geoms.append("gripper_link1_collision_1")
        if row.get("finger2_puck_contact"):
            contact_geoms.append("gripper_link2_collision_1")
        row["actual_contact_geoms"] = "|".join(contact_geoms)
        if row.get("unrecoverable") or row.get("oob") or row.get("numerical_anomaly"):
            row["safety_reset"] = True
            row["reset_reason"] = row.get("reason", "oob_or_numerical")
            env.reset(config["task"]["puck_center_xy_m"])
            current_run = 0
        else:
            row["reset_reason"] = ""
            current_run += 1
            longest = max(longest, current_run)
        rows.append(row)
        if interaction < 12 or interaction % max(1, count // 20) == 0:
            all_trace.extend({"interaction_id": interaction, **sample} for sample in trace[::5])
    fields = sorted({key for row in rows for key in row})
    with (output / "interactions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    if all_trace:
        with (output / "replay_trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_trace[0]))
            writer.writeheader(); writer.writerows(all_trace)
    valid = [r for r in rows if not r.get("unrecoverable")]
    directions = np.array([r["command_direction_rad"] for r in valid]) if valid else np.array([])
    displacements = np.array([[r["puck_delta_x"], r["puck_delta_y"]] for r in valid]) if valid else np.empty((0, 2))
    bins = np.histogram(directions, bins=8, range=(-math.pi, math.pi))[0].tolist() if valid else [0] * 8
    summary = {
        "interactions": len(rows), "valid_interactions": len(valid), "longest_continuous_run": longest,
        "contact_rate": float(np.mean([r["gripper_puck_contact"] for r in valid])) if valid else 0.0,
        "gripper_table_contact_rate": float(np.mean([r["gripper_table_contact"] for r in valid])) if valid else 0.0,
        "oob_count": sum(bool(r.get("oob")) for r in rows),
        "safety_reset_count": sum(bool(r.get("safety_reset")) for r in rows),
        "numerical_anomaly_count": sum(bool(r.get("numerical_anomaly")) for r in rows),
        "direction_bin_counts": bins,
        "magnitude_range_m": [min((r["command_magnitude_m"] for r in valid), default=float("nan")), max((r["command_magnitude_m"] for r in valid), default=float("nan"))],
        "displacement_norm_quantiles_m": np.quantile(np.linalg.norm(displacements, axis=1), [0, .25, .5, .75, 1]).tolist() if valid else [],
        "workspace_x_range_m": [min((r["puck_before_x"] for r in valid), default=float("nan")), max((r["puck_before_x"] for r in valid), default=float("nan"))],
        "workspace_y_range_m": [min((r["puck_before_y"] for r in valid), default=float("nan")), max((r["puck_before_y"] for r in valid), default=float("nan"))],
        "proxy_physics_disabled": True,
        "native_contact_geoms": sorted(env.finger_ids),
        "finger1_contact_rate": float(np.mean([r.get("finger1_puck_contact", False) for r in valid])) if valid else 0.0,
        "finger2_contact_rate": float(np.mean([r.get("finger2_puck_contact", False) for r in valid])) if valid else 0.0,
        "dual_finger_contact_rate": float(np.mean([r.get("dual_finger_contact", False) for r in valid])) if valid else 0.0,
        "gripper_aperture_command_m": float(
            config["robot"]["gripper_joint_positions_m"]["gripper_joint1"]
            - config["robot"]["gripper_joint_positions_m"]["gripper_joint2"]
        ),
        "puck_half_height_m": float(config["task"]["puck_half_height_m"]),
        "friction": friction,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260811)
    parser.add_argument("--friction", type=float, default=0.20)
    parser.add_argument("--marker-offset", type=float, default=0.0115)
    parser.add_argument("--puck-half-height", type=float)
    parser.add_argument("--gripper-opening", type=float, default=0.025, help="Per-finger joint position in meters; aperture is twice this value.")
    parser.add_argument("--exploration-bounds", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config["direct_gripper"] = {"marker_above_puck_center_m": args.marker_offset}
    if args.exploration_bounds is not None:
        config["direct_gripper"]["exploration_bounds_xy_m"] = args.exploration_bounds
    if args.puck_half_height is not None:
        config["task"]["puck_half_height_m"] = args.puck_half_height
    config["robot"]["gripper_joint_positions_m"] = {
        "gripper_joint1": args.gripper_opening,
        "gripper_joint2": -args.gripper_opening,
    }
    run_collection(config, args.count, args.seed, Path(args.output), args.friction)


if __name__ == "__main__":
    main()
