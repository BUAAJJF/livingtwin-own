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
        self.reset([*self.config["task"]["puck_center_xy_m"]])

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

    def goal(self, center: np.ndarray, direction: np.ndarray, phase: str, magnitude: float) -> np.ndarray:
        task = self.config["task"]
        strike = self.config["strike_interface"]
        table_z = float(task["table_center_xyz_m"][2]) + float(task["table_half_size_xyz_m"][2])
        # The kinematic marker is 11.5 mm above the finger distal face in the source asset.
        # This height was selected from native mesh geometry, not target performance.
        marker_offset = float(self.config.get("direct_gripper", {}).get("marker_above_puck_center_m", 0.0115))
        z = table_z + float(task["puck_half_height_m"]) + marker_offset
        radius = float(task["puck_radius_m"])
        gap = float(strike["approach_gap_m"])
        if phase == "pre":
            xy = center - direction * (radius + gap)
            return np.r_[xy, z]
        if phase == "post":
            xy = center + direction * (radius + magnitude)
            return np.r_[xy, z]
        if phase == "high":
            xy = center - direction * (radius + gap + 0.02)
            return np.r_[xy, z + 0.05]
        if phase == "retract":
            xy = center - direction * (radius + gap + 0.02)
            return np.r_[xy, z + 0.05]
        raise ValueError(phase)

    def solve(self, goal: np.ndarray, seed: np.ndarray, *, orientation: bool = True) -> np.ndarray | None:
        solver = solve_strike_ik if orientation else solve_position_ik
        result = solver(self.model, self.config, goal, seed)
        if not result["converged"]:
            home = np.asarray(self.config["robot"]["home_joint_positions_rad"], dtype=float)
            result = solver(self.model, self.config, goal, home)
        return np.asarray(result["q"]) if result["converged"] else None

    def move(self, target: np.ndarray, duration: float, trace: list[dict[str, Any]], phase: str) -> None:
        start = self.current_q.copy()
        steps = max(1, round(duration / self.model.opt.timestep))
        for step in range(steps):
            alpha = _smoothstep((step + 1) / steps)
            self.data.ctrl[:] = (1.0 - alpha) * start + alpha * target
            mujoco.mj_step(self.model, self.data)
            self.time_step_index += 1
            trace.append(self.sample(phase))
        self.current_q = target.copy()

    def move_push_after_touch(
        self,
        target: np.ndarray,
        duration: float,
        trace: list[dict[str, Any]],
        direction: np.ndarray,
        after_touch_displacement: float,
    ) -> dict[str, Any]:
        """Stop the existing push path after a fixed EE displacement from first contact."""
        start = self.current_q.copy()
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
                    self.current_q = command.copy()
                    reached = True
                    break
        if not reached:
            self.current_q = target.copy()
        return {
            "first_contact_detected": first_contact_site is not None,
            "after_touch_displacement_achieved_m": achieved,
            "after_touch_displacement_reached": reached,
        }

    def sample(self, phase: str) -> dict[str, Any]:
        finger1_puck = pair_contact(self.model, self.data, {self.finger_ids["gripper_link1_collision_1"]}, {self.puck_id})
        finger2_puck = pair_contact(self.model, self.data, {self.finger_ids["gripper_link2_collision_1"]}, {self.puck_id})
        finger_puck = finger1_puck or finger2_puck
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
            "finite": bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))),
        }

    def push(self, direction_rad: float, magnitude: float, speed: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        center = self.puck_xy()
        before_v = self.puck_v()
        unit = np.array([math.cos(direction_rad), math.sin(direction_rad)])
        # Reposition/retract do not require the strike-axis constraint. Position-only IK
        # removes an unnecessary failure mode while keeping pre/post push IK unchanged.
        high_q = self.solve(self.goal(center, unit, "high", magnitude), self.current_q, orientation=False)
        if high_q is None:
            return {
                "unrecoverable": True, "reason": "high_ik",
                "puck_before_x": float(center[0]), "puck_before_y": float(center[1]),
                "command_direction_rad": float(direction_rad), "command_magnitude_m": float(magnitude),
                "command_speed_mps": float(speed),
            }, []
        pre_q = self.solve(self.goal(center, unit, "pre", magnitude), high_q)
        post_q = self.solve(self.goal(center, unit, "post", magnitude), pre_q if pre_q is not None else high_q)
        retract_q = self.solve(
            self.goal(center, unit, "retract", magnitude),
            post_q if post_q is not None else high_q,
            orientation=False,
        )
        if any(q is None for q in (pre_q, post_q, retract_q)):
            return {
                "unrecoverable": True, "reason": "push_ik",
                "puck_before_x": float(center[0]), "puck_before_y": float(center[1]),
                "command_direction_rad": float(direction_rad), "command_magnitude_m": float(magnitude),
                "command_speed_mps": float(speed),
            }, []
        trace: list[dict[str, Any]] = []
        self.move(high_q, 0.30, trace, "reposition_high")
        self.move(pre_q, 0.18, trace, "approach")
        path = float(np.linalg.norm(self.goal(center, unit, "post", magnitude) - self.goal(center, unit, "pre", magnitude)))
        after_touch_displacement = (
            float(magnitude)
            if self.config.get("sustained_push", False)
            else self.config.get("direct_gripper", {}).get("contact_after_touch_displacement_m")
        )
        if after_touch_displacement is None:
            self.move(post_q, max(0.06, path / speed), trace, "push")
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
        self.move(retract_q, 0.12, trace, "retract")
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
            "simulation_step_before": trace[0]["step"] if trace else self.time_step_index,
            "simulation_step_after": self.time_step_index,
            "puck_before_x": float(center[0]), "puck_before_y": float(center[1]),
            "puck_after_x": float(after[0]), "puck_after_y": float(after[1]),
            "puck_delta_x": float(after[0] - center[0]), "puck_delta_y": float(after[1] - center[1]),
            "puck_velocity_before_x": float(before_v[0]), "puck_velocity_before_y": float(before_v[1]),
            "puck_velocity_after_x": float(after_v[0]), "puck_velocity_after_y": float(after_v[1]),
            "gripper_before_q": self.current_q.tolist(),
            "command_direction_rad": float(direction_rad), "command_magnitude_m": float(magnitude), "command_speed_mps": float(speed),
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
