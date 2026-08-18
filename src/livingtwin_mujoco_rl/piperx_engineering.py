from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from livingtwin_mujoco_rl.strike_puck_env import release_index_from_history


STRIKE_PUCK_ACTION_NAMES = (
    "direction_offset",
    "strike_speed",
    "tangential_contact_offset",
    "contact_duration",
)


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("refusing to write empty CSV")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha256(root: str | Path) -> tuple[str, int]:
    root_path = Path(root)
    hasher = hashlib.sha256()
    files = sorted(path for path in root_path.rglob("*") if path.is_file())
    for path in files:
        relative = str(path.relative_to(root_path)).encode()
        data = path.read_bytes()
        hasher.update(len(relative).to_bytes(4, "little"))
        hasher.update(relative)
        hasher.update(len(data).to_bytes(8, "little"))
        hasher.update(data)
    return hasher.hexdigest(), len(files)


def audit_asset(config: Mapping[str, Any]) -> dict[str, Any]:
    asset = config["asset"]
    root = Path(asset["root"])
    urdf = root / asset["urdf_relative_path"]
    errors: list[str] = []
    if not root.is_dir():
        raise FileNotFoundError(root)
    source_commit_path = root / "SOURCE_COMMIT.txt"
    source_commit = source_commit_path.read_text(encoding="utf-8").strip()
    if source_commit != asset["source_commit"]:
        errors.append("SOURCE_COMMIT.txt does not match frozen source commit")
    digest, file_count = tree_sha256(root)
    if digest != asset["tree_sha256"]:
        errors.append("asset tree SHA256 changed from read-only preflight")
    if file_sha256(urdf) != asset["urdf_sha256"]:
        errors.append("URDF SHA256 changed from read-only preflight")
    xml = ET.parse(urdf).getroot()
    links = {node.attrib["name"]: node for node in xml.findall("link")}
    joints = xml.findall("joint")
    parents: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    joint_rows: list[dict[str, Any]] = []
    for joint in joints:
        name = joint.attrib["name"]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        if parent not in links or child not in links:
            errors.append(f"{name}: missing parent or child link")
        if child in parents:
            errors.append(f"{child}: multiple parents")
        parents[child] = parent
        children.setdefault(parent, []).append(child)
        kind = joint.attrib.get("type")
        limit = joint.find("limit")
        row: dict[str, Any] = {"joint": name, "type": kind, "parent": parent, "child": child}
        if kind in {"revolute", "prismatic", "continuous"}:
            if limit is None:
                errors.append(f"{name}: missing limit")
            else:
                row.update({key: float(value) for key, value in limit.attrib.items()})
                if kind != "continuous" and row["lower"] >= row["upper"]:
                    errors.append(f"{name}: invalid lower/upper limit")
                if row["effort"] <= 0 or row["velocity"] <= 0:
                    errors.append(f"{name}: non-positive effort or velocity limit")
        joint_rows.append(row)
    root_links = sorted(set(links) - set(parents))
    visited: set[str] = set()
    stack = list(root_links)
    while stack:
        name = stack.pop()
        if name in visited:
            errors.append(f"kinematic cycle at {name}")
            continue
        visited.add(name)
        stack.extend(children.get(name, []))
    if visited != set(links):
        errors.append("not all URDF links are connected")

    compiler = xml.find("mujoco/compiler")
    meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    mesh_rows: list[dict[str, Any]] = []
    for role in ("visual", "collision"):
        for link_name, link in links.items():
            for mesh in link.findall(f"{role}/geometry/mesh"):
                reference = mesh.attrib["filename"]
                path = Path(reference)
                if not path.is_absolute():
                    path = urdf.parent / meshdir / path
                exists = path.is_file()
                mesh_rows.append({
                    "role": role,
                    "link": link_name,
                    "reference": reference,
                    "resolved": str(path.resolve()),
                    "exists": exists,
                    "size_bytes": path.stat().st_size if exists else -1,
                })
                if not exists:
                    errors.append(f"missing {role} mesh {reference}")
    stl_rows: list[dict[str, Any]] = []
    for path in sorted((urdf.parent / meshdir).glob("*.stl")):
        data = path.read_bytes()
        facets = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else -1
        valid = facets > 0 and 84 + 50 * facets == len(data)
        stl_rows.append({"name": path.name, "size_bytes": len(data), "facets": facets, "valid": valid})
        if not valid:
            errors.append(f"invalid binary STL: {path.name}")
    lfs_pointers = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.stat().st_size < 1024
        and b"version https://git-lfs.github.com/spec/v1" in path.read_bytes()
    ]
    if lfs_pointers:
        errors.append("Git LFS pointer files remain")
    broken_symlinks = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink() and not path.exists()]
    if broken_symlinks:
        errors.append("broken symlinks remain")
    model = mujoco.MjModel.from_xml_path(str(urdf))
    return {
        "passed": not errors,
        "asset_root": str(root),
        "source_commit": source_commit,
        "directory_is_git_worktree": (root / ".git").exists(),
        "tree_sha256": digest,
        "file_count": file_count,
        "urdf": str(urdf),
        "urdf_sha256": file_sha256(urdf),
        "links": len(links),
        "joints_in_urdf": len(joints),
        "root_links": root_links,
        "all_links_connected": visited == set(links),
        "joint_limits": joint_rows,
        "mesh_references": mesh_rows,
        "stl_integrity": stl_rows,
        "lfs_pointers": lfs_pointers,
        "broken_symlinks": broken_symlinks,
        "compiled_model": {
            "nbody": model.nbody,
            "njnt": model.njnt,
            "nq": model.nq,
            "nv": model.nv,
            "ngeom": model.ngeom,
            "nmesh": model.nmesh,
            "nu": model.nu,
        },
        "errors": errors,
    }


def build_engineering_spec(config: Mapping[str, Any]) -> mujoco.MjSpec:
    urdf = Path(config["asset"]["root"]) / config["asset"]["urdf_relative_path"]
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.option.timestep = float(config["simulation"]["timestep_s"])
    spec.option.gravity = config["simulation"]["gravity_mps2"]
    enabled = set(config["robot"]["enabled_collision_bodies"])
    for body in spec.bodies:
        if not body.name or body.name == "world":
            continue
        body.gravcomp = 1.0
        for index, geom in enumerate(body.geoms):
            if geom.contype == 0 and geom.conaffinity == 0:
                geom.name = f"{body.name}_visual_{index}"
                geom.group = 2
            else:
                geom.name = f"{body.name}_collision_{index}"
                geom.group = 3
                active = body.name in enabled
                geom.contype = int(active)
                geom.conaffinity = int(active)
    strike = config["strike_interface"]
    gripper = spec.body("gripper_base")
    gripper.add_site(
        name="strike_site",
        pos=strike["grasp_site_position_in_gripper_base_m"],
        size=[0.005, 0.005, 0.005],
        group=5,
    )
    gripper.add_geom(
        name="strike_tip",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=strike["grasp_site_position_in_gripper_base_m"],
        size=[float(strike["strike_tip_radius_m"]), float(strike["strike_tip_half_height_m"]), 0.0],
        mass=0.01,
        contype=1,
        conaffinity=1,
        friction=[0.6, 0.005, 0.0001],
        group=3,
    )
    equality = spec.add_equality()
    equality.type = mujoco.mjtEq.mjEQ_JOINT
    equality.objtype = mujoco.mjtObj.mjOBJ_JOINT
    equality.name1 = "gripper_joint2"
    equality.name2 = "gripper_joint1"
    equality.data[:5] = [0.0, -1.0, 0.0, 0.0, 0.0]
    equality.solref = [0.005, 1.0]
    exclude = spec.add_exclude()
    exclude.bodyname1 = "gripper_link1"
    exclude.bodyname2 = "gripper_link2"

    task = config["task"]
    spec.worldbody.add_geom(
        name="strike_table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=task["table_center_xyz_m"],
        size=task["table_half_size_xyz_m"],
        contype=1,
        conaffinity=1,
        friction=task["table_friction"],
        group=3,
    )
    table_surface_z = float(task["table_center_xyz_m"][2]) + float(task["table_half_size_xyz_m"][2])
    spec.worldbody.add_site(
        name="goal_site",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[*task["goal_xy_m"], table_surface_z + 0.0005],
        size=[float(task["success_radius_m"]), 0.0005, 0.0],
        rgba=[0.1, 0.8, 0.2, 0.45],
        group=5,
    )
    puck = spec.worldbody.add_body(
        name="strike_puck",
        pos=[*task["puck_center_xy_m"], table_surface_z + float(task["puck_half_height_m"])],
    )
    puck.add_freejoint(name="strike_puck_free")
    puck.add_geom(
        name="strike_puck_geom",
        type=(mujoco.mjtGeom.mjGEOM_BOX if task.get("object_type", "cylinder") == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER),
        size=(task["object_halfsize_xyz_m"] if task.get("object_type", "cylinder") == "box" else [float(task["puck_radius_m"]), float(task["puck_half_height_m"]), 0.0]),
        mass=float(task["puck_mass_kg"]),
        contype=1,
        conaffinity=1,
        friction=task["table_friction"],
        group=3,
    )
    for name, stiffness, damping in zip(
        config["robot"]["arm_joint_names"],
        config["robot"]["actuator_stiffness"],
        config["robot"]["actuator_damping"],
        strict=True,
    ):
        joint = spec.joint(name)
        joint.frictionloss = 0.3
        joint.armature = 0.005
        actuator = spec.add_actuator()
        actuator.name = f"position_{name}"
        actuator.target = name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.gainprm[0] = float(stiffness)
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.biasprm[1] = -float(stiffness)
        actuator.biasprm[2] = -float(damping)
        actuator.ctrllimited = True
        actuator.ctrlrange = joint.range
        actuator.forcelimited = True
        force = float(config["robot"]["actuator_force_limit"])
        actuator.forcerange = [-force, force]
    return spec


def _joint_addresses(model: mujoco.MjModel, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.asarray([model.joint(name).qposadr[0] for name in names], dtype=int)
    dof = np.asarray([model.joint(name).dofadr[0] for name in names], dtype=int)
    ranges = np.asarray([model.joint(name).range for name in names], dtype=np.float64)
    return qpos, dof, ranges


def _linear_map(value: float, limits: Sequence[float]) -> float:
    clipped = float(np.clip(value, -1.0, 1.0))
    low, high = float(limits[0]), float(limits[1])
    return low + 0.5 * (clipped + 1.0) * (high - low)


def encode_linear(value: float, limits: Sequence[float]) -> float:
    low, high = float(limits[0]), float(limits[1])
    if high <= low:
        raise ValueError(f"invalid action limits: {limits}")
    return float(np.clip(2.0 * (float(value) - low) / (high - low) - 1.0, -1.0, 1.0))


def nominal_action(config: Mapping[str, Any], distance_index: int) -> np.ndarray:
    controller = config["nominal_controller"]
    speed = float(config["strike_interface"]["strike_speed_mps_by_distance"][distance_index])
    return np.asarray(
        [
            encode_linear(float(controller["direction_offset_rad"]), config["action"]["direction_offset_rad"]),
            encode_linear(speed, config["action"]["strike_speed_mps"]),
            encode_linear(
                float(controller["tangential_contact_offset_m"]),
                config["action"]["tangential_contact_offset_m"],
            ),
            encode_linear(
                float(controller["contact_duration_s"]),
                config["action"]["contact_duration_s"],
            ),
        ],
        dtype=np.float64,
    )


def decode_strike_action(normalized_action: Sequence[float], config: Mapping[str, Any]) -> dict[str, float]:
    action = np.clip(np.asarray(normalized_action, dtype=np.float64), -1.0, 1.0)
    if action.shape != (4,):
        raise ValueError(f"expected action shape (4,), got {action.shape}")
    action_config = config["action"]
    return {
        "direction_offset_rad": _linear_map(action[0], action_config["direction_offset_rad"]),
        "strike_speed_mps": _linear_map(action[1], action_config["strike_speed_mps"]),
        "tangential_contact_offset_m": _linear_map(action[2], action_config["tangential_contact_offset_m"]),
        "contact_duration_s": _linear_map(action[3], action_config["contact_duration_s"]),
    }


def set_robot_state(
    model: mujoco.MjModel, data: mujoco.MjData, config: Mapping[str, Any], arm_q: Sequence[float]
) -> None:
    names = config["robot"]["arm_joint_names"]
    qpos, _, _ = _joint_addresses(model, names)
    data.qpos[qpos] = arm_q
    for name, value in config["robot"]["gripper_joint_positions_m"].items():
        data.qpos[model.joint(name).qposadr[0]] = float(value)


def solve_position_ik(
    model: mujoco.MjModel,
    config: Mapping[str, Any],
    goal: Sequence[float],
    initial_q: Sequence[float],
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    names = config["robot"]["arm_joint_names"]
    qpos, dof, ranges = _joint_addresses(model, names)
    q = np.asarray(initial_q, dtype=np.float64).copy()
    margin = float(config["ik"]["hard_joint_margin_rad"])
    site_id = model.site("strike_site").id
    iterations = 0
    for iterations in range(1, int(config["ik"]["maximum_iterations"]) + 1):
        set_robot_state(model, data, config, q)
        mujoco.mj_forward(model, data)
        error = np.asarray(goal, dtype=np.float64) - data.site_xpos[site_id]
        if float(np.linalg.norm(error)) <= float(config["ik"]["position_tolerance_m"]):
            break
        jacobian_position = np.zeros((3, model.nv))
        jacobian_rotation = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)
        jacobian = jacobian_position[:, dof]
        damping = float(config["ik"]["damping"])
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(3), error
        )
        q = np.clip(
            q + float(config["ik"]["step_size"]) * delta,
            ranges[:, 0] + margin,
            ranges[:, 1] - margin,
        )
    set_robot_state(model, data, config, q)
    mujoco.mj_forward(model, data)
    error_m = float(np.linalg.norm(np.asarray(goal) - data.site_xpos[site_id]))
    joint_margin = np.minimum(q - ranges[:, 0], ranges[:, 1] - q)
    contacts = unexpected_contacts(model, data)
    return {
        "converged": error_m <= float(config["ik"]["position_tolerance_m"]),
        "q": q,
        "position": data.site_xpos[site_id].copy(),
        "position_error_m": error_m,
        "iterations": iterations,
        "minimum_joint_margin_rad": float(np.min(joint_margin)),
        "joint_margin_warning": bool(np.min(joint_margin) < float(config["ik"]["warning_joint_margin_rad"])),
        "unexpected_contacts": contacts,
    }


def solve_strike_ik(
    model: mujoco.MjModel,
    config: Mapping[str, Any],
    goal: Sequence[float],
    initial_q: Sequence[float],
    desired_opening_axis_world: Sequence[float] | None = None,
    desired_tip_axis_world: Sequence[float] | None = None,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    names = config["robot"]["arm_joint_names"]
    _, dof, ranges = _joint_addresses(model, names)
    position_seed = solve_position_ik(model, config, goal, initial_q)
    q = np.asarray(position_seed["q"], dtype=np.float64).copy()
    margin = float(config["ik"]["hard_joint_margin_rad"])
    site_id = model.site("strike_site").id
    desired_axis = np.asarray(config["strike_interface"]["strike_tip_axis_world"], dtype=np.float64)
    desired_axis = desired_axis / np.linalg.norm(desired_axis)
    if desired_tip_axis_world is not None:
        desired_axis = np.asarray(desired_tip_axis_world, dtype=np.float64)
        desired_axis = desired_axis / np.linalg.norm(desired_axis)
    desired_opening = None if desired_opening_axis_world is None else np.asarray(desired_opening_axis_world, dtype=np.float64) / np.linalg.norm(desired_opening_axis_world)
    orientation_weight = float(config["ik"]["orientation_axis_weight"])
    iterations = 0
    for iterations in range(1, int(config["ik"]["maximum_iterations"]) + 1):
        set_robot_state(model, data, config, q)
        mujoco.mj_forward(model, data)
        position_error = np.asarray(goal, dtype=np.float64) - data.site_xpos[site_id]
        current_axis = data.site_xmat[site_id].reshape(3, 3)[:, 2]
        axis_error = np.cross(current_axis, desired_axis)
        opening_error = np.zeros(3) if desired_opening is None else np.cross(data.site_xmat[site_id].reshape(3, 3)[:, 1], desired_opening)
        if (
            float(np.linalg.norm(position_error)) <= float(config["ik"]["position_tolerance_m"])
            and float(np.linalg.norm(axis_error)) <= float(config["ik"]["orientation_axis_tolerance"])
            and float(np.linalg.norm(opening_error)) <= float(config["ik"]["orientation_axis_tolerance"])
        ):
            break
        jacobian_position = np.zeros((3, model.nv))
        jacobian_rotation = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)
        jacobian = np.vstack(
            [jacobian_position[:, dof], orientation_weight * jacobian_rotation[:, dof]]
        )
        error = np.concatenate([position_error, orientation_weight * (axis_error + opening_error)])
        damping = float(config["ik"]["damping"])
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(6), error
        )
        q = np.clip(
            q + float(config["ik"]["step_size"]) * delta,
            ranges[:, 0] + margin,
            ranges[:, 1] - margin,
        )
    set_robot_state(model, data, config, q)
    mujoco.mj_forward(model, data)
    position_error_m = float(np.linalg.norm(np.asarray(goal) - data.site_xpos[site_id]))
    current_axis = data.site_xmat[site_id].reshape(3, 3)[:, 2]
    dot = float(np.clip(np.dot(current_axis, desired_axis), -1.0, 1.0))
    orientation_error_rad = float(math.acos(dot))
    opening_error_rad = 0.0 if desired_opening is None else float(math.acos(np.clip(np.dot(data.site_xmat[site_id].reshape(3, 3)[:, 1], desired_opening), -1.0, 1.0)))
    joint_margin = np.minimum(q - ranges[:, 0], ranges[:, 1] - q)
    contacts = unexpected_contacts(model, data)
    return {
        "converged": (
            position_error_m <= float(config["ik"]["position_tolerance_m"])
            and orientation_error_rad <= float(config["ik"]["orientation_axis_tolerance"])
            and opening_error_rad <= float(config["ik"]["orientation_axis_tolerance"])
        ),
        "q": q,
        "position": data.site_xpos[site_id].copy(),
        "position_error_m": position_error_m,
        "orientation_axis_error_rad": orientation_error_rad,
        "opening_axis_error_rad": opening_error_rad,
        "iterations": iterations,
        "minimum_joint_margin_rad": float(np.min(joint_margin)),
        "joint_margin_warning": bool(np.min(joint_margin) < float(config["ik"]["warning_joint_margin_rad"])),
        "unexpected_contacts": contacts,
    }


def unexpected_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    allowed = {
        frozenset(("strike_puck_geom", "strike_table")),
        frozenset(("strike_puck_geom", "strike_tip")),
    }
    result: list[str] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        names = frozenset((model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name))
        if names not in allowed:
            result.append(" <-> ".join(sorted(names)))
    return sorted(set(result))


def cell_goal(config: Mapping[str, Any], direction_index: int, phase: str) -> np.ndarray:
    return action_cell_goal(config, direction_index, phase, {
        "direction_offset_rad": 0.0,
        "tangential_contact_offset_m": 0.0,
        "follow_through_m": float(config["strike_interface"]["follow_through_m"]),
    })


def action_cell_goal(
    config: Mapping[str, Any],
    direction_index: int,
    phase: str,
    action: Mapping[str, float],
) -> np.ndarray:
    task = config["task"]
    angle = math.radians(float(task["direction_degrees"][direction_index])) + float(action["direction_offset_rad"])
    unit = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    tangent = np.asarray([-unit[1], unit[0]], dtype=np.float64)
    table_z = float(task["table_center_xyz_m"][2]) + float(task["table_half_size_xyz_m"][2])
    strike = config["strike_interface"]
    tip_extent = math.hypot(
        float(strike["strike_tip_radius_m"]),
        float(strike["strike_tip_half_height_m"]),
    )
    strike_z = table_z + tip_extent + float(strike["table_clearance_m"])
    center = np.asarray([*task["puck_center_xy_m"], strike_z], dtype=np.float64)
    tangential = tangent * float(action["tangential_contact_offset_m"])
    if phase == "prestrike":
        offset = float(task["puck_radius_m"]) + float(config["strike_interface"]["strike_tip_radius_m"]) + float(config["strike_interface"]["approach_gap_m"])
        return center - np.asarray([unit[0] * offset, unit[1] * offset, 0.0]) + np.asarray([*tangential, 0.0])
    if phase == "poststrike":
        follow = float(action.get("follow_through_m", config["strike_interface"]["follow_through_m"]))
        offset = float(task["puck_radius_m"]) + float(config["strike_interface"]["strike_tip_radius_m"]) + follow
        return center + np.asarray([unit[0] * offset, unit[1] * offset, 0.0]) + np.asarray([*tangential, 0.0])
    if phase == "retract":
        lift = float(config["nominal_controller"]["retract_lift_m"])
        backoff = float(config["nominal_controller"]["retract_backoff_m"])
        offset = (
            float(task["puck_radius_m"])
            + float(config["strike_interface"]["strike_tip_radius_m"])
            + float(config["strike_interface"]["approach_gap_m"])
            + backoff
        )
        return (
            center
            - np.asarray([unit[0] * offset, unit[1] * offset, 0.0])
            + np.asarray([*tangential, lift])
        )
    raise ValueError(phase)


def reachability_audit(model: mujoco.MjModel, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    home = np.asarray(config["robot"]["home_joint_positions_rad"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    solutions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for distance_index, distance in enumerate(config["task"]["distances_m"]):
        for direction_index, degrees in enumerate(config["task"]["direction_degrees"]):
            decoded = decode_strike_action(nominal_action(config, distance_index), config)
            decoded["follow_through_m"] = float(config["nominal_controller"]["follow_through_m"])
            pre = solve_strike_ik(model, config, action_cell_goal(config, direction_index, "prestrike", decoded), home)
            post = solve_strike_ik(model, config, action_cell_goal(config, direction_index, "poststrike", decoded), pre["q"])
            retract = solve_strike_ik(model, config, action_cell_goal(config, direction_index, "retract", decoded), post["q"])
            reachable = bool(
                pre["converged"]
                and post["converged"]
                and retract["converged"]
                and not pre["unexpected_contacts"]
                and not post["unexpected_contacts"]
                and not retract["unexpected_contacts"]
            )
            rows.append({
                "cell_index": distance_index * 8 + direction_index,
                "direction_index": direction_index,
                "direction_degrees": degrees,
                "distance_index": distance_index,
                "distance_m": distance,
                "reachable": reachable,
                "prestrike_error_m": pre["position_error_m"],
                "poststrike_error_m": post["position_error_m"],
                "retract_error_m": retract["position_error_m"],
                "prestrike_orientation_axis_error_rad": pre["orientation_axis_error_rad"],
                "poststrike_orientation_axis_error_rad": post["orientation_axis_error_rad"],
                "retract_orientation_axis_error_rad": retract["orientation_axis_error_rad"],
                "minimum_joint_margin_rad": min(
                    pre["minimum_joint_margin_rad"],
                    post["minimum_joint_margin_rad"],
                    retract["minimum_joint_margin_rad"],
                ),
                "joint_margin_warning": pre["joint_margin_warning"] or post["joint_margin_warning"] or retract["joint_margin_warning"],
                "prestrike_unexpected_contacts": "|".join(pre["unexpected_contacts"]),
                "poststrike_unexpected_contacts": "|".join(post["unexpected_contacts"]),
                "retract_unexpected_contacts": "|".join(retract["unexpected_contacts"]),
                **{f"pre_q_{index + 1}": float(value) for index, value in enumerate(pre["q"])},
                **{f"post_q_{index + 1}": float(value) for index, value in enumerate(post["q"])},
                **{f"retract_q_{index + 1}": float(value) for index, value in enumerate(retract["q"])},
            })
            solutions[(direction_index, distance_index)] = (pre["q"], post["q"], retract["q"])
    return rows, solutions


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _contact_between(model: mujoco.MjModel, data: mujoco.MjData, first: str, second: str) -> bool:
    pair = {model.geom(first).id, model.geom(second).id}
    return any(
        {int(data.contact[index].geom1), int(data.contact[index].geom2)} == pair
        for index in range(data.ncon)
    )


def replay_cell(
    model: mujoco.MjModel,
    config: Mapping[str, Any],
    direction_index: int,
    distance_index: int,
    pre_q: np.ndarray,
    post_q: np.ndarray,
    retract_q: np.ndarray,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    set_robot_state(model, data, config, pre_q)
    data.ctrl[:] = pre_q
    mujoco.mj_forward(model, data)
    puck_joint = model.joint("strike_puck_free")
    puck_qpos = int(puck_joint.qposadr[0])
    puck_dof = int(puck_joint.dofadr[0])
    initial_xy = data.qpos[puck_qpos : puck_qpos + 2].copy()
    timestep = float(model.opt.timestep)
    for _ in range(round(float(config["strike_interface"]["prestrike_settle_s"]) / timestep)):
        data.ctrl[:] = pre_q
        mujoco.mj_step(model, data)
    action = nominal_action(config, distance_index)
    decoded = decode_strike_action(action, config)
    decoded["follow_through_m"] = float(config["nominal_controller"]["follow_through_m"])
    pre_goal = action_cell_goal(config, direction_index, "prestrike", decoded)
    post_goal = action_cell_goal(config, direction_index, "poststrike", decoded)
    strike_angle = math.radians(float(config["task"]["direction_degrees"][direction_index])) + float(decoded["direction_offset_rad"])
    strike_unit = np.asarray([math.cos(strike_angle), math.sin(strike_angle)], dtype=np.float64)
    path_length = float(np.linalg.norm(post_goal - pre_goal))
    speed = float(decoded["strike_speed_mps"])
    strike_duration = path_length / speed
    strike_steps = max(1, round(strike_duration / timestep))
    contacts: list[bool] = []
    retracting: list[bool] = []
    states: list[tuple[np.ndarray, np.ndarray, float]] = []
    contact_start_index: int | None = None
    site_id = model.site("strike_site").id
    previous_site_xy = data.site_xpos[site_id, :2].copy()
    ee_projected_speeds: list[float] = []
    precontact_ee_projected_speeds: list[float] = []
    contact_ee_speed_mps = float("nan")
    numerical_anomaly = False
    last_strike_q = pre_q.copy()
    for step in range(strike_steps):
        if (
            contact_start_index is not None
            and (step - contact_start_index) * timestep >= float(decoded["contact_duration_s"])
        ):
            break
        alpha = _smoothstep((step + 1) / strike_steps)
        last_strike_q = (1.0 - alpha) * pre_q + alpha * post_q
        data.ctrl[:] = last_strike_q
        mujoco.mj_step(model, data)
        contact = _contact_between(model, data, "strike_tip", "strike_puck_geom")
        site_xy = data.site_xpos[site_id, :2].copy()
        ee_projected_speed = float(np.dot((site_xy - previous_site_xy) / timestep, strike_unit))
        previous_site_xy = site_xy
        ee_projected_speeds.append(ee_projected_speed)
        if contact_start_index is None:
            precontact_ee_projected_speeds.append(ee_projected_speed)
        if contact and not np.isfinite(contact_ee_speed_mps):
            contact_ee_speed_mps = ee_projected_speed
        sample_index = len(contacts)
        if contact and contact_start_index is None:
            contact_start_index = sample_index
        early_release = bool(contact_start_index is not None and not contact and any(contacts))
        contacts.append(contact)
        retracting.append(early_release)
        states.append((
            data.qpos[puck_qpos : puck_qpos + 2].copy(),
            data.qvel[puck_dof : puck_dof + 2].copy(),
            float(data.qvel[puck_dof + 5]),
        ))
        numerical_anomaly = numerical_anomaly or not (
            np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        )
        if early_release:
            break
    retract_steps = max(1, round(float(config["strike_interface"]["retract_duration_s"]) / timestep))
    for step in range(retract_steps):
        alpha = _smoothstep((step + 1) / retract_steps)
        data.ctrl[:] = (1.0 - alpha) * last_strike_q + alpha * retract_q
        mujoco.mj_step(model, data)
        contact = _contact_between(model, data, "strike_tip", "strike_puck_geom")
        contacts.append(contact)
        retracting.append(True)
        states.append((
            data.qpos[puck_qpos : puck_qpos + 2].copy(),
            data.qvel[puck_dof : puck_dof + 2].copy(),
            float(data.qvel[puck_dof + 5]),
        ))
        numerical_anomaly = numerical_anomaly or not (
            np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        )
    observation_steps = max(1, round(float(config["strike_interface"]["poststrike_observation_s"]) / timestep))
    for _ in range(observation_steps):
        data.ctrl[:] = retract_q
        mujoco.mj_step(model, data)
        contact = _contact_between(model, data, "strike_tip", "strike_puck_geom")
        contacts.append(contact)
        retracting.append(True)
        states.append((
            data.qpos[puck_qpos : puck_qpos + 2].copy(),
            data.qvel[puck_dof : puck_dof + 2].copy(),
            float(data.qvel[puck_dof + 5]),
        ))
        numerical_anomaly = numerical_anomaly or not (
            np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        )
    release_config = config["release"]
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
        release_time = release_index * timestep
        check_from = release_index + int(release_config["consecutive_absent_frames"])
        recontact = any(contacts[check_from:])
    final_xy = data.qpos[puck_qpos : puck_qpos + 2].copy()
    final_velocity = data.qvel[puck_dof : puck_dof + 2].copy()
    final_omega = float(data.qvel[puck_dof + 5])
    angle = math.radians(float(config["task"]["direction_degrees"][direction_index]))
    target_xy = initial_xy + float(config["task"]["distances_m"][distance_index]) * np.asarray([math.cos(angle), math.sin(angle)])
    final_error = float(np.linalg.norm(final_xy - target_xy))
    final_speed = float(np.linalg.norm(final_velocity))
    motion = float(np.linalg.norm(final_xy - initial_xy))
    success = bool(
        release_index is not None
        and
        final_error <= float(config["task"]["success_radius_m"])
        and final_speed <= float(config["task"]["success_speed_mps"])
        and not recontact
        and not numerical_anomaly
    )
    release_speed = float(np.linalg.norm(release_velocity)) if np.all(np.isfinite(release_velocity)) else float("nan")
    release_direction = (
        math.degrees(math.atan2(release_velocity[1], release_velocity[0])) % 360.0
        if release_speed > 1.0e-9
        else float("nan")
    )
    return {
        "cell_index": distance_index * 8 + direction_index,
        "direction_index": direction_index,
        "direction_degrees": config["task"]["direction_degrees"][direction_index],
        "distance_index": distance_index,
        "distance_m": config["task"]["distances_m"][distance_index],
        "normalized_action": action.tolist(),
        "action_size": len(STRIKE_PUCK_ACTION_NAMES),
        **{name: float(action[index]) for index, name in enumerate(STRIKE_PUCK_ACTION_NAMES)},
        **{key: float(value) for key, value in decoded.items() if key != "follow_through_m"},
        "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.95)),
        "commanded_strike_speed_mps": speed,
        "realized_ee_speed_mps": (
            float(np.mean(precontact_ee_projected_speeds))
            if precontact_ee_projected_speeds
            else float(np.mean(ee_projected_speeds)) if ee_projected_speeds else float("nan")
        ),
        "contact_ee_speed_mps": contact_ee_speed_mps,
        "ee_speed_execution_scale": (
            float(np.mean(precontact_ee_projected_speeds)) / speed
            if precontact_ee_projected_speeds and speed > 0.0
            else float("nan")
        ),
        "contact_ee_speed_scale": contact_ee_speed_mps / speed if speed > 0.0 else float("nan"),
        "strike_duration_s": strike_duration,
        "contact_detected": any(contacts),
        "recontact": recontact,
        "puck_motion_m": motion,
        "target_distance_m": float(config["task"]["distances_m"][distance_index]),
        "initial_puck_xy": initial_xy.tolist(),
        "target_xy": target_xy.tolist(),
        "contact_start_time_s": (
            contact_start_index * timestep if contact_start_index is not None else float("nan")
        ),
        "contact_end_time_s": (
            max(index for index, value in enumerate(contacts) if value) * timestep
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
        "final_error_m": final_error,
        "final_speed_mps": final_speed,
        "final_angular_velocity_radps": final_omega,
        "success": success,
        "out_of_bounds": False,
        "numerical_anomaly": numerical_anomaly,
        "episode_steps": len(states),
        "initial_x": float(initial_xy[0]),
        "initial_y": float(initial_xy[1]),
        "final_x": float(final_xy[0]),
        "final_y": float(final_xy[1]),
        "target_x": float(target_xy[0]),
        "target_y": float(target_xy[1]),
    }


def run_engineering(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty engineering output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = {
        "track": "C", "stage": "ASSET_AUDIT", "state": "RUNNING", "pid": os.getpid(),
        "started_unix_s": started, "updated_unix_s": started,
        "tmux_session": os.environ.get("TMUX_SESSION", ""),
    }
    write_json(output / "RUN_STATUS.json", status)
    asset_before = audit_asset(config)
    write_json(output / "ASSET_AUDIT.json", asset_before)
    if not asset_before["passed"]:
        status.update({"state": "FAIL", "failure": "ASSET_AUDIT", "updated_unix_s": time.time()})
        write_json(output / "RUN_STATUS.json", status)
        return {"passed": False, "stop_stage": "ASSET_AUDIT", "asset_audit": asset_before}
    status.update({"stage": "SIMULATION_IMPORT", "updated_unix_s": time.time()})
    write_json(output / "RUN_STATUS.json", status)
    spec = build_engineering_spec(config)
    model = spec.compile()
    (output / "piperx_strikepuck_scene.xml").write_text(spec.to_xml(), encoding="utf-8")
    model_summary = {
        "mujoco_version": mujoco.__version__, "nbody": model.nbody, "njnt": model.njnt,
        "nq": model.nq, "nv": model.nv, "ngeom": model.ngeom, "nmesh": model.nmesh,
        "nu": model.nu, "actuators": [model.actuator(index).name for index in range(model.nu)],
    }
    write_json(output / "MODEL_IMPORT.json", model_summary)
    status.update({"stage": "REACHABILITY", "updated_unix_s": time.time()})
    write_json(output / "RUN_STATUS.json", status)
    reachability, solutions = reachability_audit(model, config)
    write_csv(output / "reachability_24_cells.csv", reachability)
    reachable_count = sum(bool(row["reachable"]) for row in reachability)
    status.update({"stage": "NO_TRAINING_REPLAY", "reachable_cells": reachable_count, "updated_unix_s": time.time()})
    write_json(output / "RUN_STATUS.json", status)
    replay_rows = [
        replay_cell(model, config, direction_index, distance_index, *solutions[(direction_index, distance_index)])
        for distance_index in range(3)
        for direction_index in range(8)
        if reachability[distance_index * 8 + direction_index]["reachable"]
    ]
    if replay_rows:
        write_csv(output / "minimal_replay_24_cells.csv", replay_rows)
    else:
        (output / "minimal_replay_24_cells.csv").write_text("cell_index\n", encoding="utf-8")
    asset_after = audit_asset(config)
    write_json(output / "ASSET_AUDIT_AFTER.json", asset_after)
    gates = config["engineering_gate"]
    required_schema_fields = (
        "release_time_s",
        "release_position_x",
        "release_position_y",
        "release_velocity_x",
        "release_velocity_y",
        "release_speed_mps",
        "release_direction_deg",
        "final_position_x",
        "final_position_y",
        "final_velocity_x",
        "final_velocity_y",
        "final_speed_mps",
        "final_error_m",
    )
    checks = {
        "asset_audit": asset_before["passed"],
        "asset_unchanged": asset_before["tree_sha256"] == asset_after["tree_sha256"],
        "simulation_import": model.nu == 6 and model.nmesh >= 11,
        "all_24_cells_reachable": reachable_count >= int(gates["required_reachable_cells"]),
        "prestrike_position_error": max(row["prestrike_error_m"] for row in reachability) <= float(gates["maximum_prestrike_position_error_m"]),
        "poststrike_position_error": max(row["poststrike_error_m"] for row in reachability) <= float(gates["maximum_poststrike_position_error_m"]),
        "retract_position_error": max(row["retract_error_m"] for row in reachability) <= float(gates["maximum_retract_position_error_m"]),
        "orientation_axis_error": max(
            max(
                row["prestrike_orientation_axis_error_rad"],
                row["poststrike_orientation_axis_error_rad"],
                row["retract_orientation_axis_error_rad"],
            )
            for row in reachability
        ) <= float(gates["maximum_orientation_axis_error_rad"]),
        "no_unexpected_reachability_collision": all(
            not row["prestrike_unexpected_contacts"]
            and not row["poststrike_unexpected_contacts"]
            and not row["retract_unexpected_contacts"]
            for row in reachability
        ),
        "replay_numerical_stability": bool(replay_rows) and not any(row["numerical_anomaly"] for row in replay_rows),
        "replay_contact": bool(replay_rows) and all(row["contact_detected"] for row in replay_rows),
        "replay_puck_motion": bool(replay_rows) and all(row["puck_motion_m"] >= float(gates["replay_requires_puck_motion_m"]) for row in replay_rows),
        "replay_release_state": bool(replay_rows) and all(np.isfinite(row["release_time_s"]) for row in replay_rows),
        "replay_release_final_schema": bool(replay_rows) and all(
            all(field in row for field in required_schema_fields) for row in replay_rows
        ),
    }
    passed = bool(all(checks.values()))
    summary = {
        "track": "C", "status": "PASS" if passed else "FAIL", "passed": passed,
        "checks": checks, "asset_source_commit": asset_before["source_commit"],
        "asset_tree_sha256": asset_before["tree_sha256"], "model_import": model_summary,
        "reachable_cells": reachable_count, "joint_margin_warning_cells": sum(bool(row["joint_margin_warning"]) for row in reachability),
        "minimum_joint_margin_rad": min(float(row["minimum_joint_margin_rad"]) for row in reachability),
        "replay_count": len(replay_rows), "replay_contact_count": sum(bool(row["contact_detected"]) for row in replay_rows),
        "replay_release_state_count": sum(bool(np.isfinite(row["release_time_s"])) for row in replay_rows),
        "replay_recontact_count": sum(bool(row["recontact"]) for row in replay_rows),
        "replay_motion_count": sum(float(row["puck_motion_m"]) >= float(gates["replay_requires_puck_motion_m"]) for row in replay_rows),
        "replay_success_count": sum(bool(row["success"]) for row in replay_rows),
        "replay_mean_final_error_m": float(np.mean([row["final_error_m"] for row in replay_rows])) if replay_rows else float("nan"),
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "TRACK_C_GATE.json", summary)
    status.update({"stage": "TRACK_C_COMPLETE", "state": summary["status"], "updated_unix_s": time.time()})
    write_json(output / "RUN_STATUS.json", status)
    report = f"""# Track C Piper X Engineering Report

- Status: {summary['status']}.
- Source commit: `{summary['asset_source_commit']}` (verified from SOURCE_COMMIT.txt).
- Asset tree SHA256: `{summary['asset_tree_sha256']}`; unchanged after execution: {checks['asset_unchanged']}.
- MuJoCo import: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms, {model.nmesh} meshes, {model.nu} actuators.
- Reachable cells: {reachable_count}/24.
- Minimum joint-limit margin: {summary['minimum_joint_margin_rad']:.4f} rad; warning cells: {summary['joint_margin_warning_cells']}.
- No-training replay contact/motion: {summary['replay_contact_count']}/{len(replay_rows)} and {summary['replay_motion_count']}/{len(replay_rows)}.
- Finite release states: {summary['replay_release_state_count']}/{len(replay_rows)}.
- Re-contact count after release: {summary['replay_recontact_count']}/{len(replay_rows)}.
- Frozen task success in the engineering replay: {summary['replay_success_count']}/{len(replay_rows)} (not method evidence).
- Mean replay final error: {1000.0 * summary['replay_mean_final_error_m']:.2f} mm.
- Checks: `{json.dumps(checks, sort_keys=True)}`
"""
    (output / "TRACK_C_REPORT.md").write_text(report, encoding="utf-8")
    return summary
