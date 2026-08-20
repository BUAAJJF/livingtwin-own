#!/usr/bin/env python3
"""Viewer-only playback of the seven frozen x=.46 staged-precontact cases.

This file deliberately owns only Viser rendering and an instance-local sample
hook.  It neither changes DirectPushEnv nor writes qualification artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import viser
import yaml

from livingtwin_mujoco_rl.checkpoint import load_checkpoint
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.piperx_goal_push_env import PiperGoalPushEnv


CASES = (
    ("1  x=.46 y=-.10  180 deg", .46, -.10, 180),
    ("2  x=.46 y=-.10  225 deg", .46, -.10, 225),
    ("3  x=.46 y=+.10  135 deg", .46, +.10, 135),
    ("4  x=.46 y=+.10  180 deg", .46, +.10, 180),
    ("5  x=.46 y=-.10  315 deg", .46, -.10, 315),
    ("6  x=.46 y= .00    0 deg", .46, .00, 0),
    ("7  x=.46 y=+.10   45 deg", .46, +.10, 45),
)


def quat(matrix: np.ndarray) -> np.ndarray:
    result = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(result, np.asarray(matrix, dtype=np.float64).reshape(9))
    return result


class GateViewer:
    def __init__(self, config: dict[str, Any], server: viser.ViserServer, fps: float, sample_stride: int, policy_run: Path, checkpoint: Path) -> None:
        self.config, self.server = config, server
        self.source_mode = "manual"
        self.policy_config = json.loads((policy_run / "FROZEN_CONFIG.json").read_text())
        self.checkpoint = checkpoint
        self.goal_env = PiperGoalPushEnv(config["environment"], 0)
        self.env = self.goal_env.executor
        self.model, self.data = self.env.model, self.env.data
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stop = threading.Event()
        self.continuous_stop = threading.Event()
        self.running = False
        self.fps = max(1.0, fps)
        self.sample_stride = max(1, int(sample_stride))
        self.sample_count = 0
        self.handles: dict[int, Any] = {}
        self.contact_points: list[Any] = []
        self.contact_normals: list[Any] = []
        self.policy_model: ActorCritic | None = None
        self.policy_normalizer: RunningMeanStd | None = None
        self._build_scene()
        self._build_gui()
        self.thread = threading.Thread(target=self._render_loop, daemon=True)
        self.thread.start()
        self.reset_case()

    def _mesh(self, gid: int, path: str, color: tuple[int, int, int], *, wireframe: bool = False) -> Any:
        mesh = int(self.model.geom_dataid[gid])
        va, vn = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
        fa, fn = int(self.model.mesh_faceadr[mesh]), int(self.model.mesh_facenum[mesh])
        vertices = np.asarray(self.model.mesh_vert[va:va + vn], dtype=np.float32)
        faces = np.asarray(self.model.mesh_face[fa:fa + fn], dtype=np.int32).reshape(-1, 3)
        return self.server.scene.add_mesh_simple(
            path, vertices, faces, color=color, wireframe=wireframe, side="double",
            position=tuple(float(v) for v in self.data.geom_xpos[gid]),
            wxyz=quat(self.data.geom_xmat[gid]), material="toon3",
        )

    def _build_scene(self) -> None:
        task = self.env.config["task"]
        center = task["table_center_xyz_m"]
        half = task["table_half_size_xyz_m"]
        self.server.scene.add_box("/table", dimensions=tuple(2 * float(v) for v in half),
                                  position=tuple(float(v) for v in center), color=(65, 105, 150), material="toon3")
        self.cube_id = self.env.puck_id
        cube_half = self.model.geom_size[self.cube_id][:3]
        self.cube = self.server.scene.add_box(
            "/cube_50mm", dimensions=tuple(2 * float(v) for v in cube_half), color=(230, 70, 35), material="toon3",
            position=tuple(float(v) for v in self.data.geom_xpos[self.cube_id]), wxyz=quat(self.data.geom_xmat[self.cube_id]),
        )
        self.goal = self.server.scene.add_cylinder(
            "/goal", radius=.006, height=.003, color=(45, 225, 90), material="toon3",
            position=(.45, .0, float(self.data.geom_xpos[self.cube_id][2])),
        )
        for gid in range(self.model.ngeom):
            if gid == self.cube_id or int(self.model.geom_type[gid]) != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            name = self.model.geom(gid).name or f"geom_{gid}"
            if name.endswith("_visual_0"):
                self.handles[gid] = self._mesh(gid, f"/robot/{name}", (175, 180, 188))
        for gid in self.env.finger_ids.values():
            self.handles[gid] = self._mesh(gid, f"/real_finger_mesh/{self.model.geom(gid).name}", (255, 45, 185), wireframe=True)
        hidden = (0.0, 0.0, -1.0)
        for index in range(8):
            self.contact_points.append(self.server.scene.add_icosphere(f"/contacts/{index}/point", radius=.003, color=(255,235,20), position=hidden, visible=False))
            self.contact_normals.append(self.server.scene.add_line_segments(f"/contacts/{index}/normal", np.asarray([[hidden, hidden]]), colors=(255,235,20), line_width=3., visible=False))

    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.add_markdown("## Piper X Policy V2 + frozen-case live viewer\nViewer-only: current controller, no artifacts written.")
        self.policy_seed = gui.add_number("Policy success seed", initial_value=910045, step=1)
        self.play_policy = gui.add_button("Play Policy V2 (clean success seed)", color="blue")
        self.play_continuous_policy = gui.add_button("Start continuous Policy V2", color="green")
        self.stop_continuous_policy = gui.add_button("Stop continuous Policy V2", color="red")
        self.goal_info = gui.add_text("Policy goal", "")
        self.case = gui.add_dropdown("Frozen x=.46 case", tuple(x[0] for x in CASES), initial_value=CASES[0][0])
        self.stage = gui.add_text("Controller stage", "RESET")
        self.status = gui.add_text("Run status", "READY")
        self.direction = gui.add_text("Push direction", "")
        self.play = gui.add_button("Play selected case", color="green")
        self.reset = gui.add_button("Reset selected case", color="gray")
        self.contacts = gui.add_checkbox("Show native finger-cube contacts", initial_value=True)
        self.case.on_update(lambda _: self.reset_case())
        self.play_policy.on_click(lambda _: self.play_policy_episode())
        self.play_continuous_policy.on_click(lambda _: self.play_continuous_policy_episode())
        self.stop_continuous_policy.on_click(lambda _: self.stop_continuous_policy_episode())
        self.play.on_click(lambda _: self.play_case())
        self.reset.on_click(lambda _: self.reset_case())

    def _selected(self) -> tuple[str, float, float, int]:
        return next(item for item in CASES if item[0] == self.case.value)

    def reset_case(self) -> None:
        if self.running:
            return
        _, x, y, degree = self._selected()
        if self.source_mode != "manual":
            self._set_environment(PiperGoalPushEnv(self.config["environment"], 0))
            self.source_mode = "manual"
        self.env.reset([x, y])
        self.sample_count = 0
        self.stage.value = "RESET"
        self.status.value = "READY — select Play"
        self.direction.value = f"{degree} deg; unit d=({math.cos(math.radians(degree)):.3f}, {math.sin(math.radians(degree)):.3f})"
        self._enqueue("RESET")

    def _set_environment(self, goal_env: PiperGoalPushEnv) -> None:
        """Switch only the viewer's source model/data; production code is untouched."""
        self.goal_env = goal_env
        self.env = goal_env.executor
        self.model, self.data = self.env.model, self.env.data
        self.cube_id = self.env.puck_id

    def _load_policy(self) -> tuple[ActorCritic, RunningMeanStd]:
        if self.policy_model is None or self.policy_normalizer is None:
            probe = PiperGoalPushEnv(self.policy_config["environment"], 0)
            obs, _ = probe.reset(0)
            self.policy_model = ActorCritic(obs.shape[-1], 2, self.policy_config["network"]["hidden_sizes"])
            self.policy_normalizer = RunningMeanStd((obs.shape[-1],))
            payload = load_checkpoint(self.checkpoint, self.policy_model, None, self.policy_normalizer)
            self.policy_model.eval()
            print(f"VIEWER_POLICY_LOADED checkpoint={self.checkpoint} global_step={payload['global_step']}", flush=True)
        return self.policy_model, self.policy_normalizer

    def _enqueue(self, phase: str) -> None:
        contacts = []
        # The frozen Policy-v2 baseline predates the optional detailed-contact
        # helper.  Playback remains exact when it is unavailable; only the
        # decorative contact overlay is omitted.
        records = getattr(self.env, "_finger_puck_contact_records", lambda: [])()
        for item in records:
            contacts.append({"position": item["position"], "normal": item["normal"]})
        self.queue.put({"phase": phase, "geom_xpos": self.data.geom_xpos.copy(), "geom_xmat": self.data.geom_xmat.copy(), "contacts": contacts})

    def play_case(self) -> None:
        print("VIEWER_PLAY_CLICK", flush=True)
        if self.running:
            return
        self.running = True
        self.status.value = "STARTING PLAYBACK"
        threading.Thread(target=self._run, daemon=True).start()

    def play_policy_episode(self) -> None:
        if self.running:
            return
        self.running = True
        self.status.value = "STARTING POLICY PLAYBACK"
        threading.Thread(target=self._run_policy, kwargs={"continuous": False}, daemon=True).start()

    def play_continuous_policy_episode(self) -> None:
        if self.running:
            return
        self.running = True
        self.continuous_stop.clear()
        self.status.value = "STARTING CONTINUOUS POLICY PLAYBACK"
        threading.Thread(target=self._run_policy, kwargs={"continuous": True}, daemon=True).start()

    def stop_continuous_policy_episode(self) -> None:
        self.continuous_stop.set()
        if self.running:
            self.status.value = "STOP REQUESTED — finishes the current simulator step"

    def _set_goal_marker(self) -> None:
        self.goal.position = (
            float(self.goal_env.goal[0]), float(self.goal_env.goal[1]),
            float(self.data.geom_xpos[self.cube_id][2]),
        )
        self.goal_info.value = (
            f"goal=({self.goal_env.goal[0]:.3f}, {self.goal_env.goal[1]:.3f}); "
            f"current error={1000.0 * self.goal_env._distance():.1f} mm"
        )

    def _sample_next_legal_goal(self) -> bool:
        """Update only goal state, preserving the settled cube and robot state."""
        task = self.goal_env.config["task"]
        lo_x, hi_x, lo_y, hi_y = self.goal_env._legal_xy()
        origin = self.goal_env._object_xy()
        distance_lo, distance_hi = (float(v) for v in task["goal_distance_range_m"])
        angle_lo, angle_hi = (float(v) for v in task.get("goal_angle_range_rad", [-math.pi, math.pi]))
        for _ in range(128):
            radius = self.goal_env.rng.uniform(distance_lo, distance_hi)
            angle = self.goal_env.rng.uniform(angle_lo, angle_hi)
            candidate = origin + radius * np.asarray([math.cos(angle), math.sin(angle)])
            if lo_x <= candidate[0] <= hi_x and lo_y <= candidate[1] <= hi_y:
                self.goal_env.goal[:] = candidate
                self.model.site_pos[self.model.site("goal_site").id, :2] = candidate
                mujoco.mj_forward(self.model, self.data)
                self.goal_env.step_count = 0
                self.goal_env.last_info = {"success": False, "continuous_goal_reset": True}
                self._set_goal_marker()
                self._enqueue("NEXT_GOAL")
                return True
        return False

    def _run_policy(self, *, continuous: bool) -> None:
        seed = int(self.policy_seed.value)
        try:
            model, normalizer = self._load_policy()
            self._set_environment(PiperGoalPushEnv(self.policy_config["environment"], seed))
            self.source_mode = "policy"
            observation, _ = self.goal_env.reset(seed)
            self._set_goal_marker()
            self.sample_count = 0
            original_sample = self.env.sample
            def observed_sample(phase: str) -> dict[str, Any]:
                sample = original_sample(phase)
                self.sample_count += 1
                if self.sample_count % self.sample_stride == 0:
                    self._enqueue("FIRST_CONTACT" if phase == "contact" else phase)
                return sample
            self.env.sample = observed_sample
            self.status.value = f"PLAYING {'CONTINUOUS ' if continuous else ''}POLICY seed={seed}"
            try:
                goal_index = 1
                reset_index = 0
                while not (continuous and self.continuous_stop.is_set()):
                    for decision in range(int(self.policy_config["environment"]["task"]["episode_steps"])):
                        if continuous and self.continuous_stop.is_set():
                            break
                        with torch.no_grad():
                            action = model.deterministic(torch.as_tensor(normalizer.normalize(observation), dtype=torch.float32)).cpu().numpy()
                        observation, reward, terminated, truncated, info = self.goal_env.step(action)
                        self.direction.value = (
                            f"goal {goal_index}, decision {decision}: theta={float(info['action_theta']):.3f} rad; "
                            f"travel={1000.0 * float(info['commanded_after_touch_travel_m']):.1f} mm; reward={float(reward):.3f}"
                        )
                        self._enqueue("REOBSERVE")
                        if terminated or truncated:
                            if not continuous:
                                self.stage.value = "SUCCESS" if info.get("success") else f"TERMINATED: {info.get('terminated_reason')}"
                                self.status.value = f"POLICY DONE after {decision + 1} decisions; error={1000.0 * float(info['post_settled_goal_distance_m']):.2f} mm"
                                return
                            if info.get("success") and self._sample_next_legal_goal():
                                goal_index += 1
                                self.status.value = f"GOAL {goal_index - 1} SUCCESS — continuing from settled cube"
                                observation = self.goal_env.observation()
                                break
                            reset_index += 1
                            next_seed = seed + reset_index
                            self.status.value = f"RESET after {info.get('terminated_reason')} — starting goal {goal_index + 1}"
                            observation, _ = self.goal_env.reset(next_seed)
                            self._set_goal_marker()
                            self._enqueue("RESET_AFTER_FAILURE")
                            goal_index += 1
                            break
                    else:
                        continue
                    if not continuous or self.continuous_stop.is_set():
                        break
                if continuous:
                    self.stage.value = "CONTINUOUS PLAY STOPPED"
                    self.status.value = "CONTINUOUS POLICY STOPPED — scene retained"
            finally:
                self.env.sample = original_sample
        except Exception as error:
            self.stage.value = "VIEWER ERROR"
            self.status.value = f"{type(error).__name__}: {error}"
            print(f"VIEWER_POLICY_ERROR {type(error).__name__}: {error}", flush=True)
        finally:
            self.running = False

    def _run(self) -> None:
        _, x, y, degree = self._selected()
        self.env.reset([x, y])
        self.sample_count = 0
        original_sample = self.env.sample
        def observed_sample(phase: str) -> dict[str, Any]:
            sample = original_sample(phase)
            self.sample_count += 1
            if self.sample_count % self.sample_stride == 0:
                self._enqueue("FIRST_CONTACT" if phase == "contact" else phase)
            return sample
        self.env.sample = observed_sample  # instance-local viewer hook only
        try:
            self.status.value = "PLAYING — playback remains at failure"
            result, _ = self.env.push(
                math.radians(degree), .08,
                float(self.config["environment"]["action"]["sustained_push_speed_mps"]),
            )
            if result.get("unrecoverable"):
                self.stage.value = f"FAILED: {result.get('precontact_stage')} / {result.get('precontact_failure_cause')}"
                self.status.value = "FAILED — scene retained; Reset to inspect another case"
            else:
                self.stage.value = "DONE"
                self.status.value = "DONE — scene retained"
        except Exception as error:
            self.stage.value = "VIEWER ERROR"
            self.status.value = f"{type(error).__name__}: {error}"
            print(f"VIEWER_PLAY_ERROR {type(error).__name__}: {error}", flush=True)
        finally:
            self.env.sample = original_sample
            self.running = False

    def _apply(self, frame: dict[str, Any]) -> None:
        for gid, handle in self.handles.items():
            handle.position = tuple(float(v) for v in frame["geom_xpos"][gid])
            handle.wxyz = quat(frame["geom_xmat"][gid])
        self.cube.position = tuple(float(v) for v in frame["geom_xpos"][self.cube_id])
        self.cube.wxyz = quat(frame["geom_xmat"][self.cube_id])
        self.stage.value = frame["phase"]
        visible = bool(self.contacts.value)
        for i, (point, normal) in enumerate(zip(self.contact_points, self.contact_normals, strict=True)):
            if i >= len(frame["contacts"]) or not visible:
                point.visible = normal.visible = False
                continue
            p, n = np.asarray(frame["contacts"][i]["position"]), np.asarray(frame["contacts"][i]["normal"])
            point.position = tuple(float(v) for v in p); point.visible = True
            normal.points = np.asarray([[p, p + .035 * n]]); normal.visible = True

    def _render_loop(self) -> None:
        previous = None
        while not self.stop.is_set():
            try:
                frame = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            changed = previous is not None and frame["phase"] != previous
            self._apply(frame); self.queue.task_done()
            previous = frame["phase"]
            time.sleep(.8 if changed else 1.0 / self.fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/piperx_goal_push_dev.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--sample-stride", type=int, default=40)
    parser.add_argument("--policy-run", type=Path, default=Path("results/piperx_goalpush_ppo_policy_v2_nominal_20260818T190900Z"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/piperx_goalpush_ppo_policy_v2_nominal_20260818T190900Z/checkpoints/step_000049152.pt"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    server = viser.ViserServer(host=args.host, port=args.port)
    GateViewer(config, server, args.fps, args.sample_stride, args.policy_run.resolve(), args.checkpoint.resolve())
    print(f"VIEWER_READY http://{args.host}:{args.port}", flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
