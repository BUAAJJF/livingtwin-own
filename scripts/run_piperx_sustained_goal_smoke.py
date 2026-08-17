#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import yaml

from livingtwin_mujoco_rl.piperx_goal_push_env import PiperGoalPushEnv


DIRECTIONS = (
    ("right", 0.0), ("upper_right", math.pi / 4), ("up", math.pi / 2),
    ("upper_left", 3 * math.pi / 4), ("left", math.pi),
    ("lower_left", -3 * math.pi / 4), ("down", -math.pi / 2),
    ("lower_right", -math.pi / 4),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pushes", type=int, default=4)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["environment"]
    rows = []
    for index, (name, angle) in enumerate(DIRECTIONS):
        env = PiperGoalPushEnv(config, seed=8000 + index)
        env.reset(seed=8000 + index)
        center = np.asarray([0.45, 0.0], dtype=np.float64)
        unit = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
        env.executor.reset(center)
        env.goal[:] = center + 0.08 * unit
        env.model.site_pos[env.model.site("goal_site").id, :2] = env.goal
        mujoco.mj_forward(env.model, env.data)
        initial_distance = env._distance()
        events = []
        final_info: dict[str, object] = {}
        for _ in range(args.max_pushes):
            _, _, terminated, truncated, final_info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))
            events.append(final_info)
            if terminated or truncated:
                break
        final_distance = env._distance()
        rows.append({
            "direction": name,
            "initial_goal_distance_m": initial_distance,
            "final_goal_distance_m": final_distance,
            "pushes": len(events),
            "meaningful_distance_decrease": bool(initial_distance - final_distance >= 0.002),
            "success": bool(final_info.get("success", False)),
            "contact_failures": int(sum(bool(item.get("contact_failure")) for item in events)),
            "push_ik_failures": int(sum(bool(item.get("push_ik_failure")) for item in events)),
            "table_collision": bool(any(bool(item.get("table_collision")) for item in events)),
            "oob": bool(any(bool(item.get("oob")) for item in events)),
            "safety_termination": bool(any(bool(item.get("safety_termination")) for item in events)),
        })
    payload = {"max_pushes": args.max_pushes, "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
