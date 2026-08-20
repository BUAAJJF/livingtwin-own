#!/usr/bin/env python3
"""Reproduce selected Policy-v2 evaluation seeds with dormant executor diagnostics.

This is intentionally a standalone engineering tool.  It loads the frozen
checkpoint and normalizer, follows ``evaluate_goal_policy`` exactly for one
seed at a time, and writes only a new diagnostic artifact directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from livingtwin_mujoco_rl.checkpoint import load_checkpoint
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.piperx_goal_push_env import PiperGoalPushEnv


CASES = {
    910001: {"kind": "immediate_large_failure", "decision": 0},
    910012: {"kind": "immediate_large_failure", "decision": 0},
    910022: {"kind": "immediate_large_failure", "decision": 0},
    910004: {"kind": "small_aligned_failure", "decision": 5},
    910030: {"kind": "small_aligned_failure", "decision": 1},
    910044: {"kind": "small_aligned_failure", "decision": 20},
    910002: {"kind": "success_control", "decision": 15},
    910003: {"kind": "success_control", "decision": 3},
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_rows(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["seed"]): row for row in csv.DictReader(path.open())}


def _read_decisions(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    return {(int(row["seed"]), int(row["decision_index"])): row for row in csv.DictReader(path.open())}


class Collector:
    def __init__(self, seed: int):
        self.seed = seed
        self.decision_index = -1
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append({
            "seed": self.seed,
            "decision_index": self.decision_index,
            "event": event,
            **_jsonable(payload),
        })


def _state(env: PiperGoalPushEnv) -> dict[str, Any]:
    executor = env.executor
    site_id = env.model.site("strike_site").id
    return {
        "object_xy": env._object_xy(),
        "goal_xy": env.goal.copy(),
        "goal_error_m": env._distance(),
        "ee_position": env.data.site_xpos[site_id].copy(),
        "ee_orientation_matrix": env.data.site_xmat[site_id].reshape(3, 3).copy(),
        "arm_qpos": env.data.qpos[env.joint_qpos].copy(),
        "closed_tip_position": env.data.site_xpos[executor.closed_tip_site_id].copy(),
    }


def _match(original: dict[str, str] | None, replay: dict[str, Any], *, tolerance: float = 2.0e-6) -> dict[str, Any]:
    if original is None:
        return {"available": False}
    fields = {
        "action_theta": float(replay["info"]["action_theta"]),
        "action_magnitude": float(replay["info"]["action_magnitude"]),
        "pre_settled_goal_distance_m": float(replay["info"]["pre_settled_goal_distance_m"]),
        "commanded_after_touch_travel_m": float(replay["info"]["commanded_after_touch_travel_m"]),
    }
    deltas = {key: abs(float(original[key]) - value) for key, value in fields.items()}
    return {
        "available": True,
        "original": {key: original[key] for key in fields},
        "replay": fields,
        "absolute_deltas": deltas,
        "within_tolerance": bool(all(value <= tolerance for value in deltas.values())),
    }


def replay_case(
    *,
    seed: int,
    config: dict[str, Any],
    model: ActorCritic,
    normalizer: RunningMeanStd,
    original_episode: dict[str, str],
    original_decisions: dict[tuple[int, int], dict[str, str]],
) -> dict[str, Any]:
    env = PiperGoalPushEnv(config["environment"], seed)
    observation, _ = env.reset(seed)
    collector = Collector(seed)
    env.executor.diagnostic_hook = collector
    decisions: list[dict[str, Any]] = []
    total_reward = 0.0
    for index in range(int(config["environment"]["task"]["episode_steps"])):
        collector.decision_index = index
        before = _state(env)
        with torch.no_grad():
            action = model.deterministic(
                torch.as_tensor(normalizer.normalize(observation), dtype=torch.float32)
            ).cpu().numpy()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        row = {
            "seed": seed,
            "decision_index": index,
            "state_before": before,
            "raw_action": np.asarray(action).copy(),
            "info": dict(info),
            "state_after": _state(env),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "artifact_match": _match(original_decisions.get((seed, index)), {
                "info": info,
            }),
        }
        decisions.append(_jsonable(row))
        if terminated or truncated:
            break
    final = decisions[-1]
    outcome = {
        "success": bool(final["info"]["success"]),
        "termination_reason": str(final["info"]["terminated_reason"]),
        "ik_failure": bool(final["info"].get("ik_failure", False)),
        "oob": bool(final["info"].get("oob", False)),
        "decision_count": len(decisions),
        "episode_return": total_reward,
    }
    original_outcome = {
        "success": original_episode["success"] == "True",
        "termination_reason": original_episode["termination_reason"],
        "ik_failure": original_episode["ik_failure"] == "True",
        "oob": original_episode["oob"] == "True",
        "decision_count": int(original_episode["episode_steps"]),
    }
    target = CASES[seed]
    target_row = next(row for row in decisions if row["decision_index"] == target["decision"])
    outcome_match = all(outcome[key] == value for key, value in original_outcome.items())
    return _jsonable({
        "seed": seed,
        "case": target,
        "original_outcome": original_outcome,
        "replay_outcome": outcome,
        "outcome_exact_match": outcome_match,
        "target_decision_match": target_row["artifact_match"],
        "decisions": decisions,
        "events": collector.events,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output or Path("results") / f"piperx_policy_v2_ik_localization_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((run / "FROZEN_CONFIG.json").read_text())
    episode_rows = _read_rows(run / "evaluation_episodes_step_000049152.csv")
    decision_rows = _read_decisions(run / "evaluation_decisions_step_000049152.csv")
    probe = PiperGoalPushEnv(config["environment"], 0)
    observation, _ = probe.reset(0)
    model = ActorCritic(observation.shape[-1], 2, config["network"]["hidden_sizes"])
    normalizer = RunningMeanStd((observation.shape[-1],))
    payload = load_checkpoint(checkpoint, model, None, normalizer)
    results = []
    for seed in CASES:
        result = replay_case(
            seed=seed,
            config=config,
            model=model,
            normalizer=normalizer,
            original_episode=episode_rows[seed],
            original_decisions=decision_rows,
        )
        (output / f"case_{seed}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
    with (output / "replay_summary.csv").open("w", newline="") as handle:
        fields = ["seed", "kind", "target_decision", "outcome_exact_match", "replay_reason", "replay_ik_failure", "replay_steps", "target_match"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for result in results:
            writer.writerow({
                "seed": result["seed"], "kind": result["case"]["kind"], "target_decision": result["case"]["decision"],
                "outcome_exact_match": result["outcome_exact_match"], "replay_reason": result["replay_outcome"]["termination_reason"],
                "replay_ik_failure": result["replay_outcome"]["ik_failure"], "replay_steps": result["replay_outcome"]["decision_count"],
                "target_match": result["target_decision_match"].get("within_tolerance", False),
            })
    with (output / "phase_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "decision_index", "event", "payload_json"]); writer.writeheader()
        for result in results:
            for event in result["events"]:
                compact = dict(event); seed = compact.pop("seed"); decision = compact.pop("decision_index"); name = compact.pop("event")
                writer.writerow({"seed": seed, "decision_index": decision, "event": name, "payload_json": json.dumps(compact, sort_keys=True)})
    provenance = {
        "checkpoint": str(checkpoint), "checkpoint_global_step": int(payload["global_step"]),
        "evaluation_artifacts": str(run), "evaluation_step": 49152,
        "normalizer_restored": True, "evaluation_path": "PiperGoalPushEnv.reset(seed) -> deterministic(normalizer.normalize(observation)) -> env.step",
        "cases": CASES,
    }
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
