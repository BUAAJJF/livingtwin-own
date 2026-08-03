from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
import torch

from livingtwin_mujoco_rl.checkpoint import load_checkpoint, save_checkpoint
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.ppo import Rollout, ppo_update, set_global_seed
from livingtwin_mujoco_rl.yaw_env import (
    YAW_OBSERVATION_NAMES,
    YawPlanarPushEnv,
    compute_yaw_reward,
    wrap_to_pi,
)
from livingtwin_mujoco_rl.yaw_evaluate import evaluate_yaw_policy


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(model: ActorCritic) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _move_toward(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.zeros(2, dtype=np.float32) if norm < 1.0e-9 else np.clip(vector / norm, -1.0, 1.0).astype(np.float32)


def heuristic_action(env: YawPlanarPushEnv) -> np.ndarray:
    cube = env.cube_xy()
    pusher = env.data.mocap_pos[env.pusher_mocap_id, :2]
    position_vector = env.target[:2] - cube
    force_direction = _move_toward(position_vector).astype(np.float64)
    if np.linalg.norm(force_direction) < 1.0e-9:
        force_direction[:] = (1.0, 0.0)
    _, _, delta_yaw = env.errors()
    if abs(delta_yaw) > 0.12:
        perpendicular = np.asarray([-force_direction[1], force_direction[0]])
        desired = cube - 0.040 * force_direction - np.sign(delta_yaw) * 0.019 * perpendicular
    else:
        desired = cube - 0.040 * force_direction
    alignment = desired - pusher
    if float(np.linalg.norm(alignment)) > 0.008:
        return _move_toward(alignment)
    return force_direction.astype(np.float32)


def _signed_push_probe(
    env: YawPlanarPushEnv, *, y_offset: float, seed: int
) -> dict[str, float]:
    env.reset(seed=seed)
    cube = env.cube_xy()
    initial_yaw = env.cube_yaw()
    env.data.mocap_pos[env.pusher_mocap_id, :2] = (
        cube[0] - 0.042,
        cube[1] + y_offset,
    )
    mujoco.mj_forward(env.model, env.data)
    contact_point = env.data.mocap_pos[env.pusher_mocap_id, :2].copy()
    final_info: dict[str, Any] = {}
    for _ in range(32):
        _, _, terminated, truncated, final_info = env.step([1.0, 0.0])
        if terminated or truncated:
            break
    yaw_change = float(wrap_to_pi(env.cube_yaw() - initial_yaw))
    return {
        "contact_x_m": float(contact_point[0]),
        "contact_y_m": float(contact_point[1]),
        "push_dx": 1.0,
        "push_dy": 0.0,
        "yaw_change_rad": yaw_change,
        "contact_substeps": int(final_info.get("contact_substeps", 0)),
    }


def run_yaw_preflight(
    smoke_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    asset_path = baseline_config["asset_path"]
    env_config = baseline_config["environment"]
    env = YawPlanarPushEnv(asset_path, env_config, seed=30001)
    checks: dict[str, Any] = {}

    negative_offset = _signed_push_probe(env, y_offset=-0.020, seed=30001)
    positive_offset = _signed_push_probe(env, y_offset=0.020, seed=30002)
    signed_opposite = negative_offset["yaw_change_rad"] * positive_offset["yaw_change_rad"] < 0.0
    signed_large = min(abs(negative_offset["yaw_change_rad"]), abs(positive_offset["yaw_change_rad"])) > 0.03
    checks["A_signed_yaw_controllability"] = {
        "passed": bool(signed_opposite and signed_large),
        "negative_y_offset_probe": negative_offset,
        "positive_y_offset_probe": positive_offset,
    }

    reset_rows = []
    immediate_successes = 0
    for seed in range(31000, 32024):
        observation, info = env.reset(seed=seed)
        reset_rows.append(
            {
                "initial_yaw": float(info["cube_yaw_rad"]),
                "target_yaw": float(info["target_yaw_rad"]),
                "delta_yaw": float(info["delta_yaw_rad"]),
                "position_error": float(info["position_error_m"]),
                "finite": bool(np.all(np.isfinite(observation))),
            }
        )
        immediate_successes += int(info["instantaneous_success"])
    delta_values = np.asarray([row["delta_yaw"] for row in reset_rows])
    reset_pass = bool(
        all(row["finite"] for row in reset_rows)
        and max(row["target_yaw"] for row in reset_rows) > 3.0
        and min(row["target_yaw"] for row in reset_rows) < -3.0
        and np.percentile(np.abs(delta_values), 95) > 2.8
        and immediate_successes == 0
    )
    checks["A_reset_and_full_yaw_distribution"] = {
        "passed": reset_pass,
        "sample_count": len(reset_rows),
        "initial_yaw_min": min(row["initial_yaw"] for row in reset_rows),
        "initial_yaw_max": max(row["initial_yaw"] for row in reset_rows),
        "target_yaw_min": min(row["target_yaw"] for row in reset_rows),
        "target_yaw_max": max(row["target_yaw"] for row in reset_rows),
        "delta_yaw_p95_abs": float(np.percentile(np.abs(delta_values), 95)),
        "instantaneous_success_fraction": float(immediate_successes / len(reset_rows)),
    }

    heuristic_rows = []
    for index in range(32):
        _, initial = env.reset(seed=33000 + index)
        initial_position = float(initial["position_error_m"])
        initial_yaw = float(initial["yaw_error_rad"])
        final_info = initial
        for _ in range(int(env_config["task"]["episode_steps"])):
            _, _, terminated, truncated, final_info = env.step(heuristic_action(env))
            if terminated or truncated:
                break
        heuristic_rows.append(
            {
                "position_improvement_m": initial_position - float(final_info["position_error_m"]),
                "yaw_improvement_rad": initial_yaw - float(final_info["yaw_error_rad"]),
                "initial_yaw_error_rad": initial_yaw,
                "final_yaw_error_rad": float(final_info["yaw_error_rad"]),
                "contact_control_steps": int(final_info["contact_control_steps"]),
                "success": bool(final_info["success"]),
            }
        )
    measurable = [
        row["position_improvement_m"] > 0.005 or row["yaw_improvement_rad"] > 0.05
        for row in heuristic_rows
    ]
    yaw_improved = [row["yaw_improvement_rad"] > 0.05 for row in heuristic_rows]
    checks["B_heuristic_controller"] = {
        "passed": bool(np.mean(measurable) >= 0.80 and np.mean(yaw_improved) >= 0.25),
        "state_count": len(heuristic_rows),
        "measurable_improvement_fraction": float(np.mean(measurable)),
        "yaw_improvement_fraction": float(np.mean(yaw_improved)),
        "mean_position_improvement_m": float(np.mean([row["position_improvement_m"] for row in heuristic_rows])),
        "mean_yaw_improvement_rad": float(np.mean([row["yaw_improvement_rad"] for row in heuristic_rows])),
        "success_rate": float(np.mean([row["success"] for row in heuristic_rows])),
        "all_states_contacted": bool(all(row["contact_control_steps"] > 0 for row in heuristic_rows)),
    }

    reward_cfg = env_config["reward"]
    cases = {}
    for name, position, yaw in (
        ("reference", 0.10, 1.0),
        ("position_better", 0.05, 1.0),
        ("yaw_better", 0.10, 0.3),
        ("both_better", 0.05, 0.3),
        ("position_better_yaw_worse", 0.05, 1.5),
    ):
        total, components = compute_yaw_reward(
            position, yaw, [0.0, 0.0], success_event=False, reward_config=reward_cfg
        )
        cases[name] = {"total": total, **components}
    reference = cases["reference"]["total"]
    checks["C_reward_direction"] = {
        "passed": bool(
            cases["position_better"]["total"] > reference
            and cases["yaw_better"]["total"] > reference
            and cases["both_better"]["total"] > cases["position_better"]["total"]
            and cases["both_better"]["total"] > cases["yaw_better"]["total"]
            and cases["position_better_yaw_worse"]["position"] > cases["reference"]["position"]
            and cases["position_better_yaw_worse"]["yaw"] < cases["reference"]["yaw"]
        ),
        "cases": cases,
        "typical_initial_position_component": -2.0 * float(np.mean([row["position_error"] for row in reset_rows])),
        "typical_initial_yaw_component": -0.15 * float(np.mean(np.abs(delta_values))),
    }

    delta_a = float(wrap_to_pi(np.deg2rad(-179.0) - np.deg2rad(179.0)))
    delta_b = float(wrap_to_pi(np.deg2rad(179.0) - np.deg2rad(-179.0)))
    eps = 1.0e-7
    left = np.asarray([np.sin(np.pi - eps), np.cos(np.pi - eps)])
    right = np.asarray([np.sin(-np.pi + eps), np.cos(-np.pi + eps)])
    checks["D_angle_boundary"] = {
        "passed": bool(abs(delta_a - np.deg2rad(2.0)) < 1.0e-12 and abs(delta_b + np.deg2rad(2.0)) < 1.0e-12 and np.linalg.norm(left - right) < 1.0e-5),
        "target_minus_current_179_to_minus179_rad": delta_a,
        "target_minus_current_minus179_to_179_rad": delta_b,
        "sincos_boundary_difference": float(np.linalg.norm(left - right)),
    }

    set_global_seed(34001)
    model = ActorCritic(15, 2, baseline_config["network"]["hidden_sizes"])
    normalizer = RunningMeanStd((15,))
    observations = [env.reset(seed=34001 + index)[0] for index in range(128)]
    normalizer.update(np.stack(observations))
    eval_seeds = list(range(35001, 35009))
    before_metrics, before_rows = evaluate_yaw_policy(
        model, normalizer, asset_path=asset_path, env_config=env_config, seeds=eval_seeds
    )
    generator = torch.Generator().manual_seed(36001)
    count = 256
    synthetic = torch.randn((count, 15), generator=generator)
    with torch.no_grad():
        _, pre_tanh, old_logp, values = model.sample(synthetic)
    rollout = Rollout(
        observations=synthetic,
        pre_tanh_actions=pre_tanh,
        old_log_probabilities=old_logp,
        advantages=torch.randn(count, generator=generator),
        returns=values + 0.1 * torch.randn(count, generator=generator),
        values=values,
    )
    training = dict(baseline_config["training"])
    training["minibatch_size"] = 64
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    before_digest = _digest(model)
    ppo_update(model, optimizer, rollout, training, generator=generator)
    after_digest = _digest(model)
    after_metrics, after_rows = evaluate_yaw_policy(
        model, normalizer, asset_path=asset_path, env_config=env_config, seeds=eval_seeds
    )
    checks["E_zero_learning_rate"] = {
        "passed": bool(before_digest == after_digest and before_metrics == after_metrics and before_rows == after_rows),
        "parameter_sha256_before": before_digest,
        "parameter_sha256_after": after_digest,
        "evaluation_exact_match": before_rows == after_rows,
    }
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        path = Path(temporary) / "roundtrip.pt"
        save_checkpoint(path, model, optimizer, normalizer, global_step=0, config=baseline_config, seed=34001)
        restored = ActorCritic(15, 2, baseline_config["network"]["hidden_sizes"])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.0)
        restored_normalizer = RunningMeanStd((15,))
        payload = load_checkpoint(path, restored, restored_optimizer, restored_normalizer)
        restored_metrics, restored_rows = evaluate_yaw_policy(
            restored, restored_normalizer, asset_path=asset_path, env_config=env_config, seeds=eval_seeds
        )
        checks["E_checkpoint_roundtrip"] = {
            "passed": bool(
                _digest(restored) == _digest(model)
                and restored_rows == after_rows
                and restored_metrics == after_metrics
                and np.array_equal(restored_normalizer.mean, normalizer.mean)
                and np.array_equal(restored_normalizer.var, normalizer.var)
                and restored_normalizer.count == normalizer.count
                and bool(payload["optimizer_state"]["state"])
                and "rng_state" in payload
            ),
            "actor_critic_exact_match": _digest(restored) == _digest(model),
            "evaluation_exact_match": restored_rows == after_rows,
            "normalization_exact_match": bool(np.array_equal(restored_normalizer.mean, normalizer.mean) and np.array_equal(restored_normalizer.var, normalizer.var)),
            "optimizer_state_present": bool(payload["optimizer_state"]["state"]),
            "rng_state_present": "rng_state" in payload,
        }

    passed = all(bool(row["passed"]) for row in checks.values())
    result = {
        "schema_version": "planar_push_yaw_preflight_v1",
        "status": "PASS" if passed else "FAIL",
        "training_started": False,
        "observation_names": list(YAW_OBSERVATION_NAMES),
        "asset_sha256": _sha256(asset_path),
        "smoke_env_sha256": _sha256(smoke_config["environment"]["_source_path"]),
        "baseline_env_sha256": _sha256(env_config["_source_path"]),
        "checks": checks,
    }
    (output / "YAW_PRECHECK_RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
