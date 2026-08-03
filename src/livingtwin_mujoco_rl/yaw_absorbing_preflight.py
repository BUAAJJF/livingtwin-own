from __future__ import annotations

import hashlib
import json
import math
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
from livingtwin_mujoco_rl.yaw_absorbing_env import AbsorbingYawPlanarPushEnv
from livingtwin_mujoco_rl.yaw_absorbing_evaluate import evaluate_absorbing_policy
from livingtwin_mujoco_rl.yaw_env import YawPlanarPushEnv, wrap_to_pi
from livingtwin_mujoco_rl.yaw_preflight import heuristic_action


def _discounted(rewards: list[float], gamma: float) -> float:
    return float(sum((gamma**index) * reward for index, reward in enumerate(rewards)))


def _load_frozen_policy(root: Path, seed: int, config: Mapping[str, Any]) -> tuple[ActorCritic, RunningMeanStd]:
    model = ActorCritic(15, 2, config["network"]["hidden_sizes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    normalizer = RunningMeanStd((15,))
    load_checkpoint(root / f"results/yaw_baseline_seed_{seed}/checkpoints/step_001048576.pt", model, optimizer, normalizer)
    return model, normalizer


def _trajectory(
    model: ActorCritic,
    normalizer: RunningMeanStd,
    config: Mapping[str, Any],
    eval_seed: int,
) -> dict[str, Any]:
    env = YawPlanarPushEnv(config["asset_path"], config["environment"], seed=eval_seed)
    observation, initial = env.reset(seed=eval_seed)
    rewards: list[float] = []
    while True:
        with torch.no_grad():
            action = model.deterministic(torch.as_tensor(normalizer.normalize(observation), dtype=torch.float32).unsqueeze(0))[0].cpu().numpy()
        observation, reward, terminated, truncated, final = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    return {
        "eval_seed": eval_seed,
        "reason": final["terminated_reason"],
        "steps": len(rewards),
        "rewards": rewards,
        "position_improvement_m": float(initial["position_error_m"] - final["position_error_m"]),
        "yaw_improvement_rad": float(initial["yaw_error_rad"] - final["yaw_error_rad"]),
    }


def audit_return_order(
    root: Path, old_config: Mapping[str, Any], failure_reward: float, gamma: float
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for policy_seed in (2201, 2202, 2203):
        model, normalizer = _load_frozen_policy(root, policy_seed, old_config)
        for eval_seed in range(620000, 620100):
            row = _trajectory(model, normalizer, old_config, eval_seed)
            row["policy_seed"] = policy_seed
            row["undiscounted_return"] = float(sum(row["rewards"]))
            row["discounted_return_gamma_0_99"] = _discounted(row["rewards"], gamma)
            rows.append(row)
    categories = {
        "success": [row for row in rows if row["reason"] == "success"],
        "valid_timeout_meaningful_progress": [
            row for row in rows
            if row["reason"] == "timeout"
            and (row["position_improvement_m"] > 0.005 or row["yaw_improvement_rad"] > 0.05)
        ],
        "out_of_bounds_failure": [row for row in rows if row["reason"] == "out_of_bounds"],
    }
    representatives: dict[str, Any] = {}
    for name, candidates in categories.items():
        ordered = sorted(candidates, key=lambda row: row["discounted_return_gamma_0_99"])
        chosen = ordered[len(ordered) // 2]
        representatives[name] = {
            key: value for key, value in chosen.items() if key != "rewards"
        }
        representatives[name]["category_count"] = len(candidates)
        if name == "out_of_bounds_failure":
            fixed_rewards = chosen["rewards"][:-1] + [failure_reward] * (200 - len(chosen["rewards"]) + 1)
            representatives[name]["absorbing_fixed_undiscounted_return"] = float(sum(fixed_rewards))
            representatives[name]["absorbing_fixed_discounted_return_gamma_0_99"] = _discounted(fixed_rewards, gamma)
    success = representatives["success"]
    timeout = representatives["valid_timeout_meaningful_progress"]
    failure = representatives["out_of_bounds_failure"]
    current_undiscounted_pass = success["undiscounted_return"] > timeout["undiscounted_return"] > failure["undiscounted_return"]
    current_discounted_pass = success["discounted_return_gamma_0_99"] > timeout["discounted_return_gamma_0_99"] > failure["discounted_return_gamma_0_99"]
    fixed_undiscounted_pass = success["undiscounted_return"] > timeout["undiscounted_return"] > failure["absorbing_fixed_undiscounted_return"]
    fixed_discounted_pass = success["discounted_return_gamma_0_99"] > timeout["discounted_return_gamma_0_99"] > failure["absorbing_fixed_discounted_return_gamma_0_99"]
    return {
        "gamma": gamma,
        "representative_selection": "median gamma-discounted return within each category over 3 frozen policies x 100 common states",
        "representatives": representatives,
        "current_environment_order_passed": {
            "undiscounted": current_undiscounted_pass,
            "discounted_gamma_0_99": current_discounted_pass,
        },
        "absorbing_fix_order_passed": {
            "undiscounted": fixed_undiscounted_pass,
            "discounted_gamma_0_99": fixed_discounted_pass,
        },
        "passed": bool(not current_undiscounted_pass and not current_discounted_pass and fixed_undiscounted_pass and fixed_discounted_pass),
    }


def absorbing_semantics_check(config: Mapping[str, Any]) -> dict[str, Any]:
    env = AbsorbingYawPlanarPushEnv(config["asset_path"], config["environment"], seed=41001)
    env.reset(seed=41001)
    env.data.qpos[env.cube_qpos_adr] = float(config["environment"]["task"]["cube_x_bounds_m"][1]) + 0.001
    mujoco.mj_forward(env.model, env.data)
    observation, reward, terminated, truncated, info = env.step([0.0, 0.0])
    frozen_qpos = env.data.qpos.copy()
    frozen_mocap = env.data.mocap_pos.copy()
    checks = [
        info["failure"], info["absorption_step"] == 1, not terminated, not truncated,
        np.array_equal(observation, np.zeros(15, dtype=np.float32)),
        reward == env.failure_reward,
    ]
    nonterminal_steps = 1
    all_fixed = True
    for step in range(2, 200):
        observation, reward, terminated, truncated, info = env.step([1.0, -1.0])
        nonterminal_steps += 1
        all_fixed &= bool(
            not terminated and not truncated and reward == env.failure_reward
            and np.array_equal(observation, np.zeros(15, dtype=np.float32))
            and np.array_equal(env.data.qpos, frozen_qpos)
            and np.array_equal(env.data.mocap_pos, frozen_mocap)
            and info["action_ignored"]
        )
    _, final_reward, final_terminated, final_truncated, final_info = env.step([-1.0, 1.0])
    checks.extend(
        [
            all_fixed, nonterminal_steps == 199, not final_terminated, final_truncated,
            final_reward == env.failure_reward, final_info["step_count"] == 200,
            final_info["terminated_reason"] == "absorbing_failure_horizon",
        ]
    )
    return {
        "passed": bool(all(checks)),
        "trigger_done_mask": 0,
        "absorbing_steps_2_to_199_done_mask": 0,
        "horizon_step_200_done_mask": 1,
        "gae_nonterminal_mask_through_step_199": 1,
        "bootstrap_at_step_200": 0,
        "nonterminal_absorbing_steps": nonterminal_steps,
        "physics_and_observation_fixed": all_fixed,
        "failure_reward": env.failure_reward,
    }


def yaw_bin_reachability(config: Mapping[str, Any]) -> dict[str, Any]:
    env = AbsorbingYawPlanarPushEnv(config["asset_path"], config["environment"], seed=700000)
    bins = {
        "0_45_deg": (0.0, math.pi / 4),
        "45_90_deg": (math.pi / 4, math.pi / 2),
        "90_135_deg": (math.pi / 2, 3 * math.pi / 4),
        "135_180_deg": (3 * math.pi / 4, math.pi + 1.0e-12),
    }
    selected: dict[str, list[int]] = {name: [] for name in bins}
    seed = 700000
    while min(len(values) for values in selected.values()) < 32:
        _, info = env.reset(seed=seed)
        error = abs(float(info["delta_yaw_rad"]))
        for name, (lower, upper) in bins.items():
            if lower <= error < upper and len(selected[name]) < 32:
                selected[name].append(seed)
                break
        seed += 1
    result: dict[str, Any] = {}
    for name, seeds in selected.items():
        rows = []
        for eval_seed in seeds:
            _, initial = env.reset(seed=eval_seed)
            initial_yaw = env.cube_yaw()
            initial_delta = float(initial["delta_yaw_rad"])
            final = initial
            for _ in range(int(config["environment"]["task"]["episode_steps"])):
                _, _, terminated, truncated, final = env.step(heuristic_action(env))
                if terminated or truncated:
                    break
            yaw_motion = float(wrap_to_pi(env.cube_yaw() - initial_yaw))
            position_improvement = float(initial["position_error_m"] - final["position_error_m"])
            yaw_improvement = float(initial["yaw_error_rad"] - final["yaw_error_rad"])
            rows.append(
                {
                    "correct_rotation_direction": bool(np.sign(initial_delta) * yaw_motion > 0.03),
                    "position_improvement_m": position_improvement,
                    "yaw_improvement_rad": yaw_improvement,
                    "simultaneous_improvement": bool(position_improvement > 0.005 and yaw_improvement > 0.05),
                    "success": bool(final["success"]),
                    "failure": bool(final.get("failure", False)),
                }
            )
        result[name] = {
            "state_count": len(rows),
            "correct_rotation_direction_fraction": float(np.mean([row["correct_rotation_direction"] for row in rows])),
            "simultaneous_position_yaw_improvement_fraction": float(np.mean([row["simultaneous_improvement"] for row in rows])),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "failure_rate": float(np.mean([row["failure"] for row in rows])),
            "mean_position_improvement_m": float(np.mean([row["position_improvement_m"] for row in rows])),
            "mean_yaw_improvement_rad": float(np.mean([row["yaw_improvement_rad"] for row in rows])),
        }
    return {
        "bins": result,
        "diagnostic_conclusion": "The heuristic is a controllability probe, not a completeness proof. A zero success bin is recorded as not demonstrated within four seconds, not as proof of physical impossibility.",
        "passed": bool(all(row["correct_rotation_direction_fraction"] > 0.0 for row in result.values())),
    }


def optimizer_checkpoint_check(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    set_global_seed(43001)
    model = ActorCritic(15, 2, config["network"]["hidden_sizes"])
    normalizer = RunningMeanStd((15,))
    env = AbsorbingYawPlanarPushEnv(config["asset_path"], config["environment"], seed=43001)
    observations = [env.reset(seed=43001 + index)[0] for index in range(64)]
    normalizer.update(np.stack(observations))
    seeds = list(range(44001, 44009))
    before_metrics, before_rows = evaluate_absorbing_policy(model, normalizer, asset_path=config["asset_path"], env_config=config["environment"], seeds=seeds)
    generator = torch.Generator().manual_seed(45001)
    count = 256
    synthetic = torch.randn((count, 15), generator=generator)
    with torch.no_grad(): _, actions, logp, values = model.sample(synthetic)
    rollout = Rollout(
        observations=synthetic,
        pre_tanh_actions=actions,
        old_log_probabilities=logp,
        advantages=torch.randn(count, generator=generator),
        returns=values + 0.1 * torch.randn(count, generator=generator),
        values=values,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    before = hashlib.sha256(b"".join(value.detach().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    training = dict(config["training"]); training["minibatch_size"] = 64
    ppo_update(model, optimizer, rollout, training, generator=generator)
    after = hashlib.sha256(b"".join(value.detach().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    after_metrics, after_rows = evaluate_absorbing_policy(model, normalizer, asset_path=config["asset_path"], env_config=config["environment"], seeds=seeds)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        path = Path(temporary) / "roundtrip.pt"
        save_checkpoint(path, model, optimizer, normalizer, global_step=0, config=config, seed=43001)
        restored = ActorCritic(15, 2, config["network"]["hidden_sizes"])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.0)
        restored_normalizer = RunningMeanStd((15,))
        payload = load_checkpoint(path, restored, restored_optimizer, restored_normalizer)
        restored_metrics, restored_rows = evaluate_absorbing_policy(restored, restored_normalizer, asset_path=config["asset_path"], env_config=config["environment"], seeds=seeds)
    return {
        "passed": bool(before == after and before_metrics == after_metrics == restored_metrics and before_rows == after_rows == restored_rows and payload["optimizer_state"]["state"]),
        "zero_lr_parameter_exact_match": before == after,
        "zero_lr_evaluation_exact_match": before_rows == after_rows,
        "checkpoint_evaluation_exact_match": after_rows == restored_rows,
        "optimizer_state_present": bool(payload["optimizer_state"]["state"]),
    }


def run_absorbing_preflight(
    root: Path, smoke_config: Mapping[str, Any], full_config: Mapping[str, Any], old_config: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_reward = float(smoke_config["environment"]["absorbing_failure"]["reward_per_step"])
    task = smoke_config["environment"]["task"]
    max_dx = max(abs(task["target_x_range_m"][1] - task["cube_x_bounds_m"][0]), abs(task["cube_x_bounds_m"][1] - task["target_x_range_m"][0]))
    max_dy = max(abs(task["target_y_range_m"][1] - task["cube_y_bounds_m"][0]), abs(task["cube_y_bounds_m"][1] - task["target_y_range_m"][0]))
    lower_bound = -2.0 * math.hypot(max_dx, max_dy) - 0.15 * math.pi - 0.001 * 2.0
    bound = {"max_legal_position_error_m": math.hypot(max_dx, max_dy), "most_negative_legal_ordinary_reward": lower_bound, "frozen_failure_reward": failure_reward, "passed": failure_reward < lower_bound}
    return_audit = audit_return_order(root, old_config, failure_reward, 0.99)
    semantics = absorbing_semantics_check(smoke_config)
    reachability = yaw_bin_reachability(full_config)
    optimizer = optimizer_checkpoint_check(smoke_config, output_dir)
    result = {
        "schema_version": "yaw_absorbing_preflight_v1",
        "status": "PASS" if all(section["passed"] for section in (bound, return_audit, semantics, reachability, optimizer)) else "FAIL",
        "training_started": False,
        "failure_reward_bound": bound,
        "return_order_audit": return_audit,
        "absorbing_semantics": semantics,
        "yaw_bin_reachability": reachability,
        "optimizer_checkpoint": optimizer,
    }
    (output_dir / "RETURN_ORDER_AUDIT.json").write_text(json.dumps(return_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "YAW_REACHABILITY_BY_BIN.json").write_text(json.dumps(reachability, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "ABSORBING_PRECHECK_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
