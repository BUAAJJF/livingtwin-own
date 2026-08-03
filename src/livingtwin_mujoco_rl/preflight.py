from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from livingtwin_mujoco_rl.checkpoint import load_checkpoint, save_checkpoint
from livingtwin_mujoco_rl.env import PlanarPushEnv, compute_reward
from livingtwin_mujoco_rl.evaluate import evaluate_policy
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.ppo import Rollout, ppo_update, set_global_seed


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manual_action(env: PlanarPushEnv) -> np.ndarray:
    cube = env._cube_xy()
    target_direction = env.target - cube
    norm = float(np.linalg.norm(target_direction))
    if norm < 1.0e-9:
        return np.zeros(2, dtype=np.float32)
    direction = target_direction / norm
    pusher = env.data.mocap_pos[env.pusher_mocap_id, :2]
    desired_behind = cube - direction * 0.040
    alignment = desired_behind - pusher
    if float(np.linalg.norm(alignment)) > 0.009:
        command = alignment / max(float(np.linalg.norm(alignment)), 1.0e-9)
    else:
        command = direction
    return np.clip(command, -1.0, 1.0).astype(np.float32)


def _parameter_digest(model: ActorCritic) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_preflight(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env_config = config["environment"]
    asset_path = config["asset_path"]
    details: dict[str, Any] = {}

    # A: reset distribution and observation contract.
    env = PlanarPushEnv(asset_path, env_config, seed=7001)
    reset_rows = []
    for seed in range(7001, 7129):
        observation, info = env.reset(seed=seed)
        reset_rows.append(
            {
                "seed": seed,
                "observation_finite": bool(np.all(np.isfinite(observation))),
                "observation_size": int(observation.size),
                "cube_xy": info["cube_xy"],
                "target_xy": info["target_xy"],
                "distance_m": info["distance_m"],
            }
        )
    task = env_config["task"]
    check_a = bool(
        all(row["observation_finite"] and row["observation_size"] == 13 for row in reset_rows)
        and all(float(task["target_x_range_m"][0]) <= row["target_xy"][0] <= float(task["target_x_range_m"][1]) for row in reset_rows)
        and all(float(task["target_y_range_m"][0]) <= row["target_xy"][1] <= float(task["target_y_range_m"][1]) for row in reset_rows)
        and min(row["distance_m"] for row in reset_rows) > float(task["success_distance_m"])
    )
    details["A_reset_distribution"] = {
        "passed": check_a,
        "sample_count": len(reset_rows),
        "minimum_initial_distance_m": float(min(row["distance_m"] for row in reset_rows)),
        "maximum_initial_distance_m": float(max(row["distance_m"] for row in reset_rows)),
        "observation_size": 13,
    }

    # B: stochastic, positively directed actions must establish contact and move the cube.
    rng = np.random.default_rng(7201)
    random_rows = []
    for episode in range(32):
        env.reset(seed=7201 + episode)
        initial = env._cube_xy()
        info: dict[str, Any] = {}
        for _ in range(100):
            action = np.asarray(
                [rng.uniform(0.55, 1.0), rng.uniform(-0.35, 0.35)], dtype=np.float32
            )
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        random_rows.append(
            {
                "displacement_m": float(np.linalg.norm(env._cube_xy() - initial)),
                "contact_count": int(info.get("contact_count", 0)),
            }
        )
    check_b = bool(
        np.mean([row["displacement_m"] > 0.005 for row in random_rows]) >= 0.75
        and np.mean([row["contact_count"] > 0 for row in random_rows]) >= 0.75
    )
    details["B_random_contact_motion"] = {
        "passed": check_b,
        "episode_count": len(random_rows),
        "moved_fraction": float(np.mean([row["displacement_m"] > 0.005 for row in random_rows])),
        "contact_fraction": float(np.mean([row["contact_count"] > 0 for row in random_rows])),
        "mean_displacement_m": float(np.mean([row["displacement_m"] for row in random_rows])),
        "action_distribution": "vx~Uniform(0.55,1), vy~Uniform(-0.35,0.35)",
    }

    # C: a geometric feedback controller should reduce target error.
    manual_rows = []
    for episode in range(32):
        _, reset_info = env.reset(seed=7301 + episode)
        initial_distance = float(reset_info["distance_m"])
        final_info: dict[str, Any] = reset_info
        for _ in range(int(task["episode_steps"])):
            _, _, terminated, truncated, final_info = env.step(_manual_action(env))
            if terminated or truncated:
                break
        manual_rows.append(
            {
                "initial_distance_m": initial_distance,
                "final_distance_m": float(final_info["distance_m"]),
                "improvement_m": initial_distance - float(final_info["distance_m"]),
                "success": bool(final_info["success"]),
            }
        )
    check_c = bool(
        np.mean([row["improvement_m"] > 0.02 for row in manual_rows]) >= 0.80
        and np.mean([row["improvement_m"] for row in manual_rows]) > 0.04
    )
    details["C_manual_controller"] = {
        "passed": check_c,
        "episode_count": len(manual_rows),
        "improved_fraction": float(np.mean([row["improvement_m"] > 0.02 for row in manual_rows])),
        "mean_improvement_m": float(np.mean([row["improvement_m"] for row in manual_rows])),
        "success_rate": float(np.mean([row["success"] for row in manual_rows])),
    }

    # D: the only dense task term is strictly monotone with target distance.
    far_reward, _ = compute_reward(
        0.10, [0.0, 0.0], success_event=False, reward_config=env_config["reward"]
    )
    near_reward, _ = compute_reward(
        0.03, [0.0, 0.0], success_event=False, reward_config=env_config["reward"]
    )
    success_reward, _ = compute_reward(
        0.02, [0.0, 0.0], success_event=True, reward_config=env_config["reward"]
    )
    check_d = bool(near_reward > far_reward and success_reward > near_reward)
    details["D_reward_direction"] = {
        "passed": check_d,
        "far_reward": far_reward,
        "near_reward": near_reward,
        "success_reward": success_reward,
    }

    # Shared frozen policy/normalization fixture for E and F.
    set_global_seed(7401)
    model = ActorCritic(13, 2, config["network"]["hidden_sizes"])
    normalizer = RunningMeanStd((13,))
    observations = []
    for seed in range(7401, 7529):
        observation, _ = env.reset(seed=seed)
        observations.append(observation)
    normalizer.update(np.stack(observations))
    eval_seeds = list(range(7601, 7609))
    before_metrics, before_rows = evaluate_policy(
        model,
        normalizer,
        asset_path=asset_path,
        env_config=env_config,
        seeds=eval_seeds,
    )

    # E: a PPO optimizer step with LR=0 must not change parameters or evaluation.
    generator = torch.Generator().manual_seed(7701)
    count = 256
    synthetic_observations = torch.randn((count, 13), generator=generator)
    with torch.no_grad():
        _, pre_tanh, old_log_probability, values = model.sample(synthetic_observations)
    rollout = Rollout(
        observations=synthetic_observations,
        pre_tanh_actions=pre_tanh,
        old_log_probabilities=old_log_probability,
        advantages=torch.randn(count, generator=generator),
        returns=values + 0.1 * torch.randn(count, generator=generator),
        values=values,
    )
    training = dict(config["training"])
    training["minibatch_size"] = 64
    optimizer_zero = torch.optim.Adam(model.parameters(), lr=0.0)
    digest_before = _parameter_digest(model)
    ppo_update(model, optimizer_zero, rollout, training, generator=generator)
    digest_after = _parameter_digest(model)
    after_metrics, after_rows = evaluate_policy(
        model,
        normalizer,
        asset_path=asset_path,
        env_config=env_config,
        seeds=eval_seeds,
    )
    check_e = bool(
        digest_before == digest_after
        and json.dumps(before_metrics, sort_keys=True) == json.dumps(after_metrics, sort_keys=True)
        and json.dumps(before_rows, sort_keys=True) == json.dumps(after_rows, sort_keys=True)
    )
    details["E_zero_learning_rate"] = {
        "passed": check_e,
        "parameter_sha256_before": digest_before,
        "parameter_sha256_after": digest_after,
        "evaluation_exact_match": before_rows == after_rows,
    }

    # F: round-trip all policy, critic, optimizer, and observation RMS state.
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        checkpoint_path = Path(temporary) / "roundtrip.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer_zero,
            normalizer,
            global_step=0,
            config=config,
            seed=7401,
        )
        restored = ActorCritic(13, 2, config["network"]["hidden_sizes"])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.0)
        restored_normalizer = RunningMeanStd((13,))
        payload = load_checkpoint(
            checkpoint_path, restored, restored_optimizer, restored_normalizer
        )
        restored_metrics, restored_rows = evaluate_policy(
            restored,
            restored_normalizer,
            asset_path=asset_path,
            env_config=env_config,
            seeds=eval_seeds,
        )
        check_f = bool(
            _parameter_digest(model) == _parameter_digest(restored)
            and json.dumps(after_metrics, sort_keys=True) == json.dumps(restored_metrics, sort_keys=True)
            and json.dumps(after_rows, sort_keys=True) == json.dumps(restored_rows, sort_keys=True)
            and np.array_equal(normalizer.mean, restored_normalizer.mean)
            and np.array_equal(normalizer.var, restored_normalizer.var)
            and normalizer.count == restored_normalizer.count
            and payload["optimizer_state"]["state"]
        )
        details["F_checkpoint_roundtrip"] = {
            "passed": check_f,
            "parameter_exact_match": _parameter_digest(model) == _parameter_digest(restored),
            "evaluation_exact_match": after_rows == restored_rows,
            "normalization_exact_match": bool(
                np.array_equal(normalizer.mean, restored_normalizer.mean)
                and np.array_equal(normalizer.var, restored_normalizer.var)
                and normalizer.count == restored_normalizer.count
            ),
            "optimizer_state_present": bool(payload["optimizer_state"]["state"]),
        }

    passed = all(bool(row["passed"]) for row in details.values())
    result = {
        "schema_version": "planar_push_preflight_v1",
        "status": "PASS" if passed else "FAIL",
        "asset_sha256": _sha256(asset_path),
        "checks": details,
        "training_started": False,
    }
    (output / "PRECHECK_RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
