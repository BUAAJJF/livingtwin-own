from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from livingtwin_mujoco_rl.checkpoint import save_checkpoint
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.ppo import Rollout, append_csv, ppo_update, set_global_seed
from livingtwin_mujoco_rl.yaw_absorbing_env import AbsorbingYawPlanarPushEnv, VectorAbsorbingYawPlanarPushEnv
from livingtwin_mujoco_rl.yaw_absorbing_evaluate import evaluate_absorbing_policy


def collect_absorbing_rollout(
    vector_env: VectorAbsorbingYawPlanarPushEnv,
    model: ActorCritic,
    normalizer: RunningMeanStd,
    observation: np.ndarray,
    *,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    reset_seed_base: int,
) -> tuple[Rollout, np.ndarray, dict[str, Any]]:
    num_envs, obs_size = vector_env.num_envs, observation.shape[-1]
    observations = np.zeros((rollout_steps, num_envs, obs_size), dtype=np.float32)
    pre_tanh_actions = np.zeros((rollout_steps, num_envs, 2), dtype=np.float32)
    log_probabilities = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    raw_observations: list[np.ndarray] = []
    completed: list[dict[str, Any]] = []
    reward_component_values = {name: [] for name in ("position", "yaw", "success", "action", "failure")}
    saturated = 0
    contact_steps = 0
    absorbing_steps = 0
    for step in range(rollout_steps):
        normalized = normalizer.normalize(observation)
        observations[step] = normalized
        with torch.no_grad():
            action, pre_tanh, log_probability, value = model.sample(torch.as_tensor(normalized, dtype=torch.float32))
        action_numpy = action.cpu().numpy()
        saturated += int(np.sum(np.any(np.abs(action_numpy) >= 0.95, axis=1)))
        next_observation, reward, done, infos = vector_env.step(
            action_numpy, reset_seed_base=reset_seed_base + step * num_envs, gamma=gamma
        )
        pre_tanh_actions[step] = pre_tanh.cpu().numpy()
        log_probabilities[step] = log_probability.cpu().numpy()
        rewards[step] = reward
        dones[step] = done.astype(np.float32)
        values[step] = value.cpu().numpy()
        raw_observations.append(next_observation.copy())
        for info, completed_now in zip(infos, done, strict=True):
            for name in reward_component_values:
                reward_component_values[name].append(float(info["reward_components"].get(name, 0.0)))
            contact_steps += int(info["contact_this_step"])
            absorbing_steps += int(info["absorbing_failure"])
            if completed_now:
                completed.append(info)
        observation = next_observation
    with torch.no_grad():
        next_value = model.value(torch.as_tensor(normalizer.normalize(observation), dtype=torch.float32)).cpu().numpy()
    advantages = np.zeros_like(rewards)
    last_advantage = np.zeros(num_envs, dtype=np.float32)
    for step in reversed(range(rollout_steps)):
        following_value = next_value if step == rollout_steps - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * following_value * nonterminal - values[step]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    flat_advantages = advantages.reshape(-1)
    flat_advantages = (flat_advantages - np.mean(flat_advantages)) / (np.std(flat_advantages) + 1.0e-8)
    flat_values, flat_returns = values.reshape(-1), returns.reshape(-1)
    variance = float(np.var(flat_returns))
    explained = float(1.0 - np.var(flat_returns - flat_values) / variance) if variance > 1.0e-12 else 0.0
    rollout = Rollout(
        observations=torch.as_tensor(observations.reshape(-1, obs_size)),
        pre_tanh_actions=torch.as_tensor(pre_tanh_actions.reshape(-1, 2)),
        old_log_probabilities=torch.as_tensor(log_probabilities.reshape(-1)),
        advantages=torch.as_tensor(flat_advantages.astype(np.float32)),
        returns=torch.as_tensor(flat_returns),
        values=torch.as_tensor(flat_values),
    )
    normalizer.update(np.concatenate(raw_observations, axis=0))
    total = rollout_steps * num_envs
    summary = {
        "rollout_mean_reward": float(np.mean(rewards)),
        "completed_episodes": completed,
        "action_saturation_rate": float(saturated / total),
        "contact_presence_rate": float(contact_steps / total),
        "absorbing_state_fraction": float(absorbing_steps / total),
        "explained_variance": explained,
        **{f"mean_reward_{name}": float(np.mean(series)) for name, series in reward_component_values.items()},
    }
    return rollout, observation, summary


def train_absorbing_yaw(config: Mapping[str, Any], *, seed: int, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    set_global_seed(seed)
    started = time.time()
    env_config = config["environment"]
    num_envs = int(config["num_envs"])
    vector_env = VectorAbsorbingYawPlanarPushEnv(num_envs, config["asset_path"], env_config, seed)
    observation, _ = vector_env.reset(seed * 100_000)
    normalizer = RunningMeanStd((AbsorbingYawPlanarPushEnv.observation_size,))
    normalizer.update(observation)
    model = ActorCritic(15, 2, config["network"]["hidden_sizes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    generator = torch.Generator().manual_seed(seed + 77)
    global_step = update_index = 0
    next_evaluation = int(config["evaluation_interval_steps"])
    next_checkpoint = int(config["checkpoint_interval_steps"])
    total_steps = int(config["total_environment_steps"])
    rollout_steps = int(config["rollout_steps"])
    batch_steps = rollout_steps * num_envs
    if total_steps % batch_steps:
        raise ValueError("total steps must be divisible by rollout_steps*num_envs")
    frozen_config = json.loads(json.dumps(config))
    (output / "FROZEN_CONFIG.json").write_text(json.dumps(frozen_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    while global_step < total_steps:
        model.train()
        rollout, observation, rollout_summary = collect_absorbing_rollout(
            vector_env, model, normalizer, observation,
            rollout_steps=rollout_steps,
            gamma=float(config["training"]["gamma"]),
            gae_lambda=float(config["training"]["gae_lambda"]),
            reset_seed_base=seed * 10_000_000 + global_step,
        )
        update_metrics = ppo_update(model, optimizer, rollout, config["training"], generator=generator)
        global_step += batch_steps
        update_index += 1
        completed = rollout_summary["completed_episodes"]
        row = {
            "update": update_index,
            "global_step": global_step,
            "completed_episode_count": len(completed),
            "completed_joint_success_rate": float(np.mean([x["success"] for x in completed])) if completed else 0.0,
            "completed_failure_rate": float(np.mean([x["failure"] for x in completed])) if completed else 0.0,
            "completed_position_success_rate": float(np.mean([x["position_success"] for x in completed])) if completed else 0.0,
            "completed_yaw_success_rate": float(np.mean([x["yaw_success"] for x in completed])) if completed else 0.0,
            "mean_completed_return": float(np.mean([x["episode_return"] for x in completed])) if completed else 0.0,
            "mean_completed_discounted_return_gamma_0_99": float(np.mean([x["discounted_episode_return_gamma_0_99"] for x in completed])) if completed else 0.0,
            **{key: value for key, value in rollout_summary.items() if key != "completed_episodes"},
            **update_metrics,
        }
        append_csv(output / "training_metrics.csv", row)
        if global_step >= next_evaluation or global_step == total_steps:
            seeds = list(range(int(config["evaluation"]["common_seed_start"]), int(config["evaluation"]["common_seed_start"]) + int(config["evaluation_episodes"])))
            metrics, episodes = evaluate_absorbing_policy(
                model, normalizer, asset_path=config["asset_path"], env_config=env_config,
                seeds=seeds, gamma=float(config["training"]["gamma"]),
            )
            append_csv(output / "common_evaluations.csv", {"global_step": global_step, **metrics})
            for episode in episodes:
                append_csv(output / f"common_episodes_step_{global_step:09d}.csv", episode)
            next_evaluation += int(config["evaluation_interval_steps"])
        if global_step >= next_checkpoint or global_step == total_steps:
            save_checkpoint(
                output / "checkpoints" / f"step_{global_step:09d}.pt", model, optimizer,
                normalizer, global_step=global_step, config=frozen_config, seed=seed,
            )
            next_checkpoint += int(config["checkpoint_interval_steps"])
    seeds = list(range(int(config["evaluation"]["generalization_seed_start"]), int(config["evaluation"]["generalization_seed_start"]) + int(config["evaluation"]["generalization_episodes"])))
    general_metrics, general_rows = evaluate_absorbing_policy(
        model, normalizer, asset_path=config["asset_path"], env_config=env_config,
        seeds=seeds, gamma=float(config["training"]["gamma"]),
    )
    (output / "GENERALIZATION_METRICS.json").write_text(json.dumps(general_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for episode in general_rows:
        append_csv(output / "generalization_episodes.csv", episode)
    final_checkpoint = output / "checkpoints" / f"step_{global_step:09d}.pt"
    summary = {
        "seed": seed, "global_step": global_step, "updates": update_index,
        "elapsed_seconds": float(time.time() - started), "final_checkpoint": str(final_checkpoint),
        "generalization_metrics": general_metrics, "status": "TRAINING_COMPLETE",
    }
    (output / "TRAINING_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
