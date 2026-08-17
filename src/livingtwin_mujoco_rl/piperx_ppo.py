from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from livingtwin_mujoco_rl.checkpoint import save_checkpoint
from livingtwin_mujoco_rl.piperx_goal_push_env import PiperGoalPushEnv
from livingtwin_mujoco_rl.piperx_vector_env import VectorPiperGoalPushEnv
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd


@dataclass
class Rollout:
    observations: torch.Tensor
    pre_tanh_actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


def evaluate_goal_policy(model: ActorCritic, normalizer: RunningMeanStd, config: Mapping[str, Any], seeds: Sequence[int]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for seed in seeds:
        env = PiperGoalPushEnv(config, int(seed)); observation, _ = env.reset(int(seed)); total = 0.0
        for _ in range(int(config["task"]["episode_steps"])):
            with torch.no_grad():
                action = model.deterministic(torch.as_tensor(normalizer.normalize(observation), dtype=torch.float32)).cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action); total += reward
            if terminated or truncated: break
        rows.append({"seed": int(seed), "success": bool(info["success"]), "final_distance_m": float(info["distance_m"]), "episode_steps": int(info["step_count"]), "oob": bool(info.get("oob", False)), "ik_failure": bool(info.get("ik_failure", False)), "table_collision": bool(info.get("table_collision", False)), "episode_return": float(total)})
    distances = np.asarray([r["final_distance_m"] for r in rows], dtype=float)
    metrics = {"success_rate": float(np.mean([r["success"] for r in rows])), "median_final_distance_m": float(np.median(distances)), "p90_final_distance_m": float(np.quantile(distances, 0.9)), "mean_episode_steps": float(np.mean([r["episode_steps"] for r in rows])), "oob_rate": float(np.mean([r["oob"] for r in rows])), "ik_failure_rate": float(np.mean([r["ik_failure"] for r in rows])), "table_collision_rate": float(np.mean([r["table_collision"] for r in rows]))}
    return metrics, rows


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: Mapping[str, Any],
    *,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    count = rollout.observations.shape[0]
    batch_size = int(config["minibatch_size"])
    clip_ratio = float(config["clip_ratio"])
    metrics: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "total_loss": [],
    }
    stopped_early = False
    for _ in range(int(config["update_epochs"])):
        order = torch.randperm(count, generator=generator)
        for start in range(0, count, batch_size):
            indices = order[start : start + batch_size]
            new_log_probability, entropy, new_value = model.evaluate_pre_tanh(
                rollout.observations[indices], rollout.pre_tanh_actions[indices]
            )
            log_ratio = (
                new_log_probability - rollout.old_log_probabilities[indices]
            )
            ratio = torch.exp(log_ratio)
            advantage = rollout.advantages[indices]
            unclipped = ratio * advantage
            clipped = torch.clamp(
                ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
            ) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * torch.mean(
                (new_value - rollout.returns[indices]) ** 2
            )
            entropy_mean = entropy.mean()
            total_loss = (
                policy_loss
                + float(config["value_coefficient"]) * value_loss
                - float(config["entropy_coefficient"]) * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["maximum_gradient_norm"])
            )
            optimizer.step()
            with torch.no_grad():
                approx_kl = torch.mean((ratio - 1.0) - log_ratio)
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > clip_ratio).float()
                )
            values = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_mean,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "total_loss": total_loss,
            }
            for name, value in values.items():
                metrics[name].append(float(value.detach().cpu()))
        if metrics["approx_kl"] and np.mean(metrics["approx_kl"][-max(1, count // batch_size) :]) > float(config["target_kl"]):
            stopped_early = True
            break
    result = {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in metrics.items()
    }
    result["early_stop_kl"] = float(stopped_early)
    return result


def collect_rollout(
    vector_env: VectorPiperGoalPushEnv,
    model: ActorCritic,
    normalizer: RunningMeanStd,
    observation: np.ndarray,
    *,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    reset_seed_base: int,
) -> tuple[Rollout, np.ndarray, dict[str, Any]]:
    num_envs = vector_env.num_envs
    observations = np.zeros(
        (rollout_steps, num_envs, observation.shape[-1]), dtype=np.float32
    )
    pre_tanh_actions = np.zeros(
        (rollout_steps, num_envs, 2), dtype=np.float32
    )
    log_probabilities = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    raw_observations_for_update: list[np.ndarray] = []
    completed: list[dict[str, Any]] = []
    action_absolute_max = 0.0

    for step in range(rollout_steps):
        normalized = normalizer.normalize(observation)
        observations[step] = normalized
        with torch.no_grad():
            action, pre_tanh, log_probability, value = model.sample(
                torch.as_tensor(normalized, dtype=torch.float32)
            )
        action_numpy = action.cpu().numpy()
        next_observation, reward, done, infos = vector_env.step(
            action_numpy,
            reset_seed_base=reset_seed_base + step * num_envs,
        )
        pre_tanh_actions[step] = pre_tanh.cpu().numpy()
        log_probabilities[step] = log_probability.cpu().numpy()
        rewards[step] = reward
        dones[step] = done.astype(np.float32)
        values[step] = value.cpu().numpy()
        raw_observations_for_update.append(next_observation.copy())
        action_absolute_max = max(
            action_absolute_max, float(np.max(np.abs(action_numpy)))
        )
        for info, completed_now in zip(infos, done, strict=True):
            if completed_now:
                completed.append(info)
        observation = next_observation

    with torch.no_grad():
        next_value = model.value(
            torch.as_tensor(normalizer.normalize(observation), dtype=torch.float32)
        ).cpu().numpy()
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
    flat_advantages = (flat_advantages - np.mean(flat_advantages)) / (
        np.std(flat_advantages) + 1.0e-8
    )
    rollout = Rollout(
        observations=torch.as_tensor(observations.reshape(-1, observations.shape[-1])),
        pre_tanh_actions=torch.as_tensor(pre_tanh_actions.reshape(-1, PiperGoalPushEnv.action_size)),
        old_log_probabilities=torch.as_tensor(log_probabilities.reshape(-1)),
        advantages=torch.as_tensor(flat_advantages.astype(np.float32)),
        returns=torch.as_tensor(returns.reshape(-1)),
        values=torch.as_tensor(values.reshape(-1)),
    )
    normalizer.update(np.concatenate(raw_observations_for_update, axis=0))
    summary = {
        "rollout_mean_reward": float(np.mean(rewards)),
        "completed_episodes": completed,
        "action_absolute_max": action_absolute_max,
    }
    return rollout, observation, summary


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(
    config: Mapping[str, Any],
    *,
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    set_global_seed(seed)
    started = time.time()
    env_config = config["environment"]
    num_envs = int(config["num_envs"])
    vector_env = VectorPiperGoalPushEnv(
        num_envs,
        env_config,
        seed=seed,
    )
    observation, _ = vector_env.reset(seed=seed * 100_000)
    observation_size = int(observation.shape[-1])
    normalizer = RunningMeanStd((observation_size,))
    normalizer.update(observation)
    model = ActorCritic(
        observation_size,
        PiperGoalPushEnv.action_size,
        config["network"]["hidden_sizes"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    generator = torch.Generator().manual_seed(seed + 77)
    global_step = 0
    next_evaluation = int(config["evaluation_interval_steps"])
    next_checkpoint = int(config["checkpoint_interval_steps"])
    update_index = 0
    total_steps = int(config["total_environment_steps"])
    rollout_steps = int(config["rollout_steps"])
    batch_environment_steps = rollout_steps * num_envs
    if total_steps % batch_environment_steps != 0:
        raise ValueError("total_environment_steps must be divisible by rollout_steps*num_envs")

    frozen_config = json.loads(json.dumps(config))
    (output / "FROZEN_CONFIG.json").write_text(
        json.dumps(frozen_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    while global_step < total_steps:
        model.train()
        rollout, observation, rollout_summary = collect_rollout(
            vector_env,
            model,
            normalizer,
            observation,
            rollout_steps=rollout_steps,
            gamma=float(config["training"]["gamma"]),
            gae_lambda=float(config["training"]["gae_lambda"]),
            reset_seed_base=seed * 10_000_000 + global_step,
        )
        update_metrics = ppo_update(
            model,
            optimizer,
            rollout,
            config["training"],
            generator=generator,
        )
        global_step += batch_environment_steps
        update_index += 1
        completed = rollout_summary["completed_episodes"]
        row = {
            "update": update_index,
            "global_step": global_step,
            "rollout_mean_reward": rollout_summary["rollout_mean_reward"],
            "completed_episode_count": len(completed),
            "completed_success_rate": float(np.mean([x["success"] for x in completed])) if completed else float("nan"),
            "mean_completed_return": float(np.mean([x["episode_return"] for x in completed])) if completed else float("nan"),
            "action_absolute_max": rollout_summary["action_absolute_max"],
            **update_metrics,
        }
        append_csv(output / "training_metrics.csv", row)

        if global_step >= next_evaluation or global_step == total_steps:
            count = int(config["evaluation_episodes"])
            seed_start = int(config["evaluation"]["seed_start"])
            eval_seeds = [seed_start + index for index in range(count)]
            metrics, episode_rows = evaluate_goal_policy(model, normalizer, env_config, eval_seeds)
            append_csv(
                output / "evaluations.csv",
                {"global_step": global_step, **metrics},
            )
            detail_path = output / f"evaluation_episodes_step_{global_step:09d}.csv"
            for episode_row in episode_rows:
                append_csv(detail_path, episode_row)
            next_evaluation += int(config["evaluation_interval_steps"])

        if global_step >= next_checkpoint or global_step == total_steps:
            save_checkpoint(
                output / "checkpoints" / f"step_{global_step:09d}.pt",
                model,
                optimizer,
                normalizer,
                global_step=global_step,
                config=frozen_config,
                seed=seed,
            )
            next_checkpoint += int(config["checkpoint_interval_steps"])

    final_checkpoint = output / "checkpoints" / f"step_{global_step:09d}.pt"
    summary = {
        "seed": int(seed),
        "global_step": int(global_step),
        "updates": int(update_index),
        "elapsed_seconds": float(time.time() - started),
        "final_checkpoint": str(final_checkpoint),
        "status": "TRAINING_COMPLETE",
    }
    (output / "TRAINING_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
