from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.ppo import append_csv
from livingtwin_mujoco_rl.yaw_absorbing_env import AbsorbingYawPlanarPushEnv


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_absorbing_training(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    train, evaluation = _rows(output / "training_metrics.csv"), _rows(output / "common_evaluations.csv")
    ts = np.asarray([float(row["global_step"]) for row in train])
    es = np.asarray([float(row["global_step"]) for row in evaluation])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for key, label in (("joint_success_rate", "joint"), ("position_success_rate", "position"), ("yaw_success_rate", "yaw"), ("failure_rate", "failure")):
        axes[0, 0].plot(es, [float(row[key]) for row in evaluation], marker="o", label=label)
    axes[0, 0].set(title="Deterministic outcomes", ylim=(-0.02, 1.02)); axes[0, 0].legend()
    axes[0, 1].plot(es, [100 * float(row["mean_final_position_error_m"]) for row in evaluation], marker="o")
    axes[0, 1].set(title="Mean position error", ylabel="cm")
    axes[0, 2].plot(es, [np.degrees(float(row["mean_final_yaw_error_rad"])) for row in evaluation], marker="o")
    axes[0, 2].set(title="Mean yaw error", ylabel="degrees")
    axes[1, 0].plot(ts, [float(row["policy_loss"]) for row in train], label="policy")
    axes[1, 0].plot(ts, [float(row["value_loss"]) for row in train], label="value")
    axes[1, 0].set(title="PPO losses"); axes[1, 0].legend()
    axes[1, 1].plot(ts, [float(row["approx_kl"]) for row in train], label="KL")
    axes[1, 1].plot(ts, [float(row["clip_fraction"]) for row in train], label="clip")
    axes[1, 1].plot(ts, [float(row["absorbing_state_fraction"]) for row in train], label="absorbing")
    axes[1, 1].set(title="Optimization/failure state"); axes[1, 1].legend()
    for key, label in (("mean_reward_position", "position"), ("mean_reward_yaw", "yaw"), ("mean_reward_action", "action"), ("mean_reward_failure", "failure")):
        axes[1, 2].plot(ts, [float(row[key]) for row in train], label=label)
    axes[1, 2].set(title="Reward components"); axes[1, 2].legend()
    for axis in axes.flat:
        axis.set_xlabel("environment steps"); axis.grid(alpha=0.25)
    path = output / "ABSORBING_SMOKE_CURVES.png"
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def render_absorbing_video(
    model: ActorCritic,
    normalizer: RunningMeanStd,
    *, asset_path: str, env_config: Mapping[str, Any], seed: int, path: str | Path,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = AbsorbingYawPlanarPushEnv(asset_path, env_config, seed=seed)
    observation, _ = env.reset(seed=seed)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    frames: list[np.ndarray] = []
    trajectory: list[dict[str, Any]] = []
    episode_return = discounted = 0.0
    final_info: dict[str, Any] = {}
    for step in range(int(env_config["task"]["episode_steps"])):
        if step % 2 == 0:
            renderer.update_scene(env.data, camera="overview"); frames.append(renderer.render().copy())
        with torch.no_grad():
            action = model.deterministic(torch.as_tensor(normalizer.normalize(observation), dtype=torch.float32).unsqueeze(0))[0].cpu().numpy()
        observation, reward, terminated, truncated, final_info = env.step(action)
        episode_return += float(reward); discounted += (0.99**step) * float(reward)
        trajectory.append(
            {
                "step": step + 1, "cube_x_m": float(env.cube_xy()[0]), "cube_y_m": float(env.cube_xy()[1]),
                "cube_yaw_rad": float(env.cube_yaw()), "target_x_m": float(env.target[0]), "target_y_m": float(env.target[1]),
                "target_yaw_rad": float(env.target[2]), "position_error_m": float(final_info["position_error_m"]),
                "yaw_error_rad": float(final_info["yaw_error_rad"]), "action_x": float(action[0]), "action_y": float(action[1]),
                "absorbing_failure": bool(final_info["absorbing_failure"]), "action_ignored": bool(final_info["action_ignored"]),
            }
        )
        if terminated or truncated:
            renderer.update_scene(env.data, camera="overview"); frames.append(renderer.render().copy()); break
    renderer.close()
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(target, frames, fps=25, codec="libx264", quality=8)
    trajectory_path = target.with_name(target.stem + "_trajectory.csv")
    for row in trajectory: append_csv(trajectory_path, row)
    return {
        "seed": seed, "success": bool(final_info.get("success", False)), "failure": bool(final_info.get("failure", False)),
        "absorption_step": final_info.get("absorption_step"), "episode_return": episode_return,
        "discounted_return_gamma_0_99": discounted, "frame_count": len(frames),
        "path": str(target.resolve()), "trajectory_path": str(trajectory_path.resolve()),
    }
