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
from livingtwin_mujoco_rl.yaw_env import YawPlanarPushEnv


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_yaw_training_curves(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    training = _read_csv(output / "training_metrics.csv")
    evaluations = _read_csv(output / "common_evaluations.csv")
    train_steps = np.asarray([float(row["global_step"]) for row in training])
    eval_steps = np.asarray([float(row["global_step"]) for row in evaluations])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].plot(eval_steps, [float(row["joint_success_rate"]) for row in evaluations], marker="o", label="joint")
    axes[0, 0].plot(eval_steps, [float(row["position_success_rate"]) for row in evaluations], marker="o", label="position")
    axes[0, 0].plot(eval_steps, [float(row["yaw_success_rate"]) for row in evaluations], marker="o", label="yaw")
    axes[0, 0].axhline(0.90, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Deterministic success", ylim=(-0.02, 1.02))
    axes[0, 0].legend()
    axes[0, 1].plot(eval_steps, [100.0 * float(row["mean_final_position_error_m"]) for row in evaluations], marker="o")
    axes[0, 1].set(title="Mean position error", ylabel="cm")
    axes[0, 2].plot(eval_steps, [np.degrees(float(row["mean_final_yaw_error_rad"])) for row in evaluations], marker="o")
    axes[0, 2].set(title="Mean yaw error", ylabel="degrees")
    axes[1, 0].plot(train_steps, [float(row["policy_loss"]) for row in training], label="policy")
    axes[1, 0].plot(train_steps, [float(row["value_loss"]) for row in training], label="value")
    axes[1, 0].set(title="PPO losses")
    axes[1, 0].legend()
    axes[1, 1].plot(train_steps, [float(row["entropy"]) for row in training], label="entropy")
    axes[1, 1].plot(train_steps, [float(row["approx_kl"]) for row in training], label="approx KL")
    axes[1, 1].plot(train_steps, [float(row["clip_fraction"]) for row in training], label="clip fraction")
    axes[1, 1].set(title="Optimization diagnostics")
    axes[1, 1].legend()
    axes[1, 2].plot(train_steps, [float(row["mean_reward_position"]) for row in training], label="position")
    axes[1, 2].plot(train_steps, [float(row["mean_reward_yaw"]) for row in training], label="yaw")
    axes[1, 2].plot(train_steps, [float(row["mean_reward_action"]) for row in training], label="action")
    axes[1, 2].set(title="Mean reward components")
    axes[1, 2].legend()
    for axis in axes.flat:
        axis.set_xlabel("environment steps")
        axis.grid(alpha=0.25)
    path = output / "YAW_TRAINING_CURVES.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_yaw_policy_video(
    model: ActorCritic,
    normalizer: RunningMeanStd,
    *,
    asset_path: str,
    env_config: Mapping[str, Any],
    seed: int,
    path: str | Path,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = YawPlanarPushEnv(asset_path, env_config, seed=seed)
    observation, _ = env.reset(seed=seed)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    frames: list[np.ndarray] = []
    trajectory: list[dict[str, Any]] = []
    episode_return = 0.0
    final_info: dict[str, Any] = {}
    for step in range(int(env_config["task"]["episode_steps"])):
        if step % 2 == 0:
            renderer.update_scene(env.data, camera="overview")
            frames.append(renderer.render().copy())
        with torch.no_grad():
            action = model.deterministic(
                torch.as_tensor(normalizer.normalize(observation), dtype=torch.float32).unsqueeze(0)
            )[0].cpu().numpy()
        observation, reward, terminated, truncated, final_info = env.step(action)
        episode_return += float(reward)
        trajectory.append(
            {
                "step": step + 1,
                "pusher_x_m": float(env.data.mocap_pos[env.pusher_mocap_id, 0]),
                "pusher_y_m": float(env.data.mocap_pos[env.pusher_mocap_id, 1]),
                "cube_x_m": float(env.cube_xy()[0]),
                "cube_y_m": float(env.cube_xy()[1]),
                "cube_yaw_rad": float(env.cube_yaw()),
                "target_x_m": float(env.target[0]),
                "target_y_m": float(env.target[1]),
                "target_yaw_rad": float(env.target[2]),
                "position_error_m": float(final_info["position_error_m"]),
                "yaw_error_rad": float(final_info["yaw_error_rad"]),
                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "contact_control_steps": int(final_info["contact_control_steps"]),
            }
        )
        if terminated or truncated:
            renderer.update_scene(env.data, camera="overview")
            frames.append(renderer.render().copy())
            break
    renderer.close()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(target, frames, fps=25, codec="libx264", quality=8)
    trajectory_path = target.with_name(target.stem + "_trajectory.csv")
    for row in trajectory:
        append_csv(trajectory_path, row)
    return {
        "seed": seed,
        "success": bool(final_info.get("success", False)),
        "final_position_error_m": float(final_info.get("position_error_m", float("nan"))),
        "final_yaw_error_rad": float(final_info.get("yaw_error_rad", float("nan"))),
        "episode_return": episode_return,
        "frame_count": len(frames),
        "path": str(target.resolve()),
        "trajectory_path": str(trajectory_path.resolve()),
    }
