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

from livingtwin_mujoco_rl.env import PlanarPushEnv
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_training_curves(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    training = _read_csv(output / "training_metrics.csv")
    evaluations = _read_csv(output / "evaluations.csv")
    steps = np.asarray([float(row["global_step"]) for row in training])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(
        [float(row["global_step"]) for row in evaluations],
        [float(row["success_rate"]) for row in evaluations],
        marker="o",
    )
    axes[0, 0].axhline(0.90, color="tab:green", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Deterministic success", ylabel="rate", xlabel="steps", ylim=(-0.02, 1.02))
    axes[0, 1].plot(
        [float(row["global_step"]) for row in evaluations],
        [float(row["mean_final_position_error_m"]) for row in evaluations],
        marker="o",
    )
    axes[0, 1].set(title="Final position error", ylabel="m", xlabel="steps")
    axes[1, 0].plot(steps, [float(row["policy_loss"]) for row in training], label="policy")
    axes[1, 0].plot(steps, [float(row["value_loss"]) for row in training], label="value")
    axes[1, 0].set(title="PPO losses", xlabel="steps")
    axes[1, 0].legend()
    axes[1, 1].plot(steps, [float(row["entropy"]) for row in training], label="entropy")
    axes[1, 1].plot(steps, [float(row["approx_kl"]) for row in training], label="approx KL")
    axes[1, 1].set(title="Distribution diagnostics", xlabel="steps")
    axes[1, 1].legend()
    path = output / "TRAINING_CURVES.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_policy_video(
    model: ActorCritic,
    normalizer: RunningMeanStd,
    *,
    asset_path: str,
    env_config: Mapping[str, Any],
    seed: int,
    path: str | Path,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = PlanarPushEnv(asset_path, env_config, seed=seed)
    observation, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total_return = 0.0
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    final_info: dict[str, Any] = {}
    for step in range(int(env_config["task"]["episode_steps"])):
        if step % 2 == 0:
            renderer.update_scene(env.data, camera="overview")
            frames.append(renderer.render().copy())
        with torch.no_grad():
            action = model.deterministic(
                torch.as_tensor(
                    normalizer.normalize(observation), dtype=torch.float32
                ).unsqueeze(0)
            )[0].cpu().numpy()
        observation, reward, terminated, truncated, final_info = env.step(action)
        total_return += float(reward)
        if terminated or truncated:
            renderer.update_scene(env.data, camera="overview")
            frames.append(renderer.render().copy())
            break
    renderer.close()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(target, frames, fps=25, codec="libx264", quality=8)
    return {
        "seed": int(seed),
        "success": bool(final_info.get("success", False)),
        "final_position_error_m": float(final_info.get("distance_m", float("nan"))),
        "episode_return": float(total_return),
        "frame_count": len(frames),
        "path": str(target.resolve()),
    }

