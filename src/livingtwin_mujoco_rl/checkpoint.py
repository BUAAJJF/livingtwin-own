from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def save_checkpoint(
    path: str | Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizer: RunningMeanStd,
    *,
    global_step: int,
    config: Mapping[str, Any],
    seed: int,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "planar_push_ppo_checkpoint_v1",
        "actor_state": model.actor.state_dict(),
        "critic_state": model.critic.state_dict(),
        "log_std": model.log_std.detach().cpu(),
        "optimizer_state": optimizer.state_dict(),
        "observation_normalization": normalizer.state_dict(),
        "global_step": int(global_step),
        "seed": int(seed),
        "rng_state": capture_rng_state(),
        "config": dict(config),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_checkpoint(
    path: str | Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer | None,
    normalizer: RunningMeanStd,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "planar_push_ppo_checkpoint_v1":
        raise ValueError("unsupported checkpoint schema")
    model.actor.load_state_dict(payload["actor_state"])
    model.critic.load_state_dict(payload["critic_state"])
    with torch.no_grad():
        model.log_std.copy_(payload["log_std"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    normalizer.load_state_dict(payload["observation_normalization"])
    return payload

