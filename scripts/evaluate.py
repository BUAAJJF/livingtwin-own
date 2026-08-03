#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from livingtwin_mujoco_rl.checkpoint import load_checkpoint
from livingtwin_mujoco_rl.config import load_training_config
from livingtwin_mujoco_rl.env import PlanarPushEnv
from livingtwin_mujoco_rl.evaluate import evaluate_policy
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=710000)
    args = parser.parse_args()
    config = load_training_config(args.config)
    model = ActorCritic(13, 2, config["network"]["hidden_sizes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    normalizer = RunningMeanStd((PlanarPushEnv.observation_size,))
    load_checkpoint(args.checkpoint, model, optimizer, normalizer)
    metrics, _ = evaluate_policy(
        model,
        normalizer,
        asset_path=config["asset_path"],
        env_config=config["environment"],
        seeds=range(args.seed_start, args.seed_start + args.episodes),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

