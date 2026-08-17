#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from livingtwin_mujoco_rl.checkpoint import load_checkpoint
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.piperx_goal_push_env import PiperGoalPushEnv
from livingtwin_mujoco_rl.piperx_ppo import evaluate_goal_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=920000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    probe = PiperGoalPushEnv(config["environment"], args.seed_start)
    observation, _ = probe.reset(args.seed_start)
    model = ActorCritic(observation.shape[-1], 2, config["network"]["hidden_sizes"])
    normalizer = RunningMeanStd((observation.shape[-1],))
    load_checkpoint(args.checkpoint, model, None, normalizer)
    metrics, rows, decisions = evaluate_goal_policy(model, normalizer, config["environment"], range(args.seed_start, args.seed_start + args.episodes))
    print(json.dumps({"metrics": metrics, "episodes": rows, "decisions": decisions}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
