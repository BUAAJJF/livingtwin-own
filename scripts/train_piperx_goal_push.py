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
from livingtwin_mujoco_rl.piperx_ppo import evaluate_goal_policy, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    probe = PiperGoalPushEnv(config["environment"], args.seed)
    observation, _ = probe.reset(args.seed)
    summary = train(config, seed=args.seed, output_dir=args.output, resume_checkpoint=args.resume_checkpoint)
    model = ActorCritic(observation.shape[-1], 2, config["network"]["hidden_sizes"])
    normalizer = RunningMeanStd((observation.shape[-1],))
    load_checkpoint(summary["final_checkpoint"], model, None, normalizer)
    metrics, rows = evaluate_goal_policy(model, normalizer, config["environment"], range(config["evaluation"]["seed_start"], config["evaluation"]["seed_start"] + config["evaluation_episodes"]))
    output = Path(args.output)
    (output / "NOMINAL_EVALUATION.json").write_text(json.dumps({"metrics": metrics, "episodes": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**summary, "nominal_evaluation": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
