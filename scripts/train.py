#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from livingtwin_mujoco_rl.artifacts import plot_training_curves, render_policy_video
from livingtwin_mujoco_rl.checkpoint import load_checkpoint
from livingtwin_mujoco_rl.config import load_training_config
from livingtwin_mujoco_rl.env import PlanarPushEnv
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.ppo import train
from livingtwin_mujoco_rl.reporting import last_csv_row, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", default="results/preflight/PRECHECK_RESULTS.json")
    args = parser.parse_args()
    preflight_path = Path(args.preflight)
    if not preflight_path.is_file():
        raise SystemExit(f"missing preflight result: {preflight_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise SystemExit("preflight is not PASS")

    config = load_training_config(args.config)
    summary = train(config, seed=args.seed, output_dir=args.output)
    output = Path(args.output).resolve()
    plot_path = plot_training_curves(output)
    model = ActorCritic(13, 2, config["network"]["hidden_sizes"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    normalizer = RunningMeanStd((PlanarPushEnv.observation_size,))
    load_checkpoint(summary["final_checkpoint"], model, optimizer, normalizer)
    video = render_policy_video(
        model,
        normalizer,
        asset_path=config["asset_path"],
        env_config=config["environment"],
        seed=int(config["evaluation"]["seed_start"]),
        path=output / "POLICY_REPLAY.mp4",
    )
    final_evaluation = last_csv_row(output / "evaluations.csv")
    final = {
        **summary,
        "final_evaluation": final_evaluation,
        "training_curve": str(plot_path.resolve()),
        "video": video,
    }
    (output / "FINAL_REPORT.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(output)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

