#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from livingtwin_mujoco_rl.checkpoint import load_checkpoint
from livingtwin_mujoco_rl.config import load_training_config
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.reporting import last_csv_row, write_manifest
from livingtwin_mujoco_rl.yaw_artifacts import plot_yaw_training_curves, render_yaw_policy_video
from livingtwin_mujoco_rl.yaw_env import YawPlanarPushEnv
from livingtwin_mujoco_rl.yaw_evaluate import evaluate_yaw_policy
from livingtwin_mujoco_rl.yaw_ppo import train_yaw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", default="results/yaw_preflight/YAW_PRECHECK_RESULTS.json")
    args = parser.parse_args()
    preflight_path = Path(args.preflight)
    if not preflight_path.is_file():
        raise SystemExit(f"missing yaw preflight: {preflight_path}")
    if json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise SystemExit("yaw preflight is not PASS")
    config = load_training_config(args.config)
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=config["project_root"], text=True
    ).strip()
    summary = train_yaw(config, seed=args.seed, output_dir=args.output)
    output = Path(args.output).resolve()
    curve = plot_yaw_training_curves(output)
    model = ActorCritic(15, 2, config["network"]["hidden_sizes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    normalizer = RunningMeanStd((YawPlanarPushEnv.observation_size,))
    load_checkpoint(summary["final_checkpoint"], model, optimizer, normalizer)
    common_seeds = list(
        range(
            int(config["evaluation"]["common_seed_start"]),
            int(config["evaluation"]["common_seed_start"]) + int(config["evaluation_episodes"]),
        )
    )
    reload_metrics, _ = evaluate_yaw_policy(
        model,
        normalizer,
        asset_path=config["asset_path"],
        env_config=config["environment"],
        seeds=common_seeds,
    )
    training_metrics = last_csv_row(output / "common_evaluations.csv")
    numeric_keys = tuple(reload_metrics)
    reload_max_absolute_difference = max(
        abs(float(training_metrics[key]) - float(reload_metrics[key])) for key in numeric_keys
    )
    reload_report = {
        "code_commit": code_commit,
        "checkpoint": summary["final_checkpoint"],
        "max_metric_absolute_difference": reload_max_absolute_difference,
        "passed": reload_max_absolute_difference <= 1.0e-12,
        "metrics": reload_metrics,
    }
    (output / "CHECKPOINT_RELOAD_EVALUATION.json").write_text(
        json.dumps(reload_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    replay = render_yaw_policy_video(
        model, normalizer, asset_path=config["asset_path"], env_config=config["environment"],
        seed=int(config["evaluation"]["common_seed_start"]), path=output / "YAW_POLICY_REPLAY.mp4",
    )
    report = {
        **summary,
        "code_commit": code_commit,
        "final_common_evaluation": training_metrics,
        "checkpoint_reload": reload_report,
        "generalization_evaluation": summary["generalization_metrics"],
        "training_curve": str(curve.resolve()),
        "replay": replay,
    }
    (output / "FINAL_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
