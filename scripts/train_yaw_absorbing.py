#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from livingtwin_mujoco_rl.checkpoint import load_checkpoint
from livingtwin_mujoco_rl.config import load_training_config
from livingtwin_mujoco_rl.networks import ActorCritic
from livingtwin_mujoco_rl.normalization import RunningMeanStd
from livingtwin_mujoco_rl.reporting import last_csv_row, write_manifest
from livingtwin_mujoco_rl.yaw_absorbing_artifacts import plot_absorbing_training, render_absorbing_video
from livingtwin_mujoco_rl.yaw_absorbing_env import AbsorbingYawPlanarPushEnv
from livingtwin_mujoco_rl.yaw_absorbing_evaluate import evaluate_absorbing_policy
from livingtwin_mujoco_rl.yaw_absorbing_ppo import train_absorbing_yaw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", default="results/yaw_absorbing_preflight/ABSORBING_PRECHECK_RESULTS.json")
    args = parser.parse_args()
    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise SystemExit("absorbing preflight is not PASS")
    config = load_training_config(args.config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config["project_root"], text=True).strip()
    summary = train_absorbing_yaw(config, seed=args.seed, output_dir=args.output)
    output = Path(args.output).resolve()
    curve = plot_absorbing_training(output)
    model = ActorCritic(15, 2, config["network"]["hidden_sizes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    normalizer = RunningMeanStd((AbsorbingYawPlanarPushEnv.observation_size,))
    load_checkpoint(summary["final_checkpoint"], model, optimizer, normalizer)
    normalization_report = {
        "count": float(normalizer.count),
        "mean": normalizer.mean.tolist(),
        "variance": normalizer.var.tolist(),
        "standard_deviation": np.sqrt(normalizer.var).tolist(),
        "all_finite": bool(np.all(np.isfinite(normalizer.mean)) and np.all(np.isfinite(normalizer.var))),
        "minimum_variance": float(np.min(normalizer.var)),
        "maximum_variance": float(np.max(normalizer.var)),
        "maximum_absolute_mean": float(np.max(np.abs(normalizer.mean))),
        "absorbing_raw_observation_included_during_training": True,
    }
    (output / "OBSERVATION_NORMALIZATION.json").write_text(
        json.dumps(normalization_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    seeds = list(range(int(config["evaluation"]["common_seed_start"]), int(config["evaluation"]["common_seed_start"]) + int(config["evaluation_episodes"])))
    reload_metrics, _ = evaluate_absorbing_policy(
        model, normalizer, asset_path=config["asset_path"], env_config=config["environment"],
        seeds=seeds, gamma=float(config["training"]["gamma"]),
    )
    training_metrics = last_csv_row(output / "common_evaluations.csv")
    difference = max(abs(float(training_metrics[key]) - float(value)) for key, value in reload_metrics.items())
    reload_report = {"passed": difference <= 1.0e-12, "max_metric_absolute_difference": difference, "metrics": reload_metrics}
    (output / "CHECKPOINT_RELOAD_EVALUATION.json").write_text(json.dumps(reload_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replay = render_absorbing_video(
        model, normalizer, asset_path=config["asset_path"], env_config=config["environment"],
        seed=int(config["evaluation"]["common_seed_start"]), path=output / "ABSORBING_POLICY_REPLAY.mp4",
    )
    numeric_finite = all(math.isfinite(float(value)) for value in reload_metrics.values())
    report = {
        **summary, "code_commit": commit, "final_common_evaluation": training_metrics,
        "checkpoint_reload": reload_report, "numeric_metrics_finite": numeric_finite,
        "observation_normalization": normalization_report,
        "training_curve": str(curve.resolve()), "replay": replay,
    }
    (output / "FINAL_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
