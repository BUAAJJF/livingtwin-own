from __future__ import annotations

import argparse
import json
from pathlib import Path

from livingtwin_mujoco_rl.config import load_training_config
from livingtwin_mujoco_rl.yaw_evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically evaluate a frozen yaw PPO checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=620000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        env_config_path=cfg.env_config_path,
        episodes=args.episodes,
        seed_start=args.seed_start,
        device=cfg.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
