#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from livingtwin_mujoco_rl.config import load_training_config
from livingtwin_mujoco_rl.preflight import run_preflight
from livingtwin_mujoco_rl.reporting import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--output", default="results/preflight")
    args = parser.parse_args()
    config = load_training_config(args.config)
    result = run_preflight(config, args.output)
    write_manifest(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

