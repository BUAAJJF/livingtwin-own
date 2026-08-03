#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from livingtwin_mujoco_rl.config import load_training_config
from livingtwin_mujoco_rl.reporting import write_manifest
from livingtwin_mujoco_rl.yaw_preflight import run_yaw_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-config", default="configs/yaw_smoke.yaml")
    parser.add_argument("--baseline-config", default="configs/yaw_baseline.yaml")
    parser.add_argument("--output", default="results/yaw_preflight")
    args = parser.parse_args()
    result = run_yaw_preflight(
        load_training_config(args.smoke_config),
        load_training_config(args.baseline_config),
        args.output,
    )
    write_manifest(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
