#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from livingtwin_mujoco_rl.config import load_training_config, load_yaml
from livingtwin_mujoco_rl.reporting import write_manifest
from livingtwin_mujoco_rl.yaw_absorbing_preflight import run_absorbing_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-config", default="configs/yaw_absorbing_smoke.yaml")
    parser.add_argument("--full-env-config", default="configs/yaw_absorbing_env_full.yaml")
    parser.add_argument("--old-config", default="configs/yaw_baseline.yaml")
    parser.add_argument("--output", default="results/yaw_absorbing_preflight")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    smoke = load_training_config(args.smoke_config)
    old = load_training_config(args.old_config)
    full_env = load_yaml(args.full_env_config)
    full = {**smoke, "environment": full_env, "asset_path": str((root / full_env["asset_path"]).resolve())}
    result = run_absorbing_preflight(root, smoke, full, old, Path(args.output).resolve())
    write_manifest(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
