from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {source}")
    result = deepcopy(value)
    result["_source_path"] = str(source)
    return result


def load_training_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    source = Path(config["_source_path"])
    project_root = source.parent.parent
    env_path = (project_root / config["env_config"]).resolve()
    config["environment"] = load_yaml(env_path)
    config["project_root"] = str(project_root)
    config["asset_path"] = str(
        (project_root / config["environment"]["asset_path"]).resolve()
    )
    return config

