"""Measured active-stereo model for the installed RealSense D455.

The numbers are generated, not copied by hand.  Re-run
``hardware/depth_bench/model/fit_d455_noise.py`` after changing the camera,
emitter preset or acquisition resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

from piper_push.depth_noise import DepthNoiseCfg


MODEL_PATH = (
  Path(__file__).resolve().parents[2]
  / "hardware/depth_bench/model/d455_noise.json"
)


def load_cfg(path: Path | str = MODEL_PATH, *, strength: float = 1.0) -> DepthNoiseCfg:
  report = json.loads(Path(path).read_text())
  params = dict(report["params"])
  for name in ("texture_penalty", "surface_fill"):
    params[name] = tuple(float(x) for x in params[name])
  params["strength"] = float(strength)
  return DepthNoiseCfg(**params)


DEPTH_NOISE = load_cfg()

