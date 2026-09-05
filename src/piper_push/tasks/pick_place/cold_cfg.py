"""The -Robust task with the cold-start curriculum in place of the clock-driven one.

Everything else -- observations, actions (bounded convention), terminations,
the heavy-DR event terms and plant hooks -- is the -Robust task; the cold
curriculum sets the DR groups to nominal and the penalties to (near) zero at
construction and opens them by capability (cold_curriculum.schedule).
"""

from __future__ import annotations

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from piper_push.tasks.pick_place import cold_curriculum
from piper_push.tasks.pick_place.robust_cfg import make_robust_env_cfg


def make_cold_env_cfg(*, sight: bool = True, play: bool = False):
  cfg = make_robust_env_cfg(play=play, vision=False, wrist=False)
  if play:
    return cfg  # evaluation runs the full -Robust domain; the curriculum is a training device
  cfg.curriculum = {
    "cold_start": CurriculumTermCfg(func=cold_curriculum.cold_start_curriculum, params={"sight": bool(sight)})
  }
  # The stage-0 weights are also written into the reward cfg so that a config
  # dump (params/env.yaml) shows what training starts from.
  for name, w in cold_curriculum.schedule(sight)["stages"][0]["weights"].items():
    if name in cfg.rewards:
      cfg.rewards[name].weight = float(w)
  return cfg
