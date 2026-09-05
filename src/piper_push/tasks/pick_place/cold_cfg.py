"""The -Robust task with the cold-start curriculum in place of the clock-driven one.

Everything else -- observations, actions (bounded convention), terminations,
the heavy-DR event terms and plant hooks -- is the -Robust task; the cold
curriculum sets the DR groups to nominal and the penalties to (near) zero at
construction and opens them by capability (cold_curriculum.schedule).
"""

from __future__ import annotations

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from piper_push.tasks.pick_place import cold_curriculum
from piper_push.tasks.pick_place import mdp as pick_mdp
from piper_push.tasks.pick_place.env_cfg import TASK
from piper_push.tasks.pick_place.robust_cfg import make_robust_env_cfg


def _grasp_site() -> SceneEntityCfg:
  return SceneEntityCfg("robot", site_names=("grasp_site",))


def add_approach_terms(cfg) -> None:
  """v10d: price arriving fast, disturbing the object, and a horizontal wrist."""
  cfg.rewards["approach_speed"] = RewardTermCfg(
    func=pick_mdp.approach_speed, weight=0.0,
    params={"command_name": TASK, "asset_cfg": _grasp_site(), "near_m": 0.15, "stop_m": 0.03,
            "v_near_m_s": 0.10, "v_far_m_s": 0.60})
  cfg.rewards["object_disturbed"] = RewardTermCfg(
    func=pick_mdp.object_disturbed, weight=0.0, params={"command_name": TASK, "v_floor_m_s": 0.02})
  cfg.rewards["top_down_grasp"] = RewardTermCfg(
    func=pick_mdp.top_down_grasp, weight=0.0,
    params={"command_name": TASK, "asset_cfg": _grasp_site(), "near_m": 0.20})


def make_cold_env_cfg(*, sight: bool = True, play: bool = False, approach: bool = False):
  cfg = make_robust_env_cfg(play=play, vision=False, wrist=False)
  if approach:
    add_approach_terms(cfg)   # in play too, so the evaluation logs the same terms
  if play:
    return cfg  # evaluation runs the full -Robust domain; the curriculum is a training device
  cfg.curriculum = {
    "cold_start": CurriculumTermCfg(func=cold_curriculum.cold_start_curriculum,
                                    params={"sight": bool(sight), "approach": bool(approach)})
  }
  # The stage-0 weights are also written into the reward cfg so that a config
  # dump (params/env.yaml) shows what training starts from.
  for name, w in cold_curriculum.schedule(sight, approach=approach)["stages"][0]["weights"].items():
    if name in cfg.rewards:
      cfg.rewards[name].weight = float(w)
  return cfg
