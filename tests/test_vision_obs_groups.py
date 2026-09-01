"""Every observation group a runner asks for must exist in its environment.

This is a regression test for a specific and cheap-to-repeat mistake.  The
vision branch of ``make_pick_place_env_cfg`` ends with a block of surgery --
copying ``proprio`` into ``full_proprio`` before the deployment constraint is
applied to it, then removing the grasp flag -- and a new ``if`` inserted into
the middle of that branch instead of after it silently swallows the rest.

Nothing fails at import.  The config builds, the task registers, and the run
dies minutes later inside rsl_rl with "Observation 'full_proprio' in
observation set 'teacher' not found", which is a long way from the edit that
caused it and is only reached on a GPU.  Checking the contract here costs
milliseconds and catches it at the point of the change.
"""

from __future__ import annotations

import pytest

from piper_push.tasks.pick_place.rl_cfg import (
  pick_place_distill_runner_cfg,
  pick_place_vision_ppo_runner_cfg,
)
from piper_push.tasks.pick_place.robust_cfg import make_robust_env_cfg

# (env kwargs, runner obs_groups) for every combination that gets trained.
CASES = [
  ("distill", dict(vision=True, mask_dropout_scale=0.0),
   pick_place_distill_runner_cfg().obs_groups),
  ("distill wrist", dict(vision=True, wrist=True, mask_dropout_scale=0.0),
   pick_place_distill_runner_cfg(wrist=True).obs_groups),
  ("ppo half", dict(vision=True, mask_dropout_scale=0.5),
   pick_place_vision_ppo_runner_cfg().obs_groups),
  ("ppo threeq", dict(vision=True, mask_dropout_scale=0.75),
   pick_place_vision_ppo_runner_cfg().obs_groups),
  ("ppo wrist threeq", dict(vision=True, wrist=True, mask_dropout_scale=0.75),
   pick_place_vision_ppo_runner_cfg(wrist=True).obs_groups),
  ("ppo full", dict(vision=True, mask_dropout_scale=1.0),
   pick_place_vision_ppo_runner_cfg().obs_groups),
  ("ppo wrist half", dict(vision=True, wrist=True, mask_dropout_scale=0.5),
   pick_place_vision_ppo_runner_cfg(wrist=True).obs_groups),
  ("ppo wrist full", dict(vision=True, wrist=True, mask_dropout_scale=1.0),
   pick_place_vision_ppo_runner_cfg(wrist=True).obs_groups),
]


@pytest.mark.parametrize("name,env_kwargs,obs_groups",
                         CASES, ids=[c[0] for c in CASES])
def test_every_requested_observation_group_exists(name, env_kwargs, obs_groups):
  available = set(make_robust_env_cfg(**env_kwargs).observations.keys())
  for role, groups in obs_groups.items():
    missing = [g for g in groups if g not in available]
    assert not missing, (
      f"{name}: {role} asks for {missing}, environment has {sorted(available)}")


def test_the_wrist_is_a_second_group_and_not_extra_channels():
  """The two cameras must stay separable.

  A wrist view folded into the scene camera's channels would share one
  convolutional encoder, and the deployment would have to produce both images
  or neither.  As separate groups the model builds one encoder each, and the
  scene-only policy remains loadable without the wrist ever existing.
  """
  plain = make_robust_env_cfg(vision=True)
  wrist = make_robust_env_cfg(vision=True, wrist=True)
  assert "wrist" not in plain.observations
  assert "wrist" in wrist.observations
  assert set(plain.observations) < set(wrist.observations)
  # and the scene group is untouched by the wrist's presence
  assert (plain.observations["camera"].terms["scene"].params["sensor_name"]
          == wrist.observations["camera"].terms["scene"].params["sensor_name"])


def test_the_two_cameras_do_not_share_a_sensor_model():
  """A D405 at 120 mm is not a D455 at 1.2 m.

  The D455 config carries a 95 mm stereo baseline and a 1/32-pixel disparity
  grid; the fitted D405 model has neither.  Sharing one config would model the
  hand camera as blurrier than it is, and the policy would learn to distrust
  the better sensor.
  """
  cfg = make_robust_env_cfg(vision=True, wrist=True)
  scene = cfg.observations["camera"].terms["scene"].params["noise_cfg"]
  hand = cfg.observations["wrist"].terms["scene"].params["noise_cfg"]
  assert scene.stereo_baseline_m > 0.0 and scene.disparity_subpixel_levels > 0
  assert hand.stereo_baseline_m == 0.0 and hand.disparity_subpixel_levels == 0
  # and the hand camera does not inherit the scene camera's far clip
  assert (cfg.observations["wrist"].terms["scene"].params["cutoff_distance"]
          < cfg.observations["camera"].terms["scene"].params["cutoff_distance"])
