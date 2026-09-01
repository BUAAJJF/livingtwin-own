from __future__ import annotations

from types import SimpleNamespace

import mujoco
import pytest
import torch
import torch.nn.functional as F

from piper_push.actions import RandomizedPlantHook, RandomizedPlantHookCfg
from piper_push.depth_noise import DepthNoiseCfg
from piper_push.tasks.pick_place.mdp import CameraScene
from piper_push.tasks.pick_place.robust_cfg import (
  _scaled_dropout,
  HEAVY_DR_PROFILE,
  make_robust_env_cfg,
)


class _ActionTerm:
  device = torch.device("cpu")

  def __init__(self, n=2, j=1):
    self._default = torch.zeros(n, j)
    self._previous_target = self._default.clone()


def test_randomized_plant_has_exact_hold_and_delay_semantics():
  term = _ActionTerm()
  hook = RandomizedPlantHook(RandomizedPlantHookCfg(
    latency_weights=(0.0, 0.0, 1.0),
    hold_weights=(0.0, 1.0),
    response_range=(1.0, 1.0),
    deadband_range=(0.0, 0.0),
  ), term)
  out = []
  for value in (1.0, 2.0, 3.0, 4.0):
    y = hook(torch.full((2, 1), value), term)
    out.append(float(y[0, 0]))
  assert out == [0.0, 0.0, 1.0, 1.0]


def test_randomized_plant_reset_flushes_only_selected_environments():
  term = _ActionTerm(n=3)
  hook = RandomizedPlantHook(RandomizedPlantHookCfg(
    latency_weights=(0.0, 1.0), hold_weights=(1.0,),
    response_range=(1.0, 1.0), deadband_range=(0.0, 0.0),
  ), term)
  hook(torch.ones(3, 1), term)
  term._previous_target[1] = 0.4
  hook.reset(torch.tensor([1]))
  assert float(hook._previous[1, 0]) == pytest.approx(0.4)
  assert all(float(slot[1, 0]) == pytest.approx(0.4) for slot in hook._history)
  assert float(hook._previous[0, 0]) == pytest.approx(0.0)


def test_robust_task_keeps_nominal_task_separate_and_versioned():
  cfg = make_robust_env_cfg(play=True, vision=True)
  # v4 adds the scenery axis: what lies beyond the task's own sector.  The
  # simulated world has one piece of scenery, an infinite PLANE, and feeding a
  # recorded deployment channel 0 into this environment took the trained
  # policy from 170 objects placed to zero.  The name changes with the
  # distribution so a checkpoint can always be traced to the domain it saw.
  # v5 adds three things, all measured on the rig on 2026-08-31: the target
  # mask disappears when too little of it survives (the simulator called the
  # same scene visible 99% of the time against the rig's 20%), the command
  # rate ceiling drops, and the progress term that paid per step for closing
  # on the bin is off.
  # v6 keeps every v5 range and changes WHEN the mask dropout is applied
  # rather than how much of it there is.  Measured 2026-08-31: distilling
  # under the full dropout took the student's behaviour loss from 0.226 to
  # 0.541 and its placements from 2.06 to 0.31, because DAgger regresses onto
  # a teacher that can see what the student cannot and the minimiser of that
  # loss is a policy that commits to nothing.  The dropout is now a scale on
  # the task, 0 through distillation and ramped to 1 across fine-tuning, and
  # the profile carries the endpoint.  v6 also describes a second camera on
  # the hand, which only the -Wrist tasks build.
  assert HEAVY_DR_PROFILE["name"] == "d455_v7_visible"
  assert HEAVY_DR_PROFILE["vision"]["scenery_dr"] is True
  assert HEAVY_DR_PROFILE["vision"]["mask_dropout"] == (65.0, 330.0, 0.28, 0.50)
  assert HEAVY_DR_PROFILE["task"]["command_derate"] == 0.50
  assert HEAVY_DR_PROFILE["task"]["transport_progress_weight"] == 40.0
  assert HEAVY_DR_PROFILE["task"]["joint_vel_weight"] == -6.0e-3
  assert cfg.rewards["joint_vel"].weight == -6.0e-3
  assert scene_dropout_reaches_the_term(cfg)
  assert cfg.rewards["transport_progress"].weight == 40.0
  assert cfg.actions["arm"].velocity_limit["joint1"] < 1.9
  assert set(("robust_pd_gains", "robust_gripper_gains",
              "robust_joint_friction", "robust_link_inertia")) \
    <= set(cfg.events)
  assert cfg.actions["arm"].command_hooks
  scene = cfg.observations["camera"].terms["scene"]
  assert scene.delay_max_lag == 4
  assert scene.params["noise_cfg"].strength == 1.35
  assert scene.params["mask_jitter_px"] == 2
  assert scene.params["mask_dropout"] == (65.0, 330.0, 0.28, 0.50)


def test_the_dropout_scale_spans_a_clear_camera_and_the_measured_one():
  """0 must be the same path as "no dropout", not a spec that keeps everything.

  A spec of ``(0, 0, 1.0, 1.0)`` would compare every frame's blob against a
  zero floor and draw a Bernoulli that always passes -- correct, but it makes
  the distillation environment differ from the one that trained v4 by two
  tensor allocations per step and a code path that has never been run at that
  setting.  Returning ``None`` keeps it byte-identical to the profile that has
  no dropout at all, which is the thing being reproduced.
  """
  full = HEAVY_DR_PROFILE["vision"]["mask_dropout"]
  assert _scaled_dropout(full, 0.0) is None
  assert _scaled_dropout(None, 1.0) is None
  assert _scaled_dropout(full, 1.0) == full
  # Half is half the floor and half the way from certain detection to measured.
  lo, hi, k_lo, k_hi = _scaled_dropout(full, 0.5)
  assert (lo, hi) == (full[0] / 2, full[1] / 2)
  assert k_lo == pytest.approx(1.0 - (1.0 - full[2]) / 2)
  assert k_hi == pytest.approx(1.0 - (1.0 - full[3]) / 2)
  # Monotone: more scale is never easier.
  keeps = [_scaled_dropout(full, s)[2] for s in (0.25, 0.5, 0.75, 1.0)]
  assert keeps == sorted(keeps, reverse=True)
  # Over-driving the scale is clamped rather than extrapolated past the rig.
  assert _scaled_dropout(full, 2.0) == full


def test_the_hand_camera_is_not_the_scene_camera_with_a_new_name():
  """The wrist sub-profile must differ where the physics differs.

  Its floor is an order of magnitude smaller because the floor models the arm
  crossing the line of sight, and a camera bolted to the hand cannot be
  blocked by the arm it is bolted to.  Its keep probability is IDENTICAL
  because that was measured as the segmenter losing an object whose line was
  clear -- a property of the detector and the object, not of the viewpoint.
  Its mount jitter is a machining tolerance, not a knocked tripod.
  """
  w = HEAVY_DR_PROFILE["vision"]["wrist"]
  scene_floor = HEAVY_DR_PROFILE["vision"]["mask_dropout"][:2]
  assert w["mask_dropout"][:2] < scene_floor
  assert w["mask_dropout"][2:] == HEAVY_DR_PROFILE["vision"]["mask_dropout"][2:]
  assert w["camera_position_m"] < HEAVY_DR_PROFILE["vision"]["camera_position_m"]


def test_table_contact_is_audited_but_never_controls_the_policy():
  from piper_push.tasks.pick_place.env_cfg import make_pick_place_env_cfg

  train = make_pick_place_env_cfg()
  play = make_pick_place_env_cfg(play=True)
  for cfg in (train, play):
    assert "table_safety" not in cfg.rewards
    assert "table_safety" not in cfg.terminations
    sensor_names = {sensor.name for sensor in cfg.scene.sensors}
    assert {"table_clearance_guard", "robot_table_impact"} <= sensor_names
  assert play.curriculum == {}


class _CaptureCorruption:
  def __init__(self):
    self.featureless = None

  def __call__(self, depth, *, featureless):
    self.featureless = featureless.clone()
    return depth, torch.ones_like(depth, dtype=torch.bool)

  def reset(self, env_ids=None):
    pass


def _camera_env():
  depth = torch.full((1, 3, 3, 1), 0.5)
  seg = torch.zeros((1, 3, 3, 2), dtype=torch.long)
  seg[0, 1, 1, 0] = 7
  seg[0, 1, 1, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
  sensor = SimpleNamespace(data=SimpleNamespace(depth=depth, segmentation=seg))
  command = SimpleNamespace(target_geom_ids=torch.tensor([[7]]))
  return SimpleNamespace(
    num_envs=1,
    scene={"camera": sensor},
    command_manager=SimpleNamespace(get_term=lambda name: command),
  )


def test_featureless_objects_only_marks_object_not_table():
  env = _camera_env()
  term = CameraScene(None, env)
  corr = _CaptureCorruption()
  term._corr = corr
  term(env, "camera", "pick", noise_cfg=DepthNoiseCfg(strength=1.0),
       mask_jitter_px=0, featureless_objects_only=True)
  mask = torch.zeros(1, 1, 3, 3)
  mask[0, 0, 1, 1] = 1.0
  assert torch.equal(corr.featureless, F.avg_pool2d(mask, 3, 1, 1))


def scene_dropout_reaches_the_term(cfg) -> bool:
  """The profile value has to arrive at the observation term, not just exist.

  A DR axis that is configured and never applied is the failure this file was
  written to catch.
  """
  term = cfg.observations["camera"].terms["scene"]
  return term.params.get("mask_dropout") is not None
