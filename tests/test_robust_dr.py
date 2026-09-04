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
  # v7c retires blind_when_held for a measured PROCESS.  The coin flip had
  # the right marginal for one session and the wrong dynamics for every one:
  # scripts/measure_target_gaps.py over 20 sessions puts the median no-target
  # run at 40 control steps with a tail to 1463, against a geometric model's
  # 9 and no tail, and the aggregate marginal at 0.374 rather than 0.08.
  assert HEAVY_DR_PROFILE["name"] == "d455_v7_visible"
  assert HEAVY_DR_PROFILE["vision"]["scenery_dr"] is True
  assert "mask_dropout" not in HEAVY_DR_PROFILE["vision"]
  assert "blind_when_held" not in HEAVY_DR_PROFILE["vision"]
  tp = HEAVY_DR_PROFILE["vision"]["target_process"]
  assert tp["enabled"] is True
  # Ranges are the measured session spread, not a hand-set width; a point
  # estimate here would train against a rig that does not exist.
  assert tp["visible_held"][0] < 0.374 < tp["visible_held"][1]
  assert tp["mean_gap_held"][0] < 71.0 < tp["mean_gap_held"][1]
  assert tp["confirm_frames"] >= 1, "an instant recovery is not a tracker"
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
  assert scene.params["mask_dropout"] is None
  assert "blind_when_held" not in scene.params
  assert scene.params["target_process"] is not None, \
    "the process must reach the observation term, not just the profile"


def test_the_only_modelled_target_loss_is_the_one_that_was_measured():
  """Retiring ``mask_dropout`` is the point, not an oversight.

  It modelled the target vanishing as an open-loop draw calibrated from "the
  arm blocks the line of sight 52% of frames".  Per-stage counting on the rig
  then showed ``arm_mask`` removed 0% of the object's pixels and depth dropout
  0%; the loss was one filter constant.  Training against the fiction cost
  three campaigns: distillation gave 0.31, 0.42 and 0.10 placements against
  2.06 without it.

  What replaces it is closed loop and measured.  Once the jaws close the thing
  in the gripper is inside the arm from a fixed viewpoint -- and the policy
  chooses where to put its hand, so it is a consequence it can learn to manage
  rather than a coin flip it can only wait out.

  The first replacement, ``blind_when_held``, got the causal story right and
  the process wrong: an independent draw per frame, fitted to the single worst
  session.  Measured over twenty sessions the no-target runs have a median of
  40 control steps and reach 1463; an independent process at the same marginal
  has a median of 9 and no tail at all.  A GRU is trained by the runs, not by
  the rate, so this asserts the process is a process.
  """
  v = HEAVY_DR_PROFILE["vision"]
  assert "mask_dropout" not in v
  assert "blind_when_held" not in v, "the IID model is retired, not rescaled"
  tp = v["target_process"]
  assert tp["enabled"] is True
  assert tp["mean_gap_held"][0] >= 5.0, "a mean gap this short is a coin flip"
  cfg = make_robust_env_cfg(play=True, vision=True)
  scene = cfg.observations["camera"].terms["scene"]
  assert scene.params["mask_dropout"] is None
  assert scene.params["target_process"] == tp


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
  written to catch -- and it caught the opposite one too: ``mask_dropout``
  stayed wired long after the plan retired it, and an overnight run trained
  three vision stages against it.  So this now asserts what the profile says,
  in both directions.
  """
  term = cfg.observations["camera"].terms["scene"]
  v = HEAVY_DR_PROFILE["vision"]
  if "mask_dropout" in v:
    return term.params.get("mask_dropout") is not None
  return (term.params.get("mask_dropout") is None
          and term.params.get("target_process") == v.get("target_process"))


def _stages(cfg, name):
  return [float(s["weight"]) for s in cfg.curriculum[name].params["stages"]]


def test_the_sight_weights_scale_their_curriculum_instead_of_flattening_it():
  """The profile's number is a target, not an instruction to skip the ramp.

  This flattened all three stages onto the profile value until 2026-09-02, on
  a note claiming the curricula were inert.  They are not: stepping the robust
  env to step 120 reads ``action_rate`` at -0.02 and ``table_touch`` at -0.3,
  both first stages, correctly applied.  Flattening meant every v7 teacher
  paid the full sight penalty from iteration 0 -- charged for approaching an
  object before it had learned to grasp one -- and it shows in the behaviour:
  43% fewer grasp attempts than v5 at the same per-grasp cadence.  Whether
  that ramp yields a better teacher is the open question, not a known fact.

  A regression here is silent.  Training simply produces a teacher that
  hesitates, and the evidence arrives a day later in a distillation curve.
  """
  from mjlab.tasks.registry import load_env_cfg
  import mjlab.tasks  # noqa: F401

  cfg = load_env_cfg("Mjlab-Pick-Place-PiperX-Robust", play=False)

  hand = _stages(cfg, "sight_hand_weight")
  assert len(set(hand)) == len(hand), f"sight_hand ramp is flat: {hand}"
  assert abs(hand[0]) < abs(hand[-1]), "the penalty must grow, not shrink"

  arm = _stages(cfg, "sight_arm_weight")
  assert len(set(arm)) == len(arm), f"sight_arm ramp is flat: {arm}"

  # wrist_side_on is a bonus that FADES, so its profile value is the first
  # stage.  Scaling from the wrong end would invert the schedule.
  wrist = _stages(cfg, "wrist_decay")
  assert wrist[0] > wrist[-1] > 0.0, f"wrist bonus should decay: {wrist}"

  # And the built weight matches where the ramp starts, so a config dump does
  # not read as though there were no curriculum.
  assert float(cfg.rewards["sight_hand"].weight) == pytest.approx(hand[0])


def test_a_zero_sight_weight_switches_the_term_off_at_every_stage():
  """``SIGHT_ARM_W=0`` has to mean off, not "off for the first 200 iterations".

  The arm term is the one measurement says to retire -- the arm accounts for
  0.1-0.2% of blocked samples at every weight tried -- so switching it off has
  to be reachable without editing the ramp by hand.
  """
  from piper_push.tasks.pick_place import robust_cfg as rc
  from mjlab.tasks.registry import load_env_cfg
  import mjlab.tasks  # noqa: F401

  cfg = load_env_cfg("Mjlab-Pick-Place-PiperX-Robust", play=False)
  task = dict(rc.HEAVY_DR_PROFILE["task"])
  task["sight_arm_weight"] = 0.0
  task["wrist_weight"] = 0.0
  for key, reward, curr, ref in (
      ("sight_arm_weight", "sight_arm", "sight_arm_weight", -1),
      ("wrist_weight", "wrist_side_on", "wrist_decay", 0)):
    stages = cfg.curriculum[curr].params["stages"]
    base = float(stages[ref]["weight"])
    scale = float(task[key]) / base if base else 0.0
    for s in stages:
      s["weight"] = float(s["weight"]) * scale
    assert all(float(s["weight"]) == 0.0 for s in stages), \
      f"{curr} did not switch off: {_stages(cfg, curr)}"


def test_the_visibility_range_is_ordered_and_reaches_the_sam_domain():
  """A floor above the measured ceiling must describe a domain, not collide.

  ``TARGET_VISIBLE_FLOOR`` alone can only raise the bottom of the depth
  segmenter's spread, whose top is 0.85/0.90.  Asking for 0.95 produced
  ``(0.95, 0.85)`` -- a reversed pair that still draws, so nothing complains,
  and quietly caps the domain below what was requested.  There is now a second
  perception stack to match: SAM2.1 was measured at 97.7% of approach frames
  and 100% while held, and that is a different domain, not a different point
  in the depth stack's one.
  """
  from piper_push.tasks.pick_place.robust_cfg import _vis_range
  import piper_push.tasks.pick_place.robust_cfg as rc

  assert _vis_range(0.10, 0.85) == (0.10, 0.85), "no override, no change"

  for floor, ceil, want in ((0.95, None, (0.85, 0.95)),
                            (0.95, 1.00, (0.95, 1.00)),
                            # the floor is a floor: it cannot lower the
                            # measured bottom, only the ceiling moves down
                            (0.00, 0.50, (0.10, 0.50))):
    rc._VIS_FLOOR, rc._VIS_CEIL = floor, ceil
    got = _vis_range(0.10, 0.85)
    assert got == want, f"floor={floor} ceil={ceil}: {got} != {want}"
    assert got[0] <= got[1], "the pair must be ordered however it is set"
  rc._VIS_FLOOR, rc._VIS_CEIL = 0.0, None


def test_the_observation_latency_prior_can_be_set_from_the_measured_loop():
  """The perception loop's latency is a property of the perception stack, and
  the stack changed.

  Replayed on an idle card, the depth-only loop is 18.5 ms median / 23.5 p95;
  with SAM2.1 carrying the target it is 76.5 / 94.2.  The profile's default
  prior is a mean of 43 ms over a support that stops at 80, so a student
  distilled against it and deployed behind SAM meets, all of the time, a lag it
  saw 5% of the time.  The support cannot express 94 ms at all -- ``LAGS`` is
  five bins of one control step -- and that limit is recorded here rather than
  hidden, because the honest ceiling is 80 ms until ``latency.LAGS`` is
  widened.
  """
  import importlib
  import os

  from piper_push.latency import LAGS, STEP_MS

  assert LAGS == (0, 1, 2, 3, 4) and STEP_MS == 20.0, \
    "the support moved; the numbers in this test are in control steps"

  import piper_push.tasks.pick_place.robust_cfg as rc
  os.environ["OBS_LATENCY_PROBS"] = "0,0,0.10,0.30,0.60"
  try:
    importlib.reload(rc)
    cfg = rc.make_robust_env_cfg(play=True, vision=True)
    probs = cfg.observations["camera"].terms["scene"].params["latency_probs"]
    assert abs(sum(probs) - 1.0) < 1e-9, "must be normalised"
    mean_ms = sum(l * p for l, p in zip(LAGS, probs)) * STEP_MS
    assert abs(mean_ms - 70.0) < 1e-6, mean_ms
  finally:
    os.environ.pop("OBS_LATENCY_PROBS", None)
    importlib.reload(rc)

  # And unset, the default is untouched: every earlier result was measured on it.
  cfg = rc.make_robust_env_cfg(play=True, vision=True)
  probs = cfg.observations["camera"].terms["scene"].params["latency_probs"]
  mean_ms = sum(l * p for l, p in zip(LAGS, probs)) * STEP_MS
  assert abs(mean_ms - 43.0) < 1e-6, mean_ms


def test_smooth_scale_moves_the_whole_family_and_its_ramps():
  """One knob for every term that charges for HOW the arm moves.

  Scaling one of them alone just moves the roughness into the others, and the
  two that ramp have to be scaled through the ramp: setting only
  ``cfg.rewards[...].weight`` is undone on the curriculum manager's first step,
  which is the bug that made every v7 teacher train at a flat sight penalty
  while the config said otherwise.
  """
  import importlib
  import os

  import piper_push.tasks.pick_place.robust_cfg as rc

  base = rc.make_robust_env_cfg()
  b = {k: base.rewards[k].weight for k in
       ("action_rate", "action_acc", "joint_vel", "joint_acc",
        "joint_torques", "mech_power")}
  b_ramp = [s["weight"]
            for s in base.curriculum["action_rate_weight"].params["stages"]]

  os.environ["SMOOTH_SCALE"] = "3.0"
  try:
    importlib.reload(rc)
    cfg = rc.make_robust_env_cfg()
    for k, w in b.items():
      assert abs(cfg.rewards[k].weight - 3.0 * w) < 1e-12, k
    ramp = [s["weight"]
            for s in cfg.curriculum["action_rate_weight"].params["stages"]]
    assert all(abs(a - 3.0 * c) < 1e-12 for a, c in zip(ramp, b_ramp)), ramp
    assert ramp[0] != ramp[-1], "the ramp must still be a ramp"
  finally:
    os.environ.pop("SMOOTH_SCALE", None)
    importlib.reload(rc)

  cfg = rc.make_robust_env_cfg()
  assert cfg.rewards["action_rate"].weight == b["action_rate"], \
    "unset must leave every earlier result's domain untouched"


def test_full_range_reset_covers_the_whole_soft_limit_box():
  """The narrow reset was measured not to cover where the policy goes.

  With ``default +- 0.7 rad``, J6 is initialised over +-40.1 deg against a soft
  limit of +-108, and `student_5000` spends 63% of its late steps outside the
  box it is ever started from.  Sampling between the limits is the fix; the
  clamp is not, because clamping a wide delta piles probability onto the
  boundary and calls it coverage.
  """
  import importlib
  import os

  import piper_push.tasks.pick_place.env_cfg as ec

  cfg = ec.make_pick_place_env_cfg()
  assert cfg.events["reset_arm"].params["full_range"] is False, "off by default"

  os.environ["RESET_FULL_RANGE"] = "1"
  try:
    importlib.reload(ec)
    cfg = ec.make_pick_place_env_cfg()
    assert cfg.events["reset_arm"].params["full_range"] is True
    # position_range stays in the signature: the rejection sampler and the
    # narrow path are still the default, and one env var must not delete them.
    assert cfg.events["reset_arm"].params["position_range"] == (-0.7, 0.7)
  finally:
    os.environ.pop("RESET_FULL_RANGE", None)
    importlib.reload(ec)
