"""Conservative domain randomisation for the first D455 hardware policy.

This profile intentionally covers a wider domain than the fitted mean.  The
first real deployment is gated on success and zero table contact, not maximum
cycle speed.  It is a separate task so a checkpoint can be evaluated both on
the nominal calibrated domain and on this stress distribution without hidden
CLI mutations.
"""

from __future__ import annotations

import dataclasses
import math
import os

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from piper_push import camera
from piper_push.actions import RandomizedPlantHookCfg
from piper_push.latency import LatencyPrior, apply_latency_prior

from .env_cfg import make_pick_place_env_cfg


# Serialized by tests/reporting: changing a range changes the experiment.
_VIS_FLOOR = float(os.environ.get("TARGET_VISIBLE_FLOOR", 0.0))
"""Raises the worst sessions out of the drawn range; 0 keeps all of them."""

_VIS_CEIL = os.environ.get("TARGET_VISIBLE_CEIL")
_VIS_CEIL = float(_VIS_CEIL) if _VIS_CEIL is not None else None
"""Replaces the top of the drawn range, so a floor above the measured ceiling
describes a domain instead of colliding with one.

``TARGET_VISIBLE_FLOOR`` alone could only raise the bottom of the *depth
segmenter's* measured spread, whose top is 0.85/0.90.  Asking for a floor of
0.95 there produces ``(0.95, 0.85)`` -- still a usable interval, because the
draw is ``lo + rand * (hi - lo)``, but a reversed one that nobody wrote down
and that silently caps the domain below what was asked for.

Both ends are needed now because there is a second perception stack to match.
Measured over seven seeds with ``scripts/sim_perception_check.py --sam``,
SAM2.1 carrying the target reports it in 97.7% of approach frames and 100%
while held, against the depth stack's 47%/0%.  Those are different domains,
not different points in one, and which of them to distil into is a deployment
decision -- so it is a pair of numbers on the command line rather than an
edit."""


def _vis_range(lo: float, hi: float) -> tuple[float, float]:
  """The per-episode marginal range, after the two environment overrides.

  Ordered, so ``lo <= hi`` however the two knobs are set; the draw works either
  way but a reversed pair is a config nobody can read."""
  lo = max(lo, _VIS_FLOOR)
  hi = hi if _VIS_CEIL is None else _VIS_CEIL
  return (min(lo, hi), max(lo, hi))

_GAP_SCALE = float(os.environ.get("TARGET_GAP_SCALE", 1.0))
"""Scales the measured blackout lengths; 1.0 is the rig."""

_OBS_LATENCY = os.environ.get("OBS_LATENCY_PROBS")
"""Comma-separated probabilities over 0..N control steps of observation lag.

The profile's default is a mean of 43 ms, fitted to the depth-only loop, whose
replay measures **18.5 ms median / 23.5 ms p95** on an idle card.  Adding
SAM2.1 to the same loop measures **76.5 / 94.2 ms** -- so the median under SAM
is the 80 ms bin the default gives 5% of its mass to, and the p95 is off the
end of the support entirely.

That is a new sim-to-real gap created by fixing an old one, and it is exactly
the kind this campaign exists to stop shipping: a student distilled against the
default prior and deployed behind SAM would meet a latency it has seen 5% of
the time, all of the time.  Set this when the perception stack changes, from
the replay, not from taste."""

_SMOOTH = float(os.environ.get("SMOOTH_SCALE", 1.0))
"""Multiplies every term that charges for how the arm moves rather than where.

``action_rate``, ``action_acc``, ``joint_vel``, ``joint_acc``,
``joint_torques`` and ``mech_power`` -- the whole family, together, because
scaling one of them just moves the roughness into the others.  The two with a
curriculum are scaled THROUGH it, so a ramp that starts at a third of its final
weight still does.

1.0 is every result on record.  The measured step-to-step action change is
0.210 for `strong_teacher`, 0.244 for the -24 sight teachers and 0.311 for
`v5_baseline`, so this is the axis that separates them almost as cleanly as the
sight reward does -- and unlike the sight reward it costs nothing in
visibility.  A dose above 1 is an experiment, not a fit: v6 collapsed a policy
by over-weighting a single term."""

_SIGHT_RAMP = os.environ.get("SIGHT_RAMP", "1") not in ("0", "false", "False")
"""Whether the sight weights scale their curriculum (default) or flatten it."""

HEAVY_DR_PROFILE = {
  "name": "d455_v7_visible",
  "timing": {
    "arm_latency_weights": (0.15, 0.55, 0.30),       # 0..2 control steps
    "arm_hold_weights": (0.15, 0.70, 0.15),          # 1..3 steps
    "arm_response": (0.70, 1.00),
    "arm_deadband_rad": (0.0, 0.004),
    "gripper_latency_weights": (0.15, 0.55, 0.30),
    "gripper_hold_weights": (0.10, 0.70, 0.20),
    "gripper_response": (0.55, 1.00),
    "gripper_deadband_m": (0.0, 0.0005),
    # Mean 43 ms: close to the measured 42.23 ms real loop, with tails.
    "observation_latency_probs": (0.05, 0.15, 0.45, 0.30, 0.05),
  },
  "task": {
    # The rate ceiling the command path is allowed.  0.62 was derived from
    # kp 125 / kd 6.5 -- an underdamped second order whose 32% velocity
    # overshoot fired the safety shell.  The arm was measured on 2026-08-31 at
    # kp 280-379 with kd/kp = 56 ms, so that derivation no longer holds and the
    # number is being lowered rather than re-derived from a plant that has
    # changed underneath it.
    "command_derate": 0.50,
    # The progress term that pays per step for closing on the drop point --
    # "the engine".  Zero here at the operator's request: the run is meant to
    # be slower.  Its own comment records that at weight 25 the trip did not
    # pay for itself, so if the policy stops transporting at all, this is the
    # first number to raise.
    # Restored, at a third of the original 120.  It was set to 0 for v5 on
    # request and with no measurement behind it; the term's own comment records
    # that at weight 25 the policy judged the trip not worth taking.  With the
    # sight-cylinder in place the path to the bin is longer and more
    # constrained, and ``place``'s 250 points at the end of it are a very
    # sparse thing to steer by.  This pays for progress, not for speed.
    "transport_progress_weight": 40.0,
    # Slow the carry down.  Occlusion scales with speed on this rig -- tracing
    # the arm against the camera-to-object line, the fast run had it blocking
    # 52% of frames and the slow one 16% -- so a policy that moves less loses
    # its own target less often, and the gap it has to cross at deployment is
    # smaller to begin with.
    #
    # 6x the shipped -1.0e-3, which at 1 rad/s across six joints was -0.006 a
    # step against a 250-point placement: too small to be a gradient at all.
    # This is deliberately still an order under action_rate; the risk being
    # managed is the opposite one, because the progress term that paid for the
    # trip is now zero and standing still must not become optimal.
    "joint_vel_weight": -6.0e-3,
    # Overridable from the environment so two weightings can be trained side
    # by side without editing a file a running job has already imported.
    #
    # The defaults are what v7 launched with, and the first hundred iterations
    # say they are too small to change behaviour: sight_hand contributes -0.13
    # of episode reward against place's +2.04 and premature_touch's -0.97, and
    # it has been flat for eighty iterations while placements held at 5.0.
    # Detouring behind the object costs action_rate and action_acc, which
    # together are -1.35; not detouring costs -0.13.  The policy is right.
    "sight_arm_weight": float(os.environ.get("SIGHT_ARM_W", -2.0)),
    "sight_hand_weight": float(os.environ.get("SIGHT_HAND_W", -4.0)),
    "wrist_weight": float(os.environ.get("WRIST_W", 0.8)),
  },
  "robot": {
    "kp_scale": (0.75, 1.25),
    "kd_scale": (0.60, 1.60),
    "joint_friction_scale": (0.50, 1.80),
    "link_mass_scale": (0.80, 1.25),
    "com_shift_m": (-0.002, 0.002),
    # No hardware identification exists for the gripper loop yet.
    "gripper_kp_scale": (0.60, 1.40),
    "gripper_kd_scale": (0.50, 1.80),
  },
  "scene": {
    "table_z_m": (-0.007, 0.007),
    "table_tilt_deg": (-0.5, 0.5),
    "object_mass_kg": (0.03, 0.50),
    "object_friction": (0.25, 1.20),
    "pad_friction": (0.40, 1.30),
  },
  "vision": {
    "camera_position_m": 0.030,
    "camera_rotation_deg": 3.0,
    "depth_strength": 1.35,
    "surface_fill": (0.78, 0.995),
    "texture_penalty": (1.0, 2.8),
    "mask_jitter_px": 2,
    "scenery_dr": True,
    # mask_dropout is RETIRED.  It modelled the target vanishing as an
    # open-loop coin flip calibrated from "the arm blocks the line of sight 52%
    # of frames", and per-stage counting on the rig showed arm_mask removed 0%
    # of the object's pixels and depth dropout 0%: the loss was one filter
    # constant, `SegmenterCfg.width_range_m`, 160 mm against a 164 mm blob.
    # Training against it cost three campaigns -- distillation gave 0.31, 0.42
    # and 0.10 placements against 2.06 without it -- because DAgger regresses
    # the student onto a teacher that can see what the student cannot.
    #
    # What IS measured, and what replaces it: once the jaws close, the rig
    # reports the target in 8% of frames (155 of 1894), because from a fixed
    # viewpoint the thing in the gripper is inside the arm.  That is closed
    # loop -- the policy chooses where to put its hand -- and the simulator
    # models none of it.
    # Replaces blind_when_held.  That was an IID per-frame keep at 0.08,
    # fitted to the single worst recorded session; the rig loses the target in
    # runs with a median of 40 control steps and a tail to 29 seconds, and the
    # aggregate marginal is 0.374 rather than 0.08.  Ranges are the measured
    # session-to-session spread -- see piper_push.target_process and
    # scripts/measure_target_gaps.py.  None here disables it entirely.
    "target_process": {
      # TARGET_VISIBLE_FLOOR raises the bottom of the measured session spread
      # without touching the gap structure -- the other half of the diagnostic
      # that TARGET_GAP_SCALE started.  Gap length turned out not to be the
      # binding difficulty (gaps at 0.35x learned the same as gaps at 1.0x),
      # which leaves the marginal.  If raising it recovers v4-like learning
      # then the student's problem is how OFTEN the rig sees its target, and
      # that is a segmenter to fix rather than a domain to train through.
      "visible_held": _vis_range(0.10, 0.85),
      "visible_approach": _vis_range(0.15, 0.90),
      # TARGET_GAP_SCALE shortens the blackouts without touching the marginal,
      # which is the one knob that separates "the target is often missing"
      # from "the target is missing for a long time".  The measured process is
      # 1.0; a smaller value is a milder domain and a diagnostic, not a fit.
      "mean_gap_held": (20.0 * _GAP_SCALE, 120.0 * _GAP_SCALE),
      "mean_gap_approach": (15.0 * _GAP_SCALE, 110.0 * _GAP_SCALE),
      "confirm_frames": 3,
      "enabled": True,
    },
    # The deployment's geometric rebuild of the held object, so training sees
    # the same construction.  0 keeps the renderer's exact silhouette, which
    # is what every policy so far was trained on and what no robot can supply.
    "held_proxy_radius": float(os.environ.get("HELD_PROXY_M", 0.0)),
    # The hand camera.  Its ranges are not the scene camera's and must not be
    # copied from them: it is a D405 at 120 mm, not a D455 at 1.2 m.
    #
    # The pixel FLOOR is nearly inert here and that is the point.  The floor
    # models the scene camera's dominant loss -- the arm crossing its line of
    # sight -- and a camera bolted to the hand cannot be blocked by the arm it
    # is bolted to.  At 120 mm the object fills a large part of the frame, so
    # a floor in the tens of pixels almost never fires.  It is kept, not
    # dropped, because the wrist has its own way to see nothing: the hand
    # pointed away from the table.  That is geometry the renderer gets right,
    # and it is closed loop -- the policy chooses where to point.
    #
    # The residual KEEP is carried over unchanged, and that IS defensible
    # transfer: 0.28-0.50 was measured as the segmenter losing an object whose
    # line of sight was clear -- depth dropping out on a dark curved surface,
    # the arm mask deleting everything within 20 mm of the gripper.  Those are
    # properties of the detector and the object, not of where the camera is.
    # The one that would not transfer, occlusion by the arm, is the floor, and
    # the floor is what was scaled down.
    "wrist": {
      "camera_position_m": 0.006,
      "camera_rotation_deg": 2.5,
      "depth_strength": 1.35,
      "surface_fill": (0.78, 0.995),
      "texture_penalty": (1.0, 2.8),
      "mask_jitter_px": 2,
      "scenery_dr": True,
      "mask_dropout": (10.0, 60.0, 0.28, 0.50),
    },
  },
}


def _plant_cfg(*, gripper: bool = False) -> RandomizedPlantHookCfg:
  t = HEAVY_DR_PROFILE["timing"]
  prefix = "gripper" if gripper else "arm"
  return RandomizedPlantHookCfg(
    latency_weights=t[f"{prefix}_latency_weights"],
    hold_weights=t[f"{prefix}_hold_weights"],
    response_range=t[f"{prefix}_response"],
    deadband_range=t[f"{prefix}_deadband_m" if gripper else "arm_deadband_rad"],
  )


def _scaled_dropout(spec, scale: float):
  """Interpolate the measured dropout toward a camera that never loses anything.

  Distillation and reinforcement are not the same problem and must not get the
  same visibility.  Measured on 2026-08-31: with the profile's full dropout the
  student's behaviour loss went 0.226 -> 0.541 and it placed 0.31 objects
  against 2.06 without.  That is not the student failing to try hard enough.
  DAgger regresses the student onto the TEACHER's action, and the teacher sees
  the object on every frame; on a frame where the student is blind the target
  is not a function of the student's observation at all, so the loss has an
  irreducible floor and the minimiser is the conditional mean -- a policy that
  commits to nothing.  It reached for an object 0.46 times per episode against
  5.94.

  PPO does not have that defect.  Its critic is privileged and its objective is
  return, not agreement with somebody who can see more, which is exactly the
  asymmetric actor-critic setting partial observability calls for.  So the
  blindness belongs in the reinforcement stage and not in the imitation one,
  and this is the knob that puts it there: 0 for distillation, ramped to 1
  across fine-tuning.

  Scale 0 returns ``None`` rather than a spec that keeps everything, so the
  term takes the same path it does when the profile has no dropout at all.
  """
  if spec is None or scale <= 0.0:
    return None
  lo, hi, k_lo, k_hi = (float(x) for x in spec)
  s = min(float(scale), 1.0)
  # The floor scales from "no blob is too small" and the keep probability from
  # "the detector never drops anything", so both ends are the clear camera.
  return (lo * s, hi * s, 1.0 + (k_lo - 1.0) * s, 1.0 + (k_hi - 1.0) * s)


def apply_heavy_dr(
  cfg, *, vision: bool, wrist: bool = False, mask_dropout_scale: float = 1.0
) -> dict:
  """Mutate ``cfg`` into the versioned conservative deployment domain."""
  robot = HEAVY_DR_PROFILE["robot"]
  scene = HEAVY_DR_PROFILE["scene"]
  task = HEAVY_DR_PROFILE.get("task", {})

  if "command_derate" in task:
    from piper_push import robot as piper
    derate = float(task["command_derate"])
    cfg.actions["arm"].velocity_limit = {
      j: derate * v for j, v in piper.JOINT_TRIP_RAD_S.items()}
  if "transport_progress_weight" in task and "transport_progress" in cfg.rewards:
    cfg.rewards["transport_progress"].weight = float(
      task["transport_progress_weight"])
  if "joint_vel_weight" in task and "joint_vel" in cfg.rewards:
    cfg.rewards["joint_vel"].weight = float(task["joint_vel_weight"])
  # SIGHT_RAMP=0 restores the flat schedule, for running it deliberately as a
  # control against a ramped run at the same final weight.  Default is on.
  # The visibility weights, SCALED THROUGH their curriculum rather than
  # flattened onto it.
  #
  # This flattened them until 2026-09-02, on a note claiming the ramps were
  # inert -- "every curriculum term sitting at its final value from the first
  # iteration".  That was backwards, and measuring it says so: stepping the
  # robust env to step 120 reads ``action_rate`` at -0.02 and ``table_touch``
  # at -0.3, which are those ramps' FIRST stages, correctly applied.  The only
  # terms sitting at their final value were the three flattened here.
  #
  # Every v7 teacher therefore trained with the full sight penalty from
  # iteration 0, where the design ramps it in over 600 -- charged for
  # approaching an object before it had learned to grasp one.  Whether that
  # ramp produces a BETTER teacher is untested; what is measured is only that
  # the schedule ran differently from the one the config declares and from
  # every other shaped term in the same run.
  #
  # The strong teacher does what it was asked: 4.7% of approach samples
  # blocked against v5's 23.9%.  It is also slower (2.17 objects a rollout
  # against 4.22) -- which is wanted, smooth motion is the point -- and its
  # student reaches 0.126 of its own teacher at iteration 750 where v4's
  # reached 0.204.  That residual gap is real and its cause is NOT isolated:
  # four candidate mechanisms have been measured and refuted, including the
  # obvious one that the student sees fewer grasps.  It does not; the
  # distillation loss is behaviour cloning and the teacher labels every
  # timestep the student visits, however rarely it would grasp.
  #
  # ``ref`` is the stage the profile's number refers to, and it is not the
  # same end for every term: the two sight penalties grow (-0.6, -2.0, -4.0)
  # so the profile quotes the last, while ``wrist_decay`` is a bonus that
  # fades (0.8, 0.5, 0.30) and quotes the first.  Scaling the whole schedule
  # by ``w / stages[ref]`` keeps the shape either way, and w = 0 turns the
  # term off cleanly at every stage.
  for key, reward, curr, ref in (
      ("sight_arm_weight", "sight_arm", "sight_arm_weight", -1),
      ("sight_hand_weight", "sight_hand", "sight_hand_weight", -1),
      ("wrist_weight", "wrist_side_on", "wrist_decay", 0)):
    if key not in task or reward not in cfg.rewards:
      continue
    w = float(task[key])
    if curr in cfg.curriculum and _SIGHT_RAMP:
      stages = cfg.curriculum[curr].params["stages"]
      base = float(stages[ref]["weight"])
      scale = w / base if base else 0.0
      for stage in stages:
        stage["weight"] = float(stage["weight"]) * scale
      # The manager sets this on its first run anyway; matching stage 0 keeps
      # a config dump from reading as though the ramp were not there.
      cfg.rewards[reward].weight = float(stages[0]["weight"])
    else:
      # SIGHT_RAMP=0 lands here on purpose: the flat schedule every v7 teacher
      # accidentally trained under, kept reachable so it can be run as a
      # CONTROL rather than only as a past mistake.  A ramped run and a flat
      # one at the same final weight differ in one thing; without that pair,
      # a ramped run that works is confounded with whatever else changed
      # alongside it.
      cfg.rewards[reward].weight = w
      if curr in cfg.curriculum:
        for stage in cfg.curriculum[curr].params["stages"]:
          stage["weight"] = w

  if _SMOOTH != 1.0:
    # The two with a curriculum: scale every stage, so the ramp keeps its
    # shape.  Same treatment the sight weights get, and for the same reason --
    # setting only ``cfg.rewards[...].weight`` is silently undone on the
    # manager's first step.
    for reward, curr in (("action_rate", "action_rate_weight"),
                         ("action_acc", "action_acc_weight")):
      if reward in cfg.rewards:
        cfg.rewards[reward].weight = float(cfg.rewards[reward].weight) * _SMOOTH
      if curr in cfg.curriculum:
        for stage in cfg.curriculum[curr].params["stages"]:
          stage["weight"] = float(stage["weight"]) * _SMOOTH
    for reward in ("joint_vel", "joint_acc", "joint_torques", "mech_power"):
      if reward in cfg.rewards:
        cfg.rewards[reward].weight = float(cfg.rewards[reward].weight) * _SMOOTH

  cfg.actions["arm"].command_hooks = (_plant_cfg(),)
  cfg.actions["gripper"].command_hooks = (_plant_cfg(gripper=True),)

  base_pose = cfg.events["reset_base"].params["pose_range"]
  base_pose.update({
    "z": scene["table_z_m"],
    "roll": tuple(math.radians(x) for x in scene["table_tilt_deg"]),
    "pitch": tuple(math.radians(x) for x in scene["table_tilt_deg"]),
  })
  for name, event in cfg.events.items():
    if name.startswith("object_shape"):
      event.params["mass_range"] = scene["object_mass_kg"]
      event.params["friction_range"] = scene["object_friction"]
  cfg.events["pad_friction"].params["ranges"] = scene["pad_friction"]

  # Gains/friction are redrawn each episode; inertia is fixed per environment
  # for a run because pseudo-inertia requires a costly MuJoCo set-const.
  cfg.events["robust_pd_gains"] = EventTermCfg(
    func=dr.pd_gains,
    mode="reset",
    params={
      "kp_range": robot["kp_scale"],
      "kd_range": robot["kd_scale"],
      "operation": "scale",
      # The first four actuators are J1-6; actuator 4 is the unmeasured gripper.
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0, 1, 2, 3]),
    },
  )
  cfg.events["robust_joint_friction"] = EventTermCfg(
    func=dr.joint_friction,
    mode="reset",
    params={
      "ranges": robot["joint_friction_scale"],
      "operation": "scale",
      "asset_cfg": SceneEntityCfg("robot", joint_names=("joint[1-6]",)),
    },
  )
  cfg.events["robust_gripper_gains"] = EventTermCfg(
    func=dr.pd_gains,
    mode="reset",
    params={
      "kp_range": robot["gripper_kp_scale"],
      "kd_range": robot["gripper_kd_scale"],
      "operation": "scale",
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=[4]),
    },
  )
  mass_lo, mass_hi = robot["link_mass_scale"]
  cfg.events["robust_link_inertia"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="startup",
    params={
      # pseudo-inertia density scale is exp(2 alpha).
      "alpha_range": (0.5 * math.log(mass_lo), 0.5 * math.log(mass_hi)),
      "t_range": robot["com_shift_m"],
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("link[1-6]", "gripper_base", "gripper_link[12]"),
      ),
    },
  )

  if vision:
    v = HEAVY_DR_PROFILE["vision"]
    event = cfg.events["camera_pose"]
    event.mode = "reset"
    event.params.update({
      "pos_jitter": v["camera_position_m"],
      "rot_jitter": math.radians(v["camera_rotation_deg"]),
    })
    term = cfg.observations["camera"].terms["scene"]
    base_noise = term.params["noise_cfg"]
    term.params["noise_cfg"] = dataclasses.replace(
      camera.DEPTH_NOISE,
      strength=v["depth_strength"],
      surface_fill=v["surface_fill"],
      texture_penalty=v["texture_penalty"],
      # Retain the fitted D455 bias/quantisation and widen their effects by
      # strength instead of inventing a second independent bias source.
    )
    del base_noise
    term.params["mask_jitter_px"] = v["mask_jitter_px"]
    # What lies beyond the task's own sector.  The simulated world has one
    # piece of scenery -- an infinite PLANE -- and no room looks like that;
    # feeding a recorded deployment channel 0 into this environment took the
    # trained policy from 170 objects placed to zero.  Randomising the region
    # is what removes the deployment's need to correct the image afterwards.
    term.params["scenery_dr"] = bool(v.get("scenery_dr", False))
    term.params["mask_dropout"] = _scaled_dropout(
      v.get("mask_dropout"), mask_dropout_scale)
    term.params["target_process"] = v.get("target_process")
    term.params["held_proxy_radius"] = v.get("held_proxy_radius")
    if wrist:
      w = v["wrist"]
      wevent = cfg.events["wrist_camera_pose"]
      wevent.mode = "reset"
      wevent.params.update({
        "pos_jitter": w["camera_position_m"],
        "rot_jitter": math.radians(w["camera_rotation_deg"]),
      })
      wterm = cfg.observations["wrist"].terms["scene"]
      wterm.params["noise_cfg"] = dataclasses.replace(
        camera.WRIST_DEPTH_NOISE,
        strength=w["depth_strength"],
        surface_fill=w["surface_fill"],
        texture_penalty=w["texture_penalty"],
      )
      wterm.params["mask_jitter_px"] = w["mask_jitter_px"]
      wterm.params["scenery_dr"] = bool(w.get("scenery_dr", False))
      wterm.params["mask_dropout"] = _scaled_dropout(
        w.get("mask_dropout"), mask_dropout_scale)

    probs = HEAVY_DR_PROFILE["timing"]["observation_latency_probs"]
    if _OBS_LATENCY:
      probs = tuple(float(x) for x in _OBS_LATENCY.split(","))
      total = sum(probs)
      if total <= 0:
        raise ValueError(f"OBS_LATENCY_PROBS sums to {total}")
      probs = tuple(x / total for x in probs)
      HEAVY_DR_PROFILE["timing"]["observation_latency_probs"] = probs
    apply_latency_prior(cfg, LatencyPrior(probs), seed=20260827)

  return HEAVY_DR_PROFILE


def make_robust_env_cfg(
  *,
  play: bool = False,
  vision: bool = False,
  wrist: bool = False,
  mask_dropout_scale: float = 1.0,
):
  cfg = make_pick_place_env_cfg(play=play, vision=vision, wrist=wrist)
  apply_heavy_dr(
    cfg, vision=vision, wrist=wrist, mask_dropout_scale=mask_dropout_scale)
  return cfg
