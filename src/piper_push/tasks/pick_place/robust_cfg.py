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
    "blind_when_held": 0.08,
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
  # The visibility weights, and the curriculum entries that would otherwise
  # overwrite them.  Those ramps are inert at this environment count -- the
  # teacher log shows every curriculum term sitting at its final value from the
  # first iteration -- so leaving them in place would silently restore the
  # default the moment the manager ran.
  for key, reward, curr in (("sight_arm_weight", "sight_arm", "sight_arm_weight"),
                            ("sight_hand_weight", "sight_hand", "sight_hand_weight"),
                            ("wrist_weight", "wrist_side_on", "wrist_decay")):
    if key in task and reward in cfg.rewards:
      w = float(task[key])
      cfg.rewards[reward].weight = w
      if curr in cfg.curriculum:
        for stage in cfg.curriculum[curr].params["stages"]:
          stage["weight"] = w

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
    term.params["blind_when_held"] = v.get("blind_when_held")
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

    apply_latency_prior(
      cfg, LatencyPrior(HEAVY_DR_PROFILE["timing"]["observation_latency_probs"]),
      seed=20260827,
    )

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
