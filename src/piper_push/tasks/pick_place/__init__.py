from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

# MjlabOnPolicyRunner, not ManipulationOnPolicyRunner.  The manipulation runner
# exports ONNX on every save and stamps it with mjlab's deployment metadata,
# which assumes an action term literally named "joint_pos" and an actor group
# of 1D terms with scales.  This task has two action terms and a camera, so
# every save has been printing "ONNX export failed: 'joint_pos'" and producing
# nothing.  Exports go through scripts/check_export.py instead, which produces
# them *and* checks that they still act like the trained policy.

from .env_cfg import make_pick_place_env_cfg
from .robust_cfg import make_robust_env_cfg
from .cold_cfg import make_cold_env_cfg
from piper_push.distill import PickPlaceDistillationRunner
from piper_push.runners import PickPlaceOnPolicyRunner

from .rl_cfg import (
  pick_place_distill_runner_cfg,
  pick_place_ppo_runner_cfg,
  pick_place_vision_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX",
  env_cfg=make_pick_place_env_cfg(),
  play_env_cfg=make_pick_place_env_cfg(play=True),
  rl_cfg=pick_place_ppo_runner_cfg(),
  runner_cls=PickPlaceOnPolicyRunner,
)

# Conservative D455 deployment domain.  The ordinary tasks remain the
# calibrated nominal domain, which gives evaluation a clean A/B axis for the
# performance cost of robustness.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Robust",
  env_cfg=make_robust_env_cfg(),
  play_env_cfg=make_robust_env_cfg(play=True),
  rl_cfg=pick_place_ppo_runner_cfg(experiment_name="piperx_pick_place_robust"),
  runner_cls=PickPlaceOnPolicyRunner,
)

# The smoke variant: one fixed cube in every environment.  A reward bug is
# obvious when every environment holds the same object and invisible under a
# full distribution.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Cube",
  env_cfg=make_pick_place_env_cfg(shape_variety=0.0),
  play_env_cfg=make_pick_place_env_cfg(play=True, shape_variety=0.0),
  rl_cfg=pick_place_ppo_runner_cfg(experiment_name="piperx_pick_place_cube"),
  runner_cls=PickPlaceOnPolicyRunner,
)

# Halfway through the shape curriculum: the distribution's midpoint plus half
# its spread.  Registered rather than reachable from the CLI because
# shape_variety shapes the event ranges at build time, not a config field.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Mid",
  env_cfg=make_pick_place_env_cfg(shape_variety=0.5),
  play_env_cfg=make_pick_place_env_cfg(play=True, shape_variety=0.5),
  rl_cfg=pick_place_ppo_runner_cfg(experiment_name="piperx_pick_place_mid"),
  runner_cls=PickPlaceOnPolicyRunner,
)


# The vision stage.  The actor loses the object's state and gains the camera;
# everything else about the task is identical, which is the point -- any
# difference in the result is attributable to perception and not to a changed
# problem.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Vision",
  env_cfg=make_pick_place_env_cfg(vision=True),
  play_env_cfg=make_pick_place_env_cfg(play=True, vision=True),
  rl_cfg=pick_place_vision_ppo_runner_cfg(),
  runner_cls=PickPlaceOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Vision-Robust",
  env_cfg=make_robust_env_cfg(vision=True),
  play_env_cfg=make_robust_env_cfg(play=True, vision=True),
  rl_cfg=pick_place_vision_ppo_runner_cfg(
    experiment_name="piperx_pick_place_vision_robust"),
  runner_cls=PickPlaceOnPolicyRunner,
)


# The bootstrap.  Same environment as the vision task plus one extra
# observation group: the proprioception the state teacher was trained on, which
# the vision variant replaces and the teacher still needs.  The student acts,
# the teacher labels -- so the states being labelled are the ones a camera
# policy actually reaches, not the ones a policy that already knew the answer
# would have visited.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Distill",
  env_cfg=make_pick_place_env_cfg(vision=True),
  play_env_cfg=make_pick_place_env_cfg(play=True, vision=True),
  rl_cfg=pick_place_distill_runner_cfg(),
  runner_cls=PickPlaceDistillationRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Distill-Robust",
  env_cfg=make_robust_env_cfg(vision=True),
  play_env_cfg=make_robust_env_cfg(play=True, vision=True),
  rl_cfg=pick_place_distill_runner_cfg(
    experiment_name="piperx_pick_place_distill_robust"),
  runner_cls=PickPlaceDistillationRunner,
)


# The same three stages again, split by how much of the measured mask dropout
# each one gets.  ``robust_cfg._scaled_dropout`` carries the argument for why
# they differ; the short version is that imitation cannot learn to act blind
# and reinforcement can, so the blindness is ramped in after the imitation is
# finished rather than during it.
#
#   -Distill-Robust-Clear    scale 0.0   the student learns the skill
#   -Vision-Robust-Half      scale 0.5   PPO meets half the measured loss
#   -Vision-Robust           scale 1.0   PPO meets the rig
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Distill-Robust-Clear",
  env_cfg=make_robust_env_cfg(vision=True, mask_dropout_scale=0.0),
  play_env_cfg=make_robust_env_cfg(
    play=True, vision=True, mask_dropout_scale=0.0),
  rl_cfg=pick_place_distill_runner_cfg(
    experiment_name="piperx_pick_place_distill_robust"),
  runner_cls=PickPlaceDistillationRunner,
)

# 0.75, added after the fact and for a measured reason.  Fine-tuning at 0.5
# climbed steadily for its whole 1500 iterations -- 0.41 to 2.50 placements,
# still rising when it stopped -- and stepping straight from there to 1.0
# collapsed it: grasp attempts went 3.34 -> 0.13 and placements 2.50 -> 0.12
# within 400 iterations, the same shape as distilling blind.  The step was too
# big, not the destination unreachable.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Vision-Robust-ThreeQ",
  env_cfg=make_robust_env_cfg(vision=True, mask_dropout_scale=0.75),
  play_env_cfg=make_robust_env_cfg(
    play=True, vision=True, mask_dropout_scale=0.75),
  rl_cfg=pick_place_vision_ppo_runner_cfg(
    experiment_name="piperx_pick_place_vision_robust"),
  runner_cls=PickPlaceOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Vision-Robust-Half",
  env_cfg=make_robust_env_cfg(vision=True, mask_dropout_scale=0.5),
  play_env_cfg=make_robust_env_cfg(
    play=True, vision=True, mask_dropout_scale=0.5),
  rl_cfg=pick_place_vision_ppo_runner_cfg(
    experiment_name="piperx_pick_place_vision_robust"),
  runner_cls=PickPlaceOnPolicyRunner,
)


# And the hand camera.  Same three stages, same dropout ramp, one more
# observation group and one more convolutional encoder.  The state teacher is
# untouched by any of this -- it never looks at an image -- so both of these
# branches load the SAME teacher checkpoint, and the comparison between them
# isolates the second camera and nothing else.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Distill-Robust-Wrist",
  env_cfg=make_robust_env_cfg(
    vision=True, wrist=True, mask_dropout_scale=0.0),
  play_env_cfg=make_robust_env_cfg(
    play=True, vision=True, wrist=True, mask_dropout_scale=0.0),
  rl_cfg=pick_place_distill_runner_cfg(
    experiment_name="piperx_pick_place_distill_robust_wrist", wrist=True),
  runner_cls=PickPlaceDistillationRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Vision-Robust-Wrist-ThreeQ",
  env_cfg=make_robust_env_cfg(
    vision=True, wrist=True, mask_dropout_scale=0.75),
  play_env_cfg=make_robust_env_cfg(
    play=True, vision=True, wrist=True, mask_dropout_scale=0.75),
  rl_cfg=pick_place_vision_ppo_runner_cfg(
    experiment_name="piperx_pick_place_vision_robust_wrist", wrist=True),
  runner_cls=PickPlaceOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Vision-Robust-Wrist-Half",
  env_cfg=make_robust_env_cfg(
    vision=True, wrist=True, mask_dropout_scale=0.5),
  play_env_cfg=make_robust_env_cfg(
    play=True, vision=True, wrist=True, mask_dropout_scale=0.5),
  rl_cfg=pick_place_vision_ppo_runner_cfg(
    experiment_name="piperx_pick_place_vision_robust_wrist", wrist=True),
  runner_cls=PickPlaceOnPolicyRunner,
)

# The undomainrandomised wrist task exists for one reason: the DR-degradation
# report needs a nominal cell, and evaluating a two-camera policy on the
# one-camera nominal task would fail on a missing observation group rather
# than measure anything.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Vision-Wrist",
  env_cfg=make_pick_place_env_cfg(vision=True, wrist=True),
  play_env_cfg=make_pick_place_env_cfg(play=True, vision=True, wrist=True),
  rl_cfg=pick_place_vision_ppo_runner_cfg(
    experiment_name="piperx_pick_place_vision_wrist", wrist=True),
  runner_cls=PickPlaceOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Vision-Robust-Wrist",
  env_cfg=make_robust_env_cfg(
    vision=True, wrist=True, mask_dropout_scale=1.0),
  play_env_cfg=make_robust_env_cfg(
    play=True, vision=True, wrist=True, mask_dropout_scale=1.0),
  rl_cfg=pick_place_vision_ppo_runner_cfg(
    experiment_name="piperx_pick_place_vision_robust_wrist", wrist=True),
  runner_cls=PickPlaceOnPolicyRunner,
)


# S3: several objects on the table, cleared one at a time.  The command picks
# the target -- nearest to the hand, re-decided only when one is cleared, so
# the policy cannot change its mind by moving and drag the reward with it --
# and the table refills once it is empty.  Everything else about the task, the
# rewards, the metrics and the gate, is what it was with one object.
register_mjlab_task(
  task_id="Mjlab-Cleanup-PiperX",
  env_cfg=make_pick_place_env_cfg(num_objects=3),
  play_env_cfg=make_pick_place_env_cfg(play=True, num_objects=3),
  rl_cfg=pick_place_ppo_runner_cfg(experiment_name="piperx_cleanup"),
  runner_cls=PickPlaceOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Cleanup-PiperX-Vision",
  env_cfg=make_pick_place_env_cfg(vision=True, num_objects=3),
  play_env_cfg=make_pick_place_env_cfg(play=True, vision=True, num_objects=3),
  rl_cfg=pick_place_vision_ppo_runner_cfg(experiment_name="piperx_cleanup_vision"),
  runner_cls=PickPlaceOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Cleanup-PiperX-Distill",
  env_cfg=make_pick_place_env_cfg(vision=True, num_objects=3),
  play_env_cfg=make_pick_place_env_cfg(play=True, vision=True, num_objects=3),
  rl_cfg=pick_place_distill_runner_cfg(experiment_name="piperx_cleanup_distill"),
  runner_cls=PickPlaceDistillationRunner,
)


# ---------------------------------------------------------------------------
# The action convention before 2026-09-05: scale = PICK_ARM_SCALE about the
# home pose, a Gaussian head with nothing bounding it.  Every checkpoint from
# before that date (v3-v9 teachers, v4-v9 students, the deployed d455_v4_final)
# was trained under it and can only be evaluated under it; the default ids
# above now raise at the first step if such a policy is loaded into them.
# ---------------------------------------------------------------------------
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-V1",
  env_cfg=make_pick_place_env_cfg(bounded_actions=False),
  play_env_cfg=make_pick_place_env_cfg(play=True, bounded_actions=False),
  rl_cfg=pick_place_ppo_runner_cfg(bounded=False),
  runner_cls=PickPlaceOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Robust-V1",
  env_cfg=make_robust_env_cfg(bounded_actions=False),
  play_env_cfg=make_robust_env_cfg(play=True, bounded_actions=False),
  rl_cfg=pick_place_ppo_runner_cfg(
    experiment_name="piperx_pick_place_robust", bounded=False),
  runner_cls=PickPlaceOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Vision-V1",
  env_cfg=make_pick_place_env_cfg(vision=True, bounded_actions=False),
  play_env_cfg=make_pick_place_env_cfg(play=True, vision=True, bounded_actions=False),
  rl_cfg=pick_place_vision_ppo_runner_cfg(bounded=False),
  runner_cls=PickPlaceOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Vision-Robust-V1",
  env_cfg=make_robust_env_cfg(vision=True, bounded_actions=False),
  play_env_cfg=make_robust_env_cfg(play=True, vision=True, bounded_actions=False),
  rl_cfg=pick_place_vision_ppo_runner_cfg(
    experiment_name="piperx_pick_place_vision_robust", bounded=False),
  runner_cls=PickPlaceOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Distill-V1",
  env_cfg=make_pick_place_env_cfg(vision=True, bounded_actions=False),
  play_env_cfg=make_pick_place_env_cfg(play=True, vision=True, bounded_actions=False),
  rl_cfg=pick_place_distill_runner_cfg(bounded=False),
  runner_cls=PickPlaceDistillationRunner,
)
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Distill-Robust-V1",
  env_cfg=make_robust_env_cfg(vision=True, bounded_actions=False),
  play_env_cfg=make_robust_env_cfg(play=True, vision=True, bounded_actions=False),
  rl_cfg=pick_place_distill_runner_cfg(
    experiment_name="piperx_pick_place_distill_robust", bounded=False),
  runner_cls=PickPlaceDistillationRunner,
)


# ---------------------------------------------------------------------------
# Cold-start teachers (2026-09-05): the -Robust task under a capability-gated
# curriculum (cold_curriculum.py).  Training only; play mode is the full
# -Robust domain, so a checkpoint from here is evaluated on -Robust.
# ---------------------------------------------------------------------------
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Robust-Cold",
  env_cfg=make_cold_env_cfg(sight=True),
  play_env_cfg=make_cold_env_cfg(sight=True, play=True),
  rl_cfg=pick_place_ppo_runner_cfg(experiment_name="piperx_pick_place_robust_cold", max_iterations=9000),
  runner_cls=PickPlaceOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Robust-Cold-NoSight",
  env_cfg=make_cold_env_cfg(sight=False),
  play_env_cfg=make_cold_env_cfg(sight=False, play=True),
  rl_cfg=pick_place_ppo_runner_cfg(experiment_name="piperx_pick_place_robust_cold", max_iterations=9000),
  runner_cls=PickPlaceOnPolicyRunner,
)

# v10d: the cold-start curriculum plus the approach terms (slow arrival, object
# undisturbed, top-down wrist).  Play mode is -Robust plus the same three
# terms at zero weight, purely so they are logged during evaluation.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Robust-Cold2",
  env_cfg=make_cold_env_cfg(sight=True, approach=True),
  play_env_cfg=make_cold_env_cfg(sight=True, play=True, approach=True),
  rl_cfg=pick_place_ppo_runner_cfg(experiment_name="piperx_pick_place_robust_cold2", max_iterations=9000),
  runner_cls=PickPlaceOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Robust-Cold2-NoSight",
  env_cfg=make_cold_env_cfg(sight=False, approach=True),
  play_env_cfg=make_cold_env_cfg(sight=False, play=True, approach=True),
  rl_cfg=pick_place_ppo_runner_cfg(experiment_name="piperx_pick_place_robust_cold2", max_iterations=9000),
  runner_cls=PickPlaceOnPolicyRunner,
)
