from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import make_pick_place_env_cfg
from piper_push.distill import PickPlaceDistillationRunner

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
  runner_cls=ManipulationOnPolicyRunner,
)

# The smoke variant: one fixed cube in every environment.  A reward bug is
# obvious when every environment holds the same object and invisible under a
# full distribution.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Cube",
  env_cfg=make_pick_place_env_cfg(shape_variety=0.0),
  play_env_cfg=make_pick_place_env_cfg(play=True, shape_variety=0.0),
  rl_cfg=pick_place_ppo_runner_cfg(experiment_name="piperx_pick_place_cube"),
  runner_cls=ManipulationOnPolicyRunner,
)

# Halfway through the shape curriculum: the distribution's midpoint plus half
# its spread.  Registered rather than reachable from the CLI because
# shape_variety shapes the event ranges at build time, not a config field.
register_mjlab_task(
  task_id="Mjlab-Pick-Place-PiperX-Mid",
  env_cfg=make_pick_place_env_cfg(shape_variety=0.5),
  play_env_cfg=make_pick_place_env_cfg(play=True, shape_variety=0.5),
  rl_cfg=pick_place_ppo_runner_cfg(experiment_name="piperx_pick_place_mid"),
  runner_cls=ManipulationOnPolicyRunner,
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
  runner_cls=ManipulationOnPolicyRunner,
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
