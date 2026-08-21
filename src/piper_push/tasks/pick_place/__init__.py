from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import make_pick_place_env_cfg
from .rl_cfg import pick_place_ppo_runner_cfg

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
