from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import make_push_cube_env_cfg
from .rl_cfg import push_cube_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Push-Cube-PiperX",
  env_cfg=make_push_cube_env_cfg(),
  play_env_cfg=make_push_cube_env_cfg(play=True),
  rl_cfg=push_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
