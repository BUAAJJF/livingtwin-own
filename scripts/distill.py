"""Distill the state pick-and-place policy into the vision policy.

Its own entry point rather than mjlab's ``train`` because of where the teacher
lives.  ``train --agent.resume`` resolves a checkpoint *inside the run's own
experiment directory*, and the teacher is a different experiment by
construction -- it was trained on a different task.  Naming the file directly
is both simpler and honest about what is happening: two checkpoints, two
lineages, one of them frozen.

    python scripts/distill.py \\
        --teacher logs/rsl_rl/piperx_pick_place/<run>/model_3400.pt \\
        --num-envs 512 --iterations 1500 --run-name d1 --device cuda:4

To pick a student run back up, pass its own checkpoint instead; that restores
the student, the teacher and the optimizer together:

    python scripts/distill.py --resume logs/rsl_rl/piperx_pick_place_distill/<run>/model_800.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags

TASK = "Mjlab-Pick-Place-PiperX-Distill"


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default=TASK)
  p.add_argument("--teacher", help="state policy checkpoint to imitate")
  p.add_argument("--resume", help="distillation checkpoint to continue from")
  p.add_argument("--num-envs", type=int, default=512)
  p.add_argument("--iterations", type=int, default=1500)
  p.add_argument("--run-name", default="")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--log-root", default="logs/rsl_rl")
  p.add_argument("--logger", default="wandb", choices=("wandb", "tensorboard"))
  a = p.parse_args()

  if not a.teacher and not a.resume:
    p.error("pass --teacher to start, or --resume to continue")

  configure_torch_backends()

  env_cfg = load_env_cfg(a.task)
  agent_cfg = load_rl_cfg(a.task)
  env_cfg.scene.num_envs = a.num_envs
  env_cfg.seed = a.seed
  agent_cfg.seed = a.seed
  agent_cfg.max_iterations = a.iterations
  agent_cfg.run_name = a.run_name
  agent_cfg.logger = a.logger

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if a.run_name:
    stamp += f"_{a.run_name}"
  log_dir = (Path(a.log_root) / agent_cfg.experiment_name / stamp).resolve()
  print(f"[INFO] logging to {log_dir}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  cfg_dict = asdict(agent_cfg)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", cfg_dict)

  runner_cls = load_runner_cls(a.task)
  assert runner_cls is not None, f"{a.task} registered no runner class"
  runner = runner_cls(env, cfg_dict, str(log_dir), a.device)

  add_wandb_tags(agent_cfg.wandb_tags)
  runner.add_git_repo_to_log(__file__)

  if a.resume:
    print(f"[INFO] resuming student from {a.resume}")
    runner.load(a.resume, map_location=a.device)
  else:
    print(f"[INFO] teacher from {a.teacher}")
    # ``strict`` is the point: the teacher config in rl_cfg.py claims to be the
    # same network as the state actor, and this is where that claim is checked.
    runner.load(
      a.teacher, load_cfg={"teacher": True, "iteration": False},
      strict=True, map_location=a.device,
    )

  runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=True)
  env.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
