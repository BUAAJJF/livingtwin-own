"""PPO fine-tuning on top of the distilled vision policy.

Its own entry point for the same reason ``distill.py`` is: the weights this
starts from live in two different experiments.  The actor comes from the
distillation run; the critic comes from the *state* policy, because the vision
critic reads exactly the observation groups the state critic was trained on --
``full_proprio``, ``object``, ``privileged`` -- and estimating the value of the
same task from the same privileged state is the same problem.

That second load is the point of this script.  PPO with a good actor and a
random critic spends its first hundred iterations computing advantages from
noise and updating the actor with them, and a policy that took three hours to
distill can be destroyed in ten minutes.

    python scripts/finetune.py \\
        --student logs/rsl_rl/piperx_pick_place_distill/<run>/model_3000.pt \\
        --critic  logs/rsl_rl/piperx_pick_place/<run>/model_3400.pt \\
        --num-envs 1024 --iterations 3000 --run-name f1 --device cuda:0
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags

TASK = "Mjlab-Pick-Place-PiperX-Vision"


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default=TASK)
  p.add_argument("--student", help="distillation checkpoint to start the actor from")
  p.add_argument("--critic", help="state policy checkpoint to start the critic from")
  p.add_argument("--resume", help="fine-tuning checkpoint to continue from")
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--iterations", type=int, default=3000)
  p.add_argument("--init-std", type=float, default=0.3,
                 help="action std to restart exploration at; see below")
  p.add_argument("--run-name", default="")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--log-root", default="logs/rsl_rl")
  p.add_argument("--logger", default="wandb", choices=("wandb", "tensorboard"))
  a = p.parse_args()

  if not a.resume and not a.student:
    p.error("pass --student (and ideally --critic) to start, or --resume to continue")

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

  runner_cls = load_runner_cls(a.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, cfg_dict, str(log_dir), a.device)
  add_wandb_tags(agent_cfg.wandb_tags)
  runner.add_git_repo_to_log(__file__)

  if a.resume:
    print(f"[INFO] resuming from {a.resume}")
    runner.load(a.resume, map_location=a.device)
  else:
    print(f"[INFO] actor from {a.student}")
    runner.load(a.student, load_cfg={"actor": True, "iteration": False},
                strict=True, map_location=a.device)
    if a.critic:
      print(f"[INFO] critic from {a.critic}")
      runner.load(a.critic, load_cfg={"critic": True, "iteration": False},
                  strict=True, map_location=a.device)
    else:
      print("[WARN] no --critic: PPO will compute its first advantages from a "
            "randomly initialised value function and update the distilled "
            "actor with them.")

    # Distillation regresses the mean and never touches the standard deviation,
    # so the actor arrives still carrying the 0.6 it was initialised with --
    # +-0.6 of a 0.3-0.5 rad action scale, on a policy that already knows what
    # it is doing.  Exploration has to restart somewhere, but not there.
    if a.init_std is not None:
      actor = runner.alg.get_policy()
      with torch.no_grad():
        if hasattr(actor.distribution, "std_param"):
          actor.distribution.std_param.fill_(a.init_std)
        elif hasattr(actor.distribution, "log_std_param"):
          actor.distribution.log_std_param.fill_(float(torch.log(torch.tensor(a.init_std))))
        else:
          raise RuntimeError(
            f"{type(actor.distribution).__name__} exposes no std parameter to set"
          )
      print(f"[INFO] action std reset to {a.init_std}")

  runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=True)
  env.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
