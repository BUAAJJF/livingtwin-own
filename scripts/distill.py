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
  p.add_argument("--iterations", type=int, default=1500,
                 help="target total iterations, not additional ones: resuming "
                      "at 1600 with --iterations 3000 runs 1400 more")
  p.add_argument("--run-name", default="")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--cadence", default=None,
                 help="randomisation cadence to TRAIN under: 'object' (every "
                      "object is a new object), 'episode' (one object per "
                      "episode, re-posed), or a comma-separated subset of "
                      "shape,mass,friction.  Unset leaves the task alone.")
  p.add_argument("--sensor", default="measured",
                 choices=("measured", "clean"),
                 help="depth realism to train under.  'measured' is the fitted "
                      "active-stereo model selected by piper_push.camera; "
                      "'clean' turns it "
                      "off entirely and is the control that says how much of "
                      "the difference the model is responsible for")
  p.add_argument("--episode-length-s", type=float, default=None,
                 help="override the task's episode length.  The arm never "
                      "resets, and the teachers were found to stop working "
                      "part way through an episode three times longer than "
                      "the 12 s they were trained on -- a horizon the student "
                      "must therefore also be rolled out over, or it inherits "
                      "the blind spot instead of the behaviour.")
  p.add_argument("--log-root", default="logs/rsl_rl")
  p.add_argument("--logger", default="tensorboard", choices=("wandb", "tensorboard"))
  from piper_push import evalcfg as _evalcfg  # noqa: E402
  _evalcfg.add_action_api_arg(p)
  a = p.parse_args()

  _evalcfg.apply_action_api_arg(a)
  if not a.teacher and not a.resume:
    p.error("pass --teacher to start, or --resume to continue")

  configure_torch_backends()

  env_cfg = load_env_cfg(a.task)
  agent_cfg = load_rl_cfg(a.task)
  env_cfg.scene.num_envs = a.num_envs
  if a.episode_length_s is not None:
    env_cfg.episode_length_s = float(a.episode_length_s)
    print(f"[INFO] episode length: {env_cfg.episode_length_s} s")
  env_cfg.seed = a.seed
  agent_cfg.seed = a.seed
  if a.cadence is not None:
    from piper_push.shapes import ALL_QUANTITIES
    redraw = (ALL_QUANTITIES if a.cadence == "object"
              else () if a.cadence == "episode"
              else tuple(s.strip() for s in a.cadence.split(",") if s.strip()))
    unknown = set(redraw) - set(ALL_QUANTITIES)
    if unknown:
      p.error(f"--cadence: unknown {sorted(unknown)}")
    env_cfg.commands["pick"].redraw_on_place = redraw
    env_cfg.commands["pick"].reshape_on_place = bool(redraw)
    print(f"[INFO] training cadence: redraw_on_place={redraw}")
  if a.sensor == "clean":
    # The control.  Not "less noise" -- none, so that the comparison is
    # between a policy that was shown the sensor and one that was not, rather
    # than between two guesses about it.
    import dataclasses as _dc
    term = env_cfg.observations["camera"].terms["scene"]
    term.params["noise_cfg"] = _dc.replace(term.params["noise_cfg"],
                                           strength=0.0)
    term.params["mask_jitter_px"] = 0
  print(f"[INFO] depth realism: {a.sensor}")

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

  # Counted as a target, not a budget.  rsl_rl's learn() runs N iterations
  # *from where it is*, so a run resumed at 1600 and asked for 3000 stops at
  # 4600 -- which is how the state campaign quietly turned 3500 into 5300.
  remaining = a.iterations - runner.current_learning_iteration
  if remaining <= 0:
    print(f"[INFO] already at iteration {runner.current_learning_iteration}; "
          f"nothing to do for a target of {a.iterations}")
    env.close()
    return 0
  runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
  env.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
