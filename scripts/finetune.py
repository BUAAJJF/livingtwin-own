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

from piper_push import damping, hidden_plant, latency, perturb, residual
from piper_push.checkpoints import as_actor_checkpoint

TASK = "Mjlab-Pick-Place-PiperX-Vision"


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default=TASK)
  p.add_argument("--student", help="distillation checkpoint to start the actor from")
  p.add_argument("--critic", help="state policy checkpoint to start the critic from")
  p.add_argument("--resume", help="fine-tuning checkpoint to continue from")
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--iterations", type=int, default=3000,
                 help="target total iterations, not additional ones: resuming "
                      "at 1600 with --iterations 3000 runs 1400 more")
  # Fine-tuning hyper-parameters, and every one of them is lower than the
  # from-scratch value it replaces.  The first attempt used the training
  # numbers and destroyed the distilled policy in a single iteration: the
  # weights moved 4.4e-3 on tensors whose RMS is 5e-2, and a policy that placed
  # 49 objects a minute placed none.  Adam's first step moves every parameter
  # by roughly the learning rate no matter how small the gradient is, and there
  # are twenty of them per iteration.
  p.add_argument("--init-std", type=float, default=0.15,
                 help="action std to restart exploration at; see below")
  p.add_argument("--learning-rate", type=float, default=1.0e-4)
  p.add_argument("--desired-kl", type=float, default=0.005,
                 help="per-update KL the adaptive schedule aims at")
  p.add_argument("--entropy-coef", type=float, default=0.002,
                 help="from-scratch training wants exploration; a fine-tune "
                      "mostly wants the entropy bonus to stop pushing the "
                      "policy back towards random")
  p.add_argument("--critic-warmup", type=int, default=100,
                 help="iterations to train the value function alone before "
                      "the actor is allowed to move")
  p.add_argument("--run-name", default="")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--cadence", default=None,
                 help="randomisation cadence to TRAIN under: 'object' (every "
                      "object is a new object), 'episode' (one object per "
                      "episode, re-posed), or a comma-separated subset of "
                      "shape,mass,friction.  Unset leaves the task alone.")
  p.add_argument("--episode-length-s", type=float, default=None,
                 help="override the task's episode length.  The arm never "
                      "resets, and the teachers were found to stop working "
                      "part way through an episode three times longer than "
                      "the 12 s they were trained on -- a horizon the student "
                      "must therefore also be rolled out over, or it inherits "
                      "the blind spot instead of the behaviour.")
  p.add_argument("--log-root", default="logs/rsl_rl")
  p.add_argument("--logger", default="wandb", choices=("wandb", "tensorboard"))
  perturb.add_mismatch_args(p)
  latency.add_latency_args(p)
  damping.add_damping_args(p)
  hidden_plant.add_hidden_target_args(p)
  residual.add_residual_args(p)
  a = p.parse_args()

  if not a.resume and not a.student:
    p.error("pass --student (and ideally --critic) to start, or --resume to continue")

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

  # Session-persistent simulator mismatch to TRAIN in.  This is what makes the
  # Phase WM0 known-parameter reference possible: fine-tune with the target
  # parameters supplied, to measure what a learned calibration is being
  # compared against.  It is a reference point and not a ceiling -- WM1-A's
  # posterior-guided run beat it in both domains.
  mismatch = perturb.mismatch_from_args(a)
  applied_mismatch = perturb.apply_session_mismatch(env_cfg, mismatch)
  if applied_mismatch:
    print(f"[INFO] training under session mismatch: {applied_mismatch}")

  # Posterior-guided adaptation (Phase WM1): train under a *distribution* over
  # observation delay rather than a single value.  The oracle above is the
  # special case q = delta(3); passing --latency-probs is how a method's own
  # posterior, already mixed with the source prior, gets into the simulator.
  prior = latency.prior_from_args(a)
  applied_prior = latency.apply_latency_prior(env_cfg, prior, seed=a.seed)
  if applied_prior:
    print(f"[INFO] training under latency prior: {prior.probs} "
          f"(mean {prior.mean_lag * latency.STEP_MS:.0f} ms)")
  if applied_prior and mismatch.obs_latency_steps:
    p.error("--latency-probs and --obs-latency-steps both set the same axis")

  # Phase WM1-B's axis, installed the same way.  A point mass at nominal
  # returns {} and leaves the config untouched, so a latency-only run is
  # byte-identical to what it was before this axis existed.
  dprior = damping.prior_from_args(a)
  applied_damping = damping.apply_damping_prior(
    env_cfg, dprior, seed=getattr(a, "damping_seed", 0) or a.seed)
  if applied_damping:
    print(f"[INFO] training under servo-damping prior: {dprior.probs} "
          f"over {damping.VALUES}")
  if applied_damping and mismatch.servo_damping_scale != 1.0:
    p.error("--damping-probs and --servo-damping-scale both set the same axis")
  agent_cfg.max_iterations = a.iterations
  agent_cfg.run_name = a.run_name
  agent_cfg.logger = a.logger
  agent_cfg.algorithm.learning_rate = a.learning_rate
  agent_cfg.algorithm.desired_kl = a.desired_kl
  agent_cfg.algorithm.entropy_coef = a.entropy_coef

  # Phase RA-Sim-0: the structural target and the learned residual, both off
  # unless explicitly asked for.  They are command-path hooks, so they compose
  # with the parametric axes above rather than replacing them -- an arm that
  # trains in "the residual-augmented simulator" is nominal physics plus the
  # residual, and the oracle arm is nominal physics plus the hidden target.
  applied_hidden = hidden_plant.apply_hidden_plant(
    env_cfg, hidden_plant.hidden_from_args(a))
  if applied_hidden:
    print(f"[INFO] training under the hidden structural target: {applied_hidden}")
  applied_residual = residual.apply_residual(
    env_cfg, residual.residual_from_args(a))
  if applied_residual:
    print(f"[INFO] training under the residual: {applied_residual}")

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
    # Accepts a distillation checkpoint directly; the conversion is one key
    # rename and making the caller remember it is a way to lose an afternoon.
    actor_path = as_actor_checkpoint(a.student, log_dir / "actor_from_student.pt")
    print(f"[INFO] actor from {a.student}")
    runner.load(actor_path, load_cfg={"actor": True, "iteration": False},
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

  # Counted as a target, not a budget.  rsl_rl's learn() runs N iterations
  # *from where it is*, so a run resumed at 1600 and asked for 3000 stops at
  # 4600 -- which is how the state campaign quietly turned 3500 into 5300.
  # The value function loaded from the state run estimates the value of the
  # *state* policy's behaviour, which is better than the student's.  Letting
  # the actor move before that gap closes updates it from advantages that
  # measure the wrong thing.  So the actor is held still until the critic has
  # caught up, and the learning-rate schedule is pinned while it is: with the
  # policy frozen the measured KL is zero, and an adaptive schedule reads zero
  # KL as permission to raise the rate.
  if a.critic_warmup > 0 and not a.resume:
    alg = runner.alg
    actor_params = [q for q in alg.actor.parameters()]
    inner_update = alg.update
    warmup_until = runner.current_learning_iteration + a.critic_warmup

    def update_with_warmup():
      warming = runner.current_learning_iteration < warmup_until
      for q in actor_params:
        q.requires_grad_(not warming)
      saved = alg.schedule
      if warming:
        alg.schedule = "fixed"
      try:
        losses = inner_update()
      finally:
        alg.schedule = saved
      return losses

    alg.update = update_with_warmup
    print(f"[INFO] critic-only for the first {a.critic_warmup} iterations")

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
