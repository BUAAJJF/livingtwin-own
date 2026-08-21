"""PPO for the pick-and-place task.

Observation groups are mapped here rather than in the environment, which is
what makes the vision stage cheap: the actor's tuple changes from
``("proprio", "object")`` to ``("proprio", "camera")`` and nothing else moves.
The critic keeps the privileged group either way -- asymmetric actor-critic
costs nothing at training time and is the whole reason to build the groups.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def pick_place_ppo_runner_cfg(
  experiment_name: str = "piperx_pick_place",
  max_iterations: int = 3000,
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # Six joint targets scaled by 0.3-0.5 rad plus a gripper scaled by
        # 25 mm.  std=1.0 would explore +-0.5 rad per step, which the command
        # rate limiter would simply throw away.
        "init_std": 0.6,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # A grasp needs the hand to hold still against the object, so entropy is
      # not free here -- but 0.003 collapsed the action std from 0.6 to 0.13 by
      # iteration 529, before the policy had found that letting go over the bin
      # is worth anything, and a deterministic policy never finds it.
      entropy_coef=0.006,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      desired_kl=0.01,
      gamma=0.99,
      lam=0.95,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    logger="wandb",
    wandb_project="piper-pick-place",
    wandb_tags=("piperx", "pick-place", "state"),
    save_interval=100,
    # 0.64 s of rollout at 50 Hz: long enough to contain an approach and a
    # close, short enough to keep the batch fresh.
    num_steps_per_env=32,
    max_iterations=max_iterations,
    obs_groups={
      "actor": ("proprio", "object"),
      "critic": ("proprio", "object", "privileged"),
    },
  )
