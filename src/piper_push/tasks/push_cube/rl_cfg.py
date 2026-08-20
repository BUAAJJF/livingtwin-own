"""PPO configuration for the push task."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def push_cube_ppo_runner_cfg(
  experiment_name: str = "piperx_push_cube",
  max_iterations: int = 500,
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # Actions are absolute joint targets offset from the home pose and
        # scaled by 0.3-0.5 rad.  std=1.0 would be +/-0.5 rad of exploration
        # noise per step -- far too coarse for centimetre placement.
        "init_std": 0.7,
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
      # Lower than mjlab's manipulation default: here entropy shows up directly
      # as the action jitter the task is meant to minimise.
      entropy_coef=0.002,
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
    wandb_project="piper-push",
    wandb_tags=("piperx", "push", "joint-space"),
    save_interval=50,
    # 0.64 s of rollout at 50 Hz -- about one push stroke.
    num_steps_per_env=32,
    max_iterations=max_iterations,
  )
