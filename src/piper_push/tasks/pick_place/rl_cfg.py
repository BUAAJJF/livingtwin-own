"""PPO for the pick-and-place task.

Observation groups are mapped here rather than in the environment, which is
what makes the vision stage cheap: the actor's tuple changes from
``("proprio", "object")`` to ``("proprio", "camera")`` and nothing else moves.
The critic keeps the privileged group either way -- asymmetric actor-critic
costs nothing at training time and is the whole reason to build the groups.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from piper_push.distill import (
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationRunnerCfg,
)


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
      # 0.012: of five schedules the one with this coefficient reached the
      # highest grasp rate (0.984 against 0.91-0.94), so the extra exploration
      # is not costing the delicate part of the task anything.
      entropy_coef=0.012,
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


# Two convolutions and a spatial softmax: the softmax turns the last feature
# maps into coordinates, which is the representation a reaching task actually
# wants and a flattened feature vector makes the network rediscover.
_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_CNN_MODEL = "piper_push.models:SpatialSoftmaxRecurrentModel"


def pick_place_vision_ppo_runner_cfg(
  experiment_name: str = "piperx_pick_place_vision",
  max_iterations: int = 6000,
) -> RslRlOnPolicyRunnerCfg:
  """The same task through the camera.

  Three differences from the state configuration, and only three.  The actor
  reads ``camera`` where it read ``object``; it carries a recurrent layer,
  because the flag that told it whether it was holding something has been taken
  away and the pad contacts alone do not say whether the last squeeze worked;
  and it runs longer, because inferring the object's pose from pixels is a
  harder problem than being handed it.

  The critic is unchanged and still privileged. It never has to run on the
  robot, so there is no reason to make it work through a camera.
  """
  cfg = pick_place_ppo_runner_cfg(experiment_name, max_iterations)
  cfg.actor = RslRlModelCfg(
    hidden_dims=(256, 256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg=_CNN_CFG,
    class_name=_CNN_MODEL,
    rnn_type="gru",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.6,
      "std_type": "scalar",
    },
  )
  cfg.obs_groups = {
    "actor": ("proprio", "camera"),
    # ``full_proprio``, not ``proprio``: the critic never runs on the robot, so
    # handing it the deployment-constrained proprioception costs information
    # for nothing.  It also makes this critic dimensionally and semantically
    # identical to the state task's, which is what lets the fine-tuning stage
    # start from a trained value function instead of a random one.
    "critic": ("full_proprio", "object", "privileged"),
  }
  cfg.wandb_tags = ("piperx", "pick-place", "vision")
  return cfg


def pick_place_distill_runner_cfg(
  experiment_name: str = "piperx_pick_place_distill",
  max_iterations: int = 3000,
) -> RslRlDistillationRunnerCfg:
  """Bootstrap the vision policy off the state policy.

  The student is the same network the vision PPO stage would train from
  scratch -- same convolutions, same GRU, same head -- so the checkpoint this
  produces drops straight into that stage as an initialisation.  That is the
  whole point of doing it in this order: distillation answers "can a camera
  reach this behaviour at all" in a supervised problem with a known target,
  and only then does PPO get asked to improve on it.

  The teacher is byte-identical to ``pick_place_ppo_runner_cfg``'s actor,
  because it is loaded from that actor's weights with ``strict=True``.  Change
  one and this has to change with it.
  """
  ppo = pick_place_ppo_runner_cfg()
  vision = pick_place_vision_ppo_runner_cfg()
  return RslRlDistillationRunnerCfg(
    student=vision.actor,
    teacher=ppo.actor,
    algorithm=RslRlDistillationAlgorithmCfg(
      num_learning_epochs=1,
      # 32 steps of rollout, two optimizer steps of 16.  An uneven split would
      # silently discard the remainder.
      gradient_length=16,
      learning_rate=5.0e-4,
      max_grad_norm=1.0,
      loss_type="mse",
    ),
    experiment_name=experiment_name,
    logger="wandb",
    wandb_project="piper-pick-place",
    wandb_tags=("piperx", "pick-place", "vision", "distill"),
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=max_iterations,
    obs_groups={
      # Byte-identical to the vision PPO actor's tuple, so the student the
      # distillation produces loads into that stage without a rename.
      "student": ("proprio", "camera"),
      "teacher": ("full_proprio", "object"),
    },
  )
