"""Teacher-student distillation, which mjlab does not ship.

rsl_rl 5.4.2 has the algorithm (``rsl_rl.algorithms.Distillation``) and a
runner for it, but mjlab only defines ``RslRlOnPolicyRunnerCfg`` -- there is no
config dataclass with ``student``/``teacher`` fields and nothing wires a
non-PPO algorithm through.  This module is the missing half.

What the algorithm does, because the name undersells it: the *student* acts in
the environment and the teacher only labels what the student visited.  That is
DAgger, not behaviour cloning off a teacher's trajectories, and it is the
reason a vision student can recover from states the state teacher would never
have entered -- which is most of them, since the student starts blind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlModelCfg
from rsl_rl.algorithms import Distillation


@dataclass
class RslRlDistillationAlgorithmCfg:
  """Config for ``rsl_rl.algorithms.Distillation``."""

  num_learning_epochs: int = 1
  """Passes over the rollout.  One, because the data is on-policy and the next
  iteration collects fresh states from a slightly better student; replaying it
  buys less than moving on."""
  gradient_length: int = 16
  """Timesteps of truncated backpropagation through time per optimizer step.

  Keep ``num_steps_per_env`` an exact multiple of this.  The update accumulates
  loss across timesteps and only steps when the counter divides evenly, so a
  remainder is collected, backpropagated into nothing, and thrown away."""
  learning_rate: float = 5.0e-4
  """Lower than PPO's 1e-3: this is supervised regression through a
  convolutional encoder and a GRU, not a policy-gradient step."""
  max_grad_norm: float | None = 1.0
  loss_type: Literal["mse", "huber"] = "mse"
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  class_name: str = "Distillation"


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """What ``RslRlOnPolicyRunnerCfg`` is for PPO, for distillation.

  The field names are load-bearing: ``Distillation.construct_algorithm`` reads
  ``cfg["student"]``, ``cfg["teacher"]`` and ``cfg["obs_groups"]`` with the
  default sets ``["student", "teacher"]``, so the runner config has to present
  exactly those keys once it is turned into a dict.
  """

  class_name: str = "DistillationRunner"
  student: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
      }
    )
  )
  teacher: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  algorithm: RslRlDistillationAlgorithmCfg = field(
    default_factory=RslRlDistillationAlgorithmCfg
  )


class PickPlaceDistillationRunner(MjlabOnPolicyRunner):
  """mjlab's runner, pointed at the student and the teacher.

  Two things have to change.  ``MjlabOnPolicyRunner`` strips unset optional
  model fields for the keys ``actor`` and ``critic``; a distillation config has
  neither, and an unset ``cnn_cfg=None`` reaching ``MLPModel.__init__`` is a
  ``TypeError``.  And a distillation run with no teacher loaded trains the
  student to imitate a randomly initialised network, silently and for as long
  as you let it, so the run refuses to start without one.
  """

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    for key in ("student", "teacher"):
      if key not in train_cfg:
        continue
      for opt in ("cnn_cfg", "distribution_cfg"):
        if train_cfg[key].get(opt) is None:
          train_cfg[key].pop(opt, None)
      if train_cfg[key].get("rnn_type") is None:
        for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
          train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if not getattr(self.alg, "teacher_loaded", False):
      raise ValueError(
        "No teacher was loaded, so there is nothing to distill.  Pass the "
        "state policy's checkpoint with --teacher."
      )
    super().learn(num_learning_iterations, init_at_random_ep_len)


class BoundedDistillation(Distillation):
  """rsl_rl's Distillation with the behaviour-cloning loss taken on ``tanh``.

  Under the bounded convention both networks emit the pre-squash ``u`` and the
  arm receives ``tanh(u)``.  Regressing on ``u`` would spend the student's
  capacity matching how far past saturation the teacher sits, which is no
  behaviour at all -- the old convention's failure mode in new clothes.  The
  loss is therefore on what the arm receives.
  """

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    base = self.loss_fn

    def on_tanh(student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
      return base(torch.tanh(student_out), torch.tanh(teacher_out))

    self.loss_fn = on_tanh
