"""A learned correction to the arm's position command, and the hook that runs it.

The residual does not replace the simulator.  MuJoCo keeps integrating,
resolving contact and enforcing joint limits; all this changes is the number
the servo is asked to hold:

    y_t = u_t + clip(R_psi(history), +-DELTA_MAX)

``u_t`` is the position target the controller commanded this step and ``y_t``
what the servo receives.  Nothing here writes ``qpos`` or ``qvel``, adds a
force, or touches the contact solver, so a rollout with the residual installed
is a rollout of the same physics under a different command -- which is what
makes the comparison in Phase RA-Sim-0 a comparison of simulators rather than
of two different integrators.

**What the model may see.**  Only quantities a deployed controller has: the
measured joint positions and velocities, the commands it issued, and the
difference between them.  Not the hidden plant's internal flank or lag state,
not the target's effective command, not reward, success or the safety label.
The feature builder is the single place that could break that rule and it is
tested for it.

**Identity at initialisation.**  The output head is zero-initialised and the
bound is a ``tanh``, so an untrained residual returns exactly zero and an
augmented simulator is bit-for-bit the nominal one until the weights move.
That is what lets "residual off" be a control rather than a separate code
path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

N_JOINTS = 6
FEATURE_DIM = 5 * N_JOINTS
"""``[q, qdot, u, u_prev, q - u]`` for the six arm joints."""

DELTA_MAX = 0.05
"""Radians the residual may move a command by, per joint, per step.

Chosen against the mismatch it has to be able to express, not against
performance: the frozen structural target's backlash band is at most 0.022 rad
and its worst-case first-step shortfall on a 0.1 rad command is about 0.07 rad,
so 0.05 is the same order and cannot become a licence to drive the arm
somewhere the policy did not ask for.  It also bounds what a *wrong* residual
can do, which matters more: an unbounded correction that learns garbage is a
simulator that explodes rather than one that is merely inaccurate.
"""


def build_features(q: torch.Tensor, qd: torch.Tensor, u: torch.Tensor,
                   u_prev: torch.Tensor) -> torch.Tensor:
  """The deployable view, and only it.

  Deliberately a free function taking explicit tensors: the offline trainer
  and the in-simulator hook must build the same vector from the same
  quantities, and the only way to be sure is for there to be one builder.
  """
  return torch.cat([q, qd, u, u_prev, q - u], dim=-1)


class ResidualNet(nn.Module):
  """One GRU over the deployable history, one bounded head.

  Small on purpose.  The claim under test is that a *structural* mismatch
  needs a state-dependent correction, not that a large network can memorise a
  trajectory distribution; 19k parameters can hold a hysteresis and a
  rate-dependent lag and cannot hold much else.
  """

  def __init__(self, hidden: int = 64, delta_max: float = DELTA_MAX) -> None:
    super().__init__()
    self.hidden_size = hidden
    self.gru = nn.GRUCell(FEATURE_DIM, hidden)
    self.head = nn.Linear(hidden, N_JOINTS)
    nn.init.zeros_(self.head.weight)
    nn.init.zeros_(self.head.bias)
    self.delta_max = float(delta_max)
    self.register_buffer("x_mean", torch.zeros(FEATURE_DIM))
    self.register_buffer("x_std", torch.ones(FEATURE_DIM))

  def set_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
    self.x_mean.copy_(mean)
    self.x_std.copy_(std.clamp_min(1e-6))

  def forward(self, feat: torch.Tensor, h: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.gru((feat - self.x_mean) / self.x_std, h)
    return self.delta_max * torch.tanh(self.head(h)), h

  def zero_hidden(self, n: int, device) -> torch.Tensor:
    return torch.zeros(n, self.hidden_size, device=device)


class ResidualEnsemble(nn.Module):
  """Several residuals with different initialisations and data order.

  The mean is what the simulator uses; the spread across members is the
  model's own statement about where it has no idea, which Phase RA-Sim-0's
  accuracy gate asks to be calibrated rather than decorative.
  """

  def __init__(self, n: int = 4, hidden: int = 64,
               delta_max: float = DELTA_MAX) -> None:
    super().__init__()
    self.members = nn.ModuleList(
      [ResidualNet(hidden=hidden, delta_max=delta_max) for _ in range(n)])

  def __len__(self) -> int:
    return len(self.members)

  def forward(self, feat: torch.Tensor, hs: list[torch.Tensor]
              ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    deltas, out = [], []
    for m, h in zip(self.members, hs):
      d, h2 = m(feat, h)
      deltas.append(d)
      out.append(h2)
    stacked = torch.stack(deltas)
    return stacked.mean(0), stacked.std(0), out


@dataclass
class ResidualHookCfg:
  """Install a trained residual on the arm's command path.

  ``checkpoint`` is a file written by ``scripts/ra_sim0_train.py``.  There is
  no in-line weight field on purpose: a configuration that carried its own
  weights would make "which residual produced this result" a question about a
  Python object rather than about a file with a hash.
  """

  checkpoint: str
  scale: float = 1.0
  """Multiplies the correction.  0.0 is the off control and must reproduce the
  nominal simulator exactly, which is asserted in the audit rather than
  assumed."""

  def build(self, action_term) -> "ResidualHook":
    return ResidualHook(self, action_term)

  def to_json(self) -> dict:
    return {"checkpoint": self.checkpoint, "scale": self.scale}


class ResidualHook:
  """Runs the ensemble once per control step, batched over environments."""

  def __init__(self, cfg: ResidualHookCfg, action_term) -> None:
    dev = action_term.device
    blob = torch.load(cfg.checkpoint, map_location=dev, weights_only=False)
    ens = ResidualEnsemble(n=blob["n_members"], hidden=blob["hidden"],
                           delta_max=blob["delta_max"])
    ens.load_state_dict(blob["state_dict"])
    ens.to(dev).eval()
    for p in ens.parameters():
      p.requires_grad_(False)
    if action_term._num_targets != N_JOINTS:
      raise ValueError(
        f"the residual is defined on {N_JOINTS} arm joints, but this action "
        f"term commands {action_term._num_targets}")
    self.cfg = cfg
    self._ens = ens
    self._term = action_term
    self._scale = float(cfg.scale)
    n = action_term._default.shape[0]
    self._h = [ens.members[i].zero_hidden(n, dev) for i in range(len(ens))]
    self._u_prev = action_term._default.clone()
    self.stats = {"steps": 0, "abs_mean": 0.0, "clipped": 0.0}
    # The last step's per-joint disagreement between members.  An evaluation
    # that wants to ask whether the ensemble knows where it is wrong reads
    # this; nothing in the simulator does.
    self.last_spread = torch.zeros_like(action_term._default)
    self.last_delta = torch.zeros_like(action_term._default)

  @torch.no_grad()
  def __call__(self, target: torch.Tensor, action_term) -> torch.Tensor:
    feat = build_features(action_term.joint_pos, action_term.joint_vel,
                          target, self._u_prev)
    delta, spread, self._h = self._ens(feat, self._h)
    self.last_spread.copy_(spread)
    self.last_delta.copy_(delta)
    self._u_prev.copy_(target)
    if self._scale != 1.0:
      delta = delta * self._scale
    self.stats["steps"] += 1
    self.stats["abs_mean"] += float(delta.abs().mean())
    return target + delta

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    for h in self._h:
      h[env_ids] = 0.0
    self._u_prev[env_ids] = self._term._previous_target[env_ids]


def apply_residual(env_cfg, cfg: ResidualHookCfg | None) -> dict:
  """Install the residual on the arm.  ``None`` leaves the simulator alone."""
  if cfg is None:
    return {}
  arm = env_cfg.actions["arm"]
  arm.command_hooks = tuple(arm.command_hooks) + (cfg,)
  return cfg.to_json()


def add_residual_args(parser) -> None:
  parser.add_argument(
    "--residual", default="",
    help="path to a residual checkpoint from scripts/ra_sim0_train.py.  Empty "
         "leaves the simulator exactly as it is.")
  parser.add_argument(
    "--residual-scale", type=float, default=1.0,
    help="multiplies the correction; 0.0 is the off control and must "
         "reproduce the nominal simulator.")


def residual_from_args(args) -> ResidualHookCfg | None:
  path = getattr(args, "residual", "")
  if not path:
    return None
  return ResidualHookCfg(checkpoint=path,
                         scale=float(getattr(args, "residual_scale", 1.0)))
