"""A stateful actuator model that cannot produce a command the arm could not.

Phase RA-Sim-0 tried ``u_eff = u + clip(R(history), +-DELTA_MAX)`` and found
the bound was the binding constraint at every setting it swept: the 99th
percentile of the correction sat exactly on the bound from 0.05 rad to 0.30,
and the model only beat a parameter fit once the correction averaged **3.3x
the commanded step** -- at which point it is not correcting the command, it is
replacing it, and one run in fourteen produced non-finite states.

The diagnosis was that an *additive, memoryless, small* correction is the
wrong parameterisation for a mismatch whose signature is accumulated lag.  A
real actuator does not add an offset to its target; it **chases** it, at a
finite rate, from wherever it currently is.  So this module keeps the thing an
actuator keeps -- its own current position command -- and learns the four
numbers that govern how it chases:

    error_t   = u_t - w_t                       w = the actuator's own command
    a, r+, r-, b = net(s_t, deployable history)  all mapped into physical ranges
    delta_t   = clip(a * (error_t - b), -r- * dt, +r+ * dt)
    u_eff,t   = clip(w_t + delta_t, joint_lo, joint_hi)
    w_{t+1}   = u_eff,t

Three consequences follow from the shape rather than from a penalty:

* **The effective command is always inside the joint's command range**, because
  the last line clips it there.
* **It can never move faster than a real drive**, because ``r+`` and ``r-`` are
  mapped into a pre-registered rad/s range and multiplied by the control
  period.  A large *accumulated* lag is reachable; a large *instantaneous*
  jump is not.
* **It is a contraction towards the command** whenever ``a`` is in (0, 1] and
  ``b`` is bounded, so the state cannot run away on its own.

Ordering note.  The phase's specification writes the recursion as
``u_eff,t+1 = u_eff,t + delta_t``, i.e. emit-then-update.  That form cannot be
identity-initialised: with a = 1 and no rate limit it emits ``u_{t-1}`` at
step t, which is a one-step transport delay and a change to the simulator.
The update is therefore done *within* the step and the result emitted, which
is the same recursion with the index shifted and is the only ordering under
which "model off" and "model at initialisation" are the same simulator.  The
audit measures that rather than assuming it.

**What the model may see** is exactly what
:func:`piper_push.residual.build_features` allowed -- measured joint positions
and velocities, the commands the controller issued, and their difference --
plus the model's *own* actuator state, which is not a hidden quantity because
the model produced it.  Not the hidden target's effective command, not its
backlash flank, not its formula, not reward, success or the safety label.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

N_JOINTS = 6

# -- pre-registered physical ranges (docs/ra_sim1_experiment_plan.md) --------
ALPHA_RANGE = (0.02, 1.0)
"""Fraction of the tracking error the drive closes in one control period.
1.0 is a servo that lands on its target within the period; 0.02 is one that
takes two and a half seconds to."""

RATE_RANGE_RAD_S = (0.05, 4.0)
"""How fast the drive may move its own position command, rad/s, per direction.
The upper end is this arm's safety-shell trip speed (3.1-3.9 rad/s across the
joints, rounded up): an actuator cannot slew its target faster than the joint
it drives can turn.  The lower end is a nearly stalled drive.  Both directions
are learned separately, which is what lets an asymmetry be represented."""

BIAS_RANGE_RAD = (-0.05, 0.05)
"""A directional offset in the error the drive is chasing.  Backlash presents
this way -- the flank the teeth rest on shifts the effective target -- but
nothing here is told that; it is a general offset the state may switch."""

FEATURE_DIM = 6 * N_JOINTS
"""``[q, qdot, u, u_prev, q - u, u - w]`` for the six arm joints."""


def build_features(q: torch.Tensor, qd: torch.Tensor, u: torch.Tensor,
                   u_prev: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
  """The deployable view plus the model's own state, and nothing else.

  A free function, and the only builder, for the same reason RA-Sim-0's was:
  the offline trainer and the in-simulator hook must construct the same vector
  from the same quantities or the model is trained on one thing and run on
  another.
  """
  return torch.cat([q, qd, u, u_prev, q - u, u - w], dim=-1)


def _map(z: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
  return lo + (hi - lo) * torch.sigmoid(z)


class StableActuator(nn.Module):
  """GRU over the deployable history, four bounded coefficients per joint.

  Small on purpose: the claim under test is that the *shape* of the model
  matters, not that a large network can memorise a trajectory distribution.
  """

  # Head-bias offsets that make the model the identity at initialisation.
  # sigmoid(14) differs from 1 by 8.3e-7, so alpha starts at 1 - 8.3e-7 and
  # the deviation from a pure pass-through is ~4e-8 rad on a 0.05 rad error --
  # two orders of magnitude under the 3.6e-7 rad the simulator disagrees with
  # itself by across two builds.  Exactly 1.0 is not reachable through a
  # sigmoid and pretending otherwise would be worse than measuring it.
  ALPHA_INIT_LOGIT = 14.0
  RATE_INIT_LOGIT = 14.0

  def __init__(self, hidden: int = 64, dt: float = 0.02,
               command_lo: tuple[float, ...] | None = None,
               command_hi: tuple[float, ...] | None = None,
               alpha_range: tuple[float, float] = ALPHA_RANGE,
               rate_range: tuple[float, float] = RATE_RANGE_RAD_S,
               bias_range: tuple[float, float] = BIAS_RANGE_RAD) -> None:
    super().__init__()
    self.hidden_size = hidden
    self.dt = float(dt)
    self.alpha_range = alpha_range
    self.rate_range = rate_range
    self.bias_range = bias_range
    self.gru = nn.GRUCell(FEATURE_DIM, hidden)
    self.head = nn.Linear(hidden, 4 * N_JOINTS)
    nn.init.zeros_(self.head.weight)
    with torch.no_grad():
      b = torch.zeros(4 * N_JOINTS)
      b[0 * N_JOINTS:1 * N_JOINTS] = self.ALPHA_INIT_LOGIT   # alpha -> 1
      b[1 * N_JOINTS:2 * N_JOINTS] = self.RATE_INIT_LOGIT    # rate+ -> max
      b[2 * N_JOINTS:3 * N_JOINTS] = self.RATE_INIT_LOGIT    # rate- -> max
      b[3 * N_JOINTS:4 * N_JOINTS] = 0.0                     # bias  -> 0
      self.head.bias.copy_(b)
    self.register_buffer("x_mean", torch.zeros(FEATURE_DIM))
    self.register_buffer("x_std", torch.ones(FEATURE_DIM))
    lo = torch.tensor(command_lo if command_lo is not None
                      else (-1e9,) * N_JOINTS)
    hi = torch.tensor(command_hi if command_hi is not None
                      else (1e9,) * N_JOINTS)
    self.register_buffer("cmd_lo", lo)
    self.register_buffer("cmd_hi", hi)

  def set_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
    self.x_mean.copy_(mean)
    self.x_std.copy_(std.clamp_min(1e-6))

  def coefficients(self, feat: torch.Tensor, h: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                              torch.Tensor, torch.Tensor]:
    h = self.gru((feat - self.x_mean) / self.x_std, h)
    z = self.head(h)
    a = _map(z[..., 0 * N_JOINTS:1 * N_JOINTS], *self.alpha_range)
    rp = _map(z[..., 1 * N_JOINTS:2 * N_JOINTS], *self.rate_range)
    rn = _map(z[..., 2 * N_JOINTS:3 * N_JOINTS], *self.rate_range)
    b = _map(z[..., 3 * N_JOINTS:4 * N_JOINTS], *self.bias_range)
    return a, rp, rn, b, h

  def step(self, q, qd, u, u_prev, w, h):
    """One control step.  Returns ``(u_eff, w_next, h_next, diagnostics)``."""
    feat = build_features(q, qd, u, u_prev, w)
    a, rp, rn, b, h = self.coefficients(feat, h)
    delta = a * ((u - w) - b)
    delta = torch.maximum(torch.minimum(delta, rp * self.dt), -rn * self.dt)
    u_eff = torch.clamp(w + delta, self.cmd_lo, self.cmd_hi)
    return u_eff, u_eff, h, {"alpha": a, "rate_pos": rp, "rate_neg": rn,
                             "bias": b, "delta": delta}

  def zero_hidden(self, n: int, device) -> torch.Tensor:
    return torch.zeros(n, self.hidden_size, device=device)


@dataclass
class ActuatorHookCfg:
  """Install a trained actuator model on the arm's command path."""

  checkpoint: str
  enabled: bool = True
  """``False`` leaves the simulator untouched.  Present so that "off" is a
  configuration rather than the absence of one."""

  def build(self, action_term) -> "ActuatorHook":
    return ActuatorHook(self, action_term)

  def to_json(self) -> dict:
    return {"checkpoint": self.checkpoint, "enabled": self.enabled,
            "kind": "stable_stateful_actuator"}


class ActuatorHook:
  """Runs the actuator model once per control step, batched over environments.

  Carries three per-environment states: the GRU's hidden vector, the
  actuator's own command ``w``, and the previous issued command.  All three
  are reset per environment at the episode boundary, and the reset seeds ``w``
  from the fresh posture so a new episode does not start by unwinding the last
  one's lag.
  """

  def __init__(self, cfg: ActuatorHookCfg, action_term) -> None:
    dev = action_term.device
    blob = torch.load(cfg.checkpoint, map_location=dev, weights_only=False)
    model = StableActuator(hidden=blob["hidden"], dt=blob["dt"],
                           command_lo=tuple(blob["cmd_lo"]),
                           command_hi=tuple(blob["cmd_hi"]),
                           alpha_range=tuple(blob["alpha_range"]),
                           rate_range=tuple(blob["rate_range"]),
                           bias_range=tuple(blob["bias_range"]))
    model.load_state_dict(blob["state_dict"])
    model.to(dev).eval()
    for p in model.parameters():
      p.requires_grad_(False)
    if action_term._num_targets != N_JOINTS:
      raise ValueError(
        f"the actuator model is defined on {N_JOINTS} arm joints, but this "
        f"action term commands {action_term._num_targets}")
    self.cfg = cfg
    self.model = model
    self._term = action_term
    n = action_term._default.shape[0]
    self._h = model.zero_hidden(n, dev)
    self._w = action_term._default.clone()
    self._u_prev = action_term._default.clone()
    # Running diagnostics, for the stability gate.  Counted, not asserted:
    # a limit that never binds is not the reason anything happened.
    self.stats = {k: 0.0 for k in
                  ("steps", "nonfinite", "clipped_lo", "clipped_hi",
                   "rate_clipped", "max_abs_delta", "max_hidden_norm",
                   "max_abs_command_lag", "sum_abs_delta")}

  @torch.no_grad()
  def __call__(self, target: torch.Tensor, action_term) -> torch.Tensor:
    if not self.cfg.enabled:
      return target
    m = self.model
    u_eff, w, h, d = m.step(action_term.joint_pos, action_term.joint_vel,
                            target, self._u_prev, self._w, self._h)
    self._h = h
    self._w.copy_(w)
    self._u_prev.copy_(target)
    s = self.stats
    s["steps"] += 1
    s["nonfinite"] += float((~torch.isfinite(u_eff)).sum())
    s["clipped_lo"] += float((u_eff <= m.cmd_lo + 1e-9).sum())
    s["clipped_hi"] += float((u_eff >= m.cmd_hi - 1e-9).sum())
    s["rate_clipped"] += float(
      ((d["delta"].abs() >= torch.minimum(d["rate_pos"], d["rate_neg"])
        * m.dt - 1e-9)).sum())
    s["max_abs_delta"] = max(s["max_abs_delta"], float(d["delta"].abs().max()))
    s["sum_abs_delta"] += float(d["delta"].abs().mean())
    s["max_hidden_norm"] = max(s["max_hidden_norm"], float(h.norm(dim=-1).max()))
    s["max_abs_command_lag"] = max(s["max_abs_command_lag"], float((target - u_eff).abs().max()))
    self.last = d
    return u_eff

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._h[env_ids] = 0.0
    rest = self._term._previous_target[env_ids]
    self._w[env_ids] = rest
    self._u_prev[env_ids] = rest


def apply_actuator(env_cfg, cfg: ActuatorHookCfg | None) -> dict:
  """Install the actuator model on the arm.  ``None`` changes nothing."""
  if cfg is None:
    return {}
  arm = env_cfg.actions["arm"]
  arm.command_hooks = tuple(arm.command_hooks) + (cfg,)
  return cfg.to_json()


def add_actuator_args(parser) -> None:
  parser.add_argument(
    "--actuator", default="",
    help="path to a stable actuator checkpoint from scripts/ra_sim1_train.py. "
         "Empty leaves the simulator exactly as it is.")


def actuator_from_args(args) -> ActuatorHookCfg | None:
  path = getattr(args, "actuator", "")
  return ActuatorHookCfg(checkpoint=path) if path else None
