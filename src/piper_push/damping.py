"""Servo damping as a domain parameter, drawn per environment.

Phase WM1-B's axis. Phase WM0 measured `servo_damping_scale = 0.75` as costing
4.5% of throughput and multiplying safety-shell trips by 89 — a mismatch whose
whole cost is in the tail, which is the case the risk-aware score exists for
and the case observation latency was not.

**Why this is not `perturb.apply_session_mismatch`.** That path scales the
actuator *configuration* before the entity is built, so every environment gets
the same damping. Posterior-guided adaptation needs a mixture, which means a
different value per environment, which means writing the compiled model's
per-world `actuator_biasprm` instead. mjlab already does exactly that for
continuous randomisation in `dr.pd_gains`; this is the same write with a
categorical draw and no change to the proportional gain.

`scripts/check_damping.py` puts the two paths side by side in one simulator and
checks they produce the same field, because "the same thing by a different
route" is a claim that has to be measured rather than asserted -- Phase WM1-A
made the same claim about two delay implementations and it was true, and the
way to know was to check.

The gripper is deliberately excluded, matching Phase WM0: its closure is a
grasp parameter rather than a servo-tuning one, and the axis WM0 measured left
it alone.
"""

from __future__ import annotations

import torch
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from piper_push import prior

VALUES: tuple[float, ...] = (0.75, 1.0, 1.5)
"""The candidate set.

0.75 is the target: under-damped, the arm overshoots a velocity ramp and trips
the shell.  1.0 is nominal.  1.5 is the **counter-direction control** -- damping
the spec asks for so that a method cannot score by always guessing "less than
nominal", and so that a policy adapted towards 0.75 can be shown to be adapting
to the domain rather than to a direction."""

TARGET = 0.75
"""The hidden target for WM1-B.  Named as one symbol so that "who reads the
answer" is greppable; no estimator imports it."""

NOMINAL = 1.0
COUNTER = 1.5

EVENT_NAME = "wm1b_servo_damping"


class DampingPrior(prior.CategoricalPrior):
  """A categorical distribution over :data:`VALUES`, a multiplier on kd."""

  VALUES = VALUES
  UNIT = "x nominal kd"


P_SOURCE = DampingPrior.point(NOMINAL)
"""What the deployed policy was trained under.

`delta(1.0)`: the training distribution randomises many things and the servo's
damping is not one of them.  Section 2 of the report says so with the config
that proves it."""


@requires_model_fields("actuator_biasprm")
def randomize_servo_damping(
  env,
  env_ids: torch.Tensor | None,
  probs: tuple[float, ...],
  values: tuple[float, ...] = VALUES,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  skip_gripper: bool = True,
) -> None:
  """Draw one damping multiplier per environment and write it into the model.

  A position actuator's ``biasprm`` is ``(0, -kp, -kd)``, so scaling entry 2
  scales the derivative gain and leaves the proportional one alone -- which is
  what ``servo_damping_scale`` means and what Phase WM0 measured.  The write is
  relative to the *default* field rather than to the current one, so repeated
  resets do not compound.
  """
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)
  if env_ids.numel() == 0:
    return

  w = torch.tensor(probs, dtype=torch.float64, device=env.device)
  idx = torch.multinomial(w, int(env_ids.numel()), replacement=True)
  scale = torch.tensor(values, device=env.device, dtype=torch.float32)[idx]

  default_biasprm = env.sim.get_default_field("actuator_biasprm")
  for actuator in asset.actuators:
    names = getattr(actuator.cfg, "target_names_expr", ())
    if skip_gripper and any("gripper" in n for n in names):
      continue
    ctrl_ids = actuator.global_ctrl_ids
    env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
      default_biasprm[ctrl_ids, 2] * scale[:, None]
    )


def apply_damping_prior(env_cfg, p: DampingPrior, seed: int = 0,
                        mode: str = "reset") -> dict:
  """Install ``p`` on the task as a per-environment reset event.

  A point mass at nominal leaves the config untouched and returns ``{}``, so
  the unperturbed path stays byte-identical to the one every earlier result was
  measured on.
  """
  if p.is_point_at(NOMINAL):
    return {}
  env_cfg.events[EVENT_NAME] = EventTermCfg(
    func=randomize_servo_damping,
    mode=mode,
    params={"probs": tuple(p.probs), "values": tuple(VALUES),
            "asset_cfg": SceneEntityCfg("robot"), "skip_gripper": True},
  )
  return {"damping_prior": p.to_json(), "damping_seed": seed, "mode": mode}


def add_damping_args(parser) -> None:
  parser.add_argument(
    "--damping-probs", default=None,
    help="comma-separated categorical over servo damping scales "
         f"{VALUES}, e.g. '1,0,0' for the target")
  parser.add_argument(
    "--damping-value", type=float, default=None,
    help="shorthand for a point mass at this multiplier")
  parser.add_argument("--damping-seed", type=int, default=0)


def prior_from_args(args) -> DampingPrior:
  if getattr(args, "damping_probs", None):
    raw = [float(x) for x in args.damping_probs.split(",")]
    total = sum(raw)
    if total <= 0:
      raise ValueError("--damping-probs sums to zero")
    return DampingPrior(tuple(x / total for x in raw))
  if getattr(args, "damping_value", None) is not None:
    return DampingPrior.point(float(args.damping_value))
  return P_SOURCE
