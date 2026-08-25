"""Teacher-forced replay: how wrong is a candidate simulator, in MJWarp.

The comparison Phase RA-Sim-0 rests on is not a comparison of learned
predictors.  It is a comparison of *simulators*, so it is run in the
simulator: a candidate is built, the recorded physical state is written into
it every ``P`` control steps, the recorded action stream is played through it,
and the states it produces are compared with the ones the target produced.

``P = 1`` gives the one-step error at every step; ``P = 25`` gives horizons 1
to 25 from each resynchronisation.  Nothing about the command path is
resynchronised -- the slew memory, the delay pipeline and any hook state are
deterministic functions of the action history, which every candidate sees
identically, and their divergence is part of the mismatch under test.

Writing state *between* control steps to re-anchor an evaluation is not the
same thing as writing state *after* a physics step to fake dynamics.  No
candidate here produces its dynamics that way; the phase forbids it and the
injection audit records that the residual acts on the command instead.

**Segments are dropped, not patched.**  A segment is discarded whole if the
recording reset inside it, if the candidate terminated inside it, or if the
object was replaced inside it -- the last because the replacement draws a new
shape and mass, and a candidate holding a different object is answering a
different question.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Recording:
  """One collected session, as the fields a replay needs."""

  q: torch.Tensor       # [T, E, 6] measured arm position
  qd: torch.Tensor      # [T, E, 6] measured arm velocity
  a: torch.Tensor       # [T, E, A] policy action, the thing replayed
  gq: torch.Tensor      # [T, E, 1] gripper position
  obj: torch.Tensor     # [T, E, 13] object root state, world frame
  done: torch.Tensor    # [T, E]    the recording's own episode boundary
  shape: torch.Tensor   # [T, E]    object shape class
  u: torch.Tensor       # [T, E, 6] commanded target, for the calibration fit
  mode: torch.Tensor    # [T, E]    0 natural, 1 perturbed, 2 probe

  @classmethod
  def load(cls, path, steps: int | None = None, device: str = "cpu"
           ) -> tuple["Recording", dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    d = blob["data"]
    sl = slice(None) if steps is None else slice(0, steps)
    return cls(**{k: d[k][sl].to(device) for k in
                  ("q", "qd", "a", "gq", "obj", "done", "shape", "u", "mode")}
               ), blob["meta"]

  @property
  def steps(self) -> int:
    return self.q.shape[0]

  @property
  def num_envs(self) -> int:
    return self.q.shape[1]


class ReplayHarness:
  """Binds an already-built environment to the indices a replay needs."""

  def __init__(self, env, object_name: str = "object") -> None:
    self.env = env
    robot = env.scene["robot"]
    self.robot = robot
    self.arm_ids, _ = robot.find_joints([f"joint{i}" for i in range(1, 7)],
                                        preserve_order=True)
    self.grip_ids, _ = robot.find_joints(("gripper_joint1",))
    self.obj = env.scene[object_name]
    self.arm_ids_t = torch.tensor(self.arm_ids, device=env.device)
    self.grip_ids_t = torch.tensor(self.grip_ids, device=env.device)

  def write_state(self, q, qd, gq, obj) -> None:
    """Put the candidate back on the recorded state, arm and object both.

    Gripper velocity is written as zero rather than from the log: the
    recording does not carry it, and a slide joint at 50 Hz whose position is
    resynchronised every step recovers its velocity within one step.  It is
    written identically for every candidate.
    """
    self.robot.write_joint_state_to_sim(q, qd, joint_ids=self.arm_ids_t)
    self.robot.write_joint_state_to_sim(
      gq, torch.zeros_like(gq), joint_ids=self.grip_ids_t)
    self.obj.write_root_state_to_sim(obj)

  @torch.no_grad()
  def run(self, rec: Recording, period: int, log_every: int = 0
          ) -> dict[str, torch.Tensor]:
    env = self.env
    dev = env.device
    T = rec.steps
    pq, pqd, cdone = [], [], []
    env.reset()
    for t in range(T):
      if t % period == 0:
        self.write_state(rec.q[t].to(dev), rec.qd[t].to(dev),
                         rec.gq[t].to(dev), rec.obj[t].to(dev))
      out = env.step(rec.a[t].to(dev))
      cdone.append(out[2].bool().cpu() if len(out) > 2 else
                   torch.zeros(rec.num_envs, dtype=torch.bool))
      pq.append(self.robot.data.joint_pos[:, self.arm_ids].clone().cpu())
      pqd.append(self.robot.data.joint_vel[:, self.arm_ids].clone().cpu())
      if log_every and (t + 1) % log_every == 0:
        print(f"      replay P={period} step {t + 1}/{T}", flush=True)
    return {"q": torch.stack(pq), "qd": torch.stack(pqd),
            "done": torch.stack(cdone)}


def segment_mask(rec: Recording, cand_done: torch.Tensor, period: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
  """Which (step, env) predictions are usable, and at what horizon.

  Returns a boolean ``[T-1, E]`` and an integer horizon of the same shape.
  A step is usable only if every step of its segment up to and including it
  was clean, which is what makes a 25-step number a 25-step number.
  """
  T, E = rec.done.shape
  T = T - 1  # the last step has no recorded successor
  ok = torch.ones(T, E, dtype=torch.bool)
  hor = torch.zeros(T, E, dtype=torch.long)
  running = torch.zeros(E, dtype=torch.bool)
  for t in range(T):
    if t % period == 0:
      running = torch.ones(E, dtype=torch.bool)
    running = running & ~rec.done[t].cpu() & ~cand_done[t].cpu()
    running = running & (rec.shape[t + 1].cpu() == rec.shape[t].cpu())
    ok[t] = running
    hor[t] = (t % period) + 1
  return ok, hor


def nrms(pred: torch.Tensor, rec: Recording, ok: torch.Tensor,
         hor: torch.Tensor, period: int, horizon: int, field: str = "q"
         ) -> dict:
  """RMS error at one horizon, over the RMS motion the target actually made.

  1.0 is "as wrong as predicting that nothing moved", which makes the number
  comparable across horizons and across joints without a further choice.
  """
  ref = getattr(rec, field)
  T = ref.shape[0] - 1
  sel = ok[:T] & (hor[:T] == horizon)
  if not bool(sel.any()):
    return {"n": 0, "nrms": float("nan"), "rms": float("nan"),
            "rms_motion": float("nan")}
  idx = sel.nonzero(as_tuple=False)
  t, e = idx[:, 0], idx[:, 1]
  p = pred[t, e]
  y = ref[t + 1, e]
  anchor_t = (t // period) * period
  y0 = ref[anchor_t, e]
  err = (p - y)
  motion = (y - y0)
  rms = float(err.pow(2).mean().sqrt())
  rmm = float(motion.pow(2).mean().sqrt())
  per_joint = err.pow(2).mean(0).sqrt()
  return {"n": int(sel.sum()), "rms": rms, "rms_motion": rmm,
          "nrms": rms / max(rmm, 1e-12),
          "per_joint_rms": [float(v) for v in per_joint],
          "max_abs": float(err.abs().max())}


def reversal_mask(rec: Recording, lookback: int = 3) -> torch.Tensor:
  """Steps at which the commanded direction has just flipped, per joint.

  Backlash lives here and nowhere else, so an error that is flat across this
  mask is an error the hidden target is not responsible for.
  """
  du = rec.u[1:] - rec.u[:-1]
  sign = torch.sign(du)
  flip = torch.zeros_like(sign, dtype=torch.bool)
  flip[lookback:] = (sign[lookback:] * sign[:-lookback]) < 0
  return flip


def deadband_mask(rec: Recording, width: float = 0.004) -> torch.Tensor:
  """Steps whose commanded displacement is small enough to sit in the band."""
  du = (rec.u[1:] - rec.u[:-1]).abs()
  return du < width
