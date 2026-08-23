"""Reward-free session logs: what a deployed robot could actually record.

The whole calibration idea rests on inferring a simulator parameter from data
a real arm produces while doing its job.  So the recorded channels are exactly
the ones a real arm has -- its joint states, the commands it issued, its
gripper's servo error, and the two internal quantities of its own policy: the
encoder latent that the recurrent layer consumes, and that layer's hidden
state.  Both of those are computed on the robot, from the robot's own camera,
by weights that ship with the policy.

What is *not* here is as important:

* reward, return, success, placement counts;
* safety-shell trips;
* object pose, mass, friction, contact flags, anything privileged;
* the latency itself, which is the label.

:data:`SIM_ONLY` names the fields that exist for training and offline scoring
and must never reach an estimator that will run on the robot.
:func:`assert_deployable` is the check, and it is called by the loaders rather
than left as a convention.

One session is one environment's contiguous timeline.  That is the unit a real
deployment gives you: one arm, one afternoon.  Splitting by environment (and
by generation seed, and by object shape class) is what keeps a window in the
test set from being a near-copy of one in the training set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DEPLOYABLE: tuple[str, ...] = ("enc", "hidden", "proprio", "action", "servo", "done")
"""Channels an estimator may read.  All of them exist on the real robot."""

SIM_ONLY: tuple[str, ...] = ("lag", "shape")
"""Channels that exist only in simulation.  ``lag`` is the label; ``shape`` is
metadata used to build held-out splits.  Neither is ever an input."""

PROPRIO_NAMES: tuple[str, ...] = (
  *(f"joint{i}_pos" for i in range(1, 7)),
  *(f"joint{i}_vel" for i in range(1, 7)),
  "gripper_pos",
)

CHANNEL_DOC: dict[str, str] = {
  "enc": "actor encoder output, (T, N, D): the tensor the policy's GRU "
         "consumes.  Normalised proprioception concatenated with the "
         "spatial-softmax encoding of the (delayed) depth image.",
  "hidden": "actor GRU hidden state BEFORE the step, (T, N, H).  Recorded "
            "rather than recomputed so that the frozen actor can be replayed "
            "on a predicted latent from any point in the log.",
  "proprio": f"(T, N, {len(PROPRIO_NAMES)}): {', '.join(PROPRIO_NAMES)}.",
  "action": "commanded joint targets as the policy emitted them, (T, N, A), "
            "before the action manager's rate limits.",
  "servo": "gripper position minus gripper position target, (T, N, 1).  The "
           "quantity a real drive reports as current.",
  "done": "(T, N) bool: this step ended an episode.  Deployable -- the robot "
          "knows when the task restarts -- but excluded from every feature "
          "set, with a done-only control reported instead, because reset "
          "cadence is a physical consequence of the domain and would be an "
          "accuracy the estimator did not earn from the plant.",
  "lag": "SIMULATOR LABEL.  Observation delay in control steps.",
  "shape": "SIMULATOR METADATA, (T, N) uint8: object shape class index.",
}


def assert_deployable(keys) -> None:
  """Raise if an estimator input set contains a simulator-only channel."""
  bad = sorted(set(keys) & set(SIM_ONLY))
  if bad:
    raise ValueError(
      f"{bad} is a simulator label, not something a robot records. "
      f"Deployable channels are {DEPLOYABLE}.")
  unknown = sorted(set(keys) - set(DEPLOYABLE) - set(SIM_ONLY))
  if unknown:
    raise ValueError(f"unknown channel(s) {unknown}")


# ---------------------------------------------------------------------------
# Replaying the frozen actor
# ---------------------------------------------------------------------------


def actor_head(policy, enc: torch.Tensor, hidden: torch.Tensor | None):
  """The policy from the encoder latent onward: ``(action, next_hidden)``.

  This is ``SpatialSoftmaxRecurrentModel.forward`` with the convolutional
  stage factored out, so that a *predicted* latent can be pushed through the
  same recurrent layer, MLP and output squashing that the real one goes
  through.  ``S_action`` is the difference between the two.

  ``tests/test_wm_data.py`` checks it against ``policy(obs)`` on a model built
  the way the task builds it, because "it is the same code" is a claim that
  stops being true the first time somebody edits one of them.
  """
  out, h_new = policy.rnn.rnn(enc.unsqueeze(0), hidden)
  latent = out.squeeze(0)
  mlp_out = policy.mlp(latent)
  dist = getattr(policy, "distribution", None)
  action = dist.deterministic_output(mlp_out) if dist is not None else mlp_out
  return action, h_new


def zero_hidden(policy, n: int, device=None) -> torch.Tensor:
  """The state a fresh episode starts from.

  The device comes from the recurrent layer's own weights unless one is given.
  A TensorDict built by mjlab reports ``device=None`` even when its entries are
  on the GPU, so deriving it from the observation puts the hidden state on the
  CPU and the first forward pass dies on a device mismatch.
  """
  rnn = policy.rnn.rnn
  if device is None:
    device = next(rnn.parameters()).device
  return torch.zeros(rnn.num_layers, n, rnn.hidden_size, device=device)


# ---------------------------------------------------------------------------
# A stored session set
# ---------------------------------------------------------------------------


@dataclass
class SessionSet:
  """One rollout: ``steps`` control steps of ``n_envs`` independent sessions."""

  enc: torch.Tensor       # (T, N, D)  fp16
  hidden: torch.Tensor    # (T, N, H)  fp16
  proprio: torch.Tensor   # (T, N, P)  fp32
  action: torch.Tensor    # (T, N, A)  fp32
  servo: torch.Tensor     # (T, N, 1)  fp32
  done: torch.Tensor      # (T, N)     bool
  shape: torch.Tensor     # (T, N)     uint8
  lag: int
  meta: dict = field(default_factory=dict)

  @property
  def steps(self) -> int:
    return int(self.enc.shape[0])

  @property
  def n_envs(self) -> int:
    return int(self.enc.shape[1])

  def to_dict(self) -> dict:
    return {
      "enc": self.enc, "hidden": self.hidden, "proprio": self.proprio,
      "action": self.action, "servo": self.servo, "done": self.done,
      "shape": self.shape, "lag": self.lag, "meta": self.meta,
    }

  def save(self, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(self.to_dict(), path)

  @classmethod
  def load(cls, path: str | Path, device="cpu") -> "SessionSet":
    d = torch.load(path, map_location=device, weights_only=False)
    meta = d.pop("meta", {})
    lag = int(d.pop("lag"))
    return cls(lag=lag, meta=meta, **d)

  def describe(self) -> dict:
    """Shape and content summary, for the manifest."""
    cls_counts = torch.bincount(self.shape.reshape(-1).long(), minlength=8)
    return {
      "steps": self.steps,
      "n_envs": self.n_envs,
      "arm_seconds": self.steps * self.n_envs / 50.0,
      "dims": {"enc": int(self.enc.shape[-1]),
               "hidden": int(self.hidden.shape[-1]),
               "proprio": int(self.proprio.shape[-1]),
               "action": int(self.action.shape[-1])},
      "lag": self.lag,
      "episode_boundaries": int(self.done.sum()),
      "shape_class_counts": [int(x) for x in cls_counts[:5]],
      "bytes": int(sum(t.numel() * t.element_size() for t in
                       (self.enc, self.hidden, self.proprio, self.action,
                        self.servo, self.done, self.shape))),
    }


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def windows(s: SessionSet, length: int, burn_in: int, stride: int,
            envs: torch.Tensor | None = None,
            cross_episode: bool = False) -> list[tuple[int, int]]:
  """``(t0, env)`` pairs for every usable window.

  A window is ``burn_in + length`` steps: the burn-in exists so the recurrent
  state a model is scored from was built the way it is built on the robot --
  by running forward from wherever the arm happened to be -- rather than from
  a zero state that only occurs at a reset.

  Windows that straddle an episode boundary are dropped by default.  The
  policy's hidden state is zeroed there, so a window spanning one contains a
  discontinuity that has nothing to do with the domain, and both the model and
  any trajectory-matching score would be fitting the reset.
  """
  total = burn_in + length
  out = []
  env_list = range(s.n_envs) if envs is None else [int(e) for e in envs]
  for b in env_list:
    dones = s.done[:, b]
    for t0 in range(0, s.steps - total + 1, stride):
      if not cross_episode and bool(dones[t0:t0 + total - 1].any()):
        continue
      out.append((t0, b))
  return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(root: str | Path, entries: list[dict], extra: dict) -> Path:
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  path = root / "manifest.json"
  doc = {
    "schema": CHANNEL_DOC,
    "deployable_channels": list(DEPLOYABLE),
    "simulator_only_channels": list(SIM_ONLY),
    "proprio_names": list(PROPRIO_NAMES),
    "control_hz": 50.0,
    "files": entries,
    **extra,
  }
  path.write_text(json.dumps(doc, indent=1))
  return path
