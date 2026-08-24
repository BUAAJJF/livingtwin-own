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

SIM_ONLY: tuple[str, ...] = ("lag", "shape", "trip")
"""Channels that exist only in simulation and are never an estimator input.

``lag`` is the domain parameter's value -- the label.  It keeps the name it had
in Phase WM1-A, where the axis was a lag in control steps, so that phase's
files still load; ``meta["axis"]`` says which parameter it is, and for Phase
WM1-B it holds a damping multiplier.

``shape`` is the object class, used to build held-out splits.

``trip`` is whether the safety shell fired on that step.  It is here because
the deployable risk head of Phase WM1-B is *trained* on simulator safety
labels, which the phase specification allows; what it must never do is read
them in the target domain, and it cannot, because the risk head takes only
:data:`DEPLOYABLE` channels as input."""

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
  "lag": "SIMULATOR LABEL, scalar: the domain parameter's value.  Named for "
         "Phase WM1-A's axis, where it was an observation delay in control "
         "steps; meta['axis'] says which parameter it is.",
  "shape": "SIMULATOR METADATA, (T, N) uint8: object shape class index.",
  "trip": "SIMULATOR LABEL, (T, N) bool: the safety shell fired on this step. "
          "Trains the risk head; never an input to it.",
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
  lag: float
  trip: torch.Tensor | None = None    # (T, N) bool, absent in WM1-A files
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
      "shape": self.shape, "lag": self.lag, "trip": self.trip,
      "meta": self.meta,
    }

  def save(self, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(self.to_dict(), path)

  @classmethod
  def load(cls, path: str | Path, device="cpu") -> "SessionSet":
    d = torch.load(path, map_location=device, weights_only=False)
    meta = d.pop("meta", {})
    lag = d.pop("lag")
    # Phase WM1-A files predate the trip channel; they load without it.
    d.setdefault("trip", None)
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
      "axis": self.meta.get("axis", "obs_latency_steps"),
      "episode_boundaries": int(self.done.sum()),
      "safety_trips": (int(self.trip.sum()) if self.trip is not None else None),
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


def gather(s: "SessionSet", idx: list[tuple[int, int]], length: int,
           keys=("enc", "proprio", "action", "servo"),
           device=None) -> dict[str, torch.Tensor]:
  """Stack the windows named by ``idx`` into ``(length, B, C)`` tensors.

  ``assert_deployable`` runs on the requested keys, so a batch containing the
  label cannot be assembled at all -- the guard is at the point of use rather
  than in a docstring.
  """
  assert_deployable(keys)
  t0 = torch.tensor([a for a, _ in idx])
  bs = torch.tensor([b for _, b in idx])
  # (length, B) index grids, so the gather is one advanced-indexing op per
  # channel rather than a Python loop over tens of thousands of windows.
  ts = t0.unsqueeze(0) + torch.arange(length).unsqueeze(1)
  out = {}
  for k in keys:
    x = getattr(s, k)
    v = x[ts, bs.unsqueeze(0).expand_as(ts)]
    if v.dtype == torch.half:
      v = v.float()
    if device is not None:
      v = v.to(device)
    out[k] = v
  return out


def shapes_of(s: "SessionSet", idx: list[tuple[int, int]], at: int) -> torch.Tensor:
  """Shape class in the hand at offset ``at`` of each window.  Metadata only."""
  t0 = torch.tensor([a for a, _ in idx]) + at
  bs = torch.tensor([b for _, b in idx])
  return s.shape[t0, bs]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def load_split(root, split: str) -> list[tuple["SessionSet", str]]:
  """Every file of one split, in a fixed order.

  The split is a property of the file, decided when the data was generated:
  different seed, different environments, and for the held-out splits
  different object shape classes.  Nothing downstream re-partitions a tensor.
  """
  root = Path(root)
  out = [(SessionSet.load(f), f.name)
         for f in sorted(root.glob(f"{split}__lag*__seed*.pt"))]
  if not out:
    raise FileNotFoundError(f"no {split} files under {root}")
  return out


def make_index(sessions, stride: int, length: int, burn_in: int):
  """``(session_id, t0, env)`` for every usable window, and its lag."""
  idx, lags = [], []
  for si, (s, _) in enumerate(sessions):
    w = windows(s, length=length - burn_in, burn_in=burn_in, stride=stride)
    idx.extend((si, t0, b) for t0, b in w)
    lags.extend([s.lag] * len(w))
  return idx, torch.tensor(lags)


def batch_from(sessions, idx, sel, length: int, device,
               keys=("enc", "proprio", "action", "servo")) -> dict:
  """One batch, assembled per source session and concatenated on the batch
  axis.  ``_lag`` rides along as the conditioning value, which is an input to
  the model and never a feature."""
  parts: dict[str, list] = {}
  order = []
  for si in sorted({idx[i][0] for i in sel}):
    rows = [idx[i] for i in sel if idx[i][0] == si]
    order.extend(rows)
    g = gather(sessions[si][0], [(t0, b) for _, t0, b in rows], length,
               keys=keys, device=device)
    for k, v in g.items():
      parts.setdefault(k, []).append(v)
  out = {k: torch.cat(v, dim=1) for k, v in parts.items()}
  out["_lag"] = torch.tensor([sessions[si][0].lag for si, _, _ in order],
                             device=device)
  out["_rows"] = order
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
