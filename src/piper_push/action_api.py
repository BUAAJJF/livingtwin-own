"""Which action convention a checkpoint or an exported policy was trained under, stamped and checked.

Two conventions exist (``piper_push.robot.action_spec``):

* version 1, ``"v1"``: a is unbounded, target = default + PICK_ARM_SCALE * a,
  clipped.  Every checkpoint written before 2026-09-05.
* version 2, ``"bounded"``: the policy emits u, the action term applies
  a = tanh(u), target = centre + half_span * a.  Since 2026-09-05.

A checkpoint carries ``infos["action_api"] = {"version", "convention",
"spec_hash"}``, written by the runners in ``piper_push.runners``; an exported
``obs_spec.json`` carries the same block.  Every loader -- resume, evaluation,
distillation (teacher), fine-tuning (student, critic), export, deployment --
calls :func:`check` against what the task or the mapper expects.  A checkpoint
with no block is refused: it is a v1 checkpoint from before the stamp existed,
and loading it silently onto a bounded task drives a different robot.  Loading
one is a deliberate act: a ``-V1`` task id AND ``PIPER_ALLOW_LEGACY_ACTION_API=1``
(every script exposes it as ``--allow-legacy-action-api``).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

# piper_push.robot is imported inside the functions, never here: robot.py
# imports mjlab, whose entry-point loader imports piper_push.tasks, which
# imports this module through the runners -- importing robot at module level
# from here would put robot half-initialised under the task registration
# (see piper_push.checkpoints for the same rule).

VERSION = {"v1": 1, "bounded": 2}
ENV_FLAG = "PIPER_ALLOW_LEGACY_ACTION_API"


class ActionApiError(RuntimeError):
  pass


def _rounded(x):
  """Floats to 6 decimals before hashing: the offsets come from float32 joint
  defaults on one side and Python floats on the other, and a hash that
  differs on the seventh decimal would refuse a checkpoint for nothing."""
  if isinstance(x, float):
    return round(x, 6)
  if isinstance(x, dict):
    return {k: _rounded(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_rounded(v) for v in x]
  return x


def spec_hash(spec: dict) -> str:
  return hashlib.sha256(json.dumps(_rounded(spec), sort_keys=True).encode()).hexdigest()[:16]


def for_convention(convention: str, default_joint_pos: dict[str, float] | None = None) -> dict[str, Any]:
  from piper_push import robot as piper
  spec = piper.action_spec(convention, default_joint_pos)
  return {"version": VERSION[convention], "convention": convention, "spec_hash": spec_hash(spec)}


def for_env(env) -> dict[str, Any]:
  """The convention a built environment runs (``ManagerBasedRlEnv`` or its wrapper)."""
  env = getattr(env, "unwrapped", env)
  bounded = bool(env.cfg.actions["arm"].bounded)
  robot = env.scene["robot"]
  defaults = dict(zip(robot.joint_names, robot.data.default_joint_pos[0].tolist()))
  return for_convention("bounded" if bounded else "v1", defaults)


def for_env_cfg(env_cfg, default_joint_pos: dict[str, float]) -> dict[str, Any]:
  bounded = bool(env_cfg.actions["arm"].bounded)
  return for_convention("bounded" if bounded else "v1", default_joint_pos)


def legacy_allowed() -> bool:
  return os.environ.get(ENV_FLAG, "0") not in ("", "0", "false", "False")


def stamp(infos: dict | None, api: dict[str, Any]) -> dict:
  return {**(infos or {}), "action_api": dict(api)}


def read(loaded: dict) -> dict | None:
  infos = loaded.get("infos") if isinstance(loaded, dict) else None
  api = infos.get("action_api") if isinstance(infos, dict) else None
  return dict(api) if isinstance(api, dict) else None


def check(loaded: dict, expected: dict[str, Any], *, where: str = "checkpoint",
          allow_legacy: bool | None = None) -> dict[str, Any]:
  """Refuse a checkpoint whose convention is not the one it is being loaded into.

  Returns a small record for provenance.  ``allow_legacy`` defaults to the
  environment flag.
  """
  if allow_legacy is None:
    allow_legacy = legacy_allowed()
  api = read(loaded)
  if api is None:
    if expected["version"] == VERSION["v1"] and allow_legacy:
      return {"status": "legacy-unstamped", "expected": expected}
    raise ActionApiError(
      f"{where} carries no action_api metadata.  Every checkpoint written before "
      f"2026-09-05 is the unbounded v1 convention; the task here expects "
      f"{expected['convention']} (version {expected['version']}).  To load a v1 "
      f"checkpoint deliberately, use a '-V1' task id and set {ENV_FLAG}=1 "
      f"(--allow-legacy-action-api).")
  if api.get("version") != expected["version"] or api.get("spec_hash") != expected["spec_hash"]:
    raise ActionApiError(
      f"{where} was trained under action convention {api.get('convention')!r} "
      f"(version {api.get('version')}, spec {api.get('spec_hash')}); this task runs "
      f"{expected['convention']!r} (version {expected['version']}, spec "
      f"{expected['spec_hash']}).  Refusing: the same number would move a different joint.")
  return {"status": "ok", "api": api}
