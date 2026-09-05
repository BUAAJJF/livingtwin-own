"""What an evaluation actually ran under, made explicit.

Three things an evaluation script has to get right, and each of which went
wrong once in this repository without any error being raised:

1. **The weights arrived.**  rsl_rl's ``Distillation.load`` only recognises
   ``student``/``teacher``/``optimizer``/``iteration`` in its ``load_cfg``, so
   ``runner.load(ckpt, load_cfg={"actor": True})`` on a ``-Distill*`` task
   loads *nothing* and returns a randomly initialised student.  That is where
   ``results/decay/v4_final.json`` (twelve windows of 0.0) came from.
   :func:`load_policy` loads whichever network the runner evaluates, directly,
   and then reads the weights back and compares them to the file.

2. **"measured" means the task's own sensor.**  The scripts used to replace the
   camera term's ``noise_cfg`` with ``replace(camera.DEPTH_NOISE, strength=1.0)``,
   which on a ``-Robust`` task *downgrades* the sensor from the robust profile
   (strength 1.35, wider surface-fill and texture penalties) to the nominal
   one; and "clean" left the robust profile on, because ``robust_cfg`` sets it
   after ``play`` has had its say.  :func:`apply_sensor` takes "measured" from
   the task's training configuration and makes "clean" actually clean.

3. **The domain is more than the task id.**  ``robust_cfg`` and ``env_cfg``
   read a dozen environment variables at import time (visible-fraction floor,
   gap scale, latency mixture, smoothness dose, sight weights, reset range).
   None of them reaches a checkpoint or a task name, so a student distilled
   under one setting is silently evaluated under another.  :func:`env_knobs`
   records them; every JSON an evaluation writes should carry it.

Kept free of mjlab imports at module level so it can be imported from the
deployment side and from tests that do not build an environment.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import torch

from piper_push import action_api

# Every environment variable the task configuration reads at import time.  Add
# to this list when adding an ``os.environ`` read to ``env_cfg``/``robust_cfg``;
# ``tests/test_evalcfg.py`` checks the two stay in step.
ENV_KNOBS: tuple[str, ...] = (
  "TARGET_VISIBLE_FLOOR",
  "TARGET_VISIBLE_CEIL",
  "TARGET_GAP_SCALE",
  "OBS_LATENCY_PROBS",
  "SMOOTH_SCALE",
  "SIGHT_RAMP",
  "SIGHT_ARM_W",
  "SIGHT_HAND_W",
  "WRIST_W",
  "HELD_PROXY_M",
  "RESET_FULL_RANGE",
  "PIPER_X_URDF",
)

# Environment variables the code reads that are NOT domain knobs (telemetry
# sinks, the legacy-loading flag); listed so the knob audit can tell them apart.
NON_DOMAIN_ENV: tuple[str, ...] = ("PIPER_U_TELEMETRY", "PIPER_U_TELEMETRY_TAG", "PIPER_ALLOW_LEGACY_ACTION_API", "PIPER_CURRICULUM_LOG")

SENSOR_CHOICES: tuple[str, ...] = ("clean", "measured", "task")

WEIGHT_KEYS: tuple[str, ...] = ("student_state_dict", "actor_state_dict")


def env_knobs(environ: dict[str, str] | None = None) -> dict[str, str | None]:
  """The import-time knobs as they were set for this process.

  ``None`` marks "unset", which is a value in its own right: it is the
  default, and the default is what most of the recorded results ran under.
  """
  src = os.environ if environ is None else environ
  return {k: src.get(k) for k in ENV_KNOBS}


def env_knobs_prefix(knobs: dict[str, str | None]) -> str:
  """The shell prefix that reproduces ``knobs``; empty when all are unset."""
  return " ".join(f"{k}={v}" for k, v in knobs.items() if v is not None)


# --- weights ---------------------------------------------------------------


def weights_in(checkpoint: dict[str, Any]) -> tuple[str, dict[str, torch.Tensor]]:
  """The policy weights a checkpoint carries, and the key they were under."""
  for key in WEIGHT_KEYS:
    if key in checkpoint:
      return key, checkpoint[key]
  raise ValueError(
    f"checkpoint holds none of {WEIGHT_KEYS}; keys are {sorted(checkpoint)}")


def evaluated_network(runner) -> torch.nn.Module:
  """The network ``runner.get_inference_policy`` will hand back.

  rsl_rl keeps the un-compiled module under ``_raw_student`` (Distillation)
  or ``_raw_actor`` (PPO); ``load_state_dict`` has to go to that object, since
  a ``torch.compile`` wrapper prefixes every key with ``_orig_mod.``.
  """
  alg = getattr(runner, "alg", None)
  for name in ("_raw_student", "_raw_actor", "student", "actor"):
    net = getattr(alg, name, None)
    if isinstance(net, torch.nn.Module):
      return net
  raise TypeError(
    f"{type(runner).__name__}.alg ({type(alg).__name__}) has neither a "
    "student nor an actor to load into")


def load_weights(runner, checkpoint_path: str, device: str | None = None
                 ) -> dict[str, Any]:
  """Load the policy weights into the runner's evaluated network, and check.

  Returns ``{"key": <which state dict>, "n_tensors": ..., "iter": ...}`` for
  provenance.  Raises rather than returning an untrained network: a missing
  key, a shape mismatch and a tensor that did not arrive are all errors.
  """
  raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  key, weights = weights_in(raw)
  # The convention the checkpoint was trained under has to be the one the
  # task runs; a checkpoint with no stamp is refused unless legacy loading
  # was asked for explicitly (piper_push.action_api).
  env = getattr(runner, "env", None)
  api_record = (action_api.check(raw, action_api.for_env(env), where=checkpoint_path)
                if env is not None else {"status": "unchecked-no-env"})
  net = evaluated_network(runner)
  net.load_state_dict(weights, strict=True)
  after = net.state_dict()
  missing = [k for k in weights if k not in after]
  if missing:
    raise RuntimeError(f"{missing[:3]} not in the network after loading")
  unequal = [k for k, v in weights.items()
             if not torch.equal(after[k].detach().cpu(), v.detach().cpu())]
  if unequal:
    raise RuntimeError(
      f"{len(unequal)} of {len(weights)} tensors did not arrive "
      f"(first: {unequal[0]}); the runner's network is not the one loaded")
  return {"key": key, "n_tensors": len(weights), "iter": raw.get("iter"), "action_api": api_record}


def load_policy(runner, checkpoint_path: str, device: str | None = None):
  """``load_weights`` followed by ``runner.get_inference_policy``."""
  load_weights(runner, checkpoint_path, device)
  return runner.get_inference_policy(device=device)


# --- sensor ----------------------------------------------------------------


def _camera_term(env_cfg):
  obs = env_cfg.observations
  group = obs.get("camera") if isinstance(obs, dict) else getattr(obs, "camera", None)
  if group is None:
    return None
  return group.terms["scene"]


def sensor_provenance(env_cfg, setting: str | None = None) -> dict[str, Any]:
  """The sensor parameters an environment config will actually run with."""
  term = _camera_term(env_cfg)
  if term is None:
    return {"setting": setting, "camera": None}
  noise = term.params.get("noise_cfg")
  out: dict[str, Any] = {
    "setting": setting,
    "camera": {
      "noise_model": type(noise).__name__ if noise is not None else None,
      "mask_jitter_px": term.params.get("mask_jitter_px"),
    },
  }
  for f in ("strength", "surface_fill", "texture_penalty"):
    if noise is not None and hasattr(noise, f):
      v = getattr(noise, f)
      out["camera"][f] = list(v) if isinstance(v, (tuple, list)) else v
  pose = (env_cfg.events or {}).get("camera_pose") if hasattr(env_cfg, "events") else None
  if pose is not None:
    out["camera_pose"] = {"mode": getattr(pose, "mode", None),
                          **{k: pose.params.get(k) for k in ("pos_jitter", "rot_jitter")}}
  return out


def apply_sensor(env_cfg, task: str, setting: str) -> dict[str, Any]:
  """Set the depth sensor of a ``play`` config to what ``setting`` names.

  ``clean``     no depth noise, no mask jitter -- on every task, including the
                ``-Robust`` ones, where ``play`` alone leaves the robust
                profile on.
  ``measured``  exactly the sensor the task TRAINS under: the camera term of
                ``load_env_cfg(task, play=False)`` is copied over.  On a
                nominal task that is the fitted D455 at strength 1.0; on a
                robust task it is the robust profile, not a downgrade of it.
  ``task``      leave the play config alone (what the scripts did before
                this function existed, for anyone who needs that number).

  Returns :func:`sensor_provenance` of the resulting config.
  """
  if setting not in SENSOR_CHOICES:
    raise ValueError(f"sensor must be one of {SENSOR_CHOICES}, not {setting!r}")
  term = _camera_term(env_cfg)
  if term is None:
    return sensor_provenance(env_cfg, setting)
  if setting == "clean":
    term.params["noise_cfg"] = dataclasses.replace(term.params["noise_cfg"], strength=0.0)
    term.params["mask_jitter_px"] = 0
  elif setting == "measured":
    from mjlab.tasks.registry import load_env_cfg
    train_term = _camera_term(load_env_cfg(task, play=False))
    if train_term is None:
      raise ValueError(f"{task} has no camera term in its training config")
    term.params["noise_cfg"] = train_term.params["noise_cfg"]
    term.params["mask_jitter_px"] = train_term.params["mask_jitter_px"]
  return sensor_provenance(env_cfg, setting)


def add_sensor_arg(parser, default: str = "clean") -> None:
  parser.add_argument(
    "--sensor", default=default, choices=SENSOR_CHOICES,
    help="depth realism to EVALUATE under.  'clean': no depth noise and no "
         "mask jitter, on every task -- a student distilled under the fitted "
         "sensor collapses here (results/audit_20260904), so it is a control, "
         "not a baseline.  'measured': the sensor the task "
         "trains with -- the fitted D455 at strength 1.0 on a nominal task, "
         "the robust profile (strength 1.35, wider surface-fill and texture "
         "penalties) on a -Robust task.  'task': whatever the play config "
         "says, which on a -Robust task is the robust sensor and on a nominal "
         "one is clean.  Only 'measured' says anything about the robot.")


def add_action_api_arg(parser) -> None:
  parser.add_argument(
    "--allow-legacy-action-api", action="store_true",
    help="load a checkpoint that carries no action_api stamp (every checkpoint "
         "from before 2026-09-05, unbounded v1 convention) -- only meaningful "
         f"on a '-V1' task id.  Equivalent to {action_api.ENV_FLAG}=1.")


def apply_action_api_arg(args) -> None:
  if getattr(args, "allow_legacy_action_api", False):
    os.environ[action_api.ENV_FLAG] = "1"


def provenance(sensor: dict[str, Any] | None = None, **extra) -> dict[str, Any]:
  """The domain block every evaluation JSON should carry.

  ``sensor`` is what :func:`apply_sensor` returned; ``extra`` is anything
  else worth stamping (argv, which state dict was loaded, ...).
  """
  out: dict[str, Any] = {"env_knobs": env_knobs()}
  if sensor is not None:
    out["sensor"] = sensor
  out.update(extra)
  return out
