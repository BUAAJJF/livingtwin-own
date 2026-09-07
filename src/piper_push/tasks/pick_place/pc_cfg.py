"""The point-cloud routes: environments, runner configs and task ids (branch ``yf/pc``).

Four routes on ONE environment -- the ``-Robust`` domain with its measured
sensor, plant and latency -- differing only in what the actor is shown:

  P0    metric depth (2 channels: metres, validity) -> DepthResNetLite -> GRU
  P1A   workspace point cloud (512 x 4)              -> PointNet          -> GRU
  P1B   the same cloud                               -> point-patch transformer -> GRU
  P2    top-K analytic grasp candidates + the locked candidate -> set MLP -> GRU

Every route reads ``vision_meta`` (age, fresh, valid).  None reads a mask, a
target channel, an object pose or an instance label.  The critic is the state
critic, unchanged.  The student and the actor are the same network, so a
distillation checkpoint is a fine-tuning initialisation.

Second generation (``piper_push.pc.routes``): P1BZ and P1BT are P1B with a
fifth per-point column -- always zero, or the renderer's label of the
commanded object on the points the cloud already has.  Same encoder, same
widths, same everything else; P1BT is oracle-only and is refused by the
bundle and by every deployment entry point.

Held-out objects: the ``capped`` shape class (12% of the trained
distribution) is never drawn in training; the ``-Heldout`` ids draw only it.
"""

from __future__ import annotations

import dataclasses

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlModelCfg
from mjlab.tasks.registry import register_mjlab_task

from piper_push import camera, objects
from piper_push.distill import PickPlaceDistillationRunner, RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg
from piper_push.pc import cloud, grasp
from piper_push.pc import routes as pc_routes
from piper_push.runners import PickPlaceOnPolicyRunner
from piper_push.tasks.pick_place.rl_cfg import _distribution_cfg, pick_place_ppo_runner_cfg
from piper_push.tasks.pick_place.robust_cfg import make_robust_env_cfg

import os

ROUTES = pc_routes.ROUTES
ORACLE_ROUTES = pc_routes.ORACLE_ROUTES
NUM_POINTS = 512
# Round-4 knobs (2026-09-07), read at import like the rest and recorded by evalcfg:
# PC_CLOUD_DR=1 puts cloud.CloudDrCfg on the TRAINING config (plane offset, point
# dropout, wider frame offset and jitter); RECON_W is the weight of the
# masked-reconstruction loss in distillation for a route whose encoder has the
# head (P1BZ6R); on a route without it the knob does nothing.
PC_CLOUD_DR = os.environ.get("PC_CLOUD_DR", "0") not in ("0", "", "false", "False")
RECON_W = float(os.environ.get("RECON_W", "0"))
HELDOUT_CLASS = "capped"
_MODEL = "piper_push.pc.models:SetRecurrentModel"


def shape_weights(split: str) -> tuple[float, ...] | None:
  """``train`` excludes the held-out class; ``heldout`` draws only it; ``all`` is the task's own."""
  w = list(objects.SHAPE_WEIGHTS)
  i = objects.SHAPE_CLASSES.index(HELDOUT_CLASS)
  if split == "all":
    return None
  if split == "train":
    w[i] = 0.0
  elif split == "heldout":
    w = [0.0] * len(w)
    w[i] = 1.0
  else:
    raise ValueError(split)
  s = sum(w)
  return tuple(x / s for x in w)


def actor_groups(route: str) -> tuple[str, ...]:
  if pc_routes.base_route(route) == "P2":
    return ("proprio", "grasp_locked", "grasp_topk", "vision_meta")
  return ("proprio", "camera", "vision_meta")


def encoder_cfg(route: str) -> dict:
  recon = pc_routes.has_recon_head(route)
  route = pc_routes.base_route(route)
  if route == "P0":
    return {"camera": {"type": "depthresnet", "out_dim": 256}}
  if route == "P1A":
    return {"camera": {"type": "pointnet", "out_dim": 256}}
  if route == "P1B":
    spec = {"camera": {"type": "pointpatch", "out_dim": 256, "n_groups": 32, "group_size": 16, "dim": 128, "n_layers": 2}}
    if recon:
      spec["camera"].update({"recon": True, "mask_ratio": 0.4})
    return spec
  if route == "P2":
    return {"grasp_topk": {"type": "setmlp", "out_dim": 128}}
  raise ValueError(route)


def make_pc_env_cfg(route: str, play: bool = False, split: str = "train"):
  if route not in ROUTES:
    raise ValueError(f"route must be one of {ROUTES}")
  base = pc_routes.base_route(route)
  cfg = make_robust_env_cfg(play=play, vision=True)
  old = cfg.observations["camera"].terms["scene"]
  params = dict(old.params)
  latency = tuple(params.get("latency_probs", cloud.DEFAULT_LATENCY_PROBS))
  term_params = {
    "sensor_name": camera.CAMERA_NAME,
    "command_name": "pick",
    "num_points": NUM_POINTS,
    "cutoff_distance": params.get("cutoff_distance", camera.CUTOFF_M),
    "min_depth": params.get("min_depth", 0.05),
    "noise_cfg": params["noise_cfg"],
    "mask_jitter_px": 0,
    "scenery_dr": bool(params.get("scenery_dr", False)),
    "latency_probs": latency,
    "augment": not play,
    "mode": "depth" if base == "P0" else "cloud",
    "target_channel": pc_routes.target_channel(route),
    "workspace": dataclasses.replace(cloud.WORKSPACE, z_min=pc_routes.crop_z_min(route)),
    "dr": (cloud.CloudDrCfg() if (PC_CLOUD_DR and not play and base != "P0") else None),
  }
  # The ``camera`` group keeps its name and its ``scene`` term so that
  # evalcfg.apply_sensor and robust_cfg address the sensor the same way.
  obs = {}
  for name, group in cfg.observations.items():
    if name == "camera":
      obs["camera"] = ObservationGroupCfg(
        terms={"scene": ObservationTermCfg(func=cloud.WorkspaceCloud, params=term_params)},
        enable_corruption=False, concatenate_terms=True)
      obs["vision_meta"] = ObservationGroupCfg(
        terms={"meta": ObservationTermCfg(func=cloud.vision_meta)},
        enable_corruption=False, concatenate_terms=True)
      if base == "P2":
        obs["grasp_topk"] = ObservationGroupCfg(
          terms={"topk": ObservationTermCfg(func=grasp.GraspCandidates, params={"command_name": "pick"})},
          enable_corruption=False, concatenate_terms=True)
        obs["grasp_locked"] = ObservationGroupCfg(
          terms={"locked": ObservationTermCfg(func=grasp.locked_candidate)},
          enable_corruption=False, concatenate_terms=True)
    else:
      obs[name] = group
  cfg.observations = obs
  w = shape_weights(split)
  for name, event in cfg.events.items():
    if name.startswith("object_shape") and w is not None:
      event.params["shape_weights"] = w
  return cfg


def pc_model_cfg(route: str, bounded: bool = True) -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(256, 256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg=encoder_cfg(route),
    class_name=_MODEL,
    rnn_type="gru",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
    distribution_cfg=_distribution_cfg(bounded),
  )


def pc_vision_ppo_runner_cfg(route: str, max_iterations: int = 6000):
  cfg = pick_place_ppo_runner_cfg(f"piperx_pc_{route.lower()}_vision", max_iterations, bounded=True)
  cfg.actor = pc_model_cfg(route)
  cfg.obs_groups = {
    "actor": actor_groups(route),
    "critic": ("full_proprio", "object", "privileged"),
  }
  cfg.wandb_tags = ("piperx", "pick-place", "pc", route.lower())
  return cfg


def pc_distill_runner_cfg(route: str, max_iterations: int = 3000) -> RslRlDistillationRunnerCfg:
  ppo = pick_place_ppo_runner_cfg(bounded=True)
  return RslRlDistillationRunnerCfg(
    student=pc_model_cfg(route),
    teacher=ppo.actor,
    algorithm=RslRlDistillationAlgorithmCfg(
      num_learning_epochs=1,
      gradient_length=8 if pc_routes.base_route(route) == "P0" else 16,
      learning_rate=5.0e-4,
      class_name="piper_push.distill:BoundedDistillation",
      max_grad_norm=1.0,
      loss_type="mse",
      recon_w=RECON_W,
    ),
    experiment_name=f"piperx_pc_{route.lower()}_distill",
    logger="wandb",
    wandb_project="piper-pick-place",
    wandb_tags=("piperx", "pick-place", "pc", route.lower(), "distill"),
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=max_iterations,
    obs_groups={
      "student": actor_groups(route),
      "teacher": ("full_proprio", "object"),
    },
  )


def register() -> None:
  for route in ROUTES:
    register_mjlab_task(
      task_id=f"Mjlab-Pick-Place-PiperX-PC-{route}-Distill",
      env_cfg=make_pc_env_cfg(route),
      play_env_cfg=make_pc_env_cfg(route, play=True),
      rl_cfg=pc_distill_runner_cfg(route),
      runner_cls=PickPlaceDistillationRunner,
    )
    register_mjlab_task(
      task_id=f"Mjlab-Pick-Place-PiperX-PC-{route}-Vision",
      env_cfg=make_pc_env_cfg(route),
      play_env_cfg=make_pc_env_cfg(route, play=True),
      rl_cfg=pc_vision_ppo_runner_cfg(route),
      runner_cls=PickPlaceOnPolicyRunner,
    )
    # Evaluation on the held-out class only.  Training under this id is not a
    # thing anyone should do; its env_cfg is the play config on purpose.
    register_mjlab_task(
      task_id=f"Mjlab-Pick-Place-PiperX-PC-{route}-Vision-Heldout",
      env_cfg=make_pc_env_cfg(route, play=True, split="heldout"),
      play_env_cfg=make_pc_env_cfg(route, play=True, split="heldout"),
      rl_cfg=pc_vision_ppo_runner_cfg(route),
      runner_cls=PickPlaceOnPolicyRunner,
    )


register()
