"""A capability-gated curriculum for training the state teacher from random weights.

The default -Robust task ramps its penalties in by iteration count and decays
its guidance by iteration count (600 -> 1400).  Those schedules were tuned on
teachers that were warm-started from a policy that already grasped, and on a
cold start they fire before the policy has touched an object: the sight and
table penalties reach full weight at iteration 600 while grasp_attempts is
still zero, and the reach/lift/transport guidance is a quarter of its weight
by 1400.  d455_v7_cold (old convention, cold, 3000 iterations) never grasped;
v10b_sight (bounded convention, cold) never grasped either.

This curriculum advances on what the policy can DO, never on the clock:

  stage      gate (rolling, sustained)                    DR (actuator, scene)   penalties
  reach      --                                           nominal                near zero
  grasp      grasp_attempts/episode >= 0.5                half actuator          low
  place      placed/episode >= 0.5                        full actuator, half scene   medium
  robust     placed/episode >= 1.5 and safe episodes >= 0.9   full heavy DR      full, guidance decayed

Gate metrics are rolling means over the last ``window_iterations`` of finished
episodes; a gate has to hold for ``persist_iterations`` consecutive
iterations after at least ``min_dwell_iterations`` in the stage (the
hysteresis), and the stage index never goes down.  Every quantity changes
along a linear ramp of ``ramp_iterations`` after a transition.

The state teacher has no camera, so the vision DR group is 0 on the teacher
and is carried in the configuration only so the student stages can be gated
the same way; the link mass/COM randomisation is a startup event and stays at
its heavy value from iteration 0 (+-25% mass, +-2 mm COM).
"""

from __future__ import annotations

import collections
import json
import math
import os
import time
from typing import Any

import torch

from piper_push.tasks.pick_place import robust_cfg

STEPS_PER_ITERATION = 32  # rsl_rl num_steps_per_env for this task

# --- the schedule --------------------------------------------------------------

GUIDANCE = ("reach", "pads_touching", "holding", "lift", "transport", "object_in_bin", "jaws_ready")

def _weights(guidance_scale: float, action_rate: float, action_acc: float, premature: float,
             table: float, sight_arm: float, sight_hand: float, wrist: float) -> dict[str, float]:
  full = {"reach": 1.0, "pads_touching": 0.5, "holding": 0.8, "lift": 0.8,
          "transport": 1.5, "object_in_bin": 3.0, "jaws_ready": 0.4}
  w = {k: v * guidance_scale for k, v in full.items()}
  w.update({"action_rate": action_rate, "action_acc": action_acc, "premature_touch": premature,
            "table_touch": table, "sight_arm": sight_arm, "sight_hand": sight_hand,
            "wrist_side_on": wrist})
  return w


# v10d: the approach terms (mdp.approach_speed, object_disturbed, top_down_grasp),
# per stage.  The two penalties start small and reach full weight with the
# other style penalties; the top-down posture reward is guidance that is NOT
# decayed, because it is the deployment posture and not a hint on the way to
# one.  Weights: approach_speed is per m/s of excess, object_disturbed per m/s
# of object speed, top_down_grasp in [0, 1].
# command_acc is per (rad/s^2)^2 summed over six joints: the v10b/v10c teachers
# sit at ~17 rad/s^2 mean joint acceleration, so -1e-4 charges roughly 0.2/step
# at that level and nothing for a smooth full-speed move.  command_reversal is
# the fraction of joints that flipped direction (0.09-0.14 measured), charged
# so that a flip every ten steps costs about what reach pays.  action_rate/acc
# are the -Robust values x3, restoring the per-radian weight the bounded scale
# diluted.  joint_acc goes from a nominal -2e-7 to a weight that is felt.
APPROACH_TERMS = {
  "reach":  {"approach_speed": -0.3, "object_disturbed": -0.5, "top_down_grasp": 0.3,
             "command_acc": -1e-5, "command_reversal": -0.2, "joint_acc": -2e-5,
             "action_rate": -0.06, "action_acc": -0.03},
  "grasp":  {"approach_speed": -0.6, "object_disturbed": -1.0, "top_down_grasp": 0.3,
             "command_acc": -3e-5, "command_reversal": -0.5, "joint_acc": -5e-5,
             "action_rate": -0.12, "action_acc": -0.06},
  "place":  {"approach_speed": -1.0, "object_disturbed": -1.5, "top_down_grasp": 0.3,
             "command_acc": -6e-5, "command_reversal": -1.0, "joint_acc": -1e-4,
             "action_rate": -0.18, "action_acc": -0.09},
  "robust": {"approach_speed": -1.5, "object_disturbed": -2.0, "top_down_grasp": 0.3,
             "command_acc": -1e-4, "command_reversal": -1.5, "joint_acc": -2e-4,
             "action_rate": -0.45, "action_acc": -0.24},
}
# The penalties whose ramp the throughput guard may hold (v10d).
PENALTY_KEYS = ("approach_speed", "object_disturbed", "command_acc", "command_reversal", "joint_acc",
                "action_rate", "action_acc", "premature_touch", "table_touch", "sight_arm", "sight_hand")


def schedule(sight: bool, approach: bool = False) -> dict[str, Any]:
  """The whole curriculum as data, so it can be written next to the run.

  ``approach=True`` is the v10d variant: the same stages and gates with the
  approach-speed, object-disturbance and top-down terms added.
  """
  sched = _schedule(sight)
  if approach:
    sched["version"] = "v10d-1"
    sched["throughput_guard"] = {"hold_penalty_ramp_below_fraction_of_entry": 0.9,
                                 "note": "penalty weights stop ramping while rolling placed/episode is under 90% of "
                                         "its value when the stage was entered; DR and guidance ramps continue; "
                                         "stages never regress"}
    sched["plant"] = {"command_derate": 0.35}
    sched["exploration"] = {"entropy_coef": 0.004, "std_min": "0.1 x old joint-space noise per joint, 0.1 gripper"}
    for st in sched["stages"]:
      st["weights"].update(APPROACH_TERMS[st["name"]])
    sched["notes"]["approach_terms"] = ("approach_speed: grasp-site speed above an allowance falling from 0.6 m/s at "
                                        "15 cm to 0.1 m/s at 3 cm; object_disturbed: object speed while not held; "
                                        "top_down_grasp: verticality of the approach axis within 20 cm and while held")
  return sched


def _schedule(sight: bool) -> dict[str, Any]:
  s = 1.0 if sight else 0.0
  return {
    "version": "v10c-1",
    "budget_iterations": 9000,
    "min_full_dr_iterations": 1000,
    "window_iterations": 20,
    "persist_iterations": 10,
    "min_dwell_iterations": 100,
    "ramp_iterations": 200,
    "steps_per_iteration": STEPS_PER_ITERATION,
    "stages": [
      {"name": "reach", "gate": {},
       "dr": {"actuator": 0.0, "scene": 0.0, "vision": 0.0},
       "weights": _weights(1.0, -0.02, -0.01, -0.5, 0.0, 0.0, 0.0, 0.0)},
      {"name": "grasp", "gate": {"grasp_attempts_per_episode": 0.5},
       "dr": {"actuator": 0.5, "scene": 0.0, "vision": 0.0},
       "weights": _weights(1.0, -0.04, -0.02, -2.0, -0.3, -0.3 * s, -0.6 * s, 0.0)},
      {"name": "place", "gate": {"placed_per_episode": 0.5},
       "dr": {"actuator": 1.0, "scene": 0.5, "vision": 0.0},
       "weights": _weights(0.5, -0.06, -0.03, -6.0, -1.0, -1.0 * s, -2.0 * s, 0.3 * s)},
      {"name": "robust", "gate": {"placed_per_episode": 1.5, "safe_episode_fraction": 0.90},
       "dr": {"actuator": 1.0, "scene": 1.0, "vision": 0.0},
       "weights": _weights(0.25, -0.15, -0.08, -6.0, -2.0, -2.0 * s, -4.0 * s, 0.3 * s)},
    ],
    "notes": {
      "wrist_side_on": "0 until the place stage, then its final decayed value 0.3 (sight arm only): "
                       "a posture reward is not available before the policy grasps",
      "vision_dr": "the state teacher has no camera; the vision group is carried for the student stages",
      "link_inertia": "startup event, heavy value from iteration 0",
      "command_derate": "0.5 x trip speed, a safety-shell constant, not DR",
    },
  }


# --- domain-randomisation levels ------------------------------------------------

NOMINAL_TIMING = {
  "arm_latency_weights": (1.0, 0.0, 0.0), "arm_hold_weights": (1.0, 0.0, 0.0),
  "arm_response": (1.0, 1.0), "arm_deadband_rad": (0.0, 0.0),
  "gripper_latency_weights": (1.0, 0.0, 0.0), "gripper_hold_weights": (1.0, 0.0, 0.0),
  "gripper_response": (1.0, 1.0), "gripper_deadband_m": (0.0, 0.0),
}
NOMINAL_ROBOT = {"kp_scale": (1.0, 1.0), "kd_scale": (1.0, 1.0), "joint_friction_scale": (1.0, 1.0),
                 "gripper_kp_scale": (1.0, 1.0), "gripper_kd_scale": (1.0, 1.0)}
NOMINAL_SCENE = {"table_z_m": (-0.004, 0.004), "table_tilt_deg": (-0.25, 0.25),
                 "object_mass_kg": (0.05, 0.40), "object_friction": (0.4, 1.0),
                 "pad_friction": (0.55, 1.15)}


def _lerp(a, b, t: float):
  if isinstance(a, (tuple, list)):
    return tuple(_lerp(x, y, t) for x, y in zip(a, b))
  return float(a) + (float(b) - float(a)) * t


def _blend_probs(nominal, heavy, t: float) -> torch.Tensor:
  p = torch.tensor([_lerp(a, b, t) for a, b in zip(nominal, heavy)], dtype=torch.float32)
  return p / p.sum()


def apply_dr_level(env, actuator: float, scene: float) -> dict[str, Any]:
  """Set the actuator and scene DR groups to a fraction of the heavy profile."""
  timing, robot, sc = robust_cfg.HEAVY_DR_PROFILE["timing"], robust_cfg.HEAVY_DR_PROFILE["robot"], robust_cfg.HEAVY_DR_PROFILE["scene"]
  applied: dict[str, Any] = {}
  for term_name, prefix in (("arm", "arm"), ("gripper", "gripper")):
    term = env.action_manager.get_term(term_name)
    for hook in getattr(term, "_hooks", ()):
      hook._latency_probs = _blend_probs(NOMINAL_TIMING[f"{prefix}_latency_weights"], timing[f"{prefix}_latency_weights"], actuator).to(hook.device)
      hook._hold_probs = _blend_probs(NOMINAL_TIMING[f"{prefix}_hold_weights"], timing[f"{prefix}_hold_weights"], actuator).to(hook.device)
      hook.cfg.response_range = _lerp(NOMINAL_TIMING[f"{prefix}_response"], timing[f"{prefix}_response"], actuator)
      dkey = "gripper_deadband_m" if prefix == "gripper" else "arm_deadband_rad"
      hook.cfg.deadband_range = _lerp(NOMINAL_TIMING[dkey], timing[dkey], actuator)
      applied[f"{prefix}_response"] = hook.cfg.response_range
      applied[f"{prefix}_latency_probs"] = [round(float(x), 3) for x in hook._latency_probs.tolist()]
  ev = env.event_manager
  ev.get_term_cfg("robust_pd_gains").params["kp_range"] = _lerp(NOMINAL_ROBOT["kp_scale"], robot["kp_scale"], actuator)
  ev.get_term_cfg("robust_pd_gains").params["kd_range"] = _lerp(NOMINAL_ROBOT["kd_scale"], robot["kd_scale"], actuator)
  ev.get_term_cfg("robust_joint_friction").params["ranges"] = _lerp(NOMINAL_ROBOT["joint_friction_scale"], robot["joint_friction_scale"], actuator)
  ev.get_term_cfg("robust_gripper_gains").params["kp_range"] = _lerp(NOMINAL_ROBOT["gripper_kp_scale"], robot["gripper_kp_scale"], actuator)
  ev.get_term_cfg("robust_gripper_gains").params["kd_range"] = _lerp(NOMINAL_ROBOT["gripper_kd_scale"], robot["gripper_kd_scale"], actuator)
  applied["kp_range"] = ev.get_term_cfg("robust_pd_gains").params["kp_range"]
  pose = ev.get_term_cfg("reset_base").params["pose_range"]
  pose["z"] = _lerp(NOMINAL_SCENE["table_z_m"], sc["table_z_m"], scene)
  tilt = _lerp(NOMINAL_SCENE["table_tilt_deg"], sc["table_tilt_deg"], scene)
  pose["roll"] = pose["pitch"] = tuple(math.radians(x) for x in tilt)
  for name in [n for names in ev.active_terms.values() for n in names]:
    if name.startswith("object_shape"):
      ev.get_term_cfg(name).params["mass_range"] = _lerp(NOMINAL_SCENE["object_mass_kg"], sc["object_mass_kg"], scene)
      ev.get_term_cfg(name).params["friction_range"] = _lerp(NOMINAL_SCENE["object_friction"], sc["object_friction"], scene)
  ev.get_term_cfg("pad_friction").params["ranges"] = _lerp(NOMINAL_SCENE["pad_friction"], sc["pad_friction"], scene)
  applied["object_mass_kg"] = _lerp(NOMINAL_SCENE["object_mass_kg"], sc["object_mass_kg"], scene)
  applied["table_z_m"] = pose["z"]
  return applied


def apply_weights(env, weights: dict[str, float]) -> None:
  for name, w in weights.items():
    try:
      env.reward_manager.get_term_cfg(name).weight = float(w)
    except (KeyError, ValueError):
      pass  # the nosight variant has no sight terms to set


# --- the term ------------------------------------------------------------------


class cold_start_curriculum:
  """mjlab curriculum term; ``params={"sight": bool}``.  Returns a dict that
  mjlab logs under ``Curriculum/cold_start/*``."""

  def __init__(self, cfg, env) -> None:
    self.sched = schedule(bool(cfg.params.get("sight", True)), approach=bool(cfg.params.get("approach", False)))
    # A continuation (yf/pc, 2026-09-06) resumes AT the stage the checkpoint had
    # reached instead of re-opening every gate from nominal DR: the stage's own
    # weights and DR level apply from the first step, the blend is flat, and the
    # remaining gates (if any) work as they always did.  Unset means stage 0.
    start = int(os.environ.get("PIPER_COLD_START_STAGE", "0"))
    if not 0 <= start < len(self.sched["stages"]):
      raise ValueError(f"PIPER_COLD_START_STAGE={start} outside 0..{len(self.sched['stages']) - 1}")
    self.stage = start
    self.stage_entered_it = 0
    self.persist = 0
    self.iteration = -1
    self.full_dr_entered_it: int | None = 0 if start == len(self.sched["stages"]) - 1 else None
    self.window = collections.deque(maxlen=self.sched["window_iterations"])
    self._acc = {"episodes": 0.0, "placed": 0.0, "grasp_attempts": 0.0, "grasped_at_end": 0.0,
                 "over_speed": 0.0, "object_lost": 0.0}
    self._prev_targets = self._targets(start)
    self._guard = self.sched.get("throughput_guard")
    self._placed_at_entry = 0.0
    self._penalty_f = 1.0 if start > 0 else 0.0
    self._last_log_it = -1000
    self.log_path = os.environ.get("PIPER_CURRICULUM_LOG")
    first = self._targets(start)
    apply_weights(env, first["weights"])
    apply_dr_level(env, first["dr"]["actuator"], first["dr"]["scene"])
    self._current_dr = {k: round(float(v), 3) for k, v in first["dr"].items()}
    self._say(env, "start", extra={"stage": self.sched["stages"][start]["name"], "start_stage": start})

  # -- helpers --
  def _targets(self, stage: int) -> dict[str, Any]:
    st = self.sched["stages"][stage]
    return {"weights": dict(st["weights"]), "dr": dict(st["dr"])}

  def _blend(self, f: float, f_penalty: float | None = None) -> dict[str, Any]:
    new = self._targets(self.stage); old = self._prev_targets
    fp = f if f_penalty is None else f_penalty
    return {"weights": {k: _lerp(old["weights"][k], new["weights"][k], fp if k in PENALTY_KEYS else f)
                        for k in new["weights"]},
            "dr": {k: _lerp(old["dr"][k], new["dr"][k], f) for k in new["dr"]}}

  def metrics(self) -> dict[str, float]:
    n = sum(w["episodes"] for w in self.window)
    if n <= 0:
      return {"episodes": 0.0}
    m = {"episodes": n}
    for k in ("placed", "grasp_attempts", "grasped_at_end", "over_speed", "object_lost"):
      m[k] = sum(w[k] for w in self.window) / n
    m["placed_per_episode"] = m["placed"]
    m["grasp_attempts_per_episode"] = m["grasp_attempts"]
    m["safe_episode_fraction"] = 1.0 - m["over_speed"]
    return m

  def _gate_open(self, m: dict[str, float]) -> bool:
    if self.stage + 1 >= len(self.sched["stages"]):
      return False
    gate = self.sched["stages"][self.stage + 1]["gate"]
    if m.get("episodes", 0.0) < 1.0:
      return False
    return all(m.get(k, -1.0) >= v for k, v in gate.items())

  def _say(self, env, event: str, extra: dict | None = None) -> None:
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, "iteration": self.iteration,
           "stage": self.stage, "stage_name": self.sched["stages"][self.stage]["name"],
           "persist": self.persist, "metrics": self.metrics(),
           "weights": {k: round(float(env.reward_manager.get_term_cfg(k).weight), 4)
                       for k in self._targets(self.stage)["weights"] if k in env.reward_manager.active_terms},
           "dr": self._current_dr, "penalty_ramp": round(self._penalty_f, 3), "placed_at_entry": round(self._placed_at_entry, 3)}
    if extra:
      rec.update(extra)
    print(f"[cold-curriculum] {event} it {self.iteration} stage {rec['stage_name']} persist {self.persist} "
          f"metrics {json.dumps({k: round(v, 3) for k, v in rec['metrics'].items()})} dr {rec['dr']}", flush=True)
    if self.log_path:
      try:
        with open(self.log_path, "a") as fh:
          fh.write(json.dumps(rec) + "\n")
      except OSError:
        pass

  _current_dr: dict[str, float] = {"actuator": 0.0, "scene": 0.0, "vision": 0.0}

  # -- the per-reset call --
  def __call__(self, env, env_ids: torch.Tensor, sight: bool = True, approach: bool = False) -> dict[str, torch.Tensor]:
    cmd = env.command_manager.get_term("pick")
    tm = env.termination_manager
    n = float(len(env_ids)) if env_ids is not None else float(env.num_envs)
    ids = env_ids if env_ids is not None else slice(None)
    a = self._acc
    a["episodes"] += n
    a["placed"] += float(cmd.objects_placed[ids].sum())
    a["grasp_attempts"] += float(cmd.grasp_attempts[ids].sum())
    a["grasped_at_end"] += float(cmd.grasped[ids].float().sum())
    for k in ("over_speed", "object_lost"):
      if k in tm.active_terms:
        a[k] += float(tm.get_term(k)[ids].float().sum())

    it = int(env.common_step_counter) // self.sched["steps_per_iteration"]
    if it != self.iteration:
      # an iteration boundary: close the window entry, check the gate, apply the ramp
      self.iteration = it
      if a["episodes"] > 0:
        self.window.append(dict(a))
      self._acc = {k: 0.0 for k in a}
      m = self.metrics()
      dwell = it - self.stage_entered_it
      if self._gate_open(m) and dwell >= self.sched["min_dwell_iterations"]:
        self.persist += 1
      else:
        self.persist = 0
      if self.persist >= self.sched["persist_iterations"]:
        self._prev_targets = self._blend(min(1.0, dwell / self.sched["ramp_iterations"]), self._penalty_f)
        self._placed_at_entry = float(m.get("placed_per_episode", 0.0))
        self._penalty_f = 0.0
        self.stage += 1
        self.stage_entered_it = it
        self.persist = 0
        self.window.clear()
        if self.stage == len(self.sched["stages"]) - 1:
          self.full_dr_entered_it = it
        self._say(env, "advance", extra={"to": self.sched["stages"][self.stage]["name"]})
      f = min(1.0, (it - self.stage_entered_it) / self.sched["ramp_iterations"])
      if self._guard is None:
        self._penalty_f = f
      else:
        # Hold the penalty ramp while throughput has dropped below the guard;
        # it resumes from where it was, never goes back.
        floor = self._guard["hold_penalty_ramp_below_fraction_of_entry"] * self._placed_at_entry
        healthy = self._placed_at_entry <= 0.0 or m.get("placed_per_episode", 0.0) >= floor
        if healthy:
          self._penalty_f = max(self._penalty_f, f)
      cur = self._blend(f, self._penalty_f)
      apply_weights(env, cur["weights"])
      self._current_dr = {k: round(v, 3) for k, v in cur["dr"].items()}
      apply_dr_level(env, cur["dr"]["actuator"], cur["dr"]["scene"])
      if it - self._last_log_it >= 50:
        self._last_log_it = it
        self._say(env, "status")
    m = self.metrics()
    dev = env.device
    return {"stage": torch.tensor(float(self.stage), device=dev),
            "dr_actuator": torch.tensor(self._current_dr["actuator"], device=dev),
            "dr_scene": torch.tensor(self._current_dr["scene"], device=dev),
            "placed_per_episode": torch.tensor(m.get("placed_per_episode", 0.0), device=dev),
            "grasp_attempts_per_episode": torch.tensor(m.get("grasp_attempts_per_episode", 0.0), device=dev),
            "safe_episode_fraction": torch.tensor(m.get("safe_episode_fraction", 1.0), device=dev),
            "persist": torch.tensor(float(self.persist), device=dev)}
