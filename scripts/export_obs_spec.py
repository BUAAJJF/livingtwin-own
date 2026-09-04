"""Write down exactly what the actor expects, so the robot cannot guess wrong.

The deployed pipeline has to rebuild the actor's input vector from scratch: no
observation manager, no simulator, just joint feedback and a camera.  Getting
the *values* right is the interesting part of that job.  Getting the *order*
right is not interesting at all, and it is the one that will silently ruin a
deployment -- a policy fed a correct vector in the wrong order does not crash,
it produces confident nonsense.

The order is also not something anyone should read off ``env_cfg.py``.  The
vision variant deletes the ``grasped`` term and adds ``squeeze``, and because
Python dicts keep insertion order that puts ``squeeze`` last, after
``actions``, rather than where ``grasped`` used to be.  That is correct
behaviour and it is invisible.

So the spec is exported from a built environment, which is the only thing that
knows.  ``hardware/deploy/proprio.py`` reads this file and refuses to run if it
cannot fill every term.

    micromamba run -n mjlab python scripts/export_obs_spec.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from piper_push import robot as piper

p = argparse.ArgumentParser()
p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Vision")
p.add_argument("--device", default="cuda:0")
p.add_argument("--out", default="hardware/deploy/obs_spec.json")
a = p.parse_args()

cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
agent = load_rl_cfg(a.task)
om = env.observation_manager
robot = env.scene["robot"]

obs, _ = env.reset()

groups = {}
for name in om.active_terms:
  terms = []
  offset = 0
  for term in om.active_terms[name]:
    cfg_t = om.get_term_cfg(name, term)
    value = cfg_t.func(env, **cfg_t.params)
    width = int(value.reshape(1, -1).shape[1])
    terms.append({"name": term, "offset": offset, "width": width})
    offset += width
  groups[name] = {"terms": terms, "total": offset,
                  "shape": list(obs[name].shape[1:])}

spec = {
  "task": a.task,
  "actor_groups": list(agent.obs_groups["actor"]),
  "groups": groups,
  "actions": {
    "terms": [{"name": n,
               "width": int(env.action_manager.get_term(n).action_dim)}
              for n in env.action_manager.active_terms],
    "total": int(env.action_manager.total_action_dim),
  },
  "joint_names": list(robot.joint_names),
  "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
  # The mapping from the policy's seven numbers to joint targets, as the
  # policy was trained with it.  hardware.deploy.robot.ActionMapper reads this
  # back; a spec without it (every export before 2026-09-05) is the v1
  # convention, which the mapper reconstructs from PICK_ARM_SCALE and the
  # default joint positions above.
  "action_spec": piper.action_spec(
    "bounded" if cfg.actions["arm"].bounded else "v1",
    dict(zip(robot.joint_names, robot.data.default_joint_pos[0].tolist()))),
  "control_hz": 1.0 / (env.cfg.sim.mujoco.timestep * env.cfg.decimation),
}

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(spec, indent=2))
print(json.dumps(spec, indent=2))
print(f"\nwrote {out}")
del torch
