"""Verify in simulation that a cadence setting changes what it claims to.

tests/test_cadence.py checks the randomiser in isolation.  This checks the
plumbing: that the command term calls it at the right moment, with the right
subset, and that the object the policy is handed after a placement really did
or really did not change.

Every Phase 2 result rests on this, and a silent failure here -- the redraw
never firing, or firing for a quantity that was supposed to be held -- would
look exactly like "cadence does not matter".

    python scripts/check_cadence.py Mjlab-Pick-Place-PiperX --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from piper_push import shapes

CADENCES = {
  "object": ("shape", "mass", "friction"),
  "episode": (),
  "shape": ("shape",),
  "mass": ("mass",),
  "friction": ("friction",),
}


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("task", nargs="?", default="Mjlab-Pick-Place-PiperX")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--steps", type=int, default=900)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--json", default=None)
  a = p.parse_args()

  report = {}
  for name, redraw in CADENCES.items():
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = 4242
    cfg.commands["pick"].redraw_on_place = redraw
    cfg.commands["pick"].reshape_on_place = bool(redraw)
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
    u, pick = env, env.command_manager.get_term("pick")
    st = shapes._state(u, "object")

    # A zero action is enough: the question is what happens at a PLACEMENT,
    # and placements are forced here rather than earned, so no policy is
    # needed and the check costs seconds.
    act = torch.zeros(a.num_envs, env.action_manager.total_action_dim,
                      device=a.device)
    counts = {k: 0 for k in ("placements", "shape_changed", "mass_changed",
                             "fric_changed")}
    with torch.inference_mode():
      env.reset()
      for i in range(a.steps):
        env.step(act)
        if i % 30 != 29:
          continue
        # Force a placement on half the environments and see what moved.
        ids = torch.arange(0, a.num_envs, 2, device=a.device)
        before = {k: st[k][ids].clone() for k in ("half", "mass", "fric")}
        pick._place_object(ids)
        counts["placements"] += len(ids)
        counts["shape_changed"] += int(
          (st["half"][ids] != before["half"]).any(dim=-1).sum())
        counts["mass_changed"] += int((st["mass"][ids] != before["mass"]).sum())
        counts["fric_changed"] += int((st["fric"][ids] != before["fric"]).sum())

    n = max(counts["placements"], 1)
    frac = {k: counts[k] / n for k in ("shape_changed", "mass_changed",
                                       "fric_changed")}
    report[name] = {"redraw": list(redraw), **counts, "fraction": frac}
    print(f"  {name:9s} redraw={str(list(redraw)):34s} "
          f"shape {100 * frac['shape_changed']:5.1f}%  "
          f"mass {100 * frac['mass_changed']:5.1f}%  "
          f"friction {100 * frac['fric_changed']:5.1f}%")
    env.close()

  print()
  ok = True
  for name, redraw in CADENCES.items():
    f = report[name]["fraction"]
    for q, key in (("shape", "shape_changed"), ("mass", "mass_changed"),
                   ("friction", "fric_changed")):
      want = q in redraw
      # A redrawn quantity should change essentially always; a held one never.
      # "Essentially" because two uniform draws can coincide, which for the
      # mass and friction scalars is a float-equality event and for the shape
      # requires the class and three continuous sizes to repeat.
      got = f[key] > 0.9 if want else f[key] < 0.02
      if not got:
        ok = False
        print(f"  MISMATCH {name}: {q} wanted redraw={want}, "
              f"changed {100 * f[key]:.1f}% of placements")
  print(f"  cadence plumbing {'OK' if ok else 'BROKEN'}")

  if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(
      {"task": a.task, "num_envs": a.num_envs, "verdict": "OK" if ok else "BROKEN",
       "cadences": report}, indent=1))
    print(f"  wrote {a.json}")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
