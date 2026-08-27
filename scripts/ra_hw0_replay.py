"""Preview and envelope-check every RA-HW-0 trajectory, in simulation only.

    python scripts/ra_hw0_replay.py                     # preview + sim check
    python scripts/ra_hw0_replay.py --no-sim            # arithmetic only

Two jobs, and neither of them involves a robot:

1. **Preview.**  For each segment, print its duration, the joints it moves,
   the commanded amplitude and peak speed, and its per-joint envelope.  Gate
   H-A requires this to have been shown to a human before any motion, and the
   collector refuses to transmit a segment whose preview has not been written.
2. **Envelope check in the simulator.**  Drive the trained model's own
   kinematics through the command stream and record where the arm actually
   goes: the lowest point on any collision geometry, the closest approach to
   the table plane, and whether any joint leaves the limit table.  A
   trajectory that is safe on paper and puts a link through the table is a
   trajectory this catches.

The simulator used is the *state* task, which shares the vision task's scene,
robot and physics and skips the camera.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK = "Mjlab-Pick-Place-PiperX"


def sim_envelope(segments, table, device="cuda:0"):
  """Where the arm goes, in the simulator, for each command stream."""
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  cfg.seed = 0
  cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  robot = env.scene["robot"]
  jids, _ = robot.find_joints([f"joint{i}" for i in range(1, 7)],
                              preserve_order=True)
  # The base is bolted to the table at +10.9 mm and is meant to be there.
  # Measuring it as "the lowest link" makes every trajectory fail for the one
  # reason that has nothing to do with the trajectory.
  fixed = {"mocap_base", "base_link"}
  names = list(getattr(robot, "body_names", []))
  moving = [i for i, n in enumerate(names) if n not in fixed] or \
           list(range(len(names)))
  out = {"fixed_bodies_excluded": sorted(fixed & set(names)),
         "moving_bodies": [names[i] for i in moving]}
  try:
    for seg in segments:
      if not seg.q:
        out[seg.name] = {"transmitted": False,
                         "note": "H0 sends no command; nothing to check"}
        continue
      best = (float("inf"), None)
      span: dict[str, list[float]] = {}
      env.reset()
      for cmd in seg.q:
        q = torch.tensor([cmd], dtype=torch.float32, device=device)
        robot.write_joint_state_to_sim(
          q, torch.zeros_like(q),
          joint_ids=torch.tensor(jids, device=device))
        env.sim.forward()
        pos = robot.data.body_link_pos_w[0]
        for i in moving:
          zi = float(pos[i, 2])
          nm = names[i] if names else str(i)
          if zi < best[0]:
            best = (zi, nm)
          # The full Cartesian extent, not just height: a wrist pitch at this
          # posture moves the gripper almost horizontally, and a z-only
          # "did anything move" check would call that nothing.
          box = span.setdefault(nm, [[float("inf")] * 3, [float("-inf")] * 3])
          for k in range(3):
            v = float(pos[i, k])
            box[0][k] = min(box[0][k], v)
            box[1][k] = max(box[1][k], v)
      # A body whose height never changes across the whole command stream is
      # how a state write that silently did nothing would look, so the range
      # is reported rather than only the minimum.
      travel = {k: max(hi - lo for lo, hi in zip(*v)) for k, v in span.items()}
      moved = {k: v for k, v in travel.items() if v > 1e-6}
      out[seg.name] = {
        "bodies_that_moved_mm": {k: round(v * 1000, 3) for k, v in moved.items()},
        "cartesian_envelope_m": {k: {"min": [round(x, 4) for x in v[0]],
                                     "max": [round(x, 4) for x in v[1]]}
                                 for k, v in span.items()
                                 if travel[k] > 1e-6},
        "transmitted": True,
        "min_moving_link_z_m": best[0],
        "lowest_body": best[1],
        "table_z_m": 0.0,
        "clears_table": best[0] > 0.05,
        "n_commands": len(seg.q),
      }
  finally:
    env.close()
  return out


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--limits", default="configs/ra_hw0_safety_limits.json")
  ap.add_argument("--joint", default=None,
                  help="which joint H1 exercises; default is the first of the "
                       "pre-registered energy ordering")
  ap.add_argument("--no-sim", action="store_true")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", default="results/ra_hw0/preview.json")
  a = ap.parse_args()
  sys.path.insert(0, str(ROOT))
  sys.path.insert(0, str(ROOT / "src"))
  from hardware.ra_hw0 import limits as L
  from hardware.ra_hw0 import trajectories as T

  table = L.load(a.limits)
  joint = a.joint or T.H1_JOINT_ORDER[0]
  segments = [T.h0_hold()] + T.h1_plan(joint)

  report = {"phase": "RA-HW-0", "limits_sha_note": a.limits,
            "h1_joint": joint, "h1_joint_order": list(T.H1_JOINT_ORDER),
            "generated_host_wall_s": time.time(), "segments": []}
  print(f"H1 joint order (increasing energy): {', '.join(T.H1_JOINT_ORDER)}")
  print(f"this preview exercises: {joint}\n")
  all_reasons = []
  for seg in segments:
    reasons = T.check(seg, table)
    all_reasons += reasons
    print(" ", seg.describe())
    for r in reasons:
      print("      REFUSED:", r)
    report["segments"].append({
      "name": seg.name, "stage": seg.stage, "joint": seg.joint,
      "duration_s": seg.duration_s, "n_commands": seg.n,
      "amplitude_rad": seg.amplitude_rad,
      "peak_commanded_speed_rad_s": seg.peak_speed_rad_s,
      "envelope_rad": {k: list(v) for k, v in seg.envelope().items()},
      "refused": reasons,
      "description": seg.describe(),
    })

  seg_amplitude = {s_.name: s_.amplitude_rad for s_ in segments}
  if not a.no_sim:
    try:
      report["sim_envelope"] = sim_envelope(segments, table, a.device)
      for k, v in report["sim_envelope"].items():
        if isinstance(v, dict) and v.get("transmitted"):
          moved = v.get("bodies_that_moved_mm") or {}
          print(f"  {k}: lowest moving link ({v['lowest_body']}) "
                f"{v['min_moving_link_z_m']*1000:.1f} mm above the table "
                f"-> {'clear' if v['clears_table'] else 'TOO LOW'}; "
                f"{len(moved)} bodies moved"
                + (f", max travel {max(moved.values()):.1f} mm" if moved
                   else (" -- as commanded" if seg_amplitude.get(k, 0.0) == 0
                         else " -- NOTHING MOVED, which a working state write "
                              "cannot do")))
          if v["n_commands"] > 1 and not moved and seg_amplitude.get(k, 0.0) > 0:
            all_reasons.append(f"{k}: no body moved; the envelope check did "
                               "not exercise anything")
          if not v["clears_table"]:
            all_reasons.append(
              f"{k}: lowest moving link {v['lowest_body']} is "
              f"{v['min_moving_link_z_m']*1000:.1f} mm above the table, under "
              "the 50 mm preview minimum")
    except Exception as exc:
      report["sim_envelope"] = {"status": "FAILED",
                                "error": f"{type(exc).__name__}: {exc}"}
      all_reasons.append(f"simulated envelope check failed: {type(exc).__name__}")
      print("  simulated envelope check FAILED:", exc)
  else:
    report["sim_envelope"] = {"status": "SKIPPED (--no-sim)"}
    all_reasons.append("simulated envelope check was skipped")

  report["refusals"] = all_reasons
  report["preview_ok"] = not all_reasons
  report["motion_authorised"] = False
  out = Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2, default=str))
  print(f"\npreview {'OK' if report['preview_ok'] else 'REFUSED'} -> {out}")
  return 0 if report["preview_ok"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
