"""Camera depth against computed depth, frame by frame, through the deployment's own cloud.

A session recorded with ``--depth-source stereo`` archives two depth maps per
frame: the D455's own (``depth``) and the one the policy was given
(``policy_depth``, Fast-FoundationStereo).  This builds the point-cloud
observation from each with the same ``CloudObs`` the loop used -- same rig,
same cut above the plane -- and reports what the policy would have seen from
either, on identical frames:

    fill                 fraction of pixels with a depth
    workspace points     survivors inside the sector above the cut (the ``count`` the hold checks)
    object points        ``pc_obs.object_points``: > 15 mm above the plane, outside the arm and the bin
    near-table points    survivors between the cut and 12 mm -- table noise leaking through the cut
    agreement            median |camera - computed| where both are defined

The near-table row is the one the point-cloud line cares about: the 180 s
decay is objects ~24 mm tall lost at the cut, and the simulator's stress sweep
says a plane 4-8 mm off and noise x2 are the two things that hurt the policy.

    python scripts/pc/compare_depth_sources.py recordings/pc_noarm_<bundle>_stereo_<stamp> \\
        [--every 3] [--out compare.json]

Frames without ``policy_depth`` (a ``sensor`` session) are counted and skipped;
the script refuses a session that has none, rather than comparing depth with itself.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _pct(x, q):
  return float(np.percentile(x, q)) if len(x) else float("nan")


def summarise(rows: list[dict]) -> dict:
  """Per-source percentiles over the frames, plus the paired differences."""
  out = {"frames": len(rows)}
  for src in ("camera", "computed"):
    for key in ("fill", "workspace", "object", "near_table", "near_table_share"):
      v = np.asarray([r[src][key] for r in rows], dtype=np.float64)
      out[f"{src}_{key}"] = {"p10": _pct(v, 10), "p50": _pct(v, 50), "p90": _pct(v, 90)}
  for key in ("workspace", "object", "near_table"):
    d = np.asarray([r["computed"][key] - r["camera"][key] for r in rows], dtype=np.float64)
    out[f"delta_{key}"] = {"p10": _pct(d, 10), "p50": _pct(d, 50), "p90": _pct(d, 90)}
  agree = np.asarray([r["agreement_mm"] for r in rows if np.isfinite(r["agreement_mm"])])
  out["agreement_mm"] = {"p50": _pct(agree, 50), "p90": _pct(agree, 90)}
  both = np.asarray([r["both_defined"] for r in rows], dtype=np.float64)
  out["both_defined"] = {"p50": _pct(both, 50)}
  return out


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("session")
  p.add_argument("--every", type=int, default=1, help="use every n-th archived frame")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=None, help="JSON path; default <session>/depth_compare.json")
  a = p.parse_args()

  import torch
  from hardware.deploy import config, proprio
  from hardware.deploy.pc_obs import ARM_BODIES, CloudObs, object_points
  from piper_push.pc import cloud as pc_cloud
  from piper_push.pc import routes as pc_routes
  import mujoco

  session = pathlib.Path(a.session)
  run = json.loads((session / "run.json").read_text())
  args = run.get("args", {})
  if args.get("obs") != "pc":
    p.error(f"{session} is not a point-cloud session (obs={args.get('obs')!r})")
  camera = str(args.get("camera", "d455"))
  rig_file = args.get("rig_file") or str(pathlib.Path(config.RIG_FILE).with_name(
    f"rig{'' if camera == 'd405' else '_' + camera}.json"))
  rig = config.Rig.load(rig_file)
  bundle = pathlib.Path(args["policy"])
  route = json.loads((bundle / "manifest.json").read_text())["route"]
  cut = args.get("cloud_height_min")
  cut = float(cut) if cut is not None else pc_routes.crop_z_min(route)
  obs = CloudObs(rig, mode="cloud", num_points=pc_cloud.POINT_DIM * 128, device=a.device, height_min_m=cut)
  kin = proprio.Kinematics()
  body_ids = [mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ARM_BODIES]

  control = json.loads((session / "control.json").read_text())
  with_frame = [r for r in control if r.get("frame_file")]
  rows, skipped = [], 0
  for i, rec in enumerate(with_frame):
    if i % max(1, a.every):
      continue
    with np.load(session / rec["frame_file"], allow_pickle=False) as z:
      if "policy_depth" not in z:
        skipped += 1
        continue
      cam = np.asarray(z["depth"], dtype=np.float32) / 10000.0
      comp = np.asarray(z["policy_depth"], dtype=np.float32) / 10000.0
    q = np.asarray(rec["joint_pos"], dtype=np.float64)
    kin.update(q)
    arm = torch.as_tensor(np.asarray(kin.data.xpos[body_ids], dtype=np.float32), device=obs.device)
    row = {"frame": rec["frame_file"], "t": rec["t"]}
    for name, depth in (("camera", cam), ("computed", comp)):
      cf = obs(depth)
      inside = cf.inside[0]
      height = ((cf.points_base[0] - obs._p0) * obs._n).sum(-1)
      near = int((inside & (height < 0.012)).sum())
      row[name] = {
        "fill": float((depth > 0).mean()),
        "workspace": int(cf.count),
        "object": int(object_points(cf, arm, obs._p0, obs._n)) if cf.valid else 0,
        "near_table": near,
        "near_table_share": near / max(1, int(cf.count)),
      }
    both = (cam > 0) & (comp > 0)
    row["both_defined"] = float(both.mean())
    row["agreement_mm"] = float(np.median(np.abs(cam[both] - comp[both])) * 1000.0) if both.any() else float("nan")
    rows.append(row)

  if not rows:
    print(f"{session}: no frame carries policy_depth ({skipped} frames archived); this session ran on the "
          f"camera's depth, record one with DEPTH=stereo scripts/pc/deploy_pc.sh noarm", file=sys.stderr)
    return 2
  summary = summarise(rows)
  summary.update({"session": str(session), "route": route, "cut_m": cut, "rig_file": rig_file,
                  "frames_without_policy_depth": skipped, "every": a.every,
                  "depth_source_arg": args.get("depth_source"), "stereo_engine": args.get("stereo_engine")})
  out = pathlib.Path(a.out) if a.out else session / "depth_compare.json"
  out.write_text(json.dumps({"summary": summary, "frames": rows}, indent=1) + "\n")

  f = lambda d: f"{d['p10']:9.3f} {d['p50']:9.3f} {d['p90']:9.3f}"
  print(f"{session.name}: {len(rows)} frames compared, cut {cut * 1000:.0f} mm, route {route}")
  print(f"{'':28s}{'p10':>9s} {'p50':>9s} {'p90':>9s}")
  for key, label in (("fill", "fill"), ("workspace", "workspace points"), ("object", "object points"),
                     ("near_table", "near-table points"), ("near_table_share", "near-table share")):
    for src in ("camera", "computed"):
      print(f"{src + ' ' + label:28s}{f(summary[f'{src}_{key}'])}")
  print(f"{'computed - camera':22s}workspace {summary['delta_workspace']['p50']:+.0f}  object "
        f"{summary['delta_object']['p50']:+.0f}  near-table {summary['delta_near_table']['p50']:+.0f} (medians)")
  print(f"agreement where both defined: median {summary['agreement_mm']['p50']:.1f} mm, p90 "
        f"{summary['agreement_mm']['p90']:.1f} mm; both defined on {summary['both_defined']['p50'] * 100:.0f}% of pixels")
  print(f"wrote {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
