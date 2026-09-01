"""Put the simulator's channels and the rig's channels side by side.

``simscene`` answers "would this scene work in simulation".  When the answer is
yes and the rig still fails, the remaining difference is the observation, and
this measures it: for frames out of a recorded session, the simulator is driven
to the *same* arm pose with the object pinned at its measured place, its camera
is rendered, and the three channels are compared against the ones the
deployment pipeline built from the recording at that instant.

Nothing here is closed loop.  Both sides are anchored to the same measured
configuration, so every difference in the pictures is a difference in the
observation and not in what the two runs did.

    python -m hardware.deploy.simcompare logs/deploy/<session> \\
        --object-xy -0.169 0.386 --object-size 0.065 0.065 0.082 \\
        --shape cylinder --out review.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import pathlib
import sys
from dataclasses import asdict

import cv2
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from piper_push import robot as sim_robot

from . import config, mask, obs as dobs, proprio, rectify
from .simscene import CLASS_INDEX, TASK, pin_place, pin_shape

ARM = tuple(f"joint{i}" for i in range(1, 7))
GRIP = ("gripper_joint1", "gripper_joint2")


def _sim_camera(env, cmd, q: np.ndarray, device: str) -> np.ndarray:
  """The three channels the simulator would show for this arm pose."""
  ent = env.scene["robot"]
  arm_ids = ent.find_joints(list(ARM), preserve_order=True)[0]
  grip_ids = ent.find_joints(list(GRIP), preserve_order=True)[0]
  t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)[None]
  ent.write_joint_state_to_sim(t(q[:6]), t(np.zeros(6)), joint_ids=arm_ids)
  ent.write_joint_state_to_sim(t([q[6], -q[6]]), t(np.zeros(2)),
                               joint_ids=grip_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.scene.update(dt=0.0)
  env.sim.sense()
  out = env.observation_manager.compute(update_history=False)
  return out["camera"][0].detach().cpu().numpy()


def _real_camera(z, q, rig, reproj, seg, tracker, kin, plane):
  """The three channels ``run.py`` built from this frame."""
  kin.update(np.array([*q[:6], q[6], -q[6]]))
  depth = z["depth"].astype(np.float32) / 10000.0
  s = seg(depth, rgb=z["gray"], arm=kin.link_spheres())
  label = tracker.update(s, kin.site_pos)
  payload = mask.full_mask(s, label, seg.decimate) if label else None
  d, valid, target = reproj(depth, payload=payload)
  if plane is not None:
    d, valid = dobs.flatten_scene(
      d, valid, plane, points_base=reproj.virtual_points_base(d, rig),
      arm=kin.link_spheres())
  if target is None:
    target = np.zeros_like(valid)
  return dobs.camera_obs(d, valid, target > 0), s, label


def _png(a: np.ndarray) -> str:
  ok, buf = cv2.imencode(".png", a)
  if not ok:
    raise RuntimeError("encode failed")
  return "data:image/png;base64," + base64.b64encode(buf).decode()


def _grey(ch: np.ndarray) -> np.ndarray:
  return (np.clip(ch, 0, 1) * 255).astype(np.uint8)


def compare(session: pathlib.Path, xy, size, shape, device, every, limit,
            task: str = TASK):
  rig_file = session / "rig.json"
  rig = config.Rig.load(rig_file if rig_file.exists() else config.RIG_FILE)
  reproj = rectify.Reprojector(rig, device="cpu")
  seg = mask.DepthSegmenter(rig, reproj)
  tracker = mask.TargetTracker()
  kin = proprio.Kinematics()

  plane = None
  rj = session / "run.json"
  if rj.exists() and json.loads(rj.read_text()).get("args", {}).get(
      "flatten_scene"):
    plane = reproj.ground_plane_depth(rig)

  half = np.asarray(size, dtype=np.float64) / 2.0
  restore = pin_shape(half, CLASS_INDEX[shape])
  try:
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    cmd = env.command_manager.get_term("pick")
    pin_place(cmd, np.asarray(xy, dtype=np.float64), 0.0)
    env.reset()

    meta = {m["i"]: m for m in json.loads((session / "meta.json").read_text())
            if "i" in m and "joint_pos" in m}
    rows, cards = [], []
    for k, f in enumerate(sorted(glob.glob(str(session / "*.npz")))):
      m = meta.get(int(os.path.basename(f).split(".")[0]))
      if m is None:
        continue
      q = np.asarray(m["joint_pos"], dtype=np.float64)
      real, s, label = _real_camera(np.load(f), q, rig, reproj, seg, tracker,
                                    kin, plane)
      # The tracker's 3-of-5 confirmation is a filter over consecutive frames,
      # so every frame is processed even when only every Nth is compared.
      if k % every or len(rows) >= limit:
        continue
      sim = _sim_camera(env, cmd, q, device)
      d0 = np.abs(sim[0] - real[0])
      inter = float((sim[1] > 0.5).astype(np.float32).ravel()
                    @ (real[1] > 0.5).astype(np.float32).ravel())
      union = float(((sim[1] > 0.5) | (real[1] > 0.5)).sum())
      def centre(ch):
        v, u = np.nonzero(ch > 0.5)
        return (float(u.mean()), float(v.mean())) if u.size else (np.nan, np.nan)
      su, sv = centre(sim[1])
      ru, rv = centre(real[1])
      rows.append({
        "i": int(m["i"]),
        "sim_uv": [round(su, 1), round(sv, 1)],
        "rig_uv": [round(ru, 1), round(rv, 1)],
        "duv": [round(ru - su, 1), round(rv - sv, 1)],
        "scene_mad": float(d0.mean()),
        "scene_p90": float(np.percentile(d0, 90)),
        "sim_px": int((sim[1] > 0.5).sum()),
        "real_px": int((real[1] > 0.5).sum()),
        "iou": (inter / union) if union else float("nan"),
        "site_mm": round(float(kin.site_pos[2]) * 1000, 1),
      })
      cards.append((rows[-1], sim, real))
    return rows, cards, rig
  finally:
    shapes_restore(restore)
    try:
      env.close()
    except Exception:
      pass


def shapes_restore(original):
  from piper_push import shapes
  shapes._compose = original


def build_page(rows, cards, session, xy, size, shape) -> str:
  head = ("<meta charset='utf-8'><title>sim vs rig channels</title>"
          "<style>body{background:#111;color:#ddd;font:13px system-ui;"
          "margin:18px}h1{font-size:17px}table{border-collapse:collapse;"
          "margin:12px 0}td,th{padding:3px 9px;border-bottom:1px solid #333;"
          "text-align:right}th{color:#8ab}img{image-rendering:pixelated;"
          "width:224px;height:168px;border:1px solid #333}"
          ".r{display:flex;gap:6px;margin:10px 0;align-items:center}"
          ".c{color:#8ab;font-size:11px;width:70px}</style>")
  a = np.array([r["scene_mad"] for r in rows])
  iou = np.array([r["iou"] for r in rows])
  out = [head, f"<h1>{session.name} &mdash; simulator vs rig, same arm pose</h1>",
         f"<p>object pinned at {xy[0]:+.3f}, {xy[1]:+.3f} m &nbsp; "
         f"{shape} {size[0]*1000:.0f}&times;{size[1]*1000:.0f}&times;"
         f"{size[2]*1000:.0f} mm &nbsp; {len(rows)} frames compared</p>",
         "<table><tr><th>channel 0 mean |&Delta;|</th><th>median</th>"
         "<th>p90 frame</th><th>mask IoU median</th></tr>"
         f"<tr><td>{a.mean():.3f}</td><td>{np.median(a):.3f}</td>"
         f"<td>{np.percentile(a,90):.3f}</td>"
         f"<td>{np.nanmedian(iou):.3f}</td></tr></table>"]
  for r, sim, real in cards:
    out.append("<div class=r><div class=c>frame %d<br>hand %.0f mm<br>"
               "IoU %.2f</div>" % (r["i"], r["site_mm"], r["iou"]))
    for label, img in (("sim ch0", sim[0]), ("rig ch0", real[0]),
                       ("sim ch1", sim[1]), ("rig ch1", real[1]),
                       ("|&Delta;| ch0", np.abs(sim[0] - real[0]) * 3.0)):
      out.append(f"<div><div class=c>{label}</div>"
                 f"<img src='{_png(_grey(img))}'></div>")
    out.append("</div>")
  return "".join(out)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("session", type=pathlib.Path)
  p.add_argument("--object-xy", type=float, nargs=2, required=True)
  p.add_argument("--object-size", type=float, nargs=3, required=True)
  p.add_argument("--shape", default="cylinder")
  p.add_argument("--task", default=TASK,
                 help="Mjlab-Pick-Place-PiperX-Vision for the clean reference; "
                      "the -Robust variant carries the training DR, which is "
                      "noise when the question is where the object lands")
  p.add_argument("--every", type=int, default=25)
  p.add_argument("--limit", type=int, default=12)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("simcompare.html"))
  a = p.parse_args()
  rows, cards, _ = compare(a.session, a.object_xy, a.object_size, a.shape,
                           a.device, a.every, a.limit, a.task)
  if not rows:
    raise SystemExit("no frames compared")
  a.out.write_text(build_page(rows, cards, a.session, a.object_xy,
                              a.object_size, a.shape))
  sc = np.array([r["scene_mad"] for r in rows])
  iou = np.array([r["iou"] for r in rows])
  print("frames %d" % len(rows))
  print("channel 0 mean |delta|  %.4f  (median %.4f, p90 %.4f)"
        % (sc.mean(), np.median(sc), np.percentile(sc, 90)))
  print("target mask IoU         %.3f  (sim %.0f px, rig %.0f px)"
        % (np.nanmedian(iou), np.median([r["sim_px"] for r in rows]),
           np.median([r["real_px"] for r in rows])))
  du = np.array([r["duv"][0] for r in rows], dtype=float)
  dv = np.array([r["duv"][1] for r in rows], dtype=float)
  print("mask centre rig - sim   u %+.1f px   v %+.1f px  (224x168 grid)"
        % (np.nanmedian(du), np.nanmedian(dv)))
  print("wrote %s" % a.out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
