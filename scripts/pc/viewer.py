"""A frame-by-frame page of what a point-cloud policy saw and did, in one self-contained HTML.

    python scripts/pc/viewer.py --checkpoint checkpoints/pc/pc_final_P1B_model_799.pt \\
        --task Mjlab-Pick-Place-PiperX-PC-P1B-Vision --steps 1500 --out results/pc/p1b_viewer.html

One environment, the measured sensor, the policy deterministic.  Every control
step stores the cloud the policy was actually handed (after the 30 Hz hold and
the processing lag), its vision_meta, the grasp site, the TRUE object position
(privileged -- drawn for the reader, never fed to the policy), the jaw, the
action, whether the command says "held", and the running placement count.
The page draws a top view and a side view of the cloud coloured by height,
with the frame number burned into the canvas, a timeline with the placements
marked, and arrow keys / a slider to step.  No CDN, no server: file://.
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{margin:0;background:#111;color:#ddd;font:13px/1.4 system-ui,sans-serif}
#top{display:flex;gap:12px;padding:10px;align-items:flex-start;flex-wrap:wrap}
canvas{background:#181818;border:1px solid #333}
#panel{min-width:300px;max-width:360px}
#panel table{border-collapse:collapse;width:100%}
#panel td{padding:2px 6px;border-bottom:1px solid #2a2a2a;vertical-align:top}
#panel td:first-child{color:#999;white-space:nowrap}
#tl{display:block;margin:0 10px 10px 10px;background:#181818;border:1px solid #333}
input[type=range]{width:100%}
.k{color:#8cf}
</style></head><body>
<div id="top">
 <div><div>top view (x right, y up)</div><canvas id="c1" width="560" height="560"></canvas></div>
 <div><div>side view (y right, z up)</div><canvas id="c2" width="560" height="380"></canvas></div>
 <div><div id="camcap">camera frame</div><img id="rgb" width="448" height="336" style="display:block;border:1px solid #333;image-rendering:auto"><div>depth the cloud came from (0.3 m red … 1.5 m blue, holes at the far plane)</div><img id="dep" width="448" height="336" style="display:block;border:1px solid #333"></div>
 <div id="panel">
  <div><b>__TITLE__</b></div>
  <div class="k">← → step &nbsp; shift+← → ×10 &nbsp; space play &nbsp; h = held frames only</div>
  <input id="s" type="range" min="0" max="0" value="0">
  <table id="t"></table>
  <div style="margin-top:8px;color:#999">points coloured by height (blue 1 cm → red 20 cm+); green cross = grasp site; orange ring = true object (privileged, shown for you); box = bin; arcs = workspace sector.  Frame id is burned into every canvas.</div>
 </div>
</div>
<canvas id="tl" width="1140" height="70"></canvas>
<script>
const D = __DATA__;
const clouds = D.clouds.map(b64 => { const s = atob(b64); const a = new Int16Array(s.length/2); const dv = new DataView(new ArrayBuffer(s.length)); for (let i=0;i<s.length;i++) dv.setUint8(i, s.charCodeAt(i)); for (let i=0;i<a.length;i++) a[i] = dv.getInt16(2*i, true); return a; });
const S = D.steps; let i = 0, playing = false, heldOnly = false;
const c1 = document.getElementById('c1'), c2 = document.getElementById('c2'), tl = document.getElementById('tl'), sl = document.getElementById('s');
sl.max = S.length - 1;
const X = [-0.65, 0.65], Y = [-0.05, 0.75], Z = [-0.05, 0.45];
function mapTop(x, y){ return [ (x - X[0])/(X[1]-X[0]) * c1.width, c1.height - (y - Y[0])/(Y[1]-Y[0]) * c1.height ]; }
function mapSide(y, z){ return [ (y - Y[0])/(Y[1]-Y[0]) * c2.width, c2.height - (z - Z[0])/(Z[1]-Z[0]) * c2.height ]; }
function col(z){ const t = Math.max(0, Math.min(1, (z - 0.01) / 0.19)); const r = Math.round(255*t), b = Math.round(255*(1-t)); return `rgb(${r},${Math.round(120*(1-Math.abs(2*t-1)))},${b})`; }
function drawSector(ctx, map){ ctx.strokeStyle='#444'; ctx.lineWidth=1; ctx.beginPath();
  for (const r of [D.ws.r_min, D.ws.r_max]) { let first=true; for (let a=D.ws.a_lo; a<=D.ws.a_hi; a+=0.02){ const p=map(r*Math.cos(a), r*Math.sin(a)); if(first){ctx.moveTo(p[0],p[1]); first=false;} else ctx.lineTo(p[0],p[1]); } }
  ctx.stroke();
  for (const a of [D.ws.a_lo, D.ws.a_hi]) { ctx.beginPath(); let p=map(D.ws.r_min*Math.cos(a), D.ws.r_min*Math.sin(a)); ctx.moveTo(p[0],p[1]); p=map(D.ws.r_max*Math.cos(a), D.ws.r_max*Math.sin(a)); ctx.lineTo(p[0],p[1]); ctx.stroke(); }
  ctx.strokeStyle='#777'; const b=D.bin; const p0=map(b[0]-b[2], b[1]-b[3]), p1=map(b[0]+b[2], b[1]+b[3]); ctx.strokeRect(Math.min(p0[0],p1[0]), Math.min(p0[1],p1[1]), Math.abs(p1[0]-p0[0]), Math.abs(p1[1]-p0[1])); }
function draw(){
  const s = S[i]; const pts = s.cloud >= 0 ? clouds[s.cloud] : null;
  for (const [cv, map, planar] of [[c1, mapTop, true], [c2, mapSide, false]]) {
    const ctx = cv.getContext('2d'); ctx.fillStyle='#181818'; ctx.fillRect(0,0,cv.width,cv.height);
    if (planar) drawSector(ctx, map); else { ctx.strokeStyle='#444'; const p=map(Y[0],0), q=map(Y[1],0); ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke(); ctx.strokeStyle='#777'; const b=D.bin; const a0=map(b[1]-b[3],0), a1=map(b[1]+b[3],0.06); ctx.strokeRect(Math.min(a0[0],a1[0]), Math.min(a0[1],a1[1]), Math.abs(a1[0]-a0[0]), Math.abs(a1[1]-a0[1])); }
    if (pts) { for (let k=0;k<pts.length;k+=3){ const x=pts[k]/1000, y=pts[k+1]/1000, z=pts[k+2]/1000; const p = planar ? map(x,y) : map(y,z); ctx.fillStyle = col(z); ctx.fillRect(p[0]-1, p[1]-1, 2.5, 2.5); } }
    const o = s.obj, e = s.ee;
    let p = planar ? map(o[0],o[1]) : map(o[1],o[2]); ctx.strokeStyle='#ffa500'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(p[0],p[1],7,0,6.283); ctx.stroke();
    p = planar ? map(e[0],e[1]) : map(e[1],e[2]); ctx.strokeStyle = s.held ? '#ff0' : '#0f0'; ctx.beginPath(); ctx.moveTo(p[0]-8,p[1]); ctx.lineTo(p[0]+8,p[1]); ctx.moveTo(p[0],p[1]-8); ctx.lineTo(p[0],p[1]+8); ctx.stroke();
    ctx.fillStyle='#fff'; ctx.font='bold 16px monospace'; ctx.fillText(`frame ${i}  t=${s.t.toFixed(2)}s  ${s.fresh?'FRESH':'held'} age=${s.age}  ${s.valid?'':'NO VALID DEPTH'}`, 8, 20);
  }
  const rows = [['step / t', `${i} / ${s.t.toFixed(2)} s`], ['vision', `${s.fresh?'fresh':'held frame'}, age ${s.age} steps, ${s.valid?'valid':'INVALID'}, ${s.npts} pts`],
    ['grasp site', s.ee.map(v=>v.toFixed(3)).join(', ')], ['object (true)', s.obj.map(v=>v.toFixed(3)).join(', ')], ['ee→object', `${(s.dist*1000).toFixed(0)} mm`],
    ['held / placed', `${s.held ? 'HELD' : 'no'} / ${s.placed}`], ['jaw', `${(s.jaw*1000).toFixed(1)} mm  (cmd ${(s.jaw_cmd*1000).toFixed(1)})`],
    ['a = tanh(u)', s.a.map(v=>v.toFixed(2)).join(' ')], ['|u| max', s.umax.toFixed(2)], ['termination', s.term || '—']];
  document.getElementById('t').innerHTML = rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
  const im = D.images[String(s.img)]; if (im) { document.getElementById('rgb').src = 'data:image/jpeg;base64,' + im.rgb; if (im.depth) document.getElementById('dep').src = 'data:image/jpeg;base64,' + im.depth; }
  document.getElementById('camcap').textContent = `camera frame captured at step ${s.img} (the frame this cloud came from; ${s.age} step(s) old)`;
  sl.value = i; drawTimeline();
}
function drawTimeline(){ const ctx = tl.getContext('2d'); ctx.fillStyle='#181818'; ctx.fillRect(0,0,tl.width,tl.height);
  const n = S.length; for (let k=0;k<n;k++){ const x = k/n*tl.width; if (S[k].held){ ctx.fillStyle='#664'; ctx.fillRect(x, 30, Math.max(1,tl.width/n), 20);} if (!S[k].valid){ ctx.fillStyle='#833'; ctx.fillRect(x, 52, Math.max(1,tl.width/n), 8);} if (k>0 && S[k].placed>S[k-1].placed){ ctx.fillStyle='#4f4'; ctx.fillRect(x-1, 4, 3, 24);} if (S[k].term){ ctx.fillStyle='#f44'; ctx.fillRect(x-1, 4, 2, 60);} }
  ctx.fillStyle='#fff'; ctx.fillRect(i/n*tl.width, 0, 2, tl.height); ctx.fillStyle='#aaa'; ctx.font='11px monospace'; ctx.fillText('green = placement, yellow = held, red bar = invalid frame, red line = termination', 6, 66); }
function go(k){ i = Math.max(0, Math.min(S.length-1, k)); draw(); }
document.addEventListener('keydown', e => { const d = e.shiftKey ? 10 : 1; if (e.key==='ArrowRight') go(i+d); else if (e.key==='ArrowLeft') go(i-d); else if (e.key===' ') { playing=!playing; e.preventDefault(); } else if (e.key==='h') heldOnly=!heldOnly; });
sl.addEventListener('input', () => go(parseInt(sl.value)));
tl.addEventListener('click', e => go(Math.floor(e.offsetX / tl.width * S.length)));
setInterval(() => { if (playing) { let k = i + 1; if (heldOnly) { while (k < S.length && !S[k].held) k++; } if (k >= S.length) playing = false; else go(k); } }, 40);
draw();
</script></body></html>
"""


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-PC-P1B-Vision")
  p.add_argument("--steps", type=int, default=1500)
  p.add_argument("--seed", type=int, default=101)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", required=True)
  from piper_push import evalcfg
  evalcfg.add_sensor_arg(p, default="measured")
  evalcfg.add_action_api_arg(p)
  a = p.parse_args()
  evalcfg.apply_action_api_arg(a)

  import torch
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from piper_push import objects
  from piper_push import robot as piper
  from piper_push.pc import cloud as pc_cloud
  from piper_push.tasks.pick_place import env_cfg as task_cfg

  import cv2
  import dataclasses as _dc
  from piper_push import camera as camera_mod

  torch.manual_seed(a.seed)
  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = 1
  cfg.seed = a.seed
  # The policy's camera renders depth and segmentation; for the reader the
  # same camera also renders RGB here.  Rendering only -- nothing reaches the
  # observation.
  cfg.scene.sensors = tuple(
    _dc.replace(s, data_types=tuple(dict.fromkeys(list(s.data_types) + ["rgb"])))
    if getattr(s, "name", None) == camera_mod.CAMERA_NAME else s
    for s in cfg.scene.sensors)
  sensor_prov = evalcfg.apply_sensor(cfg, a.task, a.sensor)
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  agent = load_rl_cfg(a.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(wrapped, asdict(agent), None, a.device)
  loaded = evalcfg.load_weights(runner, a.checkpoint, a.device)
  policy = runner.get_inference_policy(device=a.device)
  cmd = env.command_manager.get_term("pick")
  robot = env.scene["robot"]
  obj = env.scene["object"]
  tm = env.termination_manager
  causes = [n for n in ("over_speed", "object_lost", "nan", "time_out") if n in tm.active_terms]
  origin = env.scene.env_origins[0]

  env.reset()
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]
  steps, clouds = [], []
  images: dict[int, dict] = {}
  owner = env._pc_cloud_owner
  sensor = env.scene[camera_mod.CAMERA_NAME]

  def jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

  def snapshot(t: int) -> None:
    """The camera frame captured this step: RGB and the corrupted depth the cloud came from."""
    rgb = sensor.data.rgb[0].cpu().numpy()
    d = owner.last_depth[0].cpu().numpy() if getattr(owner, "last_depth", None) is not None else None
    dep = ""
    if d is not None:
      g = np.clip((d - 0.30) / (1.50 - 0.30), 0.0, 1.0)
      dep = jpeg(cv2.applyColorMap((255 * (1.0 - g)).astype(np.uint8), cv2.COLORMAP_TURBO))
    images[t] = {"rgb": jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), "depth": dep}

  last_cloud_id = -1
  last_cloud_sig = None
  snapshot(-1)
  for t in range(a.steps):
    with torch.inference_mode():
      u = policy(obs)
    act = torch.tanh(u)[0]
    cam = obs["camera"][0]                      # what the policy was handed this step
    meta = obs["vision_meta"][0]
    fresh = bool(meta[1] > 0.5)
    sig = float(cam[:, :3].abs().sum())
    if fresh or last_cloud_sig is None or sig != last_cloud_sig:
      pts = cam[cam[:, 3] > 0.5][:, :3]
      arr = (pts * 1000.0).round().clamp(-32000, 32000).to(torch.int16).cpu().numpy().reshape(-1)
      clouds.append(base64.b64encode(arr.tobytes()).decode("ascii"))
      last_cloud_id = len(clouds) - 1
      last_cloud_sig = sig
    ee = (cmd._site_pos_w()[0] - origin).cpu().numpy()
    op = (obj.data.root_link_pos_w[0] - origin).cpu().numpy()
    jaw = float(robot.data.joint_pos[0, 6])
    steps.append({
      "t": round(t * env.step_dt, 3), "cloud": last_cloud_id, "fresh": fresh,
      "age": int(round(float(meta[0]) * pc_cloud.AGE_NORM)), "valid": bool(meta[2] > 0.5),
      "npts": int((cam[:, 3] > 0.5).sum()),
      "ee": [round(float(x), 4) for x in ee], "obj": [round(float(x), 4) for x in op],
      "dist": round(float(np.linalg.norm(ee - op)), 4),
      "held": bool(cmd.grasped[0]), "placed": int(cmd.objects_placed[0]),
      "jaw": round(jaw, 4), "jaw_cmd": round(float(piper.GRIPPER_OFFSET + piper.GRIPPER_SCALE * act[-1]), 4),
      "a": [round(float(x), 3) for x in act.cpu().numpy()], "umax": round(float(u.abs().max()), 3),
      "term": None,
    })
    obs, _, dones, _ = wrapped.step(u)
    from eval_occlusion import reset_recurrent
    reset_recurrent(policy, dones)
    for c in causes:
      if bool(tm.get_term(c)[0]):
        steps[-1]["term"] = c
    # The capture made during this step (the obs just returned is for step t+1).
    if owner.fresh is not None and bool(owner.fresh[0]):
      snapshot(t)
  env.close()
  # Which capture each step's cloud came from: the delayed age says how far back.
  keys = sorted(images)
  for t, s in enumerate(steps):
    want = t - 1 - s["age"]           # obs at step t was captured at t-1-age (t-1 is the step whose obs this is)
    k = max([x for x in keys if x <= want], default=keys[0])
    s["img"] = k
  placed = steps[-1]["placed"] if steps else 0
  ws = pc_cloud.WORKSPACE
  bx, by = objects.BIN_CENTER
  hx, hy = objects.BIN_INNER
  data = {"steps": steps, "clouds": clouds, "images": {str(k): v for k, v in images.items()},
          "ws": {"r_min": ws.r_min, "r_max": ws.r_max, "a_lo": ws.angle_lo, "a_hi": ws.angle_hi},
          "bin": [bx, by, hx + objects.BIN_WALL_THICKNESS, hy + objects.BIN_WALL_THICKNESS],
          "meta": {"checkpoint": a.checkpoint, "task": a.task, "seed": a.seed, "steps": a.steps,
                   "placed": placed, "weights": loaded, "sensor": sensor_prov}}
  title = f"{pathlib.Path(a.checkpoint).name} on {a.task} seed {a.seed}: {placed} placed in {a.steps * env.step_dt:.0f} s"
  html = PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, default=str))
  out = pathlib.Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(html)
  print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(steps)} steps, {len(clouds)} clouds, {placed} placed, "
        f"terminations {sum(1 for s in steps if s['term'])}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
