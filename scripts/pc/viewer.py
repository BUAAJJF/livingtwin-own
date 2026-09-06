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

Second generation additions: the phase the step is in (approach / engaged /
carry / place, as ``scripts/pc/eval_initiation.py`` defines them), attempts
(cyan ticks) and stalls (grey bands, approach runs of at least ``--idle-s``)
on the timeline, the target's pixels surviving the crop and its points among
the sampled set on every capture, and -- when the cloud carries a fifth
column -- the points flagged as the target drawn in magenta.  Keys ``n`` /
``p`` jump to the next / previous attempt, ``s`` to the next stall.
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
  <div class="k">← → step &nbsp; shift+← → ×10 &nbsp; space play &nbsp; h = held frames only &nbsp; n/p = next/prev attempt &nbsp; s = next stall</div>
  <input id="s" type="range" min="0" max="0" value="0">
  <table id="t"></table>
  <div style="margin-top:8px;color:#999">points coloured by height (blue 1 cm → red 20 cm+), magenta = flagged as the target in the observation (5-column routes only); green cross = grasp site; orange ring = true object (privileged, shown for you); box = bin; arcs = workspace sector.  Frame id is burned into every canvas.</div>
 </div>
</div>
<canvas id="tl" width="1140" height="70"></canvas>
<script>
const D = __DATA__;
const clouds = D.clouds.map(b64 => { const s = atob(b64); const a = new Int16Array(s.length/2); const dv = new DataView(new ArrayBuffer(s.length)); for (let i=0;i<s.length;i++) dv.setUint8(i, s.charCodeAt(i)); for (let i=0;i<a.length;i++) a[i] = dv.getInt16(2*i, true); return a; });
const flags = (D.flags || []).map(b64 => { if (!b64) return null; const s = atob(b64); const a = new Uint8Array(s.length); for (let i=0;i<s.length;i++) a[i] = s.charCodeAt(i); return a; });
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
    const fl = (s.cloud >= 0 && flags.length) ? flags[s.cloud] : null;
    if (pts) { for (let k=0;k<pts.length;k+=3){ const x=pts[k]/1000, y=pts[k+1]/1000, z=pts[k+2]/1000; const p = planar ? map(x,y) : map(y,z); const tgt = fl && fl[k/3]; ctx.fillStyle = tgt ? '#ff40ff' : col(z); const sz = tgt ? 4 : 2.5; ctx.fillRect(p[0]-sz/2, p[1]-sz/2, sz, sz); } }
    const o = s.obj, e = s.ee;
    let p = planar ? map(o[0],o[1]) : map(o[1],o[2]); ctx.strokeStyle='#ffa500'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(p[0],p[1],7,0,6.283); ctx.stroke();
    p = planar ? map(e[0],e[1]) : map(e[1],e[2]); ctx.strokeStyle = s.held ? '#ff0' : '#0f0'; ctx.beginPath(); ctx.moveTo(p[0]-8,p[1]); ctx.lineTo(p[0]+8,p[1]); ctx.moveTo(p[0],p[1]-8); ctx.lineTo(p[0],p[1]+8); ctx.stroke();
    ctx.fillStyle='#fff'; ctx.font='bold 16px monospace'; ctx.fillText(`frame ${i}  t=${s.t.toFixed(2)}s  ${s.fresh?'FRESH':'held'} age=${s.age}  ${s.valid?'':'NO VALID DEPTH'}`, 8, 20);
    ctx.fillStyle='#ccc'; ctx.font='13px monospace'; ctx.fillText(`${s.phase}${s.stall?'  STALL':''}${s.attempt?'  ATTEMPT':''}  target px ${s.tgt_px} / pts ${s.tgt_pts}`, 8, 38);
  }
  const rows = [['step / t', `${i} / ${s.t.toFixed(2)} s`], ['vision', `${s.fresh?'fresh':'held frame'}, age ${s.age} steps, ${s.valid?'valid':'INVALID'}, ${s.npts} pts`],
    ['grasp site', s.ee.map(v=>v.toFixed(3)).join(', ')], ['object (true)', s.obj.map(v=>v.toFixed(3)).join(', ')], ['ee→object', `${(s.dist*1000).toFixed(0)} mm`],
    ['held / placed', `${s.held ? 'HELD' : 'no'} / ${s.placed}`], ['phase', `${s.phase}${s.stall ? ' (stall)' : ''}${s.attempt ? ' -- attempt' : ''}`],
    ['target visible', `${s.tgt_px} px after crop, ${s.tgt_pts} of the sampled points${s.tgt_obs !== undefined ? ', ' + s.tgt_obs + ' flagged in the obs' : ''}`],
    ['jaw', `${(s.jaw*1000).toFixed(1)} mm  (cmd ${(s.jaw_cmd*1000).toFixed(1)})`],
    ['a = tanh(u)', s.a.map(v=>v.toFixed(2)).join(' ')], ['|u| max', s.umax.toFixed(2)], ['termination', s.term || '—']];
  document.getElementById('t').innerHTML = rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
  const im = D.images[String(s.img)]; if (im) { document.getElementById('rgb').src = 'data:image/jpeg;base64,' + im.rgb; if (im.depth) document.getElementById('dep').src = 'data:image/jpeg;base64,' + im.depth; }
  document.getElementById('camcap').textContent = `camera frame captured at step ${s.img} (the frame this cloud came from; ${s.age} step(s) old)`;
  sl.value = i; drawTimeline();
}
function drawTimeline(){ const ctx = tl.getContext('2d'); ctx.fillStyle='#181818'; ctx.fillRect(0,0,tl.width,tl.height);
  const n = S.length; for (let k=0;k<n;k++){ const x = k/n*tl.width; if (S[k].stall){ ctx.fillStyle='#555'; ctx.fillRect(x, 30, Math.max(1,tl.width/n), 20);} if (S[k].held){ ctx.fillStyle='#664'; ctx.fillRect(x, 30, Math.max(1,tl.width/n), 20);} if (!S[k].valid){ ctx.fillStyle='#833'; ctx.fillRect(x, 52, Math.max(1,tl.width/n), 8);} if (S[k].attempt){ ctx.fillStyle='#4ff'; ctx.fillRect(x-1, 20, 2, 12);} if (k>0 && S[k].placed>S[k-1].placed){ ctx.fillStyle='#4f4'; ctx.fillRect(x-1, 4, 3, 24);} if (S[k].term){ ctx.fillStyle='#f44'; ctx.fillRect(x-1, 4, 2, 60);} }
  ctx.fillStyle='#fff'; ctx.fillRect(i/n*tl.width, 0, 2, tl.height); ctx.fillStyle='#aaa'; ctx.font='11px monospace'; ctx.fillText('green = placement, cyan = attempt, yellow = held, grey = stall (no engagement), red bar = invalid frame, red line = termination', 6, 66); }
function go(k){ i = Math.max(0, Math.min(S.length-1, k)); draw(); }
function jump(pred, dir){ let k = i + dir; while (k >= 0 && k < S.length && !pred(S[k], k)) k += dir; if (k >= 0 && k < S.length) go(k); }
document.addEventListener('keydown', e => { const d = e.shiftKey ? 10 : 1; if (e.key==='ArrowRight') go(i+d); else if (e.key==='ArrowLeft') go(i-d); else if (e.key===' ') { playing=!playing; e.preventDefault(); } else if (e.key==='h') heldOnly=!heldOnly; else if (e.key==='n') jump(s=>s.attempt, 1); else if (e.key==='p') jump(s=>s.attempt, -1); else if (e.key==='s') jump((s,k)=>s.stall && !(k>0 && S[k-1].stall), 1); });
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
  p.add_argument("--idle-s", type=float, default=3.0, help="an approach run at least this long is marked as a stall")
  p.add_argument("--episode-length-s", type=float, default=None, help="override the play config's 40 s time-out")
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
  if a.episode_length_s is not None:
    cfg.episode_length_s = float(a.episode_length_s)
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
  causes = [n for n in ("over_speed", "object_lost", "object_astray", "nan", "time_out") if n in tm.active_terms]
  origin = env.scene.env_origins[0]

  env.reset()
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]
  steps, clouds, flags = [], [], []
  images: dict[int, dict] = {}
  owner = env._pc_cloud_owner
  sensor = env.scene[camera_mod.CAMERA_NAME]
  reach = float(cmd.cfg.grasp_reach_m)
  lift = float(cmd.cfg.grasp_lift_m)
  bin_c = np.asarray(cmd.cfg.bin_center)
  bin_in = np.asarray(cmd.cfg.bin_inner)
  placed_since_reset = 0
  has_flag = obs["camera"].shape[-1] >= 5

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
      keep = cam[:, 3] > 0.5
      pts = cam[keep][:, :3]
      arr = (pts * 1000.0).round().clamp(-32000, 32000).to(torch.int16).cpu().numpy().reshape(-1)
      clouds.append(base64.b64encode(arr.tobytes()).decode("ascii"))
      flags.append(base64.b64encode((cam[keep][:, 4] > 0.5).to(torch.uint8).cpu().numpy().tobytes()).decode("ascii")
                   if has_flag else "")
      last_cloud_id = len(clouds) - 1
      last_cloud_sig = sig
    ee = (cmd._site_pos_w()[0] - origin).cpu().numpy()
    op = (obj.data.root_link_pos_w[0] - origin).cpu().numpy()
    jaw = float(robot.data.joint_pos[0, 6])
    half = cmd.object_half_size[0].cpu().numpy()
    held = bool(cmd.grasped[0])
    dist = float(np.linalg.norm(ee - op))
    on_table = (op[2] - half[2]) < lift
    in_bin = bool((np.abs(op[:2] - bin_c) < bin_in).all())
    if held:
      phase = "carry"
    elif (not on_table) or in_bin:
      phase = "place"
    elif dist < reach:
      phase = "engaged"
    else:
      phase = "approach_re" if placed_since_reset > 0 else "approach_first"
    steps.append({
      "t": round(t * env.step_dt, 3), "cloud": last_cloud_id, "fresh": fresh,
      "age": int(round(float(meta[0]) * pc_cloud.AGE_NORM)), "valid": bool(meta[2] > 0.5),
      "npts": int((cam[:, 3] > 0.5).sum()),
      "ee": [round(float(x), 4) for x in ee], "obj": [round(float(x), 4) for x in op],
      "dist": round(float(np.linalg.norm(ee - op)), 4),
      "held": held, "placed": int(cmd.objects_placed[0]),
      "jaw": round(jaw, 4), "jaw_cmd": round(float(piper.GRIPPER_OFFSET + piper.GRIPPER_SCALE * act[-1]), 4),
      "a": [round(float(x), 3) for x in act.cpu().numpy()], "umax": round(float(u.abs().max()), 3),
      "term": None, "phase": phase, "attempt": False, "stall": False,
      "tgt_px": int(owner.target_full_count[0]) if owner.target_full_count is not None else -1,
      "tgt_pts": int(owner.target_sampled_count[0]) if owner.target_sampled_count is not None else -1,
      **({"tgt_obs": int((cam[:, 4] > 0.5).sum())} if has_flag else {}),
    })
    obs, _, dones, _ = wrapped.step(u)
    from eval_occlusion import reset_recurrent
    reset_recurrent(policy, dones)
    if float(cmd.just_placed[0]) > 0:
      placed_since_reset += 1
    if bool(dones[0]):
      placed_since_reset = 0
    for c in causes:
      if bool(tm.get_term(c)[0]):
        steps[-1]["term"] = c
    # The capture made during this step (the obs just returned is for step t+1).
    if owner.fresh is not None and bool(owner.fresh[0]):
      snapshot(t)
  env.close()
  # Attempts (engaged turning on) and stalls (approach runs of at least --idle-s), as eval_initiation defines them.
  idle_steps = int(round(a.idle_s / env.step_dt))
  run = 0
  for k, s_ in enumerate(steps):
    prev = steps[k - 1] if k else None
    s_["attempt"] = (s_["phase"] == "engaged" and (prev is None or (prev["phase"] != "engaged" and not prev["term"])))
    if s_["phase"].startswith("approach") and not (prev and prev["term"]):
      run += 1
    else:
      if run >= idle_steps:
        for j in range(k - run, k):
          steps[j]["stall"] = True
      run = 0
  if run >= idle_steps:
    for j in range(len(steps) - run, len(steps)):
      steps[j]["stall"] = True
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
  n_att = sum(1 for s_ in steps if s_["attempt"])
  n_stall_steps = sum(1 for s_ in steps if s_["stall"])
  data = {"steps": steps, "clouds": clouds, "flags": flags if has_flag else [],
          "images": {str(k): v for k, v in images.items()},
          "ws": {"r_min": ws.r_min, "r_max": ws.r_max, "a_lo": ws.angle_lo, "a_hi": ws.angle_hi},
          "bin": [bx, by, hx + objects.BIN_WALL_THICKNESS, hy + objects.BIN_WALL_THICKNESS],
          "meta": {"checkpoint": a.checkpoint, "task": a.task, "seed": a.seed, "steps": a.steps,
                   "placed": placed, "weights": loaded, "sensor": sensor_prov}}
  title = (f"{pathlib.Path(a.checkpoint).name} on {a.task} seed {a.seed}: {placed} placed, {n_att} attempts, "
           f"{n_stall_steps * env.step_dt:.0f} s stalled in {a.steps * env.step_dt:.0f} s")
  html = PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, default=str))
  out = pathlib.Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(html)
  print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(steps)} steps, {len(clouds)} clouds, {placed} placed, "
        f"{n_att} attempts, {n_stall_steps} stalled steps, terminations {sum(1 for s in steps if s['term'])}")
  # Where to look: the first success cycle and the first stall, as frame ranges.
  first_place = next((k for k in range(1, len(steps)) if steps[k]["placed"] > steps[k - 1]["placed"]), None)
  first_stall = next((k for k, s_ in enumerate(steps) if s_["stall"]), None)
  if first_place is not None:
    start = max(0, next((k for k in range(first_place, -1, -1) if steps[k]["attempt"]), 0) - 25)
    print(f"first success cycle: frames {start}-{first_place}")
  if first_stall is not None:
    end = next((k for k in range(first_stall, len(steps)) if not steps[k]["stall"]), len(steps) - 1)
    print(f"first stall: frames {first_stall}-{end} ({(end - first_stall) * env.step_dt:.1f} s)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
