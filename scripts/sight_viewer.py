"""A draggable, frame-by-frame 3D view of what the camera can and cannot see.

A fixed spectator render is enough to notice a problem and not enough to
diagnose one: when a number and a picture disagree the next question is
always "from where I am standing, what is actually in front of what", and a
still cannot answer it.  This exports the geometry itself and lets the
viewpoint move.

It also carries the evidence for why the sphere tracer was wrong, because that
is not something to take on trust.  Every frame stores three visibility
numbers over the SAME fifteen sample points:

  pixels      the renderer's segmentation, read at each point's pixel -- truth
  ray_fixed   the sphere cover, traced correctly
  ray_shipped the sphere cover as ``eval_occlusion.py`` shipped it, indexing
              ``geom_pos_w`` with global geom ids when that array is in the
              robot's own geom order -- every sphere on the next link along

Toggle the cover on in the viewer and the disagreement stops being a claim:
the gripper base's sphere is 49.9 mm and the fingers' are 30.8 mm, so a held
object sits inside the arm as far as that tracer is concerned.

    python scripts/sight_viewer.py \\
        --checkpoint v5=checkpoints/v7_teachers/v5_baseline.pt \\
        --checkpoint strong=checkpoints/v7_teachers/strong_teacher.pt \\
        --steps 240 --out sight.html

Writes one self-contained file: no CDN, no server, open it with file://.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mjlab.tasks  # noqa: F401,E402
import mujoco  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

from piper_push import camera as sim_camera  # noqa: E402
from eval_occlusion import (INSET, load_policy, ray_hits_spheres,  # noqa: E402
                            reset_recurrent, sample_box, sphere_cover)

VIS, FINGERS, BASE, ARM, OTHER = range(5)
GROUP_NAMES = ("visible", "fingers", "gripper_base", "arm", "other")


def _rotate(quat, vec):
  w, xyz = quat[..., :1], quat[..., 1:]
  t = 2.0 * torch.cross(xyz, vec, dim=-1)
  return vec + w * t + torch.cross(xyz, t, dim=-1)


def quat_to_mat(q):
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return torch.stack([
    torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
    torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
    torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
  ], dim=1)


def body_group(model, geom_id: int) -> int:
  """fingers / gripper_base / arm / other, for one geom."""
  b = int(model.geom_bodyid[geom_id])
  full = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
  if full.split("/")[0] != "robot":
    return OTHER
  nm = full.split("/")[-1]
  return (FINGERS if nm.startswith("gripper_link")
          else BASE if nm.startswith("gripper_base") else ARM)


def jpeg(img: np.ndarray, quality: int = 72) -> str:
  import cv2
  ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
  assert ok
  return base64.b64encode(buf.tobytes()).decode("ascii")



def rollout(label, ckpt, task, steps, device, seed, env_id=0, inset=INSET):
  torch.manual_seed(seed)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 8
  if not any(getattr(s, "name", "") == sim_camera.CAMERA_NAME
             for s in (cfg.scene.sensors or ())):
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (sim_camera.camera_cfg(),)
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  agent = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
    wrapped, asdict(agent), None, device)
  policy = load_policy(runner, ckpt, device)

  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("pick")
  sensor = env.scene[sim_camera.CAMERA_NAME]
  cam_idx = sensor.camera_idx
  model = env.sim.mj_model
  gids = robot.indexing.geom_ids.cpu().numpy()          # local order -> global
  g2l = {int(g): i for i, g in enumerate(gids)}         # the map that was missing

  # --- the arm's drawable shape: AABBs of the geoms the renderer draws ------
  vis_geoms, vis_meta = [], []
  for g in gids:
    g = int(g)
    if int(model.geom_group[g]) != 2:                   # 2 is the visual group
      continue
    aabb = np.asarray(model.geom_aabb[g]).reshape(2, 3)
    vis_geoms.append(g2l[g])
    vis_meta.append({"c": aabb[0].round(5).tolist(), "h": aabb[1].round(5).tolist(),
                     "g": body_group(model, g)})
  vis_idx = torch.tensor(vis_geoms, dtype=torch.long, device=device)

  # --- the sphere cover, and the indexing bug it shipped with ---------------
  local_np, radii_np, cov_geoms = sphere_cover(model)
  local = torch.tensor(local_np, dtype=torch.float32, device=device)
  radii = torch.tensor(radii_np, dtype=torch.float32, device=device)
  right = torch.tensor([g2l[int(g)] for g in cov_geoms], dtype=torch.long,
                       device=device)
  n_geom = robot.data.geom_pos_w.shape[1]
  wrong = torch.tensor(np.clip(cov_geoms, 0, n_geom - 1), dtype=torch.long,
                       device=device)
  sphere_meta = [{"r": round(float(r), 5), "g": body_group(model, int(g))}
                 for r, g in zip(radii_np, cov_geoms)]

  # --- static scenery, so the 3D view has a table to stand on --------------
  scenery = []
  for g in range(model.ngeom):
    if int(g) in g2l or int(model.geom_group[g]) not in (0, 2):
      continue
    b = int(model.geom_bodyid[g])
    full = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    if full.startswith("robot"):
      continue
    aabb = np.asarray(model.geom_aabb[g]).reshape(2, 3)
    if not np.all(np.isfinite(aabb)) or float(np.max(aabb[1])) > 3.0:
      continue                                          # skip the ground plane
    scenery.append({"g": g, "c": aabb[0].round(4).tolist(),
                    "h": aabb[1].round(4).tolist(),
                    "name": full.split("/")[-1]})

  offsets = torch.tensor(sample_box(inset=inset), dtype=torch.float32,
                         device=device)
  nom = torch.tensor(sim_camera.CAMERA_POS, dtype=torch.float32, device=device)
  W, H, FOVY = sim_camera.WIDTH, sim_camera.HEIGHT, sim_camera.FOVY_DEG
  f = 0.5 * H / np.tan(np.deg2rad(FOVY) / 2.0)
  GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
  import cv2

  env.reset()
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  frames = []
  b = env_id
  for _ in range(steps):
    with torch.inference_mode():
      action = policy(obs)
    obs, _, dones, _ = wrapped.step(action)
    reset_recurrent(policy, dones)

    org = env.scene.env_origins
    pts = (cmd._object_pos_local().unsqueeze(1)
           + offsets.unsqueeze(0) * cmd.object_half_size.unsqueeze(1))
    cpos = env.sim.model.cam_pos[:, cam_idx].to(torch.float32)
    cquat = env.sim.model.cam_quat[:, cam_idx].to(torch.float32)

    # pixel truth, and which body owns each blocked point
    seg = sensor.data.segmentation
    ids, types = seg[..., 0], seg[..., 1]
    cmat = quat_to_mat(cquat)
    rel = pts - cpos.unsqueeze(1)
    loc = torch.einsum("bij,bkj->bki", cmat.transpose(1, 2), rel)
    depth = (-loc[..., 2]).clamp_min(1e-6)
    u = (W / 2.0 + f * loc[..., 0] / depth).round().long().clamp(0, W - 1)
    v = (H / 2.0 - f * loc[..., 1] / depth).round().long().clamp(0, H - 1)
    flat = v * W + u
    gid = ids.flatten(1).gather(1, flat)
    typ = types.flatten(1).gather(1, flat)
    tgt = cmd.target_geom_ids
    is_t = (gid.unsqueeze(-1) == tgt.unsqueeze(1)).any(-1) & (typ == GEOM)
    status = np.full(pts.shape[1], OTHER, dtype=np.int64)
    for k in range(pts.shape[1]):
      status[k] = (VIS if bool(is_t[b, k])
                   else body_group(model, int(gid[b, k]))
                   if bool(typ[b, k] == GEOM) else OTHER)

    # the two sphere traces
    gp, gq = robot.data.geom_pos_w, robot.data.geom_quat_w
    def centres_from(idx):
      gx = gp[:, idx] - org.unsqueeze(1)
      return gx + _rotate(gq[:, idx], local.unsqueeze(0).expand(gx.shape[0], -1, -1))
    c_right, c_wrong = centres_from(right), centres_from(wrong)
    r_fix = float((~ray_hits_spheres(cpos[b], pts[b:b + 1], c_right[b:b + 1],
                                     radii)).float().mean())
    r_shp = float((~ray_hits_spheres(nom, pts[b:b + 1], c_wrong[b:b + 1],
                                     radii)).float().mean())

    # the camera's own picture, plus the target's exact pixels
    d = sensor.data.depth[b, ..., 0].cpu().numpy()
    q = np.clip((d - 0.35) / (1.6 - 0.35), 0, 1)
    pic = cv2.applyColorMap((q * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    pic[d <= 0] = (25, 25, 25)
    m = ((ids[b].unsqueeze(-1) == tgt[b].view(1, 1, -1)).any(-1)
         & (types[b] == GEOM)).cpu().numpy()
    ys, xs = np.nonzero(m)

    vx = (gp[b, vis_idx] - org[b]).cpu().numpy()
    vq = gq[b, vis_idx].cpu().numpy()
    frames.append({
      "geom": [[*np.round(p, 4).tolist(), *np.round(q4, 4).tolist()]
               for p, q4 in zip(vx, vq)],
      "sph": np.round(c_right[b].cpu().numpy(), 4).tolist(),
      "obj": np.round(cmd._object_pos_local()[b].cpu().numpy(), 4).tolist(),
      "objq": np.round(cmd.object_quat[b].cpu().numpy(), 4).tolist()
              if hasattr(cmd, "object_quat") else [1, 0, 0, 0],
      "objh": np.round(cmd.object_half_size[b].cpu().numpy(), 4).tolist(),
      "cam": np.round(cpos[b].cpu().numpy(), 4).tolist(),
      "camq": np.round(cquat[b].cpu().numpy(), 4).tolist(),
      "pts": np.round(pts[b].cpu().numpy(), 4).tolist(),
      "st": status.tolist(),
      "held": bool(cmd.grasped[b]),
      "pix": round(float((status == VIS).mean()), 4),
      "rfix": round(r_fix, 4),
      "rshp": round(r_shp, 4),
      "npx": int(m.sum()),
      "img": jpeg(pic),
      "mx": xs.astype(int).tolist(),
      "my": ys.astype(int).tolist(),
    })

  meta = {"vis": vis_meta, "sph": sphere_meta, "scenery": scenery,
          "W": W, "H": H, "fovy": FOVY}
  # scenery poses are static; take them from the model once
  data = env.sim.data if hasattr(env.sim, "data") else None
  for s in scenery:
    g = s.pop("g")
    s["p"] = np.round(np.asarray(model.geom_pos[g]), 4).tolist()
    s["q"] = np.round(np.asarray(model.geom_quat[g]), 4).tolist()
  env.close()
  return {"label": label, "checkpoint": ckpt, "frames": frames}, meta


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>sight — frame by frame</title>
<style>
:root{
  --bg:#101215; --panel:#171a1f; --line:#262b33; --ink:#e6e9ee; --dim:#8b94a3;
  --vis:#3ad46b; --fing:#ff5c5c; --base:#ffb020; --arm:#4aa8ff; --oth:#5d6673;
  --accent:#6ee7a8;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;
  gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.02em}
.sub{color:var(--dim)}
#wrap{display:flex;gap:10px;padding:10px;align-items:flex-start;flex-wrap:wrap}
.col{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:8px}
canvas{display:block;border-radius:4px;background:#0b0d10;cursor:grab}
canvas.drag{cursor:grabbing}
.cap{color:var(--dim);margin:0 0 6px;font-size:11px;letter-spacing:.06em;
  text-transform:uppercase}
#bar{padding:8px 14px;border-top:1px solid var(--line);display:flex;gap:10px;
  align-items:center;flex-wrap:wrap;position:sticky;bottom:0;background:var(--bg)}
input[type=range]{flex:1;min-width:240px;accent-color:var(--accent)}
button{background:#20252c;color:var(--ink);border:1px solid var(--line);
  border-radius:4px;padding:4px 10px;font:inherit;cursor:pointer}
button:hover{background:#2a3038}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);
  outline-offset:2px}
label{color:var(--dim);display:inline-flex;gap:5px;align-items:center;
  cursor:pointer;user-select:none}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{padding:2px 8px 2px 0;text-align:right}
th{color:var(--dim);font-weight:400;text-align:right}
td:first-child,th:first-child{text-align:left}
.k{display:inline-block;width:9px;height:9px;border-radius:2px;
  vertical-align:baseline;margin-right:4px}
.big{font-size:22px;font-variant-numeric:tabular-nums}
.warn{color:var(--base)}
.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--dim);font-size:11px}
</style>
<header>
  <h1>sight — what the camera can see, frame by frame</h1>
  <span class="sub" id="hdr"></span>
</header>
<div id="wrap"></div>
<div id="bar">
  <button id="play">▶ play</button>
  <button data-step="-10">−10</button>
  <button data-step="-1">−1</button>
  <button data-step="1">+1</button>
  <button data-step="10">+10</button>
  <input type="range" id="slider" min="0" value="0">
  <span id="fno" style="min-width:9ch"></span>
  <label><input type="checkbox" id="cover"> sphere cover</label>
  <label><input type="checkbox" id="rays" checked> sample rays</label>
  <label><input type="checkbox" id="tube" checked> sight cylinder</label>
  <label><input type="checkbox" id="sync" checked> sync views</label>
</div>
<script>
const DATA = __DATA__;
const META = __META__;
const RUNS = DATA.runs;
const COL = ['--vis','--fing','--base','--arm','--oth'].map(v =>
  getComputedStyle(document.documentElement).getPropertyValue(v).trim());
const NAMES = ['visible','fingers','gripper_base','arm','other'];
const RADIUS = __RADIUS__;

/* ---------- small 3d: quats, projection, painter's algorithm ---------- */
function qmat(q){
  const [w,x,y,z]=q;
  return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
          [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
          [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];
}
const mv=(M,v)=>[M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],
                 M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],
                 M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]];
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const scl=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const n=Math.hypot(...a)||1e-9;return scl(a,1/n)};

// One shared orbit state so "sync views" is just: everyone reads this.
const view = {az:-0.75, el:0.42, dist:1.9, target:[-0.05,0.42,0.18], fov:42};

function camera(){
  const {az,el,dist,target}=view;
  const eye=add(target,[dist*Math.cos(el)*Math.cos(az),
                        dist*Math.cos(el)*Math.sin(az), dist*Math.sin(el)]);
  const f=norm(sub(target,eye));
  const r=norm(cross(f,[0,0,1]));
  const u=cross(r,f);
  return {eye,R:[r,u,scl(f,-1)]};   // rows: right, up, backward
}
function makeProj(w,h){
  const c=camera(), fpx=0.5*h/Math.tan(view.fov*Math.PI/360);
  return p=>{
    const d=sub(p,c.eye);
    const x=dot(d,c.R[0]), y=dot(d,c.R[1]), z=-dot(d,c.R[2]);
    return [w/2+fpx*x/Math.max(z,1e-6), h/2-fpx*y/Math.max(z,1e-6), z];
  };
}
const CORNERS=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
               [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];
const FACES=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]];
function boxFaces(pos,quat,c,h,color){
  const M=qmat(quat), out=[];
  const pw=CORNERS.map(s=>add(pos,mv(M,[c[0]+s[0]*h[0],c[1]+s[1]*h[1],c[2]+s[2]*h[2]])));
  for(const f of FACES) out.push({pts:f.map(i=>pw[i]),color});
  return out;
}

/* ---------- one panel = one policy ---------- */
function panel(run){
  const col=document.createElement('div'); col.className='col';
  col.innerHTML=`<p class="cap">${run.label}</p>`;
  const c3=document.createElement('canvas'); c3.width=430; c3.height=330;
  const cv=document.createElement('canvas'); cv.width=META.W*2; cv.height=META.H*2;
  cv.style.marginTop='8px'; cv.style.width=META.W*2+'px';
  const info=document.createElement('div'); info.style.marginTop='8px';
  col.append(c3,cv,info);

  // drag to orbit, shift-drag to pan, wheel to zoom
  let drag=null;
  c3.addEventListener('pointerdown',e=>{
    drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};
    c3.setPointerCapture(e.pointerId); c3.classList.add('drag');
  });
  c3.addEventListener('pointerup',e=>{drag=null;c3.classList.remove('drag')});
  c3.addEventListener('contextmenu',e=>e.preventDefault());
  c3.addEventListener('pointermove',e=>{
    if(!drag) return;
    const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
    drag.x=e.clientX; drag.y=e.clientY;
    if(drag.pan){
      const c=camera(), s=view.dist*0.0016;
      view.target=add(view.target,add(scl(c.R[0],-dx*s),scl(c.R[1],dy*s)));
    }else{
      view.az-=dx*0.008;
      view.el=Math.max(-1.45,Math.min(1.45,view.el+dy*0.008));
    }
    draw();
  });
  c3.addEventListener('wheel',e=>{
    e.preventDefault();
    view.dist=Math.max(0.35,Math.min(6,view.dist*Math.exp(e.deltaY*0.0012)));
    draw();
  },{passive:false});

  return {run,c3,cv,info};
}

function drawPanel(p,k){
  const F=p.run.frames[Math.min(k,p.run.frames.length-1)];
  const g=p.c3.getContext('2d'), W=p.c3.width, H=p.c3.height;
  g.fillStyle='#0b0d10'; g.fillRect(0,0,W,H);
  const P=makeProj(W,H);
  const faces=[];

  // ground grid, so the eye has somewhere to stand
  g.strokeStyle='#1b2027'; g.lineWidth=1;
  for(let i=-6;i<=6;i++){
    for(const seg of [[[i*0.1,-0.2,0],[i*0.1,0.9,0]],
                      [[-0.6,0.1*i+0.35,0],[0.6,0.1*i+0.35,0]]]){
      const a=P(seg[0]), b=P(seg[1]);
      if(a[2]>0&&b[2]>0){g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke();}
    }
  }
  for(const s of META.scenery)
    faces.push(...boxFaces(s.p,s.q,s.c,s.h,'#232a33'));
  F.geom.forEach((gp,i)=>{
    const m=META.vis[i];
    faces.push(...boxFaces(gp.slice(0,3),gp.slice(3),m.c,m.h,
      m.g===1?'#c98a1e':m.g===2?'#b8641c':'#7a5a3a'));
  });
  faces.push(...boxFaces(F.obj,F.objq,[0,0,0],F.objh,'#2fbf5f'));

  faces.forEach(f=>{f.pp=f.pts.map(P); f.z=f.pp.reduce((s,q)=>s+q[2],0)/f.pp.length;});
  faces.filter(f=>f.pp.every(q=>q[2]>0)).sort((a,b)=>b.z-a.z).forEach(f=>{
    g.beginPath(); g.moveTo(f.pp[0][0],f.pp[0][1]);
    for(let i=1;i<f.pp.length;i++) g.lineTo(f.pp[i][0],f.pp[i][1]);
    g.closePath(); g.fillStyle=f.color; g.fill();
    g.strokeStyle='rgba(0,0,0,.45)'; g.lineWidth=1; g.stroke();
  });

  const cam=F.cam, obj=F.obj;
  const axis=sub(obj,cam), len=Math.hypot(...axis), u=norm(axis);

  if(document.getElementById('tube').checked){
    const side=norm(cross(u,[0,0,1])), up=cross(u,side);
    g.strokeStyle='rgba(110,231,168,.30)'; g.lineWidth=1;
    for(const t of [0.3,0.55,0.8,1.0]){
      g.beginPath();
      for(let i=0;i<=30;i++){
        const a=i/30*2*Math.PI;
        const q=P(add(add(cam,scl(axis,t)),
          add(scl(side,RADIUS*Math.cos(a)),scl(up,RADIUS*Math.sin(a)))));
        if(q[2]<=0) continue;
        i?g.lineTo(q[0],q[1]):g.moveTo(q[0],q[1]);
      }
      g.closePath(); g.stroke();
    }
  }
  if(document.getElementById('cover').checked){
    const c=camera(), fpx=0.5*H/Math.tan(view.fov*Math.PI/360);
    F.sph.forEach((s,i)=>{
      const q=P(s); if(q[2]<=0) return;
      const m=META.sph[i];
      g.beginPath(); g.arc(q[0],q[1],m.r*fpx/q[2],0,7);
      g.fillStyle=(m.g===1?'rgba(255,176,32,.16)':m.g===2?'rgba(74,168,255,.13)'
                   :'rgba(255,92,92,.16)');
      g.fill(); g.strokeStyle='rgba(255,255,255,.13)'; g.stroke();
    });
  }
  if(document.getElementById('rays').checked){
    F.pts.forEach((pt,i)=>{
      const a=P(cam), b=P(pt);
      if(a[2]<=0||b[2]<=0) return;
      g.strokeStyle=COL[F.st[i]]; g.globalAlpha=F.st[i]===0?.75:.9;
      g.lineWidth=F.st[i]===0?1:1.5;
      g.beginPath(); g.moveTo(a[0],a[1]); g.lineTo(b[0],b[1]); g.stroke();
      g.globalAlpha=1;
    });
  }
  const cp=P(cam);
  if(cp[2]>0){
    g.beginPath(); g.arc(cp[0],cp[1],6,0,7);
    g.fillStyle='#4fc3ff'; g.fill();
    g.strokeStyle='#0b0d10'; g.lineWidth=1.5; g.stroke();
    g.fillStyle='#4fc3ff'; g.font='11px ui-monospace,monospace';
    g.fillText('D455',cp[0]-13,cp[1]-11);
  }

  // the camera's own picture, with the target's exact pixels painted on
  const cg=p.cv.getContext('2d');
  const im=new Image();
  im.onload=()=>{
    cg.imageSmoothingEnabled=false;
    cg.drawImage(im,0,0,p.cv.width,p.cv.height);
    cg.fillStyle='#3ad46b';
    for(let i=0;i<F.mx.length;i++) cg.fillRect(F.mx[i]*2,F.my[i]*2,2,2);
    cg.fillStyle='rgba(0,0,0,.55)'; cg.fillRect(0,0,p.cv.width,20);
    cg.fillStyle='#e6e9ee'; cg.font='12px ui-monospace,monospace';
    cg.fillText(`camera view · ${F.npx} target px${F.held?' · HOLDING':''}`,6,14);
  };
  im.src='data:image/jpeg;base64,'+F.img;

  const cnt=[0,0,0,0,0]; F.st.forEach(s=>cnt[s]++);
  const n=F.st.length;
  p.info.innerHTML=`
    <div class="big" style="color:${F.pix<0.35?'var(--fing)':'var(--vis)'}">
      ${(100*F.pix).toFixed(0)}% visible</div>
    <table>
      <tr><th>method</th><th>visible</th></tr>
      <tr><td>pixels (truth)</td><td>${(100*F.pix).toFixed(0)}%</td></tr>
      <tr><td>ray, fixed</td><td>${(100*F.rfix).toFixed(0)}%</td></tr>
      <tr><td class="warn">ray, as shipped</td>
          <td class="warn">${(100*F.rshp).toFixed(0)}%</td></tr>
    </table>
    <table style="margin-top:6px">
      <tr><th>blocked by</th><th>pts</th></tr>
      ${[1,2,3,4].map(i=>`<tr><td><span class="k" style="background:${COL[i]}">
        </span>${NAMES[i]}</td><td>${cnt[i]}/${n}</td></tr>`).join('')}
    </table>`;
}

/* ---------- wiring ---------- */
const wrap=document.getElementById('wrap');
const panels=RUNS.map(r=>{const p=panel(r);wrap.append(p.c3.parentNode);return p});
const N=Math.min(...RUNS.map(r=>r.frames.length));
const slider=document.getElementById('slider');
slider.max=N-1;
document.getElementById('hdr').textContent=
  `${RUNS.length} policies · ${N} frames · drag to orbit, shift-drag to pan, wheel to zoom`;
let k=0, timer=null;
function draw(){
  panels.forEach(p=>drawPanel(p,k));
  document.getElementById('fno').textContent=`${k+1}/${N}`;
  slider.value=k;
}
slider.oninput=()=>{k=+slider.value;draw()};
document.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>{
  k=Math.max(0,Math.min(N-1,k+ +b.dataset.step)); draw();
});
document.getElementById('play').onclick=e=>{
  if(timer){clearInterval(timer);timer=null;e.target.textContent='▶ play';return}
  e.target.textContent='❚❚ pause';
  timer=setInterval(()=>{k=(k+1)%N;draw()},1000/20);
};
['cover','rays','tube'].forEach(id=>
  document.getElementById(id).onchange=draw);
addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'){k=Math.min(N-1,k+1);draw()}
  if(e.key==='ArrowLeft'){k=Math.max(0,k-1);draw()}
  if(e.key===' '){e.preventDefault();document.getElementById('play').click()}
});
draw();
</script>
"""


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", action="append", required=True,
                 metavar="LABEL=PATH[@TASK]",
                 help="repeat per policy.  Append @TASK to override --task for "
                      "that one checkpoint, which is how a state teacher and "
                      "the vision student distilled from it get onto the same "
                      "page.  Panels from different tasks are NOT frame-"
                      "matched: the two configs consume the RNG differently "
                      "and place the target differently on the same seed")
  p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Robust")
  p.add_argument("--steps", type=int, default=240)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=20260902)
  p.add_argument("--env-id", type=int, default=0,
                 help="which of the parallel environments to record")
  p.add_argument("--radius", type=float, default=0.07,
                 help="sight cylinder radius, drawn to scale")
  p.add_argument("--out", default="sight.html")
  a = p.parse_args()

  runs, meta = [], None
  for spec in a.checkpoint:
    label, _, path = spec.partition("=")
    if not path:
      label, path = pathlib.Path(spec).stem, spec
    # "@task" per checkpoint, because the comparison that matters most spans
    # two task ids: a state teacher and the vision student distilled from it
    # live in different registrations, and one page is the only way to watch
    # both.
    #
    # Panels from DIFFERENT tasks are not frame-matched, and the caption says
    # so.  Checked rather than assumed: on the same seed the two task configs
    # place the target at (-0.014, 0.395) and (0.035, 0.372), because they
    # build different observation terms and so consume the RNG differently.
    # Same-task panels stay matched; across tasks, compare behaviour and the
    # per-frame numbers, not what is happening at frame k.
    path, _, task = path.partition("@")
    task = task or a.task
    print(f"rolling out {label}: {path}  [{task}]", flush=True)
    run, meta = rollout(label, path, task, a.steps, a.device, a.seed,
                        env_id=a.env_id)
    px = np.array([f["pix"] for f in run["frames"]])
    rs = np.array([f["rshp"] for f in run["frames"]])
    print(f"  visible: pixels {px.mean():.2f}   ray-as-shipped {rs.mean():.2f}")
    runs.append(run)

  html = (PAGE
          .replace("__DATA__", json.dumps({"runs": runs}, separators=(",", ":")))
          .replace("__META__", json.dumps(meta, separators=(",", ":")))
          .replace("__RADIUS__", repr(float(a.radius))))
  out = pathlib.Path(a.out)
  out.write_text(html, encoding="utf-8")
  print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
