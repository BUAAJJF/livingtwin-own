"""Serve every deployment recording in one full-frame browser player.

Unlike :mod:`hardware.deploy.review`, this does not embed JPEGs in a gigantic
self-contained page.  The repository currently holds tens of gigabytes of
recordings; embedding every frame from every session would duplicate most of
that data and browsers would fail before the useful part loaded.

Compact v3 recordings preserve raw RGB, both infrared imagers and native D455
depth, plus sparse detector labels, SAM's native RGB mask and the accepted
source mask. Foundation depth, aligned gray and policy tensors are deliberately
left for offline reconstruction. Older schemas remain readable. The page
labels reconstructed or unavailable products instead of presenting them as
historical outputs. This server keeps the HTML small and renders one requested
frame at a time.

Playback uses every recorded perception frame and its real ``frame_stamp``.
There is no review stride.  At 1x, irregular camera cadence is preserved; MAX
advances as soon as the browser receives the next frame.

    python -m hardware.deploy.logview
    python -m hardware.deploy.logview --root recordings --port 8765
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import pathlib
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LivingTwin deployment logs</title>
<style>
:root{--bg:#0d1117;--panel:#151b23;--panel2:#0f141b;--line:#2a3441;
 --ink:#e6edf3;--muted:#8b98a5;--accent:#45b8d0;--ok:#49b47a;
 --warn:#e6aa4d;--bad:#e06a58;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:4;display:flex;gap:14px;align-items:center;
 padding:12px 18px;background:rgba(13,17,23,.96);border-bottom:1px solid var(--line)}
h1{font:600 15px/1.2 var(--mono);margin:0;white-space:nowrap} select,button,input{
 color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:5px}
select{min-width:320px;max-width:52vw;padding:7px 9px} button{padding:7px 12px;
 cursor:pointer;font-family:var(--mono)} button:hover{border-color:var(--accent)}
.status{margin-left:auto;color:var(--muted);font:12px var(--mono)}
main{max-width:1540px;margin:auto;padding:16px}.cards{display:grid;
 grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:8px;margin-bottom:12px}
.card{background:var(--panel);border:1px solid var(--line);padding:10px 12px;border-radius:6px}
.key{color:var(--muted);font:10px var(--mono);text-transform:uppercase;letter-spacing:.09em}
.val{font:16px var(--mono);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.viewer{background:#05070a;border:1px solid var(--line);border-radius:7px;overflow:hidden}
#frame{display:block;width:100%;height:auto;min-height:180px;object-fit:contain}
.timeline{height:48px;position:relative;background:var(--panel2);border-top:1px solid var(--line)}
#events{width:100%;height:100%;display:block}.cursor{position:absolute;top:0;bottom:0;
 width:2px;background:white;box-shadow:0 0 5px #000;pointer-events:none}
.controls{display:flex;align-items:center;gap:8px;padding:10px;background:var(--panel);
 border-top:1px solid var(--line)} #seek{flex:1;accent-color:var(--accent)}
.counter{min-width:188px;text-align:right;color:var(--muted);font:12px var(--mono)}
.below{display:grid;grid-template-columns:minmax(430px,1.2fr) minmax(340px,.8fr);
 gap:12px;margin-top:12px}.box{background:var(--panel);border:1px solid var(--line);
 border-radius:6px;overflow:auto}.box h2{font:11px var(--mono);text-transform:uppercase;
 letter-spacing:.1em;color:var(--muted);margin:0;padding:10px 12px;border-bottom:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font:12px var(--mono);font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;color:var(--muted)}
pre{margin:0;padding:10px 12px;max-height:310px;overflow:auto;font:11px/1.5 var(--mono);
 color:#bec9d3}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.hint{color:var(--muted);font-size:12px;margin:8px 2px 0}
@media(max-width:850px){header{flex-wrap:wrap}select{min-width:0;max-width:none;flex:1}
 .status{width:100%;margin:0}.below{grid-template-columns:1fr}.counter{min-width:0}}
</style></head><body>
<header><h1>deployment / full-frame log viewer</h1>
 <select id="sessions" aria-label="recording"></select>
 <button id="refresh">refresh</button><span class="status" id="status">loading logs…</span>
</header>
<main>
 <div class="cards" id="cards"></div>
 <div class="viewer">
  <img id="frame" alt="eight-pane RGB, infrared, depth and mask replay">
  <div class="timeline"><canvas id="events"></canvas><div class="cursor" id="cursor"></div></div>
  <div class="controls">
   <button id="prev">←</button><button id="play">play</button><button id="next">→</button>
   <select id="speed" aria-label="playback speed" style="min-width:82px;max-width:82px">
    <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
    <option value="1" selected>1×</option><option value="2">2×</option>
    <option value="max">MAX</option></select>
   <input id="seek" type="range" min="0" max="0" value="0">
   <span class="counter" id="counter">—</span>
  </div>
 </div>
 <p class="hint">Eight panes: raw RGB, legacy aligned gray, raw left/right IR,
 native/policy depth, detector instances, and RGB→depth→policy masks.
 Every recorded perception frame is available—no stride.
 Space plays/pauses; arrows step one frame; MAX waits only for decoding.</p>
 <div class="below">
  <div class="box"><h2>frame telemetry</h2><div id="telemetry"></div></div>
  <div class="box"><h2>run configuration and timing</h2><pre id="run"></pre></div>
 </div>
</main>
<script>
'use strict';
const $=id=>document.getElementById(id);
let logs=[], data=null, index=0, playing=false, timer=null, requestToken=0;

function esc(v){return String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmt(v,n=1){return v==null||!Number.isFinite(+v)?'—':(+v).toFixed(n);}
function eventClass(e){return e==='command'?'ok':e==='blind_target'?'warn':e?.includes('stale')||e?.includes('fault')?'bad':e?.startsWith('hold')?'bad':'';}
async function getJSON(url){const r=await fetch(url);if(!r.ok)throw new Error(await r.text());return r.json();}

async function loadLogs(keep=true){
 $('status').textContent='scanning recordings…';
 logs=await getJSON('/api/sessions');
 const old=keep&&data?data.name:new URLSearchParams(location.search).get('log');
 $('sessions').innerHTML='';
 for(const x of logs){const o=document.createElement('option');o.value=x.name;
  o.textContent=`${x.name}  ·  ${x.frames}f  ·  ${x.stop_reason||'incomplete'}`;$('sessions').appendChild(o);}
 if(!logs.length){$('status').textContent='no recordings with frames';return;}
 const chosen=logs.some(x=>x.name===old)?old:logs[0].name;
 $('sessions').value=chosen;await loadSession(chosen);
}

async function loadSession(name){
 pause();requestToken++;$('status').textContent=`loading ${name}…`;
 data=await getJSON('/api/session?name='+encodeURIComponent(name));index=0;
 history.replaceState(null,'','?log='+encodeURIComponent(name));
 $('seek').max=Math.max(0,data.frames.length-1);$('seek').value=0;
 const r=data.run||{}, rate=r.rates||{}, p=r.perception||{}, a=r.args||{};
 const rec=r.recording||{};
 const cards=[['frames',data.frames.length],['duration',fmt(data.duration_s,1)+' s'],
  ['recorded rate',fmt(data.fps,1)+' fps'],['stop',r.stop_reason||'incomplete'],
  ['target',a.target_tracker||'—'],['depth',a.depth_source||'—'],
  ['detections',data.visuals?.exact?'recorded exact':'legacy rebuilt'],
  ['log/frame',fmt(rec.mb_per_camera_frame,2)+' MB'],
  ['log queue',`${rec.queue_high_water_frames??'—'} / ${rec.queue_capacity_frames??'—'}`],
  ['vision p50',fmt(p.compute_ms?.p50,1)+' ms'],['obs age p50',fmt(rate.observation_age_ms?.p50,1)+' ms'],
  ['commands',data.event_counts.command||0],['no-target holds',data.event_counts.hold_no_target||0]];
 $('cards').innerHTML=cards.map(([k,v])=>`<div class="card"><div class="key">${esc(k)}</div><div class="val">${esc(v)}</div></div>`).join('');
 $('run').textContent=JSON.stringify({stop_reason:r.stop_reason,args:a,rates:rate,
  perception:p,recording:rec},null,2);
 drawTimeline();show(0);$('status').textContent=`${data.name} · full ${data.frames.length} frames`;
}

function telemetry(f){
 const rows=[['time',fmt(f.t_s,3)+' s'],['event',f.event],['record frame',f.i],
  ['camera index',f.frame_index],['tracker label',f.label||'—'],
  ['detections',f.detections?.length??'legacy / rebuilt'],
  ['mask state',f.mask_state||'legacy / unavailable'],
  ['D455 depth frame',f.sensor?.depth_frame_number??'not recorded'],
  ['D455 RGB frame',f.sensor?.color_frame_number??'not recorded'],
  ['observation age',fmt(f.observation_age_ms,1)+' ms'],
  ['publish age',fmt(f.publish_age_ms,1)+' ms'],['gripper gap',fmt(f.gripper_gap_mm,1)+' mm'],
  ['gripper effort',fmt(f.gripper_effort,3)],['table clearance',fmt(f.table_clearance_mm,1)+' mm']];
 const head=rows.map(([k,v])=>`<tr><td>${esc(k)}</td><td class="${k==='event'?eventClass(f.event):''}">${esc(v)}</td></tr>`).join('');
 const vec=(name,v,scale=1)=>!v?'':`<tr><td>${name}</td>${v.slice(0,7).map(x=>`<td>${fmt(x*scale,3)}</td>`).join('')}</tr>`;
 return `<table>${head}</table><table><tr><th>vector</th><th>J1</th><th>J2</th><th>J3</th><th>J4</th><th>J5</th><th>J6</th><th>grip</th></tr>
  ${vec('joint pos',f.joint_pos)}${vec('joint vel',f.joint_vel)}${vec('target',f.target)}${vec('action',f.action)}</table>`;
}

function frameDelay(k){
 if(!data||data.frames.length<2)return 66;
 const a=data.frames[k].t_s,b=data.frames[(k+1)%data.frames.length].t_s;
 if(b==null||a==null||b<=a)return 1000/Math.max(data.fps||15,1);
 return Math.max(1,Math.min(1000,(b-a)*1000));
}
function scheduleNext(renderMs){
 if(!playing)return;const speed=$('speed').value;
 const delay=speed==='max'?0:Math.max(0,frameDelay(index)/Number(speed)-renderMs);
 timer=setTimeout(()=>{if(index+1>=data.frames.length){pause();return;}show(index+1,true);},delay);
}
function show(k,fromPlay=false){
 if(!data||!data.frames.length)return;
 index=Math.max(0,Math.min(data.frames.length-1,k));const f=data.frames[index];
 $('seek').value=index;$('cursor').style.left=(100*index/Math.max(1,data.frames.length-1))+'%';
 $('counter').textContent=`${index+1} / ${data.frames.length} · ${fmt(f.t_s,2)}s`;
 $('telemetry').innerHTML=telemetry(f);
 const token=++requestToken,start=performance.now(),img=$('frame');
 img.onload=()=>{if(token!==requestToken)return;
  const ni=Math.min(index+1,data.frames.length-1),pre=new Image();pre.src=frameURL(ni);
  if(fromPlay)scheduleNext(performance.now()-start);};
 img.onerror=()=>{if(token===requestToken){$('status').textContent='frame decode failed';pause();}};
 img.src=frameURL(index);
}
function frameURL(k){return `/api/frame?view=rgb-depth-v3&name=${encodeURIComponent(data.name)}&i=${k}`;}
function play(){if(playing)return;playing=true;$('play').textContent='pause';show(index,true);}
function pause(){playing=false;clearTimeout(timer);timer=null;$('play').textContent='play';}
function toggle(){playing?pause():play();}

function drawTimeline(){
 const c=$('events'),g=c.getContext('2d'),dpr=devicePixelRatio||1;
 c.width=Math.max(1,c.clientWidth*dpr);c.height=Math.max(1,c.clientHeight*dpr);
 g.clearRect(0,0,c.width,c.height);const n=data.frames.length,w=c.width/Math.max(n,1);
 const color=e=>e==='command'?'#49b47a':e==='blind_target'?'#e6aa4d':e?.includes('stale')?'#b477db':e?.startsWith('hold')?'#e06a58':'#566474';
 data.frames.forEach((f,i)=>{g.fillStyle=color(f.event);g.fillRect(i*w,0,Math.max(1,w),c.height);});
}

$('sessions').addEventListener('change',e=>loadSession(e.target.value).catch(fail));
$('refresh').addEventListener('click',()=>loadLogs().catch(fail));
$('play').onclick=toggle;$('prev').onclick=()=>{pause();show(index-1)};
$('next').onclick=()=>{pause();show(index+1)};$('seek').oninput=e=>{pause();show(+e.target.value)};
$('events').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();pause();show(Math.round((e.clientX-r.left)/r.width*(data.frames.length-1)));};
addEventListener('resize',()=>{if(data)drawTimeline()});
addEventListener('keydown',e=>{if(e.target.matches('select,input'))return;
 if(e.key===' '){toggle();e.preventDefault()}else if(e.key==='ArrowLeft'){pause();show(index-1);e.preventDefault()}
 else if(e.key==='ArrowRight'){pause();show(index+1);e.preventDefault()}});
function fail(e){console.error(e);$('status').textContent=e.message||String(e);pause();}
loadLogs(false).catch(fail);
</script></body></html>"""


def _read_json(path: pathlib.Path, default):
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError, TypeError):
    return default


def _safe_session(root: pathlib.Path, name: str) -> pathlib.Path:
  """Resolve one direct child of ``root``; never accept traversal."""
  if not name or pathlib.PurePath(name).name != name:
    raise ValueError("invalid recording name")
  root = root.resolve()
  path = (root / name).resolve()
  if path.parent != root or not path.is_dir():
    raise ValueError(f"unknown recording {name!r}")
  return path


def list_sessions(root: pathlib.Path) -> list[dict]:
  """Cheap catalog: run summaries only; frame details load on selection."""
  out = []
  if not root.is_dir():
    return out
  for path in root.iterdir():
    if not path.is_dir() or not (path / "meta.json").is_file():
      continue
    run = _read_json(path / "run.json", {})
    perception = run.get("perception") or {}
    frames = perception.get("frames")
    if frames is None:
      frames = sum(1 for _ in path.glob("*.npz"))
    if not frames:
      continue
    args = run.get("args") or {}
    stamp = max((p.stat().st_mtime for p in (path / "run.json", path / "meta.json")
                 if p.exists()), default=path.stat().st_mtime)
    out.append({
      "name": path.name,
      "frames": int(frames),
      "duration_s": perception.get("elapsed_s"),
      "fps": perception.get("actual_hz"),
      "stop_reason": run.get("stop_reason"),
      "target_tracker": args.get("target_tracker"),
      "depth_source": args.get("depth_source"),
      "modified": dt.datetime.fromtimestamp(stamp).isoformat(timespec="seconds"),
      "stamp": stamp,
    })
  out.sort(key=lambda x: (x["stamp"], x["name"]), reverse=True)
  for row in out:
    row.pop("stamp", None)
  return out


def load_session(root: pathlib.Path, name: str) -> dict:
  """Load the exact recorded-frame manifest and compact frame telemetry."""
  path = _safe_session(root, name)
  run = _read_json(path / "run.json", {})
  meta = _read_json(path / "meta.json", [])
  if not isinstance(meta, list):
    raise ValueError(f"{name}/meta.json is not a list")

  rows = []
  for m in meta:
    if not isinstance(m, dict) or "i" not in m:
      continue
    frame_file = m.get("frame_file") or f"{int(m['i']):06d}.npz"
    if pathlib.PurePath(frame_file).name != frame_file:
      continue
    if not (path / frame_file).is_file():
      continue
    stamp = m.get("frame_stamp")
    rows.append((int(m["i"]), None if stamp is None else float(stamp),
                 frame_file, m))
  rows.sort(key=lambda x: x[0])
  if not rows:
    raise ValueError(f"{name} has no indexed recorded frames")

  stamps = [r[1] for r in rows if r[1] is not None and math.isfinite(r[1])]
  zero = min(stamps) if stamps else 0.0
  fallback_fps = float(((run.get("perception") or {}).get("actual_hz") or 30.0))
  frames = []
  for pos, (record_i, stamp, frame_file, m) in enumerate(rows):
    t_s = ((stamp - zero) if stamp is not None and math.isfinite(stamp)
           else pos / max(fallback_fps, 1e-6))
    q = m.get("joint_pos")
    frames.append({
      "i": record_i,
      "file": frame_file,
      "t_s": round(float(t_s), 6),
      "frame_index": m.get("frame_index"),
      "event": m.get("event", "unknown"),
      "label": m.get("label", 0),
      "detections": m.get("detections"),
      "mask_state": m.get("mask_state"),
      "sensor": m.get("sensor"),
      "observation_age_ms": _ms(m.get("observation_age_s")),
      "publish_age_ms": _ms(m.get("perception_publish_age_s")),
      "gripper_gap_mm": (None if not q or len(q) < 7 else
                          round(float(q[6]) * 2000.0, 2)),
      "gripper_effort": m.get("gripper_effort"),
      "table_clearance_mm": _metres_to_mm(m.get("measured_table_clearance_m")),
      "joint_pos": q,
      "joint_vel": m.get("joint_vel"),
      "target": m.get("target"),
      "action": m.get("action"),
    })
  duration = max(0.0, frames[-1]["t_s"] - frames[0]["t_s"])
  fps = ((len(frames) - 1) / duration if len(frames) > 1 and duration > 0
         else fallback_fps)
  with np.load(path / frames[0]["file"], allow_pickle=False) as first:
    recorded_arrays = sorted(first.files)
  exact_visuals = ("detection_labels" in recorded_arrays
                   or "detection_labels_shape" in recorded_arrays)
  return {
    "name": name,
    "frames": frames,
    "duration_s": duration,
    "fps": fps,
    "event_counts": dict(collections.Counter(f["event"] for f in frames)),
    "visuals": {"exact": exact_visuals, "arrays": recorded_arrays},
    "run": run,
  }


def _ms(value):
  return None if value is None else round(float(value) * 1000.0, 3)


def _metres_to_mm(value):
  return None if value is None else round(float(value) * 1000.0, 3)


def _caption(image: np.ndarray, title: str, detail: str = "") -> None:
  """Readable pane label over both infrared and turbo colour."""
  cv2.rectangle(image, (0, 0), (image.shape[1], 52), (14, 19, 25), -1)
  cv2.putText(image, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
              0.56, (245, 245, 245), 2, cv2.LINE_AA)
  if detail:
    cv2.putText(image, detail, (12, 43), cv2.FONT_HERSHEY_SIMPLEX,
                0.41, (175, 190, 202), 1, cv2.LINE_AA)


def _gray_tile(value: np.ndarray | None, shape, title: str,
               detail: str = "") -> np.ndarray:
  h, w = shape
  if value is None:
    out = np.full((h, w, 3), (18, 23, 29), np.uint8)
    _caption(out, title, detail or "not present in this recording")
    cv2.putText(out, "NOT RECORDED", (w // 2 - 90, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (95, 110, 122), 2,
                cv2.LINE_AA)
    return out
  gray = np.asarray(value, dtype=np.uint8)
  if gray.shape[:2] != (h, w):
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
  out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
  _caption(out, title, detail)
  return out


def _color_tile(value: np.ndarray | None, shape, title: str,
                detail: str = "") -> np.ndarray:
  h, w = shape
  if value is None:
    return _gray_tile(None, shape, title, detail)
  out = np.asarray(value, dtype=np.uint8)
  if out.shape[:2] != (h, w):
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
  if out.ndim == 2:
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
  else:
    out = out[..., :3].copy()
  _caption(out, title, detail)
  return out


def _turbo(depth: np.ndarray) -> np.ndarray:
  scaled = np.clip((depth - 0.35) / (1.20 - 0.35), 0.0, 1.0)
  out = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
  out[depth <= 0] = (25, 25, 25)
  return out


def _full_labels(labels: np.ndarray | None, shape) -> np.ndarray | None:
  if labels is None:
    return None
  h, w = shape
  value = np.asarray(labels)
  if value.shape[:2] != (h, w):
    value = cv2.resize(value.astype(np.uint16), (w, h),
                       interpolation=cv2.INTER_NEAREST)
  return value


def _decode_mask(z, name: str) -> np.ndarray | None:
  if name in z:
    return np.asarray(z[name])
  shape_key, bits_key = f"{name}_shape", f"{name}_bits"
  if shape_key not in z or bits_key not in z:
    return None
  shape = tuple(int(x) for x in np.asarray(z[shape_key]).reshape(-1))
  count = int(np.prod(shape))
  flat = np.unpackbits(np.asarray(z[bits_key], dtype=np.uint8),
                       count=count, bitorder="little")
  return flat.reshape(shape).astype(bool)


def _decode_labels(z, name: str) -> np.ndarray | None:
  if name in z:
    return np.asarray(z[name])
  shape_key = f"{name}_shape"
  if shape_key not in z:
    return None
  shape = tuple(int(x) for x in np.asarray(z[shape_key]).reshape(-1))
  flat = np.zeros(int(np.prod(shape)), dtype=np.uint8)
  starts = np.asarray(z[f"{name}_run_start"], dtype=np.uint32)
  lengths = np.asarray(z[f"{name}_run_length"], dtype=np.uint32)
  values = np.asarray(z[f"{name}_run_value"], dtype=np.uint8)
  for start, length, value in zip(starts, lengths, values):
    flat[int(start):int(start) + int(length)] = value
  return flat.reshape(shape)


def _paint_instances(image: np.ndarray, labels: np.ndarray | None,
                     selected: int, detections: list | None,
                     *, fill: bool) -> np.ndarray:
  """Overlay every accepted detector instance; chosen target is green."""
  labels = _full_labels(labels, image.shape[:2])
  if labels is None:
    return image
  layer = image.copy()
  ids = [int(x) for x in np.unique(labels) if int(x) > 0]
  for label in ids:
    hit = label == int(selected)
    colour = (65, 230, 90) if hit else (60, 145, 245)
    binary = (labels == label).astype(np.uint8)
    if fill:
      layer[binary > 0] = colour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, colour, 3 if hit else 2,
                     cv2.LINE_AA)
  if fill:
    image[:] = cv2.addWeighted(image, 0.58, layer, 0.42, 0.0)
    # Restore crisp outlines after blending.
    for label in ids:
      hit = label == int(selected)
      colour = (65, 230, 90) if hit else (60, 145, 245)
      contours, _ = cv2.findContours((labels == label).astype(np.uint8),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
      cv2.drawContours(image, contours, -1, colour, 3 if hit else 2,
                       cv2.LINE_AA)
  by_label = {int(d.get("label", 0)): d for d in (detections or [])}
  for label in ids:
    ys, xs = np.nonzero(labels == label)
    if not len(xs):
      continue
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
    hit = label == int(selected)
    colour = (65, 230, 90) if hit else (60, 145, 245)
    cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2 if hit else 1)
    d = by_label.get(label, {})
    top = d.get("top_z")
    n_px = d.get("n_px", int((labels == label).sum()))
    text = (f"target #{label}  " if hit else f"#{label}  ")
    text += (f"{float(top) * 1000:.0f}mm  " if top is not None
             and math.isfinite(float(top)) else "")
    text += f"{int(n_px)}px"
    cv2.putText(image, text, (x, max(68, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, colour, 1, cv2.LINE_AA)
  return image


def _mask_half(mask_value: np.ndarray | None, size, colour,
               missing: str) -> np.ndarray:
  w, h = size
  out = np.full((h, w, 3), (18, 23, 29), np.uint8)
  if mask_value is None:
    cv2.putText(out, missing, (18, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (105, 120, 132), 1, cv2.LINE_AA)
    return out
  m = cv2.resize((np.asarray(mask_value) > 0).astype(np.uint8), (w, h),
                 interpolation=cv2.INTER_NEAREST)
  out[m > 0] = colour
  return out


class DepthReplay:
  """Rebuild the deterministic depth detector for pre-telemetry logs."""

  def __init__(self, session: pathlib.Path, run: dict):
    self.session = session
    self.reason = ""
    self.segmenter = self.kin = self.reproj = None
    backend = (run.get("args") or {}).get("mask", "depth")
    if backend != "depth":
      self.reason = f"legacy {backend} detector output was not recorded"
      return
    try:
      from . import config, mask, proprio, rectify
      rig_file = session / "rig.json"
      rig = config.Rig.load(rig_file if rig_file.exists()
                            else config.RIG_FILE)
      self.reproj = rectify.Reprojector(rig, device="cpu")
      self.segmenter = mask.DepthSegmenter(rig, self.reproj)
      self.kin = proprio.Kinematics()
      self._mask_module = mask
    except Exception as e:
      self.reason = f"cannot rebuild depth detector: {type(e).__name__}: {e}"

  def analyse(self, depth: np.ndarray, gray: np.ndarray, row: dict) -> dict:
    if self.segmenter is None:
      return {"detail": self.reason}
    q = row.get("joint_pos")
    if q is None:
      return {"detail": "legacy frame has no joints; detector not rebuilt"}
    try:
      self.kin.update(np.asarray(q, dtype=np.float64))
      seg = self.segmenter(depth, rgb=gray, arm=self.kin.link_spheres())
      selected = int(row.get("label") or 0)
      source = (self._mask_module.full_mask(
        seg, selected, self.segmenter.decimate) if selected else None)
      policy = None
      if source is not None:
        _d, _valid, policy = self.reproj(depth, payload=source)
      detections = [{
        "label": int(x.label), "n_px": int(x.n_px),
        "centroid_base": np.asarray(x.centroid_base).tolist(),
        "top_z": float(x.top_z), "bbox": [int(v) for v in x.bbox],
      } for x in seg.instances]
      return {"labels": seg.labels, "detections": detections,
              "source": source, "policy": policy,
              "detail": "offline depth rebuild (not historical SAM output)"}
    except Exception as e:
      return {"detail": f"depth rebuild failed: {type(e).__name__}: {e}"}


def render_frame(root: pathlib.Path, manifest: dict, index: int,
                 quality: int = 84, replay: DepthReplay | None = None) -> bytes:
  """Render D455 inputs, detector instances and final target masks."""
  frames = manifest["frames"]
  if not 0 <= index < len(frames):
    raise IndexError(index)
  row = frames[index]
  session = _safe_session(root, manifest["name"])
  frame_file = pathlib.PurePath(row["file"]).name
  with np.load(session / frame_file, allow_pickle=False) as z:
    schema = int(np.asarray(z["schema_version"]).item()) \
      if "schema_version" in z else 1
    stored_depth = np.asarray(z["depth"], dtype=np.float32) / 10000.0
    rgb = np.asarray(z["rgb"], dtype=np.uint8) if "rgb" in z else None
    ir_left = np.asarray(z["ir_left"], dtype=np.uint8) if "ir_left" in z else None
    ir_right = np.asarray(z["ir_right"], dtype=np.uint8) if "ir_right" in z else None
    gray = np.asarray(z["gray"], dtype=np.uint8) if "gray" in z else None
    sensor_depth = (np.asarray(z["sensor_depth"], dtype=np.float32) / 10000.0
                    if "sensor_depth" in z else None)
    exact = ("detection_labels" in z or "detection_labels_shape" in z)
    labels = _decode_labels(z, "detection_labels") if exact else None
    rgb_labels = _decode_labels(z, "detection_rgb_labels")
    sam_rgb = _decode_mask(z, "sam_rgb_mask")
    sam_raw = _decode_mask(z, "sam_raw_mask")
    source = _decode_mask(z, "source_mask")
    policy = _decode_mask(z, "policy_mask")

  if schema >= 3:
    native_depth, policy_depth = stored_depth, None
  else:
    native_depth = sensor_depth if sensor_depth is not None else stored_depth
    policy_depth = stored_depth

  detail = "recorded live perception" if exact else ""
  detections = row.get("detections")
  if exact and source is None:
    # Presence of detection_labels marks the new schema.  In that schema an
    # omitted source_mask means perception published an empty source mask, not
    # that the log forgot to save it.
    source = np.zeros_like(native_depth, dtype=np.uint8)
  if not exact:
    replay = replay or DepthReplay(session, manifest.get("run") or {})
    rebuilt = replay.analyse(stored_depth, gray, row)
    labels = rebuilt.get("labels")
    detections = rebuilt.get("detections")
    source = rebuilt.get("source")
    policy = rebuilt.get("policy")
    detail = rebuilt.get("detail", "legacy log")

  h, w = native_depth.shape[:2]
  raw_rgb = _color_tile(rgb, (h, w), "1  synchronized raw RGB",
                        "YOLO + SAM2 input; never aligned through depth")
  _paint_instances(raw_rgb, rgb_labels, int(row.get("label") or 0),
                   detections, fill=False)
  aligned = _gray_tile(gray, (h, w), "2  legacy depth-aligned RGB -> gray",
                       "logged for comparison only; no longer model input")
  left = _gray_tile(ir_left, (h, w), "3  raw left infrared",
                    "FoundationStereo left; same optical grid as depth")
  right = _gray_tile(ir_right, (h, w), "4  raw right infrared",
                     "FoundationStereo right; not depth-aligned")

  native = _turbo(native_depth)
  native_detail = ("raw D455 ASIC depth"
                   if schema >= 3 else
                   "D455 ASIC, synchronized diagnostic"
                   if sensor_depth is not None else "same as policy depth")
  _caption(native, "5  D455 native depth", native_detail)
  depth_source = ((manifest.get("run") or {}).get("args") or {}).get(
    "depth_source", "sensor")
  if policy_depth is None:
    depth_tile = _gray_tile(
      None, (h, w), "6  depth used by geometry + policy",
      f"{depth_source}; recompute from the saved raw streams")
  else:
    depth_tile = _turbo(policy_depth)
    _caption(depth_tile, "6  depth used by geometry + policy",
             f"{depth_source}; 0.35-1.20 m turbo")

  segmented = _turbo(policy_depth if policy_depth is not None else native_depth)
  _paint_instances(segmented, labels, int(row.get("label") or 0), detections,
                   fill=True)
  n_det = len(detections or [])
  _caption(segmented, "7  instances after RGB -> depth",
           f"{detail}; {n_det} accepted; green=selected, orange=other")

  quarter = w // 4
  sam_rgb_half = _mask_half(sam_rgb, (quarter, h), (225, 80, 210),
                            "RGB SAM unavailable")
  sam_half = _mask_half(sam_raw, (quarter, h), (210, 85, 220),
                        "depth SAM unavailable")
  source_half = _mask_half(source, (quarter, h), (70, 225, 105),
                           "source unavailable")
  policy_half = _mask_half(policy, (w - 3 * quarter, h), (75, 190, 245),
                           "policy unavailable")
  masks = np.hstack([sam_rgb_half, sam_half, source_half, policy_half])
  for x in (quarter, 2 * quarter, 3 * quarter):
    cv2.line(masks, (x, 0), (x, h), (100, 112, 125), 1)
  _caption(masks, "8  target masks",
           f"RGB SAM -> depth SAM -> accepted -> policy   {row.get('mask_state') or detail}")

  image = np.vstack([np.hstack([raw_rgb, aligned, left, right]),
                     np.hstack([native, depth_tile, segmented, masks])])
  bar = np.zeros((46, image.shape[1], 3), dtype=np.uint8)
  bar[:] = (22, 28, 35)
  event = str(row.get("event") or "unknown")
  colour = ((100, 210, 125) if event == "command" else
            (70, 180, 235) if event == "blind_target" else
            (90, 105, 225) if event.startswith("hold") else (180, 180, 180))
  title = (f"{manifest['name']}   t={row['t_s']:.3f}s   "
           f"frame {index + 1}/{len(frames)}   {event}")
  cv2.putText(bar, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
              0.58, colour, 2, cv2.LINE_AA)
  image = np.vstack([bar, image])
  ok, encoded = cv2.imencode(".jpg", image,
                             [int(cv2.IMWRITE_JPEG_QUALITY),
                              int(np.clip(quality, 40, 95))])
  if not ok:
    raise RuntimeError(f"could not encode {manifest['name']}/{frame_file}")
  return encoded.tobytes()


class Store:
  """Thread-safe manifests and a small rendered-frame LRU."""

  def __init__(self, root: pathlib.Path, cache_frames: int = 96):
    self.root = root.resolve()
    self.cache_frames = max(1, int(cache_frames))
    self._lock = threading.RLock()
    self._render_lock = threading.Lock()
    self._manifests: dict[str, tuple[int, dict]] = {}
    self._frames: collections.OrderedDict[tuple[str, int], bytes] = (
      collections.OrderedDict())
    self._replays: dict[str, DepthReplay] = {}

  def sessions(self):
    return list_sessions(self.root)

  def session(self, name: str) -> dict:
    path = _safe_session(self.root, name)
    version = max((p.stat().st_mtime_ns for p in (path / "meta.json",
                                                   path / "run.json")
                   if p.exists()), default=0)
    with self._lock:
      cached = self._manifests.get(name)
      if cached is not None and cached[0] == version:
        return cached[1]
    manifest = load_session(self.root, name)
    with self._lock:
      self._manifests[name] = (version, manifest)
    return manifest

  def frame(self, name: str, index: int) -> bytes:
    key = (name, int(index))
    with self._lock:
      if key in self._frames:
        value = self._frames.pop(key)
        self._frames[key] = value
        return value
    with self._render_lock:
      manifest = self.session(name)
      replay = None
      if not (manifest.get("visuals") or {}).get("exact", False):
        replay = self._replays.get(name)
        if replay is None:
          replay = DepthReplay(_safe_session(self.root, name),
                               manifest.get("run") or {})
          self._replays[name] = replay
      value = render_frame(self.root, manifest, int(index), replay=replay)
    with self._lock:
      self._frames[key] = value
      while len(self._frames) > self.cache_frames:
        self._frames.popitem(last=False)
    return value


class Server(ThreadingHTTPServer):
  daemon_threads = True

  def __init__(self, address, store: Store):
    self.store = store
    super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
  server: Server

  def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
    parsed = urllib.parse.urlsplit(self.path)
    query = urllib.parse.parse_qs(parsed.query)
    try:
      if parsed.path in ("/", "/index.html"):
        self._send(PAGE.encode(), "text/html; charset=utf-8")
      elif parsed.path == "/api/sessions":
        self._json(self.server.store.sessions())
      elif parsed.path == "/api/session":
        self._json(self.server.store.session(_one(query, "name")))
      elif parsed.path == "/api/frame":
        name = _one(query, "name")
        index = int(_one(query, "i"))
        self._send(self.server.store.frame(name, index), "image/jpeg")
      elif parsed.path == "/favicon.ico":
        self.send_error(HTTPStatus.NO_CONTENT)
      else:
        self.send_error(HTTPStatus.NOT_FOUND)
    except (ValueError, IndexError, FileNotFoundError) as e:
      self.send_error(HTTPStatus.BAD_REQUEST, str(e))
    except Exception as e:  # keep one bad recording from ending the viewer
      self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(e))

  def _json(self, value):
    self._send(json.dumps(value, ensure_ascii=False,
                          separators=(",", ":")).encode(),
               "application/json; charset=utf-8")

  def _send(self, body: bytes, content_type: str, cache="no-store"):
    self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", cache)
    self.send_header("X-Content-Type-Options", "nosniff")
    self.end_headers()
    self.wfile.write(body)

  def log_message(self, _format, *_args):
    return


def _one(query: dict, key: str) -> str:
  values = query.get(key)
  if not values or len(values) != 1:
    raise ValueError(f"query parameter {key!r} is required exactly once")
  return values[0]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("recordings"))
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--cache-frames", type=int, default=96)
  args = parser.parse_args()
  store = Store(args.root, args.cache_frames)
  server = Server((args.host, args.port), store)
  host, port = server.server_address[:2]
  print(f"log viewer: http://{host}:{port}/")
  print(f"recordings: {store.root} ({len(store.sessions())} playable logs)")
  print("Ctrl-C stops the viewer; it never connects to the camera or arm.")
  try:
    server.serve_forever(poll_interval=0.25)
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
