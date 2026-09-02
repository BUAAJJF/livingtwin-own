"""Compare segmentation backends on a recorded session, offline.

``review.py`` replays one session through the backend the run actually used and
draws the result.  This asks a different question: *given the same recorded
frames, which backend finds the objects?*  So it replays the same session
through several backends, in the order the frames arrived, and puts their
per-frame answers side by side.

Two things make the comparison mean something.

**One tracker, one set of geometric filters.**  A backend here is only a source
of instance masks.  Everything downstream -- the workspace box, the arm
exclusion, the bin footprint, the table-plane fit, the height and footprint and
elongation gates, and the three-frames-in-five confirmation -- is
``mask.YoloSegmenter``'s and ``mask.TargetTracker``'s, unchanged and shared.
Whatever the numbers say, they are not saying that one backend has a kinder
post-processing chain than the other.

**Order is preserved and nothing is strided.**  ``TargetTracker`` confirms an
instance across consecutive frames, so a replay that skips frames hands it a
scene that teleports and it reports a worse number than the live run had.
Every frame is segmented, in order.

There is no ground truth here and the report does not pretend there is.  What
it reports is coverage (how often a target was found at all), agreement (how
often two backends found one in the same place), and plausibility (whether what
was chosen has the height and the footprint the task's objects have -- 24-90 mm
tall, ~180 px at the table's range).  A backend that finds a target in every
frame by aiming at the bin scores badly on the third and that is the point.

    python -m hardware.deploy.segbench baseline recordings/session
    python -m hardware.deploy.segbench eval recordings/session --masks DIR --name sam3_dart
    python -m hardware.deploy.segbench compare recordings/session
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

from . import config, mask, proprio, rectify

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results" / "segbench"
"""Where a run's per-frame records and summary go.  Under ``results/`` rather
than beside the recording because the recordings directory is gitignored bulk
and these are small enough to keep."""


# --------------------------------------------------------------------------
# A backend that reads masks someone else computed.


class PrecomputedDetector:
  """``yolo_backend``'s detector interface, served from disk.

  The point of this class is that ``mask.YoloSegmenter`` needs exactly one
  thing from its detector -- ``det(rgb, (h, w)) -> (K, H, W) bool`` -- so any
  model at all can be evaluated through the deployment's own geometry by
  writing its masks out first and replaying them here.  It also means the
  model does not have to run in the same Python environment as the deployment,
  which matters: SAM3 pins ``numpy < 2`` and this repository is on 2.4.
  """

  def __init__(self, store: pathlib.Path):
    self.store = pathlib.Path(store)
    if not self.store.is_dir():
      raise FileNotFoundError(f"no mask store at {self.store}")
    self._key: str | None = None
    self.scores: np.ndarray = np.zeros(0, dtype=np.float32)
    self.missing = 0

  def seek(self, key: str) -> None:
    """Name the frame the next call should answer for.

    The detector interface takes an image and nothing else, so the frame
    identity has to arrive by another route.  Set it before each call.
    """
    self._key = key

  def __call__(self, img: np.ndarray, shape=None) -> np.ndarray:
    h, w = shape if shape is not None else np.asarray(img).shape[:2]
    if self._key is None:
      raise RuntimeError("seek() the frame before segmenting it")
    f = self.store / f"{self._key}.npz"
    if not f.exists():
      self.missing += 1
      self.scores = np.zeros(0, dtype=np.float32)
      return np.zeros((0, h, w), dtype=bool)
    z = np.load(f)
    n = int(z["n"])
    self.scores = np.asarray(z["scores"], dtype=np.float32)
    if n == 0:
      return np.zeros((0, h, w), dtype=bool)
    mh, mw = (int(x) for x in z["hw"])
    m = np.unpackbits(z["masks"], axis=1, count=mh * mw)
    m = m.reshape(n, mh, mw).astype(bool)
    if (mh, mw) != (h, w):
      raise ValueError(f"{f} holds {mh}x{mw} masks, the frame is {h}x{w}")
    return m


@contextlib.contextmanager
def precomputed_segmenter(rig, reproj, store: pathlib.Path, conf=None,
                          bin_footprint=None):
  """``mask.YoloSegmenter`` reading ``store`` instead of a network.

  ``YoloSegmenter.__init__`` loads its detector through ``mask.load_detector``,
  which insists on a weights file that exists.  Patching that one name for the
  duration of the construction is the whole of the substitution, and it is done
  here rather than by editing ``mask.py`` because this is a measurement tool
  and the deployment path should not grow a branch for it.
  """
  det = PrecomputedDetector(store)
  original = mask.load_detector
  mask.load_detector = lambda *a, **k: det
  try:
    seg = mask.YoloSegmenter("<precomputed>", rig, reproj, conf=conf,
                             bin_footprint=bin_footprint)
  finally:
    mask.load_detector = original
  yield seg, det


# --------------------------------------------------------------------------
# The replay.


def _session_setup(session: pathlib.Path):
  rig_file = session / "rig.json"
  rig = config.Rig.load(rig_file if rig_file.exists() else config.RIG_FILE)
  reproj = rectify.Reprojector(rig, device="cpu")
  meta = {m["i"]: m for m in json.loads((session / "meta.json").read_text())
          if "i" in m and "joint_pos" in m}
  files = sorted(glob.glob(str(session / "*.npz")))
  if not files:
    raise ValueError(f"no frames in {session}")
  return rig, reproj, meta, files


def replay(session: pathlib.Path, backend: str, store: pathlib.Path | None,
           conf: float | None, limit: int | None,
           device: str = "cuda:0") -> dict:
  """Segment and track every frame of ``session`` with one backend."""
  rig, reproj, meta, files = _session_setup(session)
  kin = proprio.Kinematics()
  tracker = mask.TargetTracker()
  (rlo, rhi), (alo, ahi), _ = config.WORKSPACE_SECTOR

  if backend == "depth":
    ctx = contextlib.nullcontext((mask.DepthSegmenter(rig, reproj), None))
  elif backend == "yolo":
    # The incumbent appearance backend, run live rather than from a store.
    # It is here because it is what a new one has to beat, not only what the
    # depth backend does: ``--mask fused`` already exists and the question a
    # replacement has to answer is whether it is better than that.
    ctx = contextlib.nullcontext(
      (mask.YoloSegmenter(str(store), rig, reproj, conf=conf,
                          device=device), None))
  else:
    ctx = precomputed_segmenter(rig, reproj, store, conf=conf)
  rows: list[dict] = []
  with ctx as (seg, det):
    for f in (files if limit is None else files[:limit]):
      key = os.path.basename(f).split(".")[0]
      m = meta.get(int(key))
      if m is None:
        continue
      q = np.asarray(m["joint_pos"], dtype=np.float64)
      g = float(q[6]) if q.size > 6 else 0.05
      kin.update(np.array([*q[:6], g, -g]))
      z = np.load(f)
      depth = z["depth"].astype(np.float32) / 10000.0
      if det is not None:
        det.seek(key)
      t0 = time.perf_counter()
      out = seg(depth, rgb=z["gray"], arm=kin.link_spheres())
      ms = (time.perf_counter() - t0) * 1e3
      label = tracker.update(out, kin.site_pos)
      hit = next((i for i in out.instances if i.label == label), None)

      rejected: dict[str, int] = {}
      for why, _ in getattr(seg, "rejected", []):
        rejected[why] = rejected.get(why, 0) + 1

      row = {
        "i": int(m["i"]), "key": key,
        "instances": len(out.instances),
        "confirmed": len(tracker.confirmed_labels),
        "fill": round(float((depth > 0).mean()), 4),
        "ms": round(ms, 2),
        "rejected": rejected,
        "site_mm": [round(float(x) * 1000, 1) for x in kin.site_pos],
      }
      if hit is None:
        row.update(gap_mm=None, top_mm=None, n_px=None, r=None, a_deg=None,
                   in_sector=None, xy=None)
      else:
        c = np.asarray(hit.centroid_base, dtype=np.float64)
        r = float(math.hypot(c[0], c[1]))
        a = math.degrees(math.atan2(c[1], c[0]))
        row.update(
          gap_mm=round(float(np.linalg.norm(kin.site_pos - c)) * 1000, 1),
          top_mm=(None if not np.isfinite(hit.top_z)
                  else round(float(hit.top_z) * 1000, 1)),
          n_px=int(hit.n_px), r=round(r, 4), a_deg=round(a, 1),
          in_sector=bool(rlo <= r <= rhi and alo <= math.radians(a) <= ahi),
          xy=[round(float(c[0]), 4), round(float(c[1]), 4)],
        )
      rows.append(row)
  return {"backend": backend, "session": str(session),
          "store": (None if store is None else str(store)), "rows": rows,
          "missing_mask_frames": (0 if det is None else det.missing)}


# --------------------------------------------------------------------------
# The numbers.


OBJECT_TOP_MM = (24.0, 90.0)
"""What the task's objects actually are, from ``piper_push.objects``:
``OBJECT_HEIGHT_FLOOR`` 24 mm and ``OBJECT_MAX_HALF_HEIGHT`` 45 mm doubled.
The plausibility test is measured against this and not against the segmenter's
own ``max_top_z_m`` gate, which is 110 mm to leave room for smoothing and
calibration residual -- grading a backend against the gate it already passed
would only say that the gate ran."""

SEGMENTER_TOP_CEILING_MM = 110.0
"""``SegmenterCfg.max_top_z_m``.  A component taller than this was already
rejected, so the height check below only asks whether what survived is as short
as a *fragment* -- and the floor, not the ceiling, is what it tests.

There is deliberately **no** size criterion here, and the first version of this
file had one: 60-900 px, from the 180 px a 25-45 mm object covers at 0.7 m.  It
was wrong for the sessions it was applied to.  ``v4_stereo_repro_scene2`` is
not a bin of 25-45 mm blocks -- it opens with a ~200 mm black pouch, which YOLO
segments whole at 5169 px and the depth backend does not segment at all.
Scoring that as implausible marks a correct detection wrong and rewards the
backend that returned a fragment.  The size distribution is reported instead of
being graded, and the grading is done against ``presence_gt.json``, which is
somebody looking at the frames."""


def _pct(xs, q):
  return None if not xs else round(float(np.percentile(np.asarray(xs), q)), 1)


def summarise(run: dict) -> dict:
  rows = run["rows"]
  n = len(rows)
  hit = [r for r in rows if r["n_px"] is not None]
  tops = [r["top_mm"] for r in hit if r["top_mm"] is not None]
  pxs = [r["n_px"] for r in hit]
  # Tall enough to be a thing rather than a fragment of one.  Everything
  # taller than the ceiling was rejected upstream, so this is a floor test.
  tall = [r for r in hit if r["top_mm"] is not None
          and OBJECT_TOP_MM[0] <= r["top_mm"] <= SEGMENTER_TOP_CEILING_MM]
  rejected: dict[str, int] = {}
  for r in rows:
    for why, k in r["rejected"].items():
      rejected[why] = rejected.get(why, 0) + k
  ms = sorted(r["ms"] for r in rows)
  switches = 0
  prev = None
  for r in rows:
    if r["xy"] is None:
      continue
    if prev is not None and math.dist(r["xy"], prev) > 0.06:
      switches += 1
    prev = r["xy"]
  return {
    "backend": run["backend"],
    "frames": n,
    "with_target": len(hit),
    "with_target_pct": round(100.0 * len(hit) / max(n, 1), 1),
    # Frames where the backend produced at least one instance that survived
    # the geometry, whether or not the tracker chose it.  The gap between this
    # and ``with_target`` is the tracker's, not the segmenter's: three hits in
    # five frames before an instance may be chosen, and up to fifteen frames of
    # holding nothing after the previous target moves out of the association
    # gate.  Reporting only the second number credits the segmenter with a
    # latency it did not cause and hides a gain that is available without
    # touching it.
    "with_instance": sum(1 for r in rows if r["instances"] > 0),
    "with_instance_pct": round(
      100.0 * sum(1 for r in rows if r["instances"] > 0) / max(n, 1), 1),
    "tall_enough": len(tall),
    "tall_enough_pct": round(100.0 * len(tall) / max(n, 1), 1),
    "target_switches": switches,
    "instances_mean": round(float(np.mean([r["instances"] for r in rows])), 2),
    "confirmed_mean": round(float(np.mean([r["confirmed"] for r in rows])), 2),
    "in_sector_pct": (None if not hit else round(
      100.0 * sum(bool(r["in_sector"]) for r in hit) / len(hit), 1)),
    "top_mm": {"p5": _pct(tops, 5), "median": _pct(tops, 50),
               "p95": _pct(tops, 95)},
    "n_px": {"p5": _pct(pxs, 5), "median": _pct(pxs, 50), "p95": _pct(pxs, 95)},
    # What the *replay* cost, which for a precomputed backend is disk and
    # decompression and the geometry -- not the model.  A store written by
    # ``sam3_infer.py`` carries the model's own timing in its ``_meta.json``
    # and that is the number that belongs in a latency budget; conflating the
    # two would report SAM3 as costing whatever unpacking its masks costs.
    "replay_ms": {"median": round(float(np.median(ms)), 2) if ms else None,
                  "p95": round(float(np.percentile(ms, 95)), 2) if ms else None},
    "model_ms": (run.get("store_meta") or {}).get("ms"),
    "prompt": (run.get("store_meta") or {}).get("classes"),
    "rejected": dict(sorted(rejected.items(), key=lambda kv: -kv[1])),
    "missing_mask_frames": run.get("missing_mask_frames", 0),
  }


# --------------------------------------------------------------------------
# Scoring against what somebody saw in the frames.


def score_against_gt(rows: list[dict], gt: dict) -> dict:
  """Recall and false-alarm rate on the frames a person labelled.

  Two numbers, and they only mean anything together.  ``recall`` is over the
  labelled frames that had an object on the mat; ``false_alarm`` is over the
  ones that did not.  A backend can have either alone -- return a target every
  frame and recall is 1.0, return none and false alarm is 0.0 -- so the pair is
  the measurement and neither half is a score.

  Frames the label marks ``ambiguous`` (an object at the gripper that may
  already be held) are excluded from both, because ``mask.arm_mask`` removes an
  object in the hand on purpose and neither answer is wrong there.
  """
  by_i = {r["i"]: r for r in rows}
  pos = neg = tp = fp = tp_i = fp_i = 0
  missed, false = [], []
  for k, v in gt["frames"].items():
    if v.get("ambiguous"):
      continue
    r = by_i.get(int(k))
    if r is None:
      continue
    found = r["n_px"] is not None
    any_inst = r["instances"] > 0
    if v["n"] > 0:
      pos += 1
      tp += found
      tp_i += any_inst
      if not found:
        missed.append(int(k))
    else:
      neg += 1
      fp += found
      fp_i += any_inst
      if found:
        false.append(int(k))
  return {
    "labelled_with_object": pos, "labelled_empty": neg,
    "found_when_present": tp,
    "recall": None if not pos else round(tp / pos, 3),
    "found_when_empty": fp,
    "false_alarm": None if not neg else round(fp / neg, 3),
    # The same pair before the tracker: what the segmenter alone could offer.
    "recall_any_instance": None if not pos else round(tp_i / pos, 3),
    "false_alarm_any_instance": None if not neg else round(fp_i / neg, 3),
    "missed_frames": missed, "false_alarm_frames": false,
  }


def _out_dir(session: pathlib.Path) -> pathlib.Path:
  d = RESULTS / session.name
  d.mkdir(parents=True, exist_ok=True)
  return d


def _store_meta(store) -> dict | None:
  if store is None:
    return None
  f = pathlib.Path(store) / "_meta.json"
  return json.loads(f.read_text()) if f.exists() else None


def _write(session: pathlib.Path, name: str, run: dict) -> dict:
  d = _out_dir(session)
  with (d / f"{name}.jsonl").open("w") as fh:
    for r in run["rows"]:
      fh.write(json.dumps(r) + "\n")
  s = summarise(run)
  (d / f"{name}.json").write_text(json.dumps(s, indent=1))
  return s


def _report(s: dict) -> str:
  g = s.get("gt") or {}
  rc = "  -  " if g.get("recall") is None else f"{g['recall']:.2f}"
  fa = "  -  " if g.get("false_alarm") is None else f"{g['false_alarm']:.2f}"
  return (
    f"{s['backend']:<12} {s['frames']:>5} fr  "
    f"target {s['with_target_pct']:>5.1f}%  "
    f"recall {rc}  false-alarm {fa}  "
    f"seg-recall {'  -  ' if g.get('recall_any_instance') is None else format(g['recall_any_instance'], '.2f')}  "
    f"inst {s['instances_mean']:>4.2f}  "
    f"top {str(s['top_mm']['median']):>5} mm  "
    f"px {str(s['n_px']['median']):>6}  "
    + (f"model {s['model_ms']['median']:>6.1f} ms"
       if s.get("model_ms") else f"{s['replay_ms']['median']:>6.1f} ms")
  )


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("mode", choices=["baseline", "yolo", "eval", "compare", "score"])
  p.add_argument("session", type=pathlib.Path)
  p.add_argument("--masks", type=pathlib.Path, default=None,
                 help="directory of per-frame mask .npz written by sam3_infer")
  p.add_argument("--name", default=None, help="what to call this run")
  p.add_argument("--conf", type=float, default=None)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--limit", type=int, default=None,
                 help="stop after N frames -- for a smoke test only, the "
                      "numbers are not the session's")
  a = p.parse_args()

  if a.mode in ("compare", "score"):
    d = _out_dir(a.session)
    gt_file = d / "presence_gt.json"
    gt = json.loads(gt_file.read_text()) if gt_file.exists() else None
    out = []
    for f in sorted(d.glob("*.jsonl")):
      rows = [json.loads(line) for line in f.read_text().splitlines() if line]
      summary = summarise({"backend": f.stem, "rows": rows,
                           "store_meta": _store_meta(
                             d / f"masks_{f.stem}" if (d / f"masks_{f.stem}").is_dir()
                             else None)})
      if gt is not None:
        summary["gt"] = score_against_gt(rows, gt)
      (d / f"{f.stem}.json").write_text(json.dumps(summary, indent=1))
      out.append(summary)
    if not out:
      print(f"no .jsonl under {d}", file=sys.stderr)
      return 1
    if gt is None:
      print(f"note: no presence_gt.json in {d}; recall/false-alarm omitted")
    for s in out:
      print(_report(s))
    (d / "comparison.json").write_text(json.dumps(out, indent=1))
    return 0

  if a.mode == "baseline":
    name, backend, store = a.name or "depth", "depth", None
  elif a.mode == "yolo":
    w = a.masks or (pathlib.Path(__file__).resolve().parent
                    / "yolo_d455" / "best.pt")
    name, backend, store = a.name or "yolo", "yolo", w
  else:
    if a.masks is None:
      print("eval needs --masks", file=sys.stderr)
      return 1
    name, backend, store = a.name or a.masks.name, "precomputed", a.masks

  run = replay(a.session, backend, store, a.conf, a.limit, a.device)
  run["store_meta"] = _store_meta(store)
  print(_report(_write(a.session, name, run)))
  return 0


if __name__ == "__main__":
  sys.exit(main())
