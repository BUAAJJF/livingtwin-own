# SAM2.1 in the perception evaluation

The deployment's perception stack, scored against the renderer's own
segmentation buffer, with and without SAM2.1 carrying the target between
frames.  This is the first measurement in this repository where "did the
segmenter find the right object" has a truth to be checked against; on the arm
the segmenter's output was the only thing available, and in simulation the
target mask came straight from the renderer by a path the deployment code never
touched.

## Reproduce

```bash
micromamba run -n mjlab python scripts/sim_perception_check.py \
    --steps 600 --seed 33 \
    --policy checkpoints/v7_teachers/strong_teacher.pt \
    --sam --html results/sam_eval/frames.html --out results/sam_eval/seed33.json
```

`--sam` loads `hardware/deploy/sam2_predictor.Sam2StreamingPredictor` (the
pinned `sam2.1_hiera_small.pt`, see `hardware/deploy/sam2_assets/PROVENANCE.json`)
behind `hardware/deploy/sam_tracker.SamTargetTracker`.  Without it the run
scores the depth stack alone and needs no GPU model.

Open `frames.html` for the per-frame page: every frame, three outlines
(renderer truth in green, depth segmenter in orange, SAM in blue), a zoomed
crop, and a lane strip you can click to scrub.

## Variants

All scored on the policy's 224x168 grid, because that is the only grid the
policy sees.

| variant | what it is |
|---|---|
| `depth` | `DepthSegmenter` + `TargetTracker`, alone |
| `deploy` | `depth` plus `target_mask.TargetMask` -- the hold-over and the grasp-site rebuild that `run.py` really ships |
| `sam_raw` | whatever SAM returned, watchdog ignored |
| `sam` | `sam_raw` after the watchdog, i.e. what may be published |
| `deploy+sam` | `sam` plus the same two fallbacks |

`sam_raw` exists so a strict watchdog can be told apart from a bad tracker.
Both of the watchdog's original thresholds turned out to be wrong, and only
this comparison could show it.

## Caveats that must travel with the numbers

* **One environment, one episode per seed, and the spread is enormous.**  The
  depth stack's approach detection ranged 6% to 92% across seeds on the same
  policy.  Any single run of this script is anecdote; the sweep is the result.
* **Wrong-target is not measured here.**  `Mjlab-Pick-Place-PiperX-Robust` puts
  one object on the table, so there is nothing to confuse it with, and 0.0% is
  a property of the scene rather than of SAM.  TwinSight measured 13% on a hard
  reviewed subset of real frames.  A run on `Mjlab-Cleanup-PiperX` is not a
  substitute: the state teacher does not play that task and the tracker's
  nearest-to-hand target is not the command's target, so the two disagree for
  reasons that have nothing to do with either detector.
* **The timings are not deployment latencies.**  They are this laptop GPU,
  inside a simulation loop, with `vos_optimized=False` and no warm-up
  exclusion.  The benchmark matrix in TwinSight's handoff has not been run.
* **SAM is not in `run.py`.**  There is no `--target-tracker sam21` flag; the
  arm still runs the depth stack.

## Result, seven seeds x 600 steps, `strong_teacher` driving

Seed 22 is absent: the object started 9 pixels from the edge of view and the
source self-check refused to score a stack against geometry it could not
verify.  That is the guard working, not a missing run.

### Approach (jaws open), detection and IoU against the renderer

| variant | detected | per-seed | IoU mean |
|---|---|---|---|
| `depth` | 47.1% | 11 92 6 86 63 51 20 | 0.363 |
| `deploy` | 47.5% | 11 94 6 86 63 51 20 | 0.365 |
| `sam` | **98.4%** | 99 100 99 98 98 96 99 | **0.858** |

The per-seed column is the point.  The depth stack is bimodal -- it works for
an episode or it fails for an episode -- and a single run of this check would
have supported any conclusion between 6% and 92%.  SAM has no such spread.

### While the object is held

| variant | detected | IoU mean |
|---|---|---|
| `depth` | 0.0% | 0.000 |
| `deploy` (60 mm sphere at the grasp site) | 100% | 0.614 |
| `sam` | 100% | 0.804 |

`deploy+sam` scores 0.614, not 0.804, because the grasp-site rebuild
*overwrites* SAM's mask.  With a tracker that carries the object through the
grasp the rebuild should become SAM's fallback rather than its replacement.

### Detection against gripper-to-object distance (approach)

| mm | depth | sam |
|---|---|---|
| 0-30 | **7%** | **100%** |
| 30-50 | 48% | 100% |
| 50-80 | 76% | 98% |
| 80-150 | 56% | 100% |
| >150 | 47% | 90% |

This is the shape that matters and it is invisible in any episode average.
The depth stack goes blind as the hand arrives -- which is the one moment the
policy cannot afford it.

### Watchdog

Two thresholds in `sam_tracker.WatchdogCfg` were wrong, and only the
`sam_raw`/`sam` comparison could show it.  Before the fixes, `sam` scored 51.2%
against `sam_raw`'s 98.1% on the same frames: the watchdog was discarding
correct masks, at IoU 0.76-0.95 against the renderer.

* the centre-displacement test compared against a reference that only advances
  on an *accepted* mask, so one rejection froze it and every later frame looked
  like a bigger jump -- it latched.  The budget now scales with the gap.
* the arm-overlap test was fed the segmenter's sphere cover, which puts 50 mm
  on `gripper_base` and 31 mm on each finger, so a grasped object is inside
  "the arm" by construction.  Spheres near the grasp site are now spared.

After both, across seven seeds: 23 `jumped`, 14 `area`, 14 `empty` rejections,
and `sam` tracks `sam_raw` to within a point.

### Cost

28.1 ms p50 [25.3, 30.8] per frame on this laptop 5090, BF16, eager, mask
threshold and transfer included, `torch.cuda.synchronize()` at the boundary.
Bounding the conditioning set (`sam2_predictor.MAX_ANCHORS`) was necessary:
SAM2 attends to every anchor ever added, and 37 re-anchors over 1200 frames
took p50 from 26.5 ms to 41.4 ms before that was capped.

---

# Wired into the arm's pipeline

`hardware/deploy/run.py --target-tracker {depth,sam21}`, default `depth`.
Nothing about a depth-only run changes: the SAM import, torch and the 184 MB
checkpoint are all behind the flag.

The composition is shared, not copied.  `SamTargetTracker.carry()` holds the
anchor policy -- SAM never picks, and an anchor is refreshed only when the
depth stack independently agrees -- and both `run.py` and
`scripts/sim_perception_check.py` call it.  The watchdog's arm mask moved to
`DepthSegmenter.arm_image_mask`, so the checker cannot see geometry the arm
does not.

## Three decisions, each with the measurement behind it

**LOST now waits for a fresh anchor.**  It used to keep propagating, and would
recover the moment SAM's mask happened to pass the watchdog again -- on
whatever it had drifted onto, which is the confident-wrong-object case the
watchdog exists to stop.  The cost is measured: approach detection 97.8% ->
94.8% on the same three seeds, because recovery now waits for the depth stack.
That is a measured cost against a benefit this simulation *cannot* measure
(one object on the table, so there is nothing to drift onto).  Taken anyway,
because the failure it prevents is the one the policy has no defence against.

**The grasp-site sphere became a fallback, not an override**
(`TargetMask.rebuild_only_if_empty`, set when SAM is active).  TwinSight's
handoff recommends standing down to the kinematic reconstruction during a
carry; scored against the renderer, SAM's held mask is IoU 0.839 and the
sphere is 0.609, so applying the sphere on top is a downgrade.  The handoff
was written before either could be scored.

**SAM is cleared on the lifecycle's HELD -> SEARCH edge, not on the depth
tracker's window.**  Keyed on `tracker.has_target` -- which was the obvious
choice and the first thing written -- it cleared SAM on *every frame of a
carry*, because the depth segmenter produces nothing there at all and the
tracker's 15-frame window expires.  Held detection went from 100% to **1.2%**.
Without `--target-lifecycle` there is no placement edge at all, so `run.py`
prints a warning saying SAM will keep carrying an object that is already in
the bin.

## Verified end to end, except the one number that matters

A real recording replays through the whole stack with the arm dry:

```bash
python -m hardware.deploy.run --replay recordings/v4_stereo_try3 --no-arm \
    --policy hardware/deploy/policies/d455_v4_final \
    --target-tracker sam21 --target-lifecycle --held-target-radius 0.045
```

It loads, anchors, publishes and shuts down cleanly.  **The perception rate
from that run is not usable**: a 4096-environment training job owned the same
GPU, and the depth-only baseline read 438 ms/frame against the ~35 ms this
loop has actually recorded on the arm.  A rate measured under contention is
not a rate.  `scripts/run_v8.sh` now runs both replays on a free GPU between
the teacher and the distillation, and that is the number to read.

## What still does not transfer

Every SAM figure here is **textured RGB in simulation**.  The rig feeds
grayscale IR.  TwinSight measured SAM2.1 on IR and got 85.1% correct target on
a hard reviewed subset -- so it works there -- but 95-98% is not a number that
crosses that gap, and neither is the 0% wrong-target, which is a property of a
table with one object on it.

---

# On the rig's own frames, replayed

`recordings/v4_stereo_try3`, arm dry, **idle GPU** (the earlier 438 ms/frame
figure was measured against a training job and is not a rate).

| `--target-tracker` | perception median | p95 | rate |
|---|---|---|---|
| `depth` | **18.5 ms** | 23.5 ms | 54.1 Hz |
| `sam21` | **76.5 ms** | 94.2 ms | 13.1 Hz |

Against `--max-obs-age 0.20` that is a factor of two of margin: SAM does not
deadlock the control loop.

```
sam: anchors=6  published 451/515 (88%),
     of which 356 (69% of frames) had no depth mask at all
     rejections[area=8  arm=2  jumped=4]
```

**This is the first evidence for SAM on real grayscale IR in this pipeline**,
and the watchdog is not thrashing: 14 rejections in 515 frames.  On the frames
where the depth stack produced nothing -- most of them -- SAM produced a mask.

**It is coverage, not correctness.**  A recording has no truth, so nothing here
says those 356 masks were the right object; TwinSight measured 13% wrong-target
on a reviewed subset, and the 0.86 IoU from simulation was textured RGB.

## The gap this opens

The trained observation-latency prior is a mean of **43 ms** over a support of
0/20/40/60/80 ms, with 5% of its mass on 80.  SAM's **median is 76.5 ms** and
its **p95 is 94.2** -- so a student distilled against the default meets, all of
the time, a lag it saw 5% of the time, and a tail the support cannot express at
all (`latency.LAGS` is five bins of one control step; a sixth is rejected).

That is the same class of mismatch this whole campaign is about, created by
fixing the other half of it.  `OBS_LATENCY_PROBS` now sets the prior from the
replay: the SAM-domain student trains at `0,0,0.10,0.30,0.60` (mean 70 ms, the
top of the support).  **80 ms is the honest ceiling until `LAGS` is widened**,
and widening it touches the ring buffer in `CameraScene` and `perturb` as well.

## The pair being trained

Same teacher checkpoint (`v8_remote/model_14200`, late/early 0.75), same
iterations, same episode length.  Only the observation model differs, so the
difference between them is the measurement of what SAM is worth:

| | visible | gaps | latency prior |
|---|---|---|---|
| `v8_student_sam` | (0.95, 1.0) | x0.3 | mean 70 ms |
| `v8_student_depth` | measured spread | x1.0 | mean 43 ms (default) |
