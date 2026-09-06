# yf/pc — first generation: teacher, four mask-free vision routes, shadow readiness (2026-09-06)

**Verdict: NO-GO for real motion.  GO for shadow tests (camera + joint read, no motor command) with the P1B bundle.**
Every route places objects and is safe by the shell's measure, but the best one reaches a third of the teacher's
throughput and every one decays within an episode.  Nothing below lowers a threshold to change that.

Code: branch `yf/pc`, commits a355df8 … 0a01302 (training ran at 5ca60a5 / c553390 for P2).  All numbers are
`-Robust` domain, `--sensor measured`, Action API v2 (spec `51c919a9dd2e92db`), no legacy loading anywhere.

## 1. Teacher

| checkpoint | placed/min (s101/s202/s303) | success | late/early (1200 steps, no reset) | full-reset l/e | jaw when stopped | verdict |
|---|---|---|---|---|---|---|
| v10c sight `model_7400` (`1c0568d3…`) | 12.1 / 10.9 / 11.9 | 0.906 | 0.64 / 0.60 / 0.66 | 0.65 | 23 mm, open | fail (throughput, endurance) |
| v10c nosight `model_7400` (`1ddfefad…`) | 19.7 / 18.6 / 19.8 | 0.933 | 0.834 / 0.862 / 0.833 → median 0.834 | 0.83 | 30 mm, open | fail by 0.016 on the endurance gate |
| **v11_nosight_36s `model_9399`** (`d685b548…19367a`) | 21.1 / 19.0 / 19.3 → **19.3** | **0.931** | 0.835 / 0.860 / 0.858 → **0.858** | 0.94 | 28 mm, open | **qualified** |

v11 = v10c nosight continued 2000 iterations at 8192 envs: episode 36 s, `RESET_FULL_RANGE=1`, `WRIST_W=0.30`
(inert on -NoSight), the v10c capability curriculum resumed at its last stage, no smoothness rescale
(`scripts/pc/continue_teacher.sh`, `results/pc/teacher/v11_nosight_36s/`).  Approach/engaged occlusion under the
D455 viewpoint 18.6% / 22.9%, |Δa| 0.120, no saturation, no NaN, trips 34.9/arm-hour.

## 2. Routes — final policies (distill 1500 + PPO 800, 512 envs; P0 256 envs), three seeds, 256 envs

| route | encoder (as built) | placed/min median [min, max] | success | drops | trips/h | endurance early → late | late/early | held-out (capped) placed / success | export |
|---|---|---|---|---|---|---|---|---|---|
| P0 | ResNet18-lite 2-2-2-2 on (depth m, valid), random init — **not DeFM** (unobtainable here) | 3.86 [3.86, 4.03] | 0.685 | 0.14 | 17-23 | 7.4 → 3.0 | 0.41 | 3.67 / 0.63 | OK 1.2e-5 |
| P1A | PointNet 4-64-128-256, masked max | 3.08 [3.03, 3.17] | 0.672 | 0.13 | 4.7-7.9 | 6.1 → 0.9 | 0.15 | 2.38 / 0.61 | OK 2.4e-7 |
| **P1B** | FPS-32 patches × kNN-16, 2-layer transformer — **no masked-reconstruction loss, not official PointPatchRL** | **6.42 [6.24, 6.57]** | **0.784** | 0.11 | 7.3-13.2 | 10.9 → 5.1 | **0.47** | **6.33 / 0.74** | OK 3.0e-7 |
| P2 | analytic top-down candidates (height map → components → arm/bin exclusion → width/margin/envelope) K=32 + lock + set MLP — **no pretrained detector, "IK" = envelope test** | 4.87 [4.44, 5.05] | 0.736 | 0.11 | 7.6-13.8 | 9.4 → 3.6 | 0.40 | 4.80 / 0.71 | OK 2.4e-7 |
| gate | | ≥ 13.5 (70% of teacher) | | | | | ≥ 0.85 | > 0 | agree |

Common to all four: same teacher and hash, same object split (capped held out), same budget, reward, curriculum, DR,
controller, GRU + bounded head, MSE on tanh(u), seeds 42 (train) / 101-202-303 (eval), 30 Hz camera through the 50 Hz
loop with a per-episode phase (fresh fraction 0.59-0.60 measured), 0-4-step processing latency, `vision_meta`.

Distilled students before PPO (seed 101): P0 3.68, P1A 2.51, P1B 4.57, P2 4.25 placed/min, success 0.54-0.68,
late/early 0.37-0.50.  PPO fine-tuning improved every route (+5% … +40%) and halved the shell trips.

Actions (P2, 600 steps): |Δa| 0.062, no dimension above 0.999 saturation, safe-env fraction 0.78, object_lost
1.0/arm-min, over_speed 0.25/arm-min.  P2 proposals: **recall within 3 cm of the target on fresh frames 38%**,
no usable candidate on 10.7% of fresh frames, 5.2 lock switches per environment per 600 steps.

### Failure classification

| failure | routes | evidence | reading |
|---|---|---|---|
| within-episode decay, jaw open | all | late/early 0.15-0.47 with the stopped jaw at 26-32 mm; teacher 0.86 | not the teacher's latch; the student stops engaging — the point cloud carries no "which object" signal and the GRU does not hold it across the long episode |
| low initiation, not low skill | all | success 0.67-0.78 per attempt vs teacher 0.93, throughput 16-33% | same shape as the v8 depth students (attempts, not conversions) |
| proposal recall | P2 | 38% of fresh frames have a candidate within 3 cm of the true object | the analytic proposer, not the executor, bounds P2 |
| encoder capacity | P1A | weakest on every metric, decays fastest | PointNet's global max loses the object among arm and bin points |

Against the last mask-based lineage at the same stage: the v3/v4 depth+mask students scored 14.5/min (success 0.84)
after distillation and 25/min (0.94) after PPO.  First-generation mask-free students are 3-6× weaker; the three
confounded causes are the missing target channel, half the training budget, and the 36 s episodes.

## 3. Deployment readiness (shadow only)

Best candidate bundle: `hardware/deploy/policies/pc_P1B_20260906T0319/`

| file | sha256 |
|---|---|
| policy.onnx | `7575c17b…5742e` |
| policy.pt | `3d957ede…b25e5` |
| checkpoint.pt (`…P1B_20260906T0319_finetune/model_799.pt`) | `fc1c96a2…e510` |
| obs_spec.json (actor: proprio 36, camera 512×4, vision_meta 3) | `540684d0…f3a` |
| rig_d455.json (hand-eye 3.9 mm, table z −3.8 mm) | `3ee223d9…e967` |

Shadow against `recordings/v4_stereo_try3` (real D455 + recorded joints, 1500 control steps, RTX 5090):

| route | control p50/p95 | perception p50/p95 | effective vision Hz | frame age p50/p95 | holds | GPU |
|---|---|---|---|---|---|---|
| P1B (final bundle) | 2.4 / 3.8 ms | 1.1 / 2.4 ms | 29.8 | 20 / 33 ms | 2 of 1500 | CUDA |
| P1A (smoke export) | 1.2 / 1.9 ms | 1.0 / 1.4 ms | 29.7 | 16 / 30 ms | 2 | CUDA |
| P0 (smoke export) | 1.9 / 2.7 ms | 2.1 / 3.2 ms | 29.7 | 17 / 31 ms | 1 | CUDA |
| P2 (smoke export) | 1.3 / 2.0 ms | 3.6 / 4.8 ms | 29.4 | 18 / 37 ms | 22 | CUDA |

The logged targets of the P1B run stay inside `SAFE_TARGET_CLIP`; |tanh u| ≤ 0.87; mean |Δa| 0.034.

Commands, logs, rollback and the step-by-step first-motion procedure: `docs/pc_shadow_runbook.md`.
Rollback: `hardware/deploy/policies/d455_v4_final` (Action API v1, mask pipeline, `run.py`) — the only policy that
has placed objects on the arm; it is a fallback for the arm, not a checkpoint of this pipeline.

## 4. What is NOT done

* No real-motion runner for the point-cloud observation: `pc_run.py` has no motor path by design, and `run.py`'s
  guarded path still consumes the mask observation.  Wiring the cloud into it is the next piece of work.
* DeFM and the official PointPatchRL were not used (unobtainable / out of timebox); P0 and P1B are labelled with
  what they are.
* One training seed per route; the seed spread quoted is over evaluation seeds only.

## 5. Where the numbers live

`results/pc/teacher_eval/` (v10c screens), `results/pc/teacher/v11_nosight_36s/` (continuation + screen),
`results/pc/routes/pc_screen_*` (screening from the unqualified teacher), `results/pc/routes/pc_final_*`
(this table; `final_report.json`), `results/pc/specs/` (observation specs), the aborted P2 run in
`pc_final_P2_20260906T0319_aborted_widthfix`.

Log convention: every stage transcript (`*.log`), stage marker and `export/` under these directories is untracked; the JSONs, `manifest.json` and this report are what is committed (`results/README.md`).
