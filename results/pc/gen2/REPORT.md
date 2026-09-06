# yf/pc second generation -- why the first-generation student stops starting, and E0 / E2

Status: **complete** (audit, E0 / E2 run on 2026-09-06, decision in section 4).  Every number carries the
commit, the checkpoint hash, the sensor setting and the budget it was measured under.  Facts,
hypotheses and open questions are labelled as such.

Code: branch `yf/pc`, commit `db3b3ee` and after (the first generation trained at `5ca60a5`,
before the cadence fix `0127fdf`).  Teacher for every student: `v11_nosight_36s model_9399`,
sha256 `d685b54821593f724714c4969bda6d7ca8a11e7fe6aa85f4c5a5c4555e19367a`.  Domain `-Robust`,
`--sensor measured`, Action API v2 (spec `51c919a9dd2e92db`), no legacy loading anywhere.

## 1. Audit of the training problem (facts read off the code and the recorded configs)

| question | finding | where |
|---|---|---|
| What decides the teacher's target? | **There is one object on the table.**  `make_robust_env_cfg` builds `make_pick_place_env_cfg(num_objects=1)`; `PickCommand._retarget` (nearest object to the hand, re-evaluated on a clearance) is a no-op with one object.  The target is the only object, re-placed at a random pose (and redrawn in shape, mass, friction) the step a placement registers. | `robust_cfg.make_robust_env_cfg`, `mdp.PickCommand._retarget`, `_place_object`, `reshape_on_place` |
| Does the teacher's action depend on something the student's observation history cannot carry? | The teacher reads `full_proprio` + `object` (pose, velocity, ee-to-object, object-to-drop, shape) and never the `privileged` group (mass, friction).  Of those, the student's cloud carries the object's pose and shape *when its points survive the crop and the sample*; the teacher also sees `grasped` where the student sees `squeeze` + `pad_contact`.  So the hidden quantity is not "which object" but **whether the object is in the cloud at all, and how many of its points** -- the measured question of section 2. | `env_cfg.make_pick_place_env_cfg` (`object_state`), `pc_cfg.pc_distill_runner_cfg` (`teacher: full_proprio, object`) |
| DAgger labels, reward target and the student's target -- consistent? | Yes, by construction: rewards, terminations, metrics and the teacher's observation all read the target through the command (`_gather(self.target)`), and with one object the target index is constant 0.  `target_switches` is reported by the new ruler and is 0 on every run. | `mdp.PickCommand` accessors |
| Is the cadence fix in the branch? | **Yes.**  `0127fdf` is an ancestor of HEAD; `tests/test_pc.py::test_ring_holds_the_newest_frame_and_delays_by_the_lag` pins it.  Every gen-1 checkpoint (`pc_screen_*`, `pc_final_*`) trained under the alternating hold. | `git merge-base --is-ancestor 0127fdf HEAD` |
| Latency / fresh / hold consistent between simulator and deployment? | Simulator: `age = (t - t_last) + lag`, read back through the ring so the delayed cloud's meta is the meta of the frame the policy sees (`vision_meta = [min(age,10)/10, fresh, valid]`).  Deployment (`pc_run.py`, `run.py --obs pc`): `age_s = now - capture stamp`, `fresh = index changed`, the latest frame held between captures, the same `[age/10, fresh, valid]` encoding via `pc_perception.vision_meta`.  Semantics agree; the deployment holds the *newest* frame, which is what the fixed simulator does. | `cloud.WorkspaceCloud.__call__`, `pc_perception.vision_meta`, `pc_run.py:278-293` |
| What does `endurance` 1200 steps measure? | **24 s** at 50 Hz (`decimation` 10 x 2 ms).  Training episodes are 36 s, the play config's are 40 s; the endurance run therefore never crosses either horizon.  `late/early` from it is a within-24-s decay, not evidence about the horizon.  The 2400-step accept run (48 s) contains one synchronous time-out reset at step 2000. | `env_cfg` (`decimation`, `episode_length_s`), `eval_endurance.py` |
| DAgger "Clear" and the PPO degradation recipe on the point-cloud routes? | The cloud term receives no `mask_dropout` and no `target_process` (both are mask-line features); the sensor model, cadence and latency are identical in the `-Distill` and `-Vision` task ids.  **There is no student-only observation degradation at either stage on any PC route**; "Clear" and "the PPO recipe" are the same environment. | `pc_cfg.make_pc_env_cfg` (`term_params`), `cloud._capture` (`mask_dropout=None, target_process=None`) |
| Teacher and student reset distributions | The teacher trained with `RESET_FULL_RANGE=1` (whole soft-limit box); the students train with it unset (`run_route.sh` unsets it; delta 0.7 rad about home).  The student's start states are a subset of the teacher's, so the labels are in-distribution for the teacher; noted, not a defect. | `results/pc/teacher/v11_nosight_36s/manifest.json`, `run_route.sh` |

Hypothesis under test (NOT a finding): the student stops initiating because after a placement
the new object is represented by too few points in the sampled cloud for the encoder to notice,
and nothing else in the observation says "there is an object".  E2 asks whether telling the
policy *which* points are the object (on the points it already has) restores initiation; the
counts in section 2 say how many such points there are to flag.

## 2. The first-generation P1B student under the new ruler (on the fixed cadence)

Checkpoint `checkpoints/pc/pc_final_P1B_model_799.pt` (sha256 `fc1c96a2…e510`, the bundle's), trained
at `5ca60a5` under the alternating hold, rolled out here on the FIXED hold (`0127fdf`) -- so its input
distribution differs from what it trained on, in an unknown direction.  These numbers describe the
failure's shape, not its size; E0 is the same route retrained on the fixed hold.  256 envs,
`--sensor measured`, play config (40 s time-out, none reached in the 36 s runs), engagement radius
90 mm (`grasp_reach_m`), stall = an approach run of at least 3 s with the object on the table.
Files: `results/pc/gen2/audit/initiation_gen1_P1B_s{101,202,303}.json`, `long_gen1_P1B_s101.json`,
`bin_settle_gen1_P1B_s101.json`, pages `viewer_gen1_P1B_s101.html` (untracked, 35 MB).

**2.1 What decays and what does not (seed 101, 36 s, 2 s windows).**

| | window 1 | window 9 | window 18 | 180 s run, per 36 s block |
|---|---|---|---|---|
| placed / min | 10.2 | 5.6 | 1.5 | 6.1 → 0.9 → 0.3 → 0.1 → 0.1 |
| attempts / min (hand enters 90 mm of the object) | 70 | 55 | 47 | 59 → 47 → 43 → 43 → 40 |
| steps carrying | 4.8 % | 2.0 % | 0.7 % | 2.1 → 0.3 → 0.1 → 0.0 → 0.1 % |
| steps engaged (within 90 mm, not grasped) | 36 % | 31 % | 24 % | |
| steps stalled | 3 % | 29 % | 38 % | 26 → 43 → 47 → 47 → 49 % |
| target points among the 512, fresh frames | 17.7 | 13.7 | 12.8 | 13.5 → 13.1 (flat) |

Whole-run figures, three evaluation seeds, median [min, max] (36 s):

| | median [min, max] |
|---|---|
| placed / min (all arm-time) | 6.15 [6.05, 6.31] |
| placed / min over live time (stuck-object time excluded, 2.3) | 6.56 [6.43, 6.58] |
| attempts / min | 55.3 [55.3, 56.2] |
| success per attempt / per secure grasp | 0.111 [0.109, 0.112] / 0.880 [0.878, 0.882] |
| late/early, placements (raw / live time) | 0.33 [0.28, 0.39] / 0.34 [0.30, 0.41] |
| late/early, attempts | 0.85 [0.78, 0.86] |
| wait from a placement to the next attempt, p50 / p90 | 0.44 / 0.64 s |
| time from episode start to the first attempt, p50 | 0.56 s |
| stalled step fraction; stalls per arm-minute; length p50 / p90 | 0.28 [0.27, 0.31]; 1.2; 9.4 / 31 s |
| environments stalled / stuck at the end of the run | 35 % / 8.6 % |
| jaw commanded while stalled / while engaged (measured) | 26.5 mm / 24.8 mm (30.5 mm) |
| drops; `object_lost` / min; `over_speed` / min | 60 [51, 60]; 0.67; 0.22 |

The policy re-engages the new object at once (0.44 s), so the loss is not in restarting after
a success.  Over 180 s without a reset (seed 101): placed/min 1.64 overall (1.82 over live time),
per 36 s block 6.4 → 1.0 → 0.4 → 0.4 → 0.1, late/early 0.08, attempts late/early 0.84, stalled
45 % of steps, carrying 0.5 %.

*Reading (fact):* the student does not stop **approaching** -- attempts fall by a third while
placements fall twenty-fold -- it stops **grasping**.  Time within 90 mm of the object stays at
20-35 % while time carrying goes to zero; the jaw hovers half-open (26-29 mm commanded, of 50)
whether approaching, engaged or stalled.  "Stops initiating" is the wrong description of the
gen-1 decay; "reaches and does not close" is what the record shows.

**2.2 What the cloud shows of the target, by phase (seed 101 shown; the three-seed medians are
within 10 %: approach 13-14 points, engaged 10.7 [10.5, 11.2], zero-target frames while engaged
0.29 [0.29, 0.30]).  Fresh captures only.**

| phase | share of steps | target pixels after the crop, mean | target points of 512, mean / p50 / p10 | fresh frames with 0 target points |
|---|---|---|---|---|
| approach, before the first placement | 23 % | 28.7 | 15.5 / 11 / 0 | 27 % |
| approach, after a placement | 31 % | 26.1 | 13.3 / 8 / 0 | 31 % |
| engaged (within 90 mm) | 30 % | 25.4 | 10.1 / 5 / 0 | 31 % |
| carry | 2 % | 29.1 | 14.7 / 10 / 0 | 14 % |
| place / off the table | 14 % | 30.6 | 22.4 / 15 / 0 | 21 % |

*Facts:* the object is 25-45 mm wide at 1.2 m, about 26-30 pixels after the 10 mm-above-plane crop,
and 8-15 of the 512 sampled points on a typical fresh frame; on **a third of the fresh frames --
and a third of the frames while the hand is at the object -- it contributes no point at all**.
Visibility does not fall over the episode (13.5 → 13.1 points per 36 s block over 180 s); it is
low throughout.  The 512-point set is dominated by the arm and the bin.

**2.3 A scoring artefact that reads as decay: the object that will not settle in the bin.**

`PickCommand` registers a placement only when the object is inside the footprint less 12 mm,
below the rim, released, **settled (|v| < 0.06 m/s)** and grasped beforehand, for 5 consecutive
steps; until then the table is not refilled.  `scripts/pc/diag_bin_settle.py` on the same rollout
(seed 101, 36 s): of 45 276 steps with the object inside the bin footprint and not grasped, **37 193
(82 %) fail only the settled clause**; on those the object moves at 0.12 m/s median (p90 0.83, max
2.2 m/s) and spins at 5.3 rad/s median with the hand more than 150 mm away in 77 % of them and the
pads touching in 2 %.  **Cylinders are 69 % of those steps against a 23 % base rate** (median half
size 20 × 20 × 29 mm, height/width 1.4): a cylinder dropped into the bin lands on its side and
rolls between the walls.  67 of 256 environments (26 %) spent more than 3 s in that state and 15
(5.9 %) more than 20 s -- dead for the ruler, because nothing can be placed while the last object
is still "being placed".  Its size for this student: 5.9 % [3.8, 6.5] of all steps in 36 s and
10 % over 180 s; excluding it moves placed/min from 6.15 to 6.56 and late/early from 0.33 to 0.34,
so **it is real, it is the same for every policy on this task including the teacher, and it is
not what makes this student decay**.  The new ruler reports it apart (`stuck_object`,
`placed_per_live_min`, `late_over_early_placed_live`) and never as a stall; changing the
placement rule or the objects' rolling friction is an environment change and is left to the next
round (section 4).

**2.4 Audit items answered by measurement.**  Target switches: 0 in every run (one object).
Terminations in 36 s × 256 envs: `object_lost` 108 (0.70 per arm-minute -- the object pushed out of
the sector), `over_speed` 23, `nan` 0, `time_out` 0; the 180 s run had no time-out by construction
and 167 resets from the two terminations, all reported.  The jaw command tracks the measured
opening within 1 mm except while engaged (24 mm commanded vs 30 measured: the jaw is being asked
to close and the object is not in it).

**2.5 Teacher-side reference: the qualified v11 teacher under the same ruler** (state task
`Mjlab-Pick-Place-PiperX-Robust`, three seeds, 256 envs, on the training host:
`results/pc/gen2/teacher_v11_20260906T1020/`, sha256 `d685b548…`).  The local v10c nosight
`model_7400` (the checkpoint v11 was continued from) gives the same picture at 19.1 placed/min and
late/early 0.70 (`results/pc/gen2/audit/*teacher_v10c*`).

| | gen-1 P1B student (3 seeds) | v11 teacher (3 seeds) |
|---|---|---|
| placed / min, 36 s (raw / live time) | 6.15 / 6.56 | 20.7 [20.2, 20.8] / 20.9 |
| attempts / min; grasps / min | 55.3; 7.0 | 65.3; 23.8 |
| grasps per attempt; success per grasp | 0.126; 0.88 | 0.36 [0.35, 0.37]; 0.87 |
| steps engaged / carrying | 31 % / 2.2 % | 32 % / 7.9 % |
| jaw commanded while engaged (measured) | 24.8 mm (30.5) | 17.3 mm |
| stalled step fraction; envs stalled at the end | 0.28; 35 % | 0.19 [0.18, 0.20]; 31 % |
| late/early placed, 36 s (raw / live) | 0.33 / 0.34 | 0.78 [0.78, 0.80] / 0.78 |
| late/early attempts, 36 s | 0.85 | 0.83 |
| 180 s, per 36 s block, placed / min (seed 101) | 6.4 → 1.0 → 0.4 → 0.4 → 0.1 | 20.5 → 12.2 → 8.1 → 6.5 → 4.9 |
| 180 s, late/early placed / attempts; last 36 s over first 36 s | 0.08 / 0.84; 0.01 | 0.41 / 0.51; 0.24 |
| 180 s, stalled step fraction | 0.46 | 0.50 |
| stuck-object step fraction, 36 s / 180 s | 0.059 / 0.10 | 0.010 / 0.014 |
| bin-unsettled envs > 3 s (of 256, seed 101); object speed p50 | 67; 0.10 m/s | 188; 0.26 m/s |
| drops per 36 s run; `object_lost` per arm-minute | 60; 0.67 | 83; 0.38 |

*Facts:* (i) once the hand is within 90 mm, the teacher converts to a secure grasp 2.9× as often as
the student (0.36 vs 0.126 grasps per attempt) and commands the jaw to 17 mm where the student
commands 25 mm; once grasped both place about 87-88 % of the time -- the student's deficit is
**at the grasp**, not before it and not after it.  (ii) The teacher itself decays past its
horizon: 20.5 → 4.9 placed/min over 180 s, the last 36 s at a quarter of the first, stalls at
half of all steps, attempts late/early 0.51.  A student distilled from it cannot be expected to
hold up where its labels do not, so the long run scores E0 and E2 **against this row**, not
against 1.0; and the 24 s endurance number the gate uses (v11: 0.858) is a different, shorter
quantity from the 36 s late/early here (0.78).  (iii) The bin artefact hits the teacher in more
environments (it releases faster, 0.26 m/s) but for far less time (1 % of steps): it is not what
separates them.

**2.6 What the audit rules out and what it leaves.**  Ruled out as *the* cause of the gen-1
decay: a hidden "which object" (one object); a missing target *channel in the reward* (the
command is the same everywhere); the teacher's gripper latch (jaw open in every phase); an
evaluation defect (three seeds within 4 %, live-time rates, terminations counted); the bin
artefact (6 % of steps).  Left, and now measurable per phase: the student rarely closes on an
object it is next to, with the object present in the cloud as ~11 points (p50 5, absent on 29 %
of frames) at that moment; and a horizon decay the teacher shares (its own last-36-s rate is a
quarter of its first).  E0 (fixed cadence,
same everything) says how much of the gen-1 number was the cadence bug; E2 (the same 5-11 points
flagged) says whether *knowing which of the points are the object* is what the grasp is missing.

## 3. E0 (P1BZ) and E2 (P1BT): the results

Both ran on shen-teacher (RTX 6000D, GPUs 4 and 5) from 10:20 to 13:08 UTC on 2026-09-06, code
`5eb6d76`, tags `pc_gen2_P1BZ_20260906T1020` and `pc_gen2_P1BT_20260906T1020` under
`results/pc/routes/`; the launcher, the manifests and `timing.jsonl` record everything below.
Both smokes passed with the cloud 512 × 5, the zero column exactly 0 and the oracle column flagging
51 of 512 points on the start frames.  Controls, identical by construction and by manifest: teacher
`d685b548…` (v11), training seed 42, 512 envs, distill 1500 iterations (24.6 M env steps, 3000
optimizer updates at gradient length 16) + PPO 800 (13.1 M env steps, 16 000 updates), episode
36 s, the same cloud, encoder (PointPatchEncoder, 930 638 parameters), GRU, reward, curriculum, DR,
sensor and latency; no dropout at either stage.  Wall time E0 / E2: distill 72 / 65 min (E0 shared
its GPU with the teacher reference for the first 25 min), fine-tune 51 / 52 min, evaluation 13 / 14
min, initiation + long runs 24 / 25 min, viewer 3 / 3 min, export 15 / 13 s.  E2's export is a
graph-consistency check only (`export/ORACLE_ONLY`; the bundle and every deployment entry point
refuse the route).

**3.1 The accept / endurance ruler (three evaluation seeds, 256 envs, `--sensor measured`).**

| final policy | E0 P1BZ (zero column) | E2 P1BT (oracle flag) | gen-1 P1B (bug cadence) | gate |
|---|---|---|---|---|
| placed / min | 7.01 [6.45, 7.13] | 8.83 [8.59, 9.26] | 6.42 [6.24, 6.57] | ≥ 13.5 |
| success | 0.800 [0.769, 0.801] | 0.823 [0.819, 0.838] | 0.784 | |
| drop rate | 0.114 | 0.107 | 0.11 | |
| trips / h | 12.0 [11.4, 14.1] | 14.9 [14.4, 16.4] | 7.3-13.2 | |
| endurance (24 s) late/early | 0.45 [0.42, 0.51] | 0.62 [0.57, 0.65] | 0.47 | ≥ 0.85 |
| held-out (capped) placed / success | 6.82 / 0.776 | 8.05 / 0.795 | 6.33 / 0.74 | > 0 |
| actions: |Δa|, sat > 0.999, non-finite, safe-env fraction | 0.066, 0, 0, 0.73 | 0.068, 0, 0, 0.70 | 0.034-0.07 | |
| occlusion, approach / engaged | 0.188 / 0.207 | 0.156 / 0.201 | | |
| export | OK (onnx 3.6e-7, jit 1.2e-7) | OK (onnx 3.0e-7, jit 1.9e-7) | OK | agree |
| distilled student (seed 101): placed / min, success, l/e | 3.97, 0.650, 0.40 | 7.34, 0.763, 0.58 | 4.57, 0.673, 0.49 | |
| gate | **NO-GO** (throughput, late/early) | **NO-GO** (and oracle-only) | NO-GO | |

**3.2 The initiation ruler (36 s, three seeds) and the 180 s no-reset run (three seeds).**

| | E0 P1BZ | E2 P1BT | teacher v11 | gen-1 P1B |
|---|---|---|---|---|
| placed / min (raw / live time) | 6.96 / 7.49 | 9.01 / 9.32 | 20.7 / 20.9 | 6.15 / 6.56 |
| attempts / min → grasps / min | 43.8 → 8.3 | 47.0 → 10.5 | 65.3 → 23.8 | 55.3 → 7.0 |
| grasps per attempt | 0.195 [0.181, 0.204] | 0.222 [0.217, 0.245] | 0.359 [0.353, 0.368] | 0.126 |
| success per grasp | 0.844 | 0.862 | 0.870 | 0.880 |
| late/early placed (raw / live) | 0.39 [0.33, 0.40] / 0.40 | 0.48 [0.46, 0.51] / 0.49 | 0.78 / 0.78 | 0.33 / 0.34 |
| late/early attempts | 0.60 | 0.66 | 0.83 | 0.85 |
| wait from a placement to the next attempt, p50 | 0.40 s | 0.44 s | 0.36 s | 0.44 s |
| stalled step fraction; envs stalled at the end | 0.38; 55 % | 0.35; 50 % | 0.19; 30 % | 0.28; 35 % |
| stuck-object step fraction | 0.084 | 0.033 | 0.010 | 0.059 |
| steps engaged / carrying | 27 % / 2.8 % | 31 % / 3.6 % | 32 % / 7.9 % | 31 % / 2.2 % |
| jaw commanded while engaged | 24.9 mm | 24.1 mm | 17.3 mm | 24.8 mm |
| target points while engaged; zero-target frames | 9.4; 35 % | 11.1; 31 % | - | 10.7; 29 % |
| distilled student: grasps per attempt, l/e, jaw engaged | 0.106, 0.31, 31.1 mm | 0.152, 0.41, 27.5 mm | | |
| 180 s: placed / min (raw / live) | 1.67 [1.58, 1.80] / 1.86 | 2.61 [2.37, 2.69] / 2.72 | 10.4 / 10.6 | 1.64 / 1.82 |
| 180 s: per 36 s block (seed 101) | 6.4 → 1.5 → 0.7 → 0.3 → 0.04 | 8.7 → 2.1 → 0.6 → 0.3 → 0.2 | 20.5 → 12.2 → 8.1 → 6.5 → 4.9 | 6.4 → 1.0 → 0.4 → 0.4 → 0.1 |
| 180 s: last 36 s over first 36 s; late/early attempts | 0.01; 0.64 | 0.02; 0.56 | 0.24; 0.51 | 0.01; 0.84 |
| 180 s: stalled / stuck-object step fraction | 0.60 / 0.12 | 0.59 / 0.04 | 0.50 / 0.01 | 0.46 / 0.10 |

Frame-by-frame pages (untracked, 36-38 MB each, in the route directories): E0
`viewer_final_s101.html` -- a success cycle at frames 116-197, a 16 s stall at frames 1202-2000;
E0 `viewer_final_s202.html` -- six placements, first cycle at frames 12-97, no stall in 60 s; E2
`viewer_final_s101.html` -- eleven placements, first cycle at frames 2-85; E2 `viewer_final_s202.html`
-- six placements, first cycle at frames 20-99; E2 `viewer_final_s303.html` -- one placement, then a
9.6 s stall at frames 2521-2999 (the single-environment pages of seeds 101 and 202 happened not to
stall in 60 s, so a third seed was rendered).  In the E2 pages the flagged points are drawn in
magenta: on the frames where the hand is at the object they are 5-11 points on the object's top
and near face, and on roughly a third of those frames there are none.  Keys `n`/`p` step between
attempts, `s` jumps to the next stall.

**3.3 Reading (facts first).**

1. *The cadence bug was not the gen-1 number.*  E0 -- the gen-1 P1B recipe on the fixed hold --
   scores 7.0 placed/min against gen-1's 6.4, late/early 0.45 against 0.47, on the same ruler.
   The fix changed the shape (fewer attempts, 44 vs 55 per minute; more grasps per attempt, 0.195
   vs 0.126; more stall) and not the level.
2. *The oracle flag moves every row in the same direction and none of them far.*  E2 over E0:
   placed/min +26 % (accept) / +29 % (36 s ruler), grasps per attempt +14 %, late/early +38 %
   (24 s) / +23 % (36 s, raw and live), stalls -8 %; intervals separated on all of them; no row
   reaches the pre-registered 30 % on the three rows the rule names.  Over 180 s both collapse to
   the same floor (last 36 s at 1-2 % of the first) while the teacher holds a quarter.
3. *With a perfect label on the points, the student still does not close.*  E2 commands the jaw
   to 24 mm where the teacher commands 17 mm, converts 0.22 of its approaches into grasps against
   the teacher's 0.36, and carries 3.6 % of the time against 7.9 %.  Knowing *which* of the 512
   points are the object is therefore not what the grasp is missing; it is worth about a quarter
   of the gap in throughput and none of the gap in jaw behaviour.
4. *What the flag did buy is consistent with what it can see.*  It is present on 5-11 points and
   absent on a third of the frames at the grasp moment (occlusion by the hand, not sampling); the
   improvement is of the size a partial signal would give.

*Hypotheses the numbers are consistent with, not established:* (a) the object's geometry at
1.2 m through the measured D455 noise is too coarse in the sampled cloud for the closing decision
(a 25-45 mm object as ~10 displaced points), so the student regresses to a half-closed jaw
whatever the label says -- a representation-precision problem the mask line's 224×168 depth crop
does not have to the same degree; (b) the recipe (v11 labels, 36 s episodes, this budget) sets a
ceiling that the mask observation would hit too.  These two are exactly what the pre-registered
"E2 ≈ E0" branch separates.

## 4. Decision

**Outcome under the pre-registered rule: E2 ≈ E0** (every row better for E2, intervals separated,
no named row at or above 30 %).  Section 3.3 gives the stronger statement the rule does not: even
the oracle target label leaves the grasp deficit and the horizon collapse in place.

**The single next priority: the mask-line control under the identical recipe** -- the depth +
target-mask observation (`Mjlab-Pick-Place-PiperX-Distill-Robust` / `-Vision-Robust` at the
fixed cadence), distilled from the same v11 teacher with the same seed, budget, 36 s episodes,
evaluation seeds and the initiation ruler.  It is one run on one GPU (about 2.5 h by the timings
above) and it splits the two remaining hypotheses: a mask student at 15-25 placed/min with the
teacher's jaw command says the point-cloud *representation* is what is lost (then the work is
density/precision on the cloud side -- more points on the object, finer sampling near the hand,
a wrist view -- not a target channel); a mask student at 7-9 says the *recipe* is the ceiling
(then the work is the teacher's own horizon behaviour and the budget, before any observation
change).  Not first: budget × 2, shorter episodes, a new encoder, a new teacher -- each would
move the number without saying which of the two it moved.

Queued for the round after, because they change the environment and re-qualify the teacher:
the bin-settle artefact (2.3; ~1 % of the teacher's steps, 3-8 % of the students') and the
teacher's own fivefold decay over 180 s (2.5), which caps every student's long run.

Deployment: nothing changes.  E0 is NO-GO on the gate (7.0 < 13.5 placed/min, late/early 0.45 <
0.85); E2 is oracle-only and can never be a candidate.  `d455_v4_final` stays the rollback; the
gen-1 P1B bundle stays shadow-only.
