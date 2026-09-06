# yf/pc second generation -- why the first-generation student stops starting, and E0 / E2

Status: **audit complete, E0 / E2 scheduled (blocked on the training host's VPN)**.  Every number carries the
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

## 3. E0 (P1BZ) and E2 (P1BT)

**Status: not yet run.**  The training host `shen-teacher` is reachable only through the MotionPro
VPN, which was logged out for the whole of this session (`VPN Status: unknown`; the ssh proxy times
out).  The qualified v11 teacher exists only on the host, so neither route can be trained here.
Everything else is in place and smoke-tested locally (both routes: cloud 512 × 5, the zero column
exactly 0, the oracle column flagging ~48 of 512 points on the smoke's start frames, two
distillation iterations with a finite loss; `results/pc/gen2/audit/smoke_P1B{Z,T}.json`).

One command launches the pair and the teacher reference, once the host answers:

```bash
bash scripts/pc/launch_gen2.sh          # rsync named paths; refuse unless >= 2 of GPUs 4-7 are idle; launch
```

It writes `results/pc/routes/pc_gen2_P1BZ_<stamp>/`, `pc_gen2_P1BT_<stamp>/` and
`results/pc/gen2/teacher_v11_<stamp>/` on the host (stages: smoke → distill 1500 → distill eval →
finetune 800 → 3-seed accept/endurance/held-out/actions/occlusion → initiation × 3 seeds + 180 s × 3
seeds → viewer pages → export; `timing.jsonl` per stage).  A poller started in this session retries
the host every minute for 12 h and runs the launcher when it answers
(`results/pc/gen2/launch_attempts.log`).  Bring the numbers back with
`scripts/pull_results.sh pc/routes` and `scripts/pull_results.sh pc/gen2`, then
`python scripts/pc/report_routes.py results/pc/routes/pc_gen2_* --teacher-placed 19.3`.
Expected wall time from the gen-1 P1B timings on the same host: ~65 min distill, ~50 min
fine-tune, ~50 min of evaluation -- about 3 h for the pair in parallel.

Controls common to both (from `manifest.json` of each route directory): teacher above; seed 42;
512 environments; distill 1500 iterations (32 steps/env/iter = 24.6 M env steps, 3000 optimizer
updates at gradient length 16) + PPO 800 iterations (13.1 M env steps, 16 000 updates, 5 epochs x 4
minibatches); episode 36 s; same cloud (512 x 5), encoder, GRU, reward, curriculum, DR, sensor,
latency; evaluation seeds 101 / 202 / 303; no dropout at either stage (see the audit).  E0's fifth
column is constant zero; E2's is the oracle target flag.  E2 is oracle-only: `manifest.oracle_only`,
`export/ORACLE_ONLY`, `bundle.py` and every deployment entry point refuse it.

## 4. Decision rule (pre-registered) and the next priority

Read E0 and E2 on the same rows as section 2.5, three evaluation seeds each, the training seed
matched (42); the 180 s numbers against the v11 teacher's own 180 s run.  "Clearly better" means
the three-seed intervals do not overlap and the difference is at least 30 % of E0's value.

| outcome | next priority |
|---|---|
| E2 clearly better than E0 in grasps per attempt and late/early (raw and live) | a deployable target selection + persistence for the cloud (a per-point target flag the robot can produce: tracker/detector on the existing points, teacher/student target kept consistent); two more training seeds first |
| E2 ≈ E0 (within the intervals) | the mask-line control: the same teacher, budget, 36 s and cadence fix with the depth+mask observation, to split "representation" from "recipe" |
| E0 already close to the gate (≥ 13.5 placed/min, late/early ≥ 0.85) | quantify the recovery against gen-1 P1B; decide whether a target mechanism is needed at all |
| neither, and E0's per-phase profile still shows the grasp deficit of section 2 with attempts intact | only then budget × 2 or a 12 s-episode control, one at a time |

Independent of the outcome, two environment-side findings from the audit are queued for the round
after (they change the task and would re-qualify the teacher, so not this round): the bin-settle
artefact (2.3) and the fact that the teacher's own placement rate falls fivefold over 180 s (2.5),
which bounds what any student can be asked to hold.

**The single next priority, on today's evidence: run the E0 / E2 pair** -- it is built, tested,
smoke-tested and scheduled, and it is the one experiment that turns the audit's remaining
hypothesis (the grasp is missed because the ~11 target points are not recognised as the object)
into a measurement.  Blocked only by the VPN login on this machine.
