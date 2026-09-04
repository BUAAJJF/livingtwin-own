# What the observation model is worth: the v8 A/B

> **Validity (2026-09-04).** The `v8_students_*` and `v8_occ_*` JSONs this file quotes were measured without the GRU reset on episode boundaries and are marked `invalid`; the `*_fixed_*` files replace them.  The accept_compare robust rows ran under a downgraded sensor.  The `v8s_sam2` numbers were also taken outside the visible-fraction domain it was distilled in.  See `results/VALIDITY.md`; nothing below is re-measured yet.

Both students distilled from the same teacher (`v8_remote/model_15499`,
late/early 0.81), 2500 iterations, `episode_length_s = 36`, 512 envs, seed 42.
**Only the observation model differs.**  Evaluated on
`Mjlab-Pick-Place-PiperX-Distill-Robust`, 128 envs, seed 101.

| | `v8s_depth` | `v8s_sam2` | their teacher |
|---|---|---|---|
| visible domain | measured spread (0.10-0.85 / 0.15-0.90) | (0.62, 0.98) | -- |
| gaps | x1.0 | x0.3 | -- |
| latency prior | 43 ms | 70 ms | -- |
| **late/early, no reset** | 0.51 | 0.45 | **0.81** |
| **late/early, reset every 300** | 0.79 | **1.12** | -- |
| **engaged blocked** | 10.1% | **5.9%** | 8.6% |
| engaged visible | 0.644 | **0.675** | 0.632 |
| placed/min | **6.09** | 5.16 | 22.4 |
| mean abs step in action | 0.221 | 0.220 | 0.309 |

## What it says

* **SAM's domain buys occlusion behaviour.**  5.9% against 10.1% engaged-blocked
  -- a student trained expecting to see its target keeps its own gripper
  clearer than one trained to work half-blind.  It is also better than the
  teacher it imitates (8.6%), which is not something behaviour cloning is
  supposed to do and is worth a second look.
* **It costs 15% of the throughput** (5.16 against 6.09 placed/min).
* **Neither inherits the teacher's endurance.**  0.81 in, 0.45-0.51 out.  The
  horizon fix did not transfer.
* **Both are healthy inside a 300-step window** (0.79 and 1.12) and collapse
  beyond it -- and this time that is *inside* their own training horizon, which
  is 36 s = 1800 steps.  The teacher, evaluated over the same 1200 steps,
  reaches 0.81.  So the students lose something the teacher has, within a
  window both were trained on.
* **The students are smoother than their teacher** (0.220 against 0.309).
  Behaviour cloning regresses towards the mean action, which is smoothing for
  free -- and a reason to check whether the v9 smoothness cost is buying
  something the student would have got anyway.

## What it does not say

Which of the three differences (visibility, gap length, latency prior) is
responsible.  They were changed together on purpose -- the point was to compare
two *stacks*, not to decompose one.  A decomposition needs three more runs and
is only worth it if the 4-point occlusion gap matters.

Single seed on every row.

## Two follow-ups: one prediction refuted, one mechanism confirmed

### Out-of-distribution posture: largely fixed, and it was not the answer

Fraction of (step, env) outside the box the arm is ever reset into:

| policy | 0-100 | 500-600 | 1100+ | J6 alone, late |
|---|---|---|---|---|
| v7 `student_5000` | 33% | 61% | **67%** | **54%** |
| `v8s_depth` | 13% | 25% | **28%** | 8% |
| `v8s_sam2` | 11% | 15% | **19%** | 15% |

The v8 teacher's horizon fix cut this from 67% to 19-28% **without touching the
reset**, and J6 from 54% to 8-15%.  I predicted the v8 students would still be
60%+ and that posture coverage was the remaining cause of their decay.  Wrong
on both counts: they are much better placed and they still decay
(late/early 0.45-0.51).

That does not retire the v9 reset change -- "start from anywhere without
`--home-first`" is a capability that was asked for, not a fix for this -- but
it does remove the evidence I was offering for it as an explanation.

### The gripper drift: still there, weaker

| | commanded gripper action | jaw | frames holding |
|---|---|---|---|
| v7 `student_5000` | -3.67 -> -6.84 | 18.1 -> 7.3 mm | 4.0% -> 0.5% |
| `v8s_depth` | **-0.56** -> -7.94 | **27.7** -> 18.0 mm | 3.8% -> 1.5% |
| `v8s_sam2` | **-0.02** -> -6.41 | **29.4** -> 22.2 mm | 3.5% -> 1.6% |

The v8 students *start* with a neutral gripper command where v7 started already
closing, and they end with the jaw 2.5-3x more open.  But the drift itself is
unchanged in shape: monotone towards closed for the whole episode, in the
command, not the plant.

It survives because the label still contains it -- the v8 teacher at
late/early 0.81 still has its stopped environments sitting at a 0.1 mm jaw.

**Neither v9 change targets this.**  A full-range reset and a smoothness cost
do not say anything about holding the jaw shut.  If it is to be fixed, the
lever is on the teacher: a penalty or a termination for "jaws closed, nothing
held", or a small reward for an open jaw while searching.  That is a reward
change, and reward changes have collapsed a policy twice in this campaign
(v6's dropout, tonight's SMOOTH_SCALE=4), so it wants its own controlled run
rather than being folded into v9.

## Correction: every student number above was measured with a bug

`scripts/accept_s1.py` calls `policy.reset(dones)` after every step, because a
recurrent policy left alone carries its hidden state through an episode
termination and starts the next one remembering the last -- its own comment
says so.  `eval_endurance.py`, `eval_occlusion.py`, `sight_viewer.py` and
`sim_perception_check.py` all did `obs, _, _, _ = wrapped.step(action)`,
discarding `dones`, and never reset anything.  Fixed; the four now share
`eval_occlusion.reset_recurrent`.

Teachers are MLPs and were never affected -- the v9 dose table, the
0.49 -> 0.81 endurance line and the occlusion figures all stand.  **Every
student number is affected**, and re-measured:

| | before | after |
|---|---|---|
| `v8s_depth` late/early | 0.51 | **0.64** |
| `v8s_sam2` late/early | 0.45 | **0.57** |
| `v8s_depth` placed/min, engaged blocked | 6.09, 10.1% | 5.70, 9.8% |
| `v8s_sam2` placed/min, engaged blocked | 5.16, 5.9% | 5.55, **7.0%** |

* **A quarter of the students' "decay" was mine.**  0.45-0.51 -> 0.57-0.64,
  against their teacher's 0.81.  The gap is real and smaller than reported.
* **SAM's occlusion advantage shrank from 4.2 points to 2.8** (5.9 vs 10.1
  became 7.0 vs 9.8).  Single seed on both, and 2.8 points is close enough to
  the run-to-run spread on this metric that it should not be quoted as a result
  without more seeds.

### What did not move

| | placed | attempts | placed/attempt |
|---|---|---|---|
| v8 teacher | 10.16 | **11.55** | 0.880 |
| `v8s_depth` | 1.62 | **2.02** | 0.806 |
| `v8s_sam2` | 1.20 | **1.49** | 0.801 |

Still 5.7-7.8x fewer attempts and 9% worse per attempt.  The headline survives
the fix intact: **the students are nearly as good as their teacher at
converting an attempt, and almost never start one.**
