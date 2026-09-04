# v9: start from anywhere, and pay for how you move

Two changes to the teacher, both from measurements made on 2026-09-03, both
bootstrapped from the v8 teacher (`model_15499`, late/early 0.81) rather than
cold -- a cold start under the full DR is what collapsed v2.

## 1. `RESET_FULL_RANGE=1`

The old reset is `default + uniform(-0.7, +0.7) rad`, clamped to the soft
limits.  Measured against where the policy actually goes (`student_5000`, 256
envs x 1200 steps):

| step | 0 | 300 | 600 | 900 | 1100 |
|---|---|---|---|---|---|
| any joint outside the reset box | 34% | 55% | 59% | 62% | **63%** |
| J6 (wrist roll) | 32% | 44% | 44% | 46% | **44%** |
| J2 | 1% | 6% | 9% | 10% | 14% |
| J4 | 1% | 5% | 9% | 9% | 8% |

J6 is the worst and the reason is arithmetic: default 0, soft limit **±108°**,
reset **±40.1°** -- 67.9° of its travel is never an opening, while
`wrist_side_on` (+0.8) actively rewards driving it there.  The reward and the
initialisation disagree by design.

A policy cannot learn to recover from a posture it never starts in, and on the
arm there is **no reset at all**: `--home-first` exists precisely because
"three runs in a row went progressively lower because each began where the
previous ended".  Removing the need for it means covering the space.

Sampling is uniform *between the soft limits*, not a clamped wide delta --
clamping piles probability onto the boundary and calls it coverage.  The
rejection sampler that was already there does the rest: measured over 5120
fresh resets, minimum EE height above the table is **50 mm** and **0.0%** are
below it.

New support (deg), against the old:

| | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| old lo/hi | 53/134 | 58/139 | -112/-32 | 25/80 | -40/40 | -40/40 |
| **new lo/hi** | **-135/135** | **9/171** | **-162/-9** | **-80/80** | **-80/80** | **-108/108** |
| new sd | 78.2 | 39.7 | 42.8 | 46.2 | 46.6 | 62.8 |

## 2. `SMOOTH_SCALE`, as a three-point dose

Every term that charges for *how* the arm moves, scaled together --
`action_rate`, `action_acc`, `joint_vel`, `joint_acc`, `joint_torques`,
`mech_power`.  Together, because scaling one just moves the roughness into the
others; and the two with a curriculum are scaled *through* the ramp, which is
the mistake that left every v7 teacher on a flat sight penalty.

Three runs, identical but for the dose, because one dose is a guess and v6
collapsed a policy by over-weighting a single term:

| GPU | run | SMOOTH_SCALE | `action_rate` |
|---|---|---|---|
| 4 | `v9_full_s1.0` | 1.0 (control: reset fix only) | -0.15 |
| 5 | `v9_full_s2.0` | 2.0 | -0.30 |
| 6 | `v9_full_s4.0` | 4.0 | -0.60 |

Reference `|da|`: `strong_teacher` 0.210, the -24 sight teachers 0.244,
`v5_baseline` 0.311.

## Everything else held

`episode_length_s = 36` (the endurance fix), `SIGHT_RAMP=0` (required -- the
step counter restarts on resume, so the default ramp would restart the sight
penalty at -0.6 and change the reward mid-lineage), sight weights at their
defaults, 4096 envs, 4000 iterations.

## What to check, and with what

**The evaluation must set `RESET_FULL_RANGE=1` too.**  Scoring a
start-from-anywhere teacher on the narrow reset measures the thing that was
already working.

* grasps from a full-range start at all -- placed/min under the wide reset
* `scripts/eval_endurance.py` -- late/early, which must not regress below 0.81
* `scripts/eval_occlusion.py` -- `engaged` blocked, against v8's 8.6%
* `action_rate` in the occlusion JSON -- what the smoothness dose bought

Early, ~30 iterations in, `objects_placed` per 36 s episode reads 4.72 / 3.40 /
1.97 for the three doses against v8's 6-7 on the narrow reset.  Both changes
cost throughput, monotonically, and that is the trade being measured rather
than a result.

## Result

All three finished 4000 iterations from `v8_remote/model_15499`.  256 envs,
seed 101; endurance over 1200 steps with no reset, occlusion over 300
conditioned on the hand being engaged.

| teacher | reset | placed/min | late/early | engaged blocked | mean abs step |
|---|---|---|---|---|---|
| v8 `model_15499` | narrow | 22.4 | 0.81 | 8.6% | 0.309 |
| `v9_full_s1.0` | narrow | 24.8 | 0.85 | 7.7% | 0.316 |
| `v9_full_s1.0` | **full** | 21.8 | 0.91 | 7.7% | 0.303 |
| `v9_full_s2.0` | narrow | 26.3 | 0.82 | 7.6% | **0.202** |
| `v9_full_s2.0` | **full** | 22.2 | 0.90 | 7.5% | **0.197** |
| `v9_wrist30_s1.5` | narrow | 26.3 | 0.88 | 8.5% | 0.249 |
| `v9_wrist30_s1.5` | **full** | 22.3 | **0.98** | 8.6% | 0.243 |

* **Starting from anywhere costs about 13% of the throughput and *buys*
  endurance.**  26.3 -> 22.3 placed/min going from the narrow reset to the full
  soft-limit box, while late/early goes 0.82-0.88 -> 0.90-0.98.  A policy that
  has to recover from arbitrary postures apparently stops walking itself into
  the one it cannot recover from.
* **All three beat v8 on the narrow reset** (24.8-26.3 against 22.4) and match
  it on the full one (21.8-22.3), so the capability was added rather than
  traded for.
* **The smoothness cost is real and cheap.**  `SMOOTH_SCALE=2.0` gives 0.197
  against v8's 0.309, a 36% smaller step-to-step action change, at the same
  throughput and the same occlusion.
* **Occlusion did not move** (7.5-8.6% against v8's 8.6%), so neither change
  spent the thing the sight rewards were bought for.

### The wrist bonus was the attractor

`v9_wrist30_s1.5` is `v9_full_s1.5` with one variable changed -- `WRIST_W=0.30`
instead of the 0.8 that `SIGHT_RAMP=0` pins it at.  `s1.5` collapsed to zero
placements at iteration ~17600 with reward *rising* to 16-17 and its action std
down to 0.15.  `wrist30` ran through the same range and finished at 9.78
placements per episode -- the highest of the three -- on a mean reward of 5.12,
half of s1.0's 9.67.  Lower reward, more work: what removing a farmable bonus
should look like.

### Cold start does not work here

`v9_cold_ramped`, 12000 iterations, full DR, full-range reset, ramps as
designed: **0.0000 placements**, mean reward 11.48 from shaping terms alone.
A controlled modern reproduction of the v2 collapse, and the answer to whether
bootstrapping costs exploration: the bootstrapped runs held an action std of
0.42-0.62 while the cold one fell to 0.13 and committed to doing nothing.

### The confound in picking a winner

`wrist30` differs from `s2.0` in **two** ways -- `WRIST_W` 0.30 against 0.8 and
`SMOOTH_SCALE` 1.5 against 2.0 -- so "which is better" is not answered here.
The combination neither run covers is `WRIST_W=0.30` with `SMOOTH_SCALE=2.0`.
