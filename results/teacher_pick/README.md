# Which teacher, measured rather than remembered

Six v7-era state teachers, the same three seeds, 256 environments x 600 steps
each.  `checkpoints/v4/model_2999.pt` is not in the table and cannot be: its
actor takes a 36-dimensional observation and the current task builds 57, so it
is not runnable here and its historical 24.85/min is a number from a different
domain.

```bash
micromamba run -n mjlab python scripts/eval_occlusion.py \
    --checkpoint checkpoints/v7_teachers/strong_teacher.pt \
    --num-envs 256 --steps 600 --seed 101 --out results/teacher_pick/x.json
```

## Result

| teacher | placed/min | approach blocked | held blocked | approach visible | held visible | dq p99 rad/s | mean abs step in action |
|---|---|---|---|---|---|---|---|
| `v5_baseline` | **32.6** +-0.4 | 18.9% | **3.9%** | 0.565 | 0.595 | 2.75 | 0.311 |
| `sight0_2085` | 28.8 +-0.4 | 18.8% | 6.6% | 0.572 | 0.577 | 2.71 | 0.308 |
| `visible_teacher` | 25.9 +-0.4 | 15.9% | 4.6% | 0.588 | 0.577 | 2.76 | 0.283 |
| `v7c_hand24_flat_11995` | 21.1 +-0.7 | 8.5% | 23.0% | 0.646 | 0.502 | 2.78 | 0.246 |
| `v7c_hand24_11995` | 21.0 +-0.7 | 8.9% | 29.2% | 0.644 | 0.470 | 2.83 | 0.244 |
| `strong_teacher` | 15.0 +-0.5 | **2.4%** | 8.9% | **0.687** | **0.614** | **2.69** | **0.210** |

`blocked` is the fraction of frames where fewer than 35% of the fifteen sample
points on the object's box show the object in the rendered segmentation.  The
seed spread is 0.1-1.0 on every column, so 256 environments x 3 seeds is a
tight estimator here -- unlike the single-environment perception check, where
the same quantity swings by 80 points between seeds.

Three things the table settles:

* **The sight reward buys visibility with throughput, at roughly one for one.**
  The zero-sight control is 18.8% blocked at 28.8/min; `strong` is 2.4% at
  15.0/min.  There is no configuration in hand that gets both.
* **The curriculum ramp still does nothing.**  `hand24` ramped and flat are
  21.0 and 21.1 placed/min, 8.9% and 8.5% blocked.  This is the third
  measurement to say so, now at three seeds.
* **-24 is worse than `strong` on both visibility axes**, and much worse during
  the carry (23-29% blocked against 8.9%).  Pushing the weight further does not
  continue the trend it started.
* **Nothing is speed-limited.**  Every teacher's 99th-percentile joint speed is
  2.69-2.83 rad/s against the rig's 3.93 rating, and none exceeds it in any
  frame of any seed.  The three real runs killed by the speed guard at
  4.6-7.8 rad/s were not caused by any of these policies.

## What that means once SAM2.1 is in the perception path

Occlusion mattered because the depth segmenter went blind near the gripper.
SAM2.1 fixes the *identity* problem but cannot invent pixels, so the question
is whether a more occluding teacher still costs anything.  It does.  Same
perception stack (`scripts/sim_perception_check.py --sam`), seven seeds each,
only the driving policy changed:

| driving policy | SAM detected, approach | SAM IoU, approach | SAM IoU, holding |
|---|---|---|---|
| `strong_teacher` | **97.8%** | **0.856** | 0.804 |
| `v5_baseline` | 80.1% | 0.678 | **0.881** |

An 18-point gap in how often the student would see its own target, produced
entirely by how the teacher moves.  `v5` is better during the carry and worse
everywhere else.

## The page

`teachers.html` -- draggable 3D, 300 frames, three policies from the identical
opening scene (object at `(-0.187, 0.301, 0.033)` in all three panels, checked).

**Do not rank from the page's own visible numbers.**  It is one environment for
300 steps, and on this episode it reports `hand24_flat` at 1.0% blocked and
`strong` at 8.7% -- the reverse of the 256-environment table above.  The page
is for watching *what the arm does*; the table is for deciding.

---

# Correction: the 600-step table above is measured across a collapse

`strong_teacher` does not place 15.0 objects a minute.  It places 18 for the
first four seconds and 6 by the twentieth, and the 600-step window the table
was measured over sits on the way down.  One environment at a time stops
working and never restarts.

`scripts/../scratchpad/decay2.py`-style measurement, 256 envs, 1200 steps, one
long episode (play mode is `episode_length_s = 40`, so nothing resets):

| teacher | first 600 steps | last 600 steps | late/early | jaw at end |
|---|---|---|---|---|
| `v5_baseline` | 33.9/min | 36.5/min | **1.08** | 29 mm |
| `sight0_2085` | 30.2 | 31.2 | 1.03 | 33 mm |
| `visible_teacher` | 27.8 | 25.5 | 0.92 | 36 mm |
| `v7c_hand24_flat` | 21.3 | 15.2 | 0.71 | 44 mm |
| `v7c_hand24` | 21.5 | 15.4 | 0.72 | 43 mm |
| `strong_teacher` | 15.9 | **7.8** | **0.49** | **10 mm** |

The ordering is monotone in the sight weight, and the two ends fail in
opposite directions: `strong` latches its jaw **shut** (10 mm, and 0.1 mm in
the environments that have stopped), `hand24` holds it **open** and hovers
158 mm away without engaging.

## It is a policy failure, not a jam and not an exploit

* **Environments die, they do not slow down.**  Of the environments placing in
  the first 300 steps, 69% place nothing in the last 300; the ones still
  working place *more* than they did at the start (2.95 against 2.44).  Median
  jaw in the stopped ones is 0.1 mm, in the working ones 16.9 mm.
* **The jaw is commanded shut, not stuck.**  In the stopped environments the
  raw gripper action averages -13.8 (saturated closed) and the commanded target
  is 1.3 mm against an achieved 1.6 mm.  Commanded minus achieved is -0.2 mm on
  average.  It opens whenever it is told to; it is almost never told to.
* **It is not the domain randomisation.**  The same die-off is *worse* on the
  nominal task (88% stopped against 75%).
* **It is not reward hacking.**  Total per-step reward falls from 2.52 to 1.85
  between the first and last 300 steps.  `place` alone loses 1.05; the penalty
  terms it saves are worth +0.02 (`sight_hand`), +0.02 (`joint_vel`) and +0.09
  (`terminated`).  The collapsed behaviour is worse by the reward's own
  accounting, so PPO has not found an equilibrium -- it has left a hole.
* **The training horizon is 12 s.**  Play is 40 s.  The collapse is already 35%
  complete inside the horizon the teacher was optimised on, so a longer episode
  alone may not be sufficient, but it is the first thing to try.

Resetting every 300 steps removes it entirely: 19-21 placements/min, flat,
late/early **1.07**.

## The visibility advantage survives this, and one rival's does not

Occlusion re-measured over 300 steps only -- before the collapse -- and
conditioned on the hand being **engaged** (jaws open, within 150 mm of the
object), because a policy that stops reaching scores well on occlusion for the
wrong reason:

| teacher | placed/min | approach blocked | **engaged** blocked | engaged frames | mean reach |
|---|---|---|---|---|---|
| `v5_baseline` | 30.3 | 18.3% | 17.6% | 70% | 0.103 m |
| `sight0_2085` | 26.6 | 17.3% | 17.5% | 68% | 0.106 |
| `visible_teacher` | 25.2 | 15.9% | 16.0% | 68% | 0.108 |
| `v7c_hand24_flat` | 22.6 | 9.1% | **17.6%** | **47%** | **0.158** |
| `strong_teacher` | 17.8 | 2.7% | **2.5%** | **77%** | 0.107 |

* `strong` is the real thing: 2.5% blocked while its hand is *in* the working
  volume, and it is engaged more of the time than anything else on the list.
* `hand24`'s 9.1% was an artifact.  Conditioned on engagement it is 17.6% --
  identical to the teacher with no sight reward at all.  It did not learn to
  keep its hand clear of the sight line, it learned to keep its hand 158 mm
  away and be engaged less than half the time.

## And it survives at the deployed perception stack

`scripts/sim_perception_check.py --sam`, 300 steps so the comparison is before
the collapse, seven seeds each, only the driving policy changed:

| driving policy | SAM detected (approach) | SAM IoU (approach) | SAM IoU (holding) | depth detected |
|---|---|---|---|---|
| `strong_teacher` | **97.7%** | **0.836** | 0.830 | 47.4% |
| `v5_baseline` | 83.3% | 0.696 | **0.886** | 30.4% |

By gripper-to-object distance, the 30-80 mm band is where they part:
`strong` holds SAM at 100%, `v5` gives 79% and 74%.  `v5` is better during the
carry and worse everywhere the grasp is decided.

## Where that leaves the choice

`strong_teacher` is the only policy on the list that genuinely keeps its own
gripper out of the sight line while working, and the advantage is visible all
the way through to what the deployed tracker can see.  It also stops working
after twenty seconds, and **distillation cannot fix that**: the labels come
from the teacher, so in the states where it latches its jaw shut the target
action *is* to keep it shut.  The students distilled from it decay at
late/early 0.30-0.68 against its own 0.49.

So the collapse is a blocker on the teacher, not a defect of the student, and
it has to be closed before any of the distillation questions are worth asking.
