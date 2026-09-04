# Why every v7 student grasps once and stops

The question from the start of the session -- "夹取了一次之后就卡住" -- has an
answer, and it is not in the student.

## The chain, each link measured

1. **The state teacher progressively commands its gripper shut.**  Past the
   12 s horizon it was trained on, environments stop one at a time with the
   raw gripper action saturated at -13.8, a commanded jaw of 1.3 mm and an
   achieved 1.6 mm -- it opens whenever it is told to, and it is almost never
   told to.  `strong_teacher`: late/early 0.49, 69% of environments stopped by
   step 1200.
2. **Distillation copies the habit, because the labels are the habit.**  The
   loss is behaviour cloning against the teacher's actions at every state the
   student visits, so in the states where the teacher holds the jaw shut, the
   target action *is* to hold it shut.
3. **The student does exactly that.**  `student_5000`, 128 envs x 1200 steps:

   | step | 0 | 300 | 600 | 900 | 1100 |
   |---|---|---|---|---|---|
   | commanded gripper action | -3.67 | -5.81 | -6.51 | -6.37 | **-6.84** |
   | jaw, mm | 18.1 | 10.8 | 10.0 | 9.6 | **7.3** |
   | envs with jaw > 10 mm | 59% | 37% | 35% | 34% | **26%** |
   | frames holding an object | 4.0% | 1.5% | 2.0% | 0.9% | **0.5%** |

   The commanded action moves monotonically towards closed for the whole
   episode.  A gripper that is already shut cannot take another object.

## What this retires

Four mechanisms were measured and refuted before this one, and they are worth
naming so they are not re-proposed:

* **GRU hidden state carried across a placement.**  Resetting it moved
  late/early by +0.10 against a seed-to-seed spread of 0.23-0.33 -- inside the
  noise, and the decay survived with it enabled.  It could not have been the
  cause: the *state teacher* is an MLP with no memory at all and decays the
  same way.
* **Mask dropout / gap length / the held proxy.**  All are camera-side, and the
  teacher never sees a camera.
* **The single-object task failing to recover a stray.**  Measured at 0.000
  strays throughout.
* **Domain randomisation.**  The die-off is *worse* on the nominal task
  (88% stopped against 75%).

## Ranking, for the record

Distill-Robust, 128 envs x 1200 steps, no reset, seed 101:

| student | early/min | late/min | late/early | stopped |
|---|---|---|---|---|
| `student_5000` | 4.34 | 2.85 | **0.66** | 81% |
| `statemachine_5000` | 4.69 | 2.93 | 0.62 | 93% |
| `iid_control_8000` | 4.69 | 1.60 | 0.34 | 97% |
| `heldproxy_5000` | 4.38 | 0.78 | 0.18 | 100% |
| `v7b_strong_student_700` | 1.05 | 0.55 | 0.52 | 100% |

`iid_control_8000` is *not* the best, which corrects an earlier reading: its
"0.788 of teacher" was a ratio at 5000 iterations, not an endurance.

`best_v7_student.html` renders `student_5000` for 1200 frames.  Three separate
rollouts were tried; all three show one grasp inside the first 50 steps and
nothing for the remaining 23 seconds, and 5 of 8 environments never grasp at
all.  The target stays visible throughout (0.67-0.78) -- it is not a
perception failure.

## Why v8 should be different

The v8 teacher was continued at `episode_length_s = 36` until late/early
reached **0.81** with 46% of environments stopped.  That is an attack on link 1
of the chain, which is the only link a student can inherit through.
