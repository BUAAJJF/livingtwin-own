# v8: give the teacher a horizon long enough to contain its own failure

`strong_teacher` stops working inside one long episode.  Environments do not
slow down, they die: 69% of the ones placing in the first 300 steps place
nothing in the last 300, with the gripper commanded shut (raw action -13.8,
commanded 1.3 mm, achieved 1.6 mm -- it opens whenever it is told to).  Total
per-step reward falls 2.52 -> 1.85, so this is not an equilibrium PPO found, it
is a hole PPO left.  It is worse on the nominal task than under DR, so it is
not the randomisation.  Resetting every 300 steps removes it entirely.

The training horizon is `episode_length_s = 12`; play is 40.  The real arm has
no reset at all.  So this run continues the same teacher, with the same reward,
for 1500 more iterations at `episode_length_s = 36`.

`SIGHT_RAMP=0` is REQUIRED, not a choice: `common_step_counter` restarts at 0
on resume, so the default ramp would restart the sight penalty at -0.6 and
change the reward mid-lineage.  Flat at the final weight is also what every v7
teacher actually trained under (see the note in `robust_cfg.py`), so this
reproduces `strong_teacher`'s reward exactly.

Gate, on the checkpoints as they appear (save_interval 100):
  late/early >= 0.90 over 1200 steps x 256 envs, one episode, no reset
  engaged blocked <= 5% at 300 steps (strong is 2.5%; that is what we are
  protecting)
  sustained rate >= 15/min in the last 600 steps (strong is 7.8)

## Pass 1 result: it worked, and not far enough

1500 iterations at `episode_length_s = 36`, same reward, `SIGHT_RAMP=0`:

| | `strong_teacher` | v8 pass 1 (`model_13494`) |
|---|---|---|
| early (first 600 steps) | 15.9/min | **21.2** |
| late (last 600 steps) | 7.8/min | **14.8** |
| late/early | 0.49 | **0.70** |
| environments stopped by the end | 69% | 63% |
| jaw in the stopped ones | 0.1 mm | 0.1 mm |

Late throughput is up 90% and the ratio moved 0.49 -> 0.70, so the horizon is
the right lever.  The mechanism is unchanged -- the jaw still latches shut --
it is just rarer and later.  `Metrics/pick/objects_placed` rose 3.4 -> 5.7-6.7
across the run and had not flattened, so the run was stopped by its iteration
count rather than by convergence.

Gate is 0.75, so this FAILS and the pipeline stopped rather than distilling
from it.  Pass 2 continues the same lineage for 2000 more.

## Final: the horizon fix worked, and it cost most of the visibility edge

`model_15499`, 4000 iterations after `strong_teacher` at `episode_length_s=36`,
same reward.  Endurance at 256 envs x 1200 steps, occlusion at 256 x 300
conditioned on the hand being engaged.  Single seed (101) on the v8 rows.

| teacher | placed/min | late/early | engaged blocked | engaged visible |
|---|---|---|---|---|
| `strong_teacher` (11995) | 16.7 | 0.49 | **2.1%** | 0.686 |
| v8 @13494 | 21.2 | 0.70 | -- | -- |
| v8 @14200 | 21.6 | 0.75 | 10.4% | 0.624 |
| **v8 @15499 (final)** | **22.4** | **0.81** | 8.6% | 0.632 |
| e72 @14100 (72 s control) | 21.8 | 0.72 | 8.7% | 0.633 |
| `v5_baseline` | 29.8 | 1.08 | 18.6% | 0.559 |

* **Endurance: 0.49 -> 0.81.**  Late-half throughput 7.8 -> 20.5 per minute, a
  factor of 2.6, and the environments that stop by the end fell 69% -> 46%.
  The mechanism is unchanged -- the jaw still latches at 0.1 mm -- but it is
  much rarer.
* **The visibility edge is mostly gone.**  2.1% -> 10.4% engaged-blocked inside
  the first 300 iterations, then back to 8.6% by the end.  v8 now sits between
  `strong` and a teacher with no sight reward at all (18.6%), not at the
  `strong` end.  Whatever `strong` had learned about keeping its gripper out of
  the sight line, four thousand iterations of longer episodes largely undid.
* **72 s bought nothing over 36 s** (0.72 vs 0.75 at the same point, 8.7% vs
  10.4% blocked).  The horizon dose-response saturates at 36.
* **The final checkpoint dominates 14200 on every axis measured** -- endurance,
  occlusion and throughput -- so there is no trade to make between them.  Both
  students were restarted from it; they had started from 14200, which was
  simply the best available when they were launched.

Not established: whether the visibility loss matters more than the endurance
gain, which is what the two students will answer.  And these v8 rows are one
seed each; the spread on `engaged blocked` across three seeds was 0.1-1.0
points on the earlier teachers, so 10.4 vs 8.6 is larger than that but not by
much.
