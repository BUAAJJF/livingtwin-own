# Measured results

Every number below is a deterministic rollout of 512 environments for 2400
control steps (409.6 arm-minutes, ~22k object instances) in the *honest*
environment, where each object is drawn fresh rather than once per episode.
Smaller samples read 1–2% optimistic across the board, so figures from
256×1800 rollouts are not mixed in.

## Single object (S2)

| | teacher (state) | distilled | fine-tuned |
|---|---|---|---|
| throughput (objects/min) | 58.4 | 53.3 | **55.8** |
| success | 99.8% | 99.6% | 99.9% |
| post-grasp drop | 0.4% | 0.9% | 0.5% |
| time to place, p95 | 1.36 s | 1.76 s | 1.46 s |
| time with a stuck object | 2.2% | 3.9% | 1.3% |
| safety-shell trips / arm-hour | 4.2 | 16.4 | **3.7** |

Best checkpoint `f3/model_1500`; past ~1500 fine-tuning iterations the tail
degrades (stuck time 1.3% → 2.6% → 4.2% at 1500 / 2300 / 2999).

## Three objects (S3)

| | teacher (state) | distilled | fine-tuned |
|---|---|---|---|
| throughput (objects/min) | 42.4 | 38.7 | **39.8** |
| success | 99.6% | 98.9% | 99.0% |
| post-grasp drop | 1.0% | 1.2% | 1.4% |
| time to place, p95 | 1.98 s | 2.38 s | 2.16 s |
| tables cleared / arm-hour | 800.5 | 733.3 | 751.5 |
| objects batted astray / 100 placed | 0.1 | 0.4 | 0.3 |
| safety-shell trips / arm-hour | 15.7 | 44.7 | **9.5** |

## What replicates across both

- **The cost of seeing through a camera is flat at ~9%.** Distillation lands
  at 91% of its teacher with one object and 91% with three.  Adding objects
  does not make perception harder.
- **Imitation inherits speed and not caution.** The distilled student violates
  the joint-speed limit 3.9× (single) and 2.8× (clutter) as often as the
  teacher it copies, while matching 91% of its throughput.  Action-space MSE
  contains no term for what the trajectory does.
- **Fine-tuning is a tail fix, not a speed-up.** It returns 2–3 points of
  throughput and drives constraint violations *below* the teacher's own rate
  (3.7 against 4.2; 9.5 against 15.7).
- Clutter itself costs 27% of throughput (58.4 → 42.4 for the teacher).

## Isolated findings

**A recurrent policy exploits episode-constant facts.** With the object's
shape drawn once per episode rather than per object, the same vision policy
reads 58.2 objects/min instead of 54.0.  The state teacher, fed the shape
directly and with no memory to carry, loses 2.0% over the same change -- so
the remaining 5.2% is memory, and it is worth almost exactly what an ablation
says the shape channel itself is worth (7.1%).

**Fine-tuning in the wrong environment buys nothing.** 3000 PPO iterations
under episode-constant shapes score 53.0 in the honest environment -- no
better than the 53.3 of the distilled policy they started from.  1200
iterations in the honest environment score 56.3.

**Where the safety shell fires.** 91% of trips happen in free-space reach, none
during the close or the carry, and 91% of them jump from under 0.9 of the limit
to over 1.0 within one 20 ms control step.  It is an acceleration problem, not
a contact-impulse problem.

**Retraining the teacher in the honest environment does not help.** 54.3
against 58.4: the teacher is fed the shape and has no memory to correct.
