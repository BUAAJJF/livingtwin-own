# v4 against v8, on one ruler

> **Validity (2026-09-04).** `v4_robust.json`, `v8s_depth.json` and `v8s_sam2.json` are marked `superseded` in place (sensor downgrade, see the correction below and `results/VALIDITY.md`).

Every number here is `scripts/accept_s1.py`, 256 envs x 2400 steps, seed
20260904, `--sensor measured`.

> **Correction (2026-09-04).** An earlier version of this paragraph said the
> same v4 checkpoint read 4.4/min on `scripts/eval_endurance.py` and 29.95/min
> here "because accept_s1 runs the deployment's blind-step handling
> (`--hold-shim`)".  There is no such flag anywhere in the repository.  The
> two confirmed causes of the gap are: (1) `eval_endurance.py` and
> `eval_occlusion.py` did not reset the GRU hidden state on episode
> boundaries until 2026-09-04 01:26, while `accept_s1.py` always did; and
> (2) `eval_occlusion.load_policy` called `runner.load(load_cfg={"actor":
> True})`, which on a `-Distill*` task loads *nothing* (rsl_rl's
> `Distillation.load` does not know the key), so `results/decay/v4_final*.json`
> measured a randomly initialised student.  Both are fixed
> (`piper_push.evalcfg`).  A third difference -- `--sensor measured` here
> replaced the task's noise model with the *nominal* one at strength 1.0, so
> every `robust` row below ran robust dynamics under a downgraded sensor --
> is also fixed, and means the `robust` rows are not the robust domain; they
> have to be re-measured before being quoted.  The v4 row under the fixed
> `eval_endurance.py` has not been re-run yet.

| policy | domain | verdict | success | drops | throughput |
|---|---|---|---|---|---|
| v4 | nominal | **PASS** | **0.960** | 0.025 | **29.95/min** |
| v4 | robust | FAIL | 0.691 | 0.057 | 11.49/min |
| `v8s_depth` | nominal | FAIL | 0.752 | 0.074 | 7.95/min |
| `v8s_depth` | robust | FAIL | 0.504 | 0.120 | 3.87/min |
| `v8s_sam2` | robust | FAIL | 0.450 | 0.118 | 3.31/min |

## The 2x2 separates two costs, and both are real

* **The heavy DR costs x0.38-0.49** in throughput -- v4 29.95 -> 11.49, v8
  7.95 -> 3.87.  Consistent across both lineages, so it is a property of the
  domain, not of either policy.
* **The lineage costs x0.27-0.34 at matched domain** -- nominal 29.95 -> 7.95,
  robust 11.49 -> 3.87.  Larger than the domain cost.

Multiplied they give the raw 29.95 -> 3.87 (x0.13), which is what the
uncontrolled comparison shows and which attributes everything to whichever
factor the reader already suspected.

## How much of the lineage cost was bought on purpose

`command_derate = 0.50` entered with v5 and is in the robust profile; v4's
nominal task has none.  Half the commanded joint rate is roughly half the
throughput, and it was asked for -- smoother motion, smaller sim-to-real gap.
So a large part of the throughput column is a price, not a fault.

**Success and drop rate are not.**  They are rates per attempt, independent of
how fast the arm moves: a slower policy that still works has the same success
rate and simply gets fewer chances per minute.  At matched domain they went

| | success | drops |
|---|---|---|
| nominal, v4 -> v8 | 0.960 -> **0.752** | 0.025 -> **0.074** |
| robust, v4 -> v8 | 0.691 -> **0.504** | 0.057 -> **0.120** |

A fifth to a quarter of the successes gone and the drop rate two to three
times higher, with the slowdown already accounted for.  That is capability,
not pace.

## What changed between them

v4 predates the v5/v7 reward redesign.  v7 added `sight_arm`, `sight_hand` and
`wrist_side_on`; v5 added `command_derate` and `joint_vel_weight`.  One of
those has a measured pathology as of tonight: with `SIGHT_RAMP=0` -- which
every v7, v8 and v9 teacher trained under -- `wrist_side_on` is pinned at its
**initial** 0.8 instead of decaying to 0.30, and it pays 0.74-0.76 per step for
holding a pose.  Three runs tonight converged on collecting it and placing
nothing, at a *higher* total reward than the runs that work (16-17 against
10-15), with the action std collapsed to 0.13-0.15.

That is a candidate, not a conclusion: nothing yet ties it to the v4->v8
success drop, only to the three collapses.  `v9_wrist30_s1.5` (same as a run
that collapsed, with `WRIST_W=0.30`) is the first test.

## Where the loss actually is: attempts, not skill

Placed per grasp **attempt**, 128 envs x 1800 steps.  Rate-independent by
construction, so a policy that was deliberately slowed down is not punished for
being slow -- which is what makes it able to separate "worse at grasping" from
"grasps less often".

| policy | task | placed | attempts | placed/attempt |
|---|---|---|---|---|
| v8 teacher | Robust | 9.94 | **11.48** | 0.865 |
| `v8s_depth` | Distill-Robust | 1.36 | **1.74** | 0.780 |
| `v8s_sam2` | Distill-Robust | 1.41 | **1.70** | 0.826 |

**Distillation costs 5-10% of the per-attempt success and 85% of the
attempts.**  0.865 -> 0.78-0.83 is a small loss; 11.48 -> 1.7 attempts in the
same 1800 steps is a factor of 6.7.

The students are nearly as good as their teacher at converting an attempt into
a placement.  They almost never start one.  That is the same defect the rest of
tonight kept circling -- the gripper command drifting closed, the placement
rate decaying within an episode, "grasps once and stops" -- stated in the units
that make it one defect instead of three symptoms.

### The v4 row is missing on purpose

Measured the same way, v4 makes **0.20** attempts in 1800 steps, which cannot
be reconciled with the 29.95/min and 0.960 success it scores under
`accept_s1.py`.  That is the third time this checkpoint has read as dead under
this repo's own evaluation scripts and excellent under accept_s1, so the row is
withheld rather than published: something in the play-mode config path does not
run v4 correctly, and until that is found, **any cross-lineage number must come
from accept_s1 and nothing else**.  The 2x2 above does; this table does not,
which is why it compares only the v8 lineage against itself.
