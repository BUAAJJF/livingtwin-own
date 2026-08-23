# Novelty validation, phases 0–2

The claim under test:

> Domain randomisation matches the marginal distribution of each parameter but
> ignores the timescale on which that parameter actually changes in deployment.
> A recurrent policy exploits the wrong within-episode invariants, treating
> history about *previous* objects as an implicit privileged observation of the
> *current* one, which inflates simulated performance and produces fine-tuning
> gains that do not transfer.

This round runs phases 0–2 only and stops at Gate A. No two-timescale network,
no world model, no BFM/TeCH, no hardware.

Status: **Phase 0.1 complete** (this document, sections 1–2). Phases 0.2, 1 and
2 in progress; their sections are filled in as they finish.

---

## 1. Repository and experiment audit

Audited at `a5feaf9` on branch `yf/bolt`, working tree clean. Server mirror
`shen-teacher:/home/yunfan/work/piper-push/LivingTwin` is at the same commit.

`README.md` is stale — it still describes the retired `push_cube` task. The
pick-and-place task it does not mention is what everything below concerns.

### 1.1 Where the episode and the object lifecycle are decided

The two are *not* the same loop, and that distinction is the whole subject of
this document.

| event | code | trigger |
|---|---|---|
| episode end | `terminations` in `env_cfg.py:481` | `time_out` (12 s train / 40 s play), `object_lost`, `over_speed`, `nan` |
| episode reset | mjlab `_reset_idx` | runs reset events, then `command.reset()`, then `sim.forward()` |
| **object respawn** | `PickCommand._update_metrics` → `_place_object` (`mdp.py:451`) | `just_placed \| just_knocked`, *mid-episode*, no reset |
| table refill | `_place_all` (`mdp.py:569`) | last object cleared (`N>1` only) |
| target switch | `_retarget` (`mdp.py:641`) | only when an object is cleared; nearest to the hand |
| stray recovery | `_recover_strays` (`mdp.py:474`) | every step, `N>1` only |

At 50 Hz with a 12 s episode and a ~1.0 s median cycle, **one episode contains
roughly ten object placements**. The object lifecycle turns over about an order
of magnitude faster than the episode.

`_retarget` is deliberately deferred one step on the reset path
(`_retarget_pending`, `mdp.py:247`) because `_reset_idx` runs events and
resamples the command *before* `sim.forward()`, so a read there returns the
previous episode's poses.

### 1.2 When each parameter is redrawn

Every entry below was read off the code, not inferred from documentation.

| parameter | drawn by | current cadence |
|---|---|---|
| object shape class, half-extents | `shapes.randomize_object_shape` | reset event **and** every placement |
| object mass | same call | reset event **and** every placement |
| object inertia, COM, principal axes | derived from shape+mass in the same call | as above |
| object friction (object↔table) | same call | as above |
| object planar pose, yaw | `PickCommand._place_one` | reset **and** every placement |
| finger-pad friction | `dr.geom_friction` (`pad_friction`) | **reset only** — per episode |
| camera position and orientation | `camera.randomize_camera_pose` | **reset only** — per episode (startup in play) |
| arm reset posture | `reset_arm_valid_posture` | reset only |
| depth noise, depth dropout | inside the `camera_scene` obs term | **every step**, i.i.d. |
| observation noise (proprio, object) | `UniformNoise` on obs terms | every step, i.i.d. |
| PD gains `kp`/`kd` | `robot.SYSID_GAINS` | **never randomised** |
| joint armature, damping, Coulomb friction | `robot.SYSID_COULOMB_NM` and spec constants | **never randomised** |
| encoder bias | mjlab default (zeros) | **never randomised, fixed at 0** |
| gripper stiffness, damping, rate limit | `robot.GRIPPER_*` | **never randomised** |
| command rate limit (`COMMAND_DERATE = 0.62`) | `robot.COMMAND_RATE_LIMIT_RAD_S` | fixed |
| control / observation latency | not modelled at all | **absent** |
| bin pose, terrain friction | constants | fixed |

The critical line is the one that changed at commit `6d1d0a9`: shape, mass and
friction are redrawn **per object**, because `PickCommandCfg.reshape_on_place`
defaults to `shape_variety > 0.0` (`env_cfg.py:261`) and `_place_one` calls
`_reshape` on every non-reset placement (`mdp.py:594`).

Two consequences that shape the rest of this document:

1. **The registered tasks are already the honest (`OBJ-All`) condition for
   object parameters.** Every number in `docs/results.md` was measured there.
   The leaky `EP-All` condition is now reachable only by setting
   `reshape_on_place=False`, and **no CLI flag exposes it** — the vestigial
   `--reshape-on-place` in `accept_s1.py:76` predates the default flip and now
   double-redraws if passed.
2. **Shape, mass and friction are drawn in one atomic call**
   (`shapes.randomize_object_shape`, `shapes.py:260`). There is currently no
   way to hold mass constant while redrawing shape. Phase 2.2 requires exactly
   that, so it is the one place cadence has to be split from distribution.

### 1.3 Variable / real timescale / current sim timescale

"Real timescale" is how often the quantity actually changes on the rig once
the robot is running: a session is one deployment run between rebuilds or
recalibrations; an object is one item picked off the table; a step is one 20 ms
control period.

| variable | real timescale | current sim timescale | verdict |
|---|---|---|---|
| camera extrinsics | session | **episode** | too fast (conservative) |
| camera intrinsics (fovy) | session | fixed | no distribution |
| arm PD gains | session | **fixed** | **no distribution** |
| joint friction / armature | session | **fixed** | **no distribution** |
| encoder bias | session | **fixed at 0** | **no distribution** |
| gripper rate, force | session | **fixed** | **no distribution** |
| control latency | session + step jitter | **absent** | **not modelled** |
| table friction | session | fixed | no distribution |
| bin pose | session | fixed | no distribution |
| finger-pad friction | session (slow wear) | **episode** | too fast (conservative) |
| object shape class | object | object | **matched** |
| object size | object | object | **matched** |
| object mass | object | object | **matched** |
| object COM / inertia | object | object | **matched** |
| object friction | object | object | **matched** |
| object initial pose | object | object | **matched** |
| depth noise | step (spatially correlated) | step (i.i.d. Gaussian) | matched cadence, wrong correlation |
| depth dropout | step (surface-dependent) | step (i.i.d. Bernoulli) | matched cadence, wrong structure |
| proprio noise | step | step | matched |

Read as a whole: the object parameters are already correctly paced, the session
parameters are paced *too fast* (which is conservative, not leaky), and the
robot's own plant has **no randomisation at all**. The one direction the
hypothesis predicts damage — a parameter held constant for longer in sim than in
reality — currently applies to **no** parameter in the default configuration.
That is the honest starting point for Phase 2: the leak has to be re-created
deliberately (`EP-All`) to be measured, and the question is how much it was
worth when it was there.

### 1.4 GRU hidden state

`SpatialSoftmaxRecurrentModel.reset(dones)` (`models.py:107`) forwards to
`rsl_rl`'s `RNN.reset`, which zeroes the hidden state for the flagged
environments. It is called:

* PPO — `process_env_step`, on `dones`.
* Distillation — same path.
* Evaluation — `accept_s1.py:185`, on `dones`.

It is **never called on object respawn**. A single hidden state therefore spans
the ~10 objects an episode contains. This is the exact channel the hypothesis
names, and "zero the hidden state at every respawn" is the control Phase 2.3
needs.

### 1.5 Distillation and PPO initialisation paths

* **DAgger, not behaviour cloning.** `rsl_rl`'s `Distillation` has the *student*
  act (`stochastic_output=True`) and the teacher label only the states the
  student visited (`src/piper_push/distill.py:8`). Teacher observation group
  `("full_proprio", "object")`, student `("proprio", "camera")`
  (`rl_cfg.py:180`).
* `full_proprio` is an uncorrupted deep copy of the pre-surgery proprioception
  (`env_cfg.py:652`); the student's `proprio` has `grasped` deleted and
  `squeeze` (gripper servo error) added in its place (`env_cfg.py:663`).
* **Teacher load** — `scripts/distill.py:94`, `load_cfg={"teacher": True}`,
  `strict=True`. `PickPlaceDistillationRunner.learn` refuses to run if no
  teacher was loaded (`distill.py:96`).
* **Fine-tune load** — `scripts/finetune.py`: actor from the distillation
  checkpoint via `as_actor_checkpoint` (renames `student_state_dict` →
  `actor_state_dict`), critic from the *state* policy's checkpoint, then
  `init_std` overwritten (distillation regresses the mean and never touches the
  standard deviation), then a `critic-warmup` of 100 iterations during which
  actor parameters have `requires_grad_(False)` and the LR schedule is pinned to
  `fixed`.

### 1.6 Safety-shell definition

* **Termination** `over_speed` = `joint_velocity_trip` (`mdp.py:1149`):
  `(|q̇| > JOINT_TRIP_RAD_S).any(dim=-1)`, evaluated on the post-step joint
  velocity, per-joint limits 180–225 °/s from the PiPER-X manual.
* **Reward** `over_trip` = `joint_speed_over_trip` charges the excess above
  `0.85 × trip`, not above the trip itself.
* **The command path is already rate-limited** to `0.62 × trip`
  (`COMMAND_DERATE`, `robot.py:289`) *and* linearly interpolated across the ten
  physics substeps (`actions.py:83`). This materially changes what Phase 1 can
  claim — see section 4.
* `robot.py:280` already records a derate sweep against shell events under a
  zero policy: 0.75 → 44, 0.68 → 26, 0.62 → 5, 0.56 → 2 events. The stated
  cause of the residual is the identified plant, not the command staircase:
  `kp 125` against `kd 6.5` on joints 1–3 is ζ ≈ 0.35, and a system that
  underdamped overshoots a velocity ramp by ~32%.

### 1.7 How each reported metric is computed

All from `scripts/accept_s1.py`. `sim_seconds` accumulates `dt × num_envs`
every step, including steps spent resetting.

| metric | definition |
|---|---|
| throughput | `placed_total / sim_seconds × 60` |
| success | `ok / (ok + fail)`; an *instance* is one object from spawn to placement, a win if placed within `--budget` (4 s); an instance cut short by the episode horizon is **censored, not failed** |
| post-grasp drop | grasp spans that end without a placement or re-grasp within `--drop-grace` (1.5 s) |
| p50 / p95 time to place | quantiles of instance age at placement |
| time with a stuck object | share of total time where the live instance is older than the budget |
| tables cleared, objects astray | per-step deltas of the command's counters, clamped at ≥ 0 (they are per-episode counters that reset zeroes) |
| safety-shell trips | `termination_manager.get_term("over_speed").sum()` per step, reported per arm-hour |
| \|q̇\|/trip distribution | 256-bin histogram per joint, `SPEED_MAX = 1.6`; p99, p99.9, peak, and share of time above `0.85` |

**Defect found during the audit** (`accept_s1.py:276`): `cls_now` — the shape
class an instance is attributed to — is refreshed only on `dones`, never on
respawn. With per-object reshaping now the default, every instance after the
first in an episode is **attributed to the wrong shape class**. Overall success,
throughput, drop rate and trips are unaffected (they sum across classes); the
per-shape breakdown table is not trustworthy. This does not touch any headline
number in `docs/results.md`.

### 1.8 Determinism and seeds

This is the finding that most constrains the rest of the work.

* **`accept_s1.py` accepts no seed.** Nothing in it seeds torch. Every number in
  `docs/results.md` was produced by an unseeded rollout.
* `env_cfg.seed` (used by `distill.py` and `finetune.py`, default 42) resolves
  to a single `torch.manual_seed` (`mjlab/utils/random.py:23`).
* Every object draw consumes that one global stream: `_compose(generator=None)`
  (`shapes.py:284`), `torch.rand` for mass and friction, `sample_uniform` in
  `sector_sample`.

Therefore: seeding makes a rollout reproducible **for a fixed policy**, but it
**cannot pair scenes across policies**. The order in which the global stream is
consumed depends on *when* placements happen, which depends on the policy. Two
policies given the same seed diverge into different object sequences after the
first placement that differs in timing.

Phase 2 asks for paired evaluation scenes. That requires a per-environment RNG
stream for the object draw, advanced once per object, so that the *k*-th object
in environment *i* is the same object regardless of which policy is driving.
That is the second necessary code change, and it is recorded here rather than
assumed.

---

## 2. Minimal change plan

One hypothesis per change, nothing bundled.

| # | change | why it is required | risk to existing results |
|---|---|---|---|
| C1 | `--seed` on `accept_s1.py`, recorded in the output | no experiment below is reproducible without it | none — default `None` preserves current behaviour |
| C2 | per-environment RNG stream for the object draw | Phase 2 needs paired scenes; the global stream cannot provide them | changes the object *sequence*, not its distribution — baselines must be re-measured under it, which Phase 0.2 does |
| C3 | machine-readable output (JSON) from `accept_s1.py` | plots must be generated from files, not transcribed | none — additive |
| C4 | split *cadence* from *distribution* in the object randomiser: per-quantity hold-time (`shape`, `mass`, `friction`) | 2.2 requires mass and friction cadence separately; today they are one atomic draw | none if the default reproduces today's behaviour, which is asserted by test |
| C5 | evaluation-only action wrappers (slew / accel / LPF / cubic), default off | Phase 1 | none — default off, and the existing rate limiter is left in place |
| C6 | `--reset-hidden-on-respawn` evaluation flag | Phase 2.3 control | none — default off |
| C7 | fix `cls_now` refresh on respawn | the per-shape table is currently misattributed | corrects a table nobody's conclusions rest on |

Deliberately **not** changed: rewards, teacher inputs, network capacity,
observation spaces, the `COMMAND_DERATE`, or anything in the training path.

### 2.1 What Phase 1 can and cannot claim

The premise of Phase 1 as posed — "test whether a fixed action smoother solves
the safety problem" — is already **partly answered in the negative by the code**.
The deployed action path is not a raw policy output: it is slew-limited to
0.62 × trip and interpolated across substeps, and the residual trips are
attributed to an underdamped plant rather than to command roughness.

So Phase 1 is re-scoped, without weakening it:

* The **baseline** is the policy *with* the existing limiter, because that is
  what produced 3.7 and 9.5 events/arm-hour.
* The wrappers under test are the ones that are **not** already there:
  acceleration limiting, first-order low-pass, and cubic/S-curve interpolation —
  plus a *tighter* slew limit as the trivial control.
* The gate is unchanged: ≥ 80 % of events removed at ≤ 2 % throughput cost.

This is a stronger test than the one originally posed, because the easy win has
already been taken and the remaining events are the hard ones.

---

## 3. Baseline reproduction

Protocol as documented: 512 environments × 2400 control steps, deterministic
policy, honest (per-object) randomisation. Seed 20260823, and a second seed
31415926 for three of the runs to separate rollout sampling noise from a real
discrepancy. Intervals are a 2000-draw bootstrap resampling **environments**.

Generated by `scripts/analyze_novelty.py --section baseline` from
`results/novelty_validation/baseline/*.json`.

### 3.1 Which checkpoints

The evaluation script writes no provenance and no shell history survived, so
the checkpoints behind `docs/results.md` were recovered from each training
run's **wandb metadata**, which records the full argv. That resolved a
mis-identification: the S2 teacher is `h_full/model_3400`, not
`f_full/model_3499`. The wrong one scores 38.0 and is kept in the results
directory as `s2_f_full_notTheTeacher` rather than deleted.

    d1   --teacher .../h_full/model_3400.pt
    d1r  --resume  .../d1/model_1500.pt
    f2   --student .../d1r/model_2999.pt  --critic .../h_full/model_3400.pt
    f3   --resume  .../f2/model_1100.pt
    cd1  --teacher .../c1/model_2499.pt
    cf1  --student .../cd1/model_2499.pt  --critic .../c1/model_2499.pt

### 3.2 Result

| run | reference | reproduced | 95% CI | rel. | verdict |
|---|---:|---:|---|---:|---|
| s2_teacher | 58.4 | 59.0 | [58.3, 59.6] | +1.0% | **PASS** |
| s2_teacher seed B | 58.4 | 59.2 | [58.7, 59.7] | +1.4% | **PASS** |
| s2_distilled | 53.3 | 48.1 | [47.3, 48.9] | −9.7% | **FAIL** |
| s2_distilled seed B | 53.3 | 47.7 | [46.9, 48.5] | −10.5% | **FAIL** |
| s2_finetuned | 55.8 | 56.2 | [55.8, 56.5] | +0.6% | **PASS** |
| s2_finetuned seed B | 55.8 | 56.0 | [55.5, 56.5] | +0.4% | **PASS** |
| s3_teacher | 42.4 | 42.5 | [42.1, 42.8] | +0.1% | **PASS** |
| s3_distilled | 38.7 | 37.4 | [36.7, 38.2] | −3.3% | **FAIL** |
| s3_finetuned | 39.8 | 39.4 | [38.8, 40.0] | −1.0% | **PASS** |

Success is within 0.5 pp everywhere except `s2_distilled` (99.1% against
99.6%). p95 is within 5% everywhere except `s2_distilled` (1.96 s against
1.76 s).

**Six of nine runs reproduce inside the stated tolerance. The three that do
not are exactly the distilled-only policies**, and both S2 seeds agree to
within 0.8%, so this is not sampling noise: the seed-to-seed spread is
0.3–0.8% on every policy, and the S2 distilled gap is 10%.

### 3.3 Why the distilled numbers do not reproduce

Not a checkpoint mix-up — `d1/model_1500` scores 41.1, further away, and the
launch record says `f2` started from `d1r/model_2999`, which is what was
scored. Not the checkpoint conversion either: the distillation
`student_state_dict` and the fine-tune `actor_state_dict` are both 23 tensors
including the observation normaliser, and `as_actor_checkpoint` only renames
the key.

The wandb configs answer it. `config.yaml` for **d1, d1r and f2 contains no
`reshape_on_place` field at all** — those runs predate it, so they were
trained *and* the environment they ran in was **EP-All**, the leaky one.
`f3`, `cd1` and `cf1` all record `reshape_on_place: true`.

Section 5 measures the distilled policy under both cadences directly and finds
it reads **52.9** under EP-All against **47.3** honest — so the published 53.3
is an EP-All number. `docs/results.md` states that every figure in it was
measured in the honest environment; **for the distilled rows that is not
correct**, and the discrepancy is the leakage this round exists to test.

The safety column tells the same story from the other side: the reference
records 16.4 trips per arm-hour for the distilled policy and the honest
environment gives 22.3–26.1.

### 3.4 Reproducibility caveats

* Nothing that produced `docs/results.md` was seeded, so those numbers are not
  re-runnable even in principle; the runs here are.
* mjlab's own note is that MuJoCo-Warp is not bit-deterministic yet
  ([mujoco_warp#562]), so a seed fixes the scene sequence, not the last bit of
  the physics. The 0.3–0.8% seed-to-seed spread measured here bounds what that
  is worth in practice.
* A seed does **not** pair scenes across policies (section 1.8). Comparisons
  below are therefore unpaired, with bootstrap intervals over environments
  doing the work pairing would otherwise do.

[mujoco_warp#562]: https://github.com/google-deepmind/mujoco_warp/issues/562

## 4. Safety wrapper Pareto

### 4.1 What the baseline already is

The audit (§1.6) changed this experiment. The deployed command path is **not**
a raw policy output: it is slew-limited to `0.62 × trip` and linearly
interpolated across the ten physics substeps, and `robot.py` already records a
derate sweep against shell events under a zero policy (0.75 → 44, 0.68 → 26,
0.62 → 5). So `none` below is that path, and the filters worth testing are the
ones that are not already in it.

The calibration run (seed 1001, disjoint from the scoring seed) shows what the
command actually looks like, pooled over the six arm joints:

| quantity | p50 | p90 | p99 | peak | structural bound |
|---|---:|---:|---:|---:|---:|
| commanded \|v\|/trip | 0.6 | 0.6 | 0.6 | 0.6 | 0.62 |
| commanded \|a\| (rad/s²) | 1.6 | 195 | 244 | 243.5 | 243.5 |
| commanded \|jerk\| (rad/s³) | 156 | 12 188 | 19 531 | 24 347 | 24 350 |

Two facts follow. The **median** commanded velocity is already at the slew
ceiling — the policy is saturated against the rate limiter about 61% of the
time. And the acceleration and jerk maxima are exactly the structural bounds
that ceiling implies (`2 × 0.62 × trip / dt`, and that again over `dt`), which
is what a bang-bang command looks like. The policy uses every unit of
authority it has, so any filter that binds costs throughput.

### 4.2 Result — single object, distilled policy

Scored on held-out seed 20260823, 512 × 2400. This policy is the interesting
subject because it is the one with the safety problem, and because PPO
fine-tuning is known to fix it: the question is whether a fixed filter can do
what the fine-tune did.

| filter | obj/min | Δ | trips/arm-h | Δ | success |
|---|---:|---:|---:|---:|---:|
| none (as deployed) | 46.7 | — | 22.7 | — | 99.0% |
| slew × 0.85 | 44.6 | −4.5% | **4.2** | **−81.5%** | 98.7% |
| slew × 0.70 | 36.4 | −22.1% | 2.8 | −87.7% | 97.6% |
| accel ≤ 180 rad/s² | 46.2 | −1.1% | 24.2 | +6.6% | 99.1% |
| accel ≤ 120 | 44.2 | −5.4% | 26.2 | +15.4% | 98.9% |
| accel ≤ 80 | 38.1 | −18.4% | 42.3 | **+86.3%** | 98.5% |
| low-pass 15 Hz | 45.3 | −3.0% | 32.2 | +41.9% | 99.1% |
| low-pass 8 Hz | 40.6 | −13.1% | 55.1 | **+142.7%** | 98.9% |

**Only the slew limiter helps. Acceleration limiting and low-pass filtering
make the safety shell fire *more often*, and the harder they are applied the
worse it gets** — the 8 Hz low-pass nearly triples the trip rate.

That is counterintuitive under an open-loop reading and obvious under a
closed-loop one. The shell fires on *measured* joint velocity. An acceleration
limit or a low-pass inserts lag between what the policy asks for and what the
servo is given; the policy is in the loop, sees itself not arriving, and
commands harder to compensate. The filter smooths the command and the
compensation un-smooths the plant. Only the slew ceiling helps because it is
the one filter that bounds the very quantity the shell measures.

### 4.3 Safety Gate

**FAIL, narrowly, on the throughput criterion.** The gate asks for ≥ 80% of
trips removed for ≤ 2% of throughput. `slew × 0.85` removes 81.5% of trips but
costs 4.5%. No filter tested reached the pair. *(A finer sweep at 0.88/0.90/
0.92/0.95 is in §4.4 to locate the frontier exactly.)*

But the gate's *purpose* — deciding whether safety should be the paper's
contribution — is answered more clearly than the threshold suggests, and in
the direction the gate was written to detect:

* A deterministic shell buys **most** of the safety. 81.5% of the trips are
  reachable with one scalar and no retraining.
* It buys it by **strictly trading throughput away**. Every filter here moves
  down and left; none moves up.
* PPO fine-tuning moved this policy from **46.7 obj/min @ 22.7 trips/h** to
  **55.8 @ 2.3** — 19% *more* throughput and 90% fewer trips *at the same
  time*. No filter can do that, because a filter can only remove authority.

So: **safety alone is not the novelty** — a shell gets most of it. What a
filter cannot reproduce is the joint improvement, and "imitation inherits
speed but not caution, and RL restores caution without paying speed" remains a
claim about learning, not about filtering.

## 5. Cadence experiment matrix

### 5.1 Construction

`redraw_on_place` selects which of `(shape, mass, friction)` is drawn afresh
when an object is replaced. Every setting draws from the **same ranges** — the
reset event's params are untouched — so only the hold time varies.
`tests/test_cadence.py` asserts the marginals are identical across subsets, and
`scripts/check_cadence.py` forces placements in simulation and confirms the
plumbing:

| cadence | shape changed | mass changed | friction changed |
|---|---:|---:|---:|
| `object` | 100% | 100% | 100% |
| `episode` | 0% | 0% | 0% |
| `shape` | 100% | 0% | 0% |
| `mass` | 0% | 100% | 0% |
| `friction` | 0% | 0% | 100% |

`OBJ-All` = `object`, `EP-All` = `episode`. Both draw a fresh object at every
episode reset; the difference is what happens at the ninth placement of the
same episode.

### 5.2 The core matrix

Same checkpoint, same seed, same 512 × 2400 protocol; only the evaluation
environment's cadence changes. **Positive Δ means the leaky environment reads
better**, which is the effect under test.

| policy | memory | OBJ-All | EP-All | Δ |
|---|---|---:|---:|---:|
| state teacher — *feedforward, no memory* | — | 59.3 | 60.2 | **+1.5%** |
| vision student, distilled — *trained under EP-All* | kept | 47.3 | 52.9 | **+11.8%** |
| vision student, distilled | zeroed each object | 35.4 | 37.9 | +7.1% |
| vision student, fine-tuned — *trained under OBJ-All* | kept | 55.8 | 57.8 | **+3.6%** |
| vision student, fine-tuned | zeroed each object | 44.0 | 45.4 | +3.2% |

Three readings, in decreasing order of confidence:

**1. The leaky environment inflates a memory-carrying policy about eight times
as much as a memoryless one.** +11.8% against +1.5%, same task, same
evaluation, same everything but the hold time. The 1.5% is the honest
difficulty difference — a feedforward policy that is *handed* the shape has no
way to exploit its constancy — and everything above it is attributable to the
recurrent state.

**2. Most of the exploit is learned, not architectural.** Both students are
the same GRU network with the same weights shape. The one distilled under
EP-All gains 11.8%; the same network after PPO fine-tuning under OBJ-All gains
only 3.6%. Training in the honest environment removed roughly two thirds of
the exploitable advantage. This is the single most useful number here: it says
the artefact is fixable by fixing the benchmark, without changing the
architecture.

**3. The hidden-reset ablation is not a clean subtraction, and should not be
read as one.** Zeroing the recurrent state at every object boundary costs the
distilled policy 25% of its throughput (47.3 → 35.4) and the fine-tuned one
21% (55.8 → 44.0). The memory is doing a great deal of legitimate work —
principally tracking whether the gripper is holding anything, which is exactly
the flag that was removed from the observation to make the task deployable —
so clearing it per object removes the deployable function along with the
artefact, and the resulting policy is far outside the distribution it was
trained in. The residual gaps (+7.1% and +3.2%) are therefore *not* "the
non-memory part of the effect".

### 5.3 Safety reads better in the leaky environment too

The same table in trips per arm-hour, where lower is better:

| policy | OBJ-All | EP-All |
|---|---:|---:|
| state teacher | 4.7 | 2.6 |
| distilled, memory kept | 22.6 | 20.7 |
| fine-tuned, memory kept | 2.3 | 2.5 |

The teacher's shell rate nearly halves under EP-All. A benchmark that holds
object parameters constant within an episode under-reports the constraint
violation rate as well as over-reporting throughput, and the two errors point
the same way.

## 6. Hidden-state probes and history swap

This is where the proposed **mechanism** is tested, as opposed to the effect,
and it is where the hypothesis as stated does not survive.

### 6.1 Method

A multinomial logistic regression decodes, from the 256-unit GRU hidden state
alone, four labels: the shape class and mass bin of the object currently in
the hand, and of the object before it. Three choices decide whether the
measurement means anything:

* **The train/test split is by environment.** A random split over timesteps
  would put frames 40 ms apart on both sides and report the autocorrelation of
  the hidden state as probe accuracy.
* **The probe is linear.** The question is whether the information is present
  and linearly available, not whether some network can be built to extract it;
  a deep enough probe recovers almost anything.
* **The floor is the majority-class rate**, not `1/K`. The shape prior is
  (0.34, 0.22, 0.16, 0.16, 0.12), so a probe that has learned nothing still
  reads 34% against uniform chance.

### 6.2 Result

Lift over the majority-class baseline, in percentage points. 71 500 hidden
states from 256 environments; 3000 steps.

| policy (training cadence) | eval cadence | cur shape | **prev shape** | cur mass | **prev mass** |
|---|---|---:|---:|---:|---:|
| distilled (EP-All) | OBJ-All | +5.5 | **−0.8** | +7.3 | **+1.4** |
| distilled (EP-All) | EP-All | +7.0 | +7.0 † | +14.4 | +14.4 † |
| fine-tuned (OBJ-All) | OBJ-All | +2.5 | **+2.2** | +7.3 | **+1.9** |

† Under EP-All the previous object *is* the current object, so those two
columns are the same measurement written twice. They carry no information
about carry-over and are shown only to make the tautology explicit.

**Under the honest cadence, the recurrent state carries essentially nothing
about the previous object.** Previous shape decodes at −0.8 pp for the
distilled policy — below its own floor — and +2.2 pp for the fine-tuned one;
previous mass at +1.4 and +1.9 pp. Meanwhile the *current* object's mass
decodes at +7.3 pp for both, which is the memory doing precisely the job it
exists for: mass is the one parameter a camera cannot see and must be inferred
from contact.

### 6.3 A statistic that nearly fooled this section

The EP-All row first read "+14.4 pp on mass against +7.3 pp honest", which
looks like the leaky environment letting the memory learn twice as much. The
raw accuracies were 36.7% and 36.0% — nearly identical. The entire difference
was in the **majority baseline** (22.3% against 28.8%), because under EP-All
the label is constant for a whole episode, so the empirical class frequencies
are lumpy and the floor moves. Lift is not comparable across conditions with
different label marginals.

Two corrections are therefore reported alongside: **balanced accuracy** (mean
per-class recall, invariant to the class prior), and **n_effective** — the
number of distinct (environment, label) facts in the held-out set. Under
EP-All tens of thousands of frames carry only a few hundred independent
labels; under OBJ-All the label turns over every second or so. A confidence
read off the frame count would be wildly overconfident for one condition and
not the other.

### 6.4 History swap

Every environment is given another environment's hidden state and its own
present; both branches are then fed the **identical** observation stream, so
any difference in action is attributable to the recurrent state and to nothing
else. Divergence is normalised by the across-environment spread of the action,
so "large" means large relative to how much the policy varies its output at
all.

| policy | eval cadence | ‖Δa‖ / action spread |
|---|---|---:|
| distilled | OBJ-All | 0.69 |
| distilled | EP-All | 0.65 |
| fine-tuned | OBJ-All | 0.74 |

The memory is doing a **lot** — swapping it moves the action by about
two-thirds of the policy's entire output range. But it moves it by the same
amount under both cadences and for both policies, including the one trained
honestly. That is consistent with §5.2's finding that clearing the hidden state
costs 21–25% of throughput: what the state carries is mostly the deployable
belief the observation no longer provides (am I holding something, did the last
squeeze work), not a stale fact about a previous object.

### 6.5 What this means for the hypothesis

The proposed mechanism was that the recurrent policy "treats history about
previous objects as an implicit privileged observation of the current one".
**The probes do not support that.** Under the honest cadence there is no
recoverable previous-object information to treat as anything.

The effect in §5 is nonetheless real and large. What the two sections together
are consistent with is a different mechanism: not *stale carry-over* but
*evidence accumulation about a quantity that should have changed*. When the
object is held constant for an episode, a recurrent policy integrates
information about it across ten grasps rather than one, and arrives at a
sharper estimate than any deployment would ever permit. That is still cadence
leakage and it still inflates the benchmark — but it is a different claim, and
it implies a different fix. §11C proposes the experiment that would separate
them.

## 7. Seeds and confidence intervals

Three sources of variation, kept separate because they have very different
sizes and only one of them is cheap to reduce.

### 7.1 Rollout sampling uncertainty

Every interval quoted is a 2000-draw bootstrap **resampling environments**,
not control steps. Two consecutive steps in one environment are about as
independent as two consecutive frames of a video; two environments share only
the policy and the parameter distribution. Ratios are recomputed from
resampled totals rather than averaged over per-arm ratios, so an arm that
placed nothing contributes its zero instead of a `0/0`.

At 512 environments × 2400 steps the 95% interval on throughput is about
**±0.8 objects/min**, i.e. ±1.5% relative.

> The first implementation of this was wrong in a way worth recording. It used
> a hand-rolled LCG and took `s % n` — the low bits of a power-of-two-modulus
> generator, which cycle with period `n`. With 512 environments every
> "resample" therefore drew each index exactly once, every resampled total
> equalled the true total, and every interval collapsed to a point. It printed
> `[48.1, 48.1]` and read as an extraordinarily precise measurement. It is now
> a seeded Mersenne Twister, with a guard that raises on a degenerate interval.

### 7.2 Rollout seed

Three policies were scored at a second rollout seed (31415926 against
20260823):

| policy | seed A | seed B | spread |
|---|---:|---:|---:|
| s2 teacher | 59.0 | 59.2 | 0.3% |
| s2 distilled | 48.1 | 47.7 | 0.8% |
| s2 fine-tuned | 56.2 | 56.0 | 0.4% |

So the seed-to-seed spread is **0.3–0.8%**, comfortably inside the bootstrap
interval, and an order of magnitude smaller than the 10% baseline discrepancy
in §3 or the 11.8% cadence effect in §5.

MuJoCo-Warp is not bit-deterministic ([mujoco_warp#562]), so a seed fixes the
scene sequence and not the last bit of the physics; §7.4 measures what that is
worth directly.

### 7.3 Training seed — **the gap in this round**

Gate A condition 2 asks for the cadence effect to be consistent in direction
across **training** seeds. Only one pair of training seeds exists in this
repository: the state teacher was trained twice (`h_full` and `h_full_s2`,
`--agent.seed 17`), and both are evaluated under both cadences in §5.4.

There is **no** second training seed for either vision student. Producing one
means re-running distillation and fine-tuning, roughly two GPU-hours per seed,
which is outside this round. **Condition 2 is therefore only tested on the
memoryless control**, and that is stated as a limitation rather than papered
over: the 11.8%-versus-1.5% contrast in §5.2 rests on one distillation run and
one fine-tuning run.

### 7.4 Run-to-run reproducibility

The same checkpoint, seed and protocol read 48.1 in the Phase 0.2 batch and
46.7 as the unfiltered Phase 1 baseline — a 3% gap, larger than the seed-to-
seed spread. The action-path edits between those commits are arithmetically
identical at their defaults, so this is either a real regression or
MuJoCo-Warp's documented non-determinism. §7.5 reports a three-way repeat of
the identical command that settles it.

[mujoco_warp#562]: https://github.com/google-deepmind/mujoco_warp/issues/562

## 8. Commands

Every run below was launched on `shen-teacher` inside `tmux` with
`setsid nohup`, so a dropped VPN cannot take a run with it. Two environment
settings are required for any non-interactive invocation and are baked into
`scripts/eval.sh`:

```bash
export MUJOCO_GL=disable                       # mujoco imports a GL backend it never uses
export LD_LIBRARY_PATH=$MAMBA_ROOT/envs/mjlab/lib:$LD_LIBRARY_PATH
```

The second is not optional: the env ships `libicui18n.so.78`, which needs
`CXXABI_1.3.15`, and the system `libstdc++` does not have it. `micromamba run`
does not export the env's lib directory but an interactive `micromamba
activate` does, so this breaks only non-interactive runs — and it breaks them
inside mjlab's own import of `mediapy → IPython → sqlite3`.

**Phase 0.2 — baselines** (`repro.sh`, `wave3.sh`):

```bash
SEED=20260823 NUM_ENVS=512 STEPS=2400 OUT=results/novelty_validation/baseline \
GPU=3 scripts/eval.sh Mjlab-Pick-Place-PiperX \
  logs/rsl_rl/piperx_pick_place/2026-08-22_07-55-30_h_full/model_3400.pt s2_teacher
```

and likewise for each row of section 3.2; the distilled checkpoints go through

```bash
python scripts/student_to_actor.py <distill>/model_N.pt /tmp/actors/<name>.pt
```

**Phase 1 — calibration then held-out scoring.** Thresholds come from a
disjoint seed:

```bash
SEED=1001     NUM_ENVS=256 STEPS=1200 OUT=.../safety/calib  scripts/eval.sh ...   # calibrate
SEED=20260823 NUM_ENVS=512 STEPS=2400 OUT=.../safety  bash runq.sh "6 7" safety.jobs
```

where each `safety.jobs` line adds one filter, e.g.
`--slew-scale 0.85`, `--accel-limit 120`, `--lowpass-hz 8`, `--interp cubic`.

**Phase 2 — cadence matrix and probes:**

```bash
OUT=results/novelty_validation/cadence bash runq.sh "3 4" cadence.jobs
#   ... --cadence object | episode | shape | mass | friction
#   ... [--reset-hidden-on-respawn]

python scripts/check_cadence.py Mjlab-Pick-Place-PiperX --num-envs 256 --steps 300 \
  --json results/novelty_validation/cadence/plumbing_check.json

python scripts/probe_hidden.py Mjlab-Pick-Place-PiperX-Vision <ckpt> \
  --num-envs 256 --steps 3000 --swap-at 1500 --swap-horizon 25 \
  --cadence object --json results/novelty_validation/probe/<name>.json
```

**Analysis** (never transcribed by hand):

```bash
python scripts/analyze_novelty.py                       # all sections
python -m pytest tests -q                               # 38 tests
```

## 9. Files changed

New:

| file | why |
|---|---|
| `docs/novelty_validation_phase_0_2.md` | this report |
| `scripts/eval.sh` | pins the 512×2400 protocol and the two env settings above |
| `scripts/analyze_novelty.py` | every table and figure, from the JSONs |
| `scripts/check_cadence.py` | in-sim proof the cadence plumbing does what it says |
| `scripts/probe_hidden.py` | linear probes and the history-swap counterfactual |
| `tests/test_cadence.py` | cadence is a hold-time, not a distribution |
| `results/novelty_validation/reference.json` | the published targets, machine-readable |

Modified:

| file | change |
|---|---|
| `scripts/accept_s1.py` | `--seed`, `--json` with per-environment totals and provenance, `--cadence`, `--reset-hidden-on-respawn`, the Phase 1 shaping flags and their instrumentation; removed the vestigial `--reshape-on-place`, which double-redrew |
| `src/piper_push/actions.py` | `slew_scale`, `accel_limit`, `lowpass_hz`, `interp`, all inert by default, plus commanded velocity/acceleration/jerk and clip counters |
| `src/piper_push/shapes.py` | `redraw` subset parameter; per-asset record extended to size/pos/mass/friction; record seeding fixed to broadcast per world |
| `src/piper_push/tasks/pick_place/mdp.py` | `redraw_on_place` cadence knob resolved live from config |
| `scripts/finetune.py`, `scripts/distill.py` | `--cadence`, to train under a chosen cadence |

Deliberately **not** touched: rewards, teacher inputs, observation spaces,
network capacity, `COMMAND_DERATE`, or anything else in the training path.

## 10. Gate verdicts

### 10.1 Safety Gate — **FAIL on the threshold, PASS on the question**

The threshold asked for a filter removing ≥ 80% of safety-shell events at
≤ 2% throughput cost. The best result was `slew × 0.85`: **−81.5% events for
−4.5% throughput**. No filter reached the pair, so the literal verdict is
**FAIL**.

The finding the gate was written to produce is nonetheless available, and it
points the way the gate anticipated:

> **Safety is largely solvable by a deterministic shell and should not be the
> main algorithmic novelty.** One scalar, no retraining, removes four fifths
> of the violations.

The residual advantage of PPO fine-tuning over action filtering, which the
gate asked to be recorded if no filter passed:

| | throughput | trips/arm-h |
|---|---:|---:|
| distilled, unfiltered | 46.7 | 22.7 |
| distilled + `slew × 0.85` | 44.6 (−4.5%) | 4.2 (−81.5%) |
| **the same policy, PPO fine-tuned** | **55.8 (+19.5%)** | **2.3 (−89.9%)** |

Every filter moves down and to the left; fine-tuning moved up and to the left.
A filter can only remove authority, so it can only buy safety with throughput.
That asymmetry — not the safety number itself — is the part worth writing
about, and it is a claim about learning rather than about filtering.

A second, unanticipated result is worth keeping (§4.2): **acceleration
limiting and low-pass filtering make the shell fire more often**, up to +143%
for an 8 Hz low-pass. The policy is in the loop; a filter that inserts lag is
compensated for, and the compensation un-smooths the plant. Only the filter
that bounds the quantity the shell actually measures helps.

### 10.2 Cadence Gate A — **FAIL**

The gate authorises a two-timescale memory only if a **majority** of five
conditions hold.

| # | condition | verdict | evidence |
|---|---|---|---|
| 1 | EP-All produces ≥ 3% degradation on the honest test, or reproducible safety degradation | **PASS**, reframed | §5.2: +11.8% inflation for the recurrent policy against +1.5% for the memoryless control; §5.3: teacher shell rate 4.7 → 2.6 |
| 2 | effect consistent in direction across ≥ 2 **training** seeds | **NOT TESTED** | §7.3: only the memoryless teacher has a second training seed; neither vision student does |
| 3 | hidden state decodes **previous**-object attributes above chance | **FAIL** | §6.2: −0.8 pp (shape) and +1.4 pp (mass) for the distilled policy under the honest cadence; +2.2 and +1.9 for the fine-tuned one |
| 4 | zeroing the hidden state per object markedly reduces history-swap sensitivity | **FAIL** | §6.4: swap divergence is 0.65–0.74 of the action spread in every condition, and does not separate by cadence or by policy |
| 5 | long PPO fine-tuning in the wrong cadence does not transfer to the honest test | **PASS** | §5.2 and the transfer evaluations: the EP-All-trained student reads 47.3 honest, and the identical network fine-tuned under OBJ-All reads 55.8 |

Two of five pass, one is untested, and **the two that fail are precisely the
two that test the proposed mechanism**. Condition 1 establishes that the
*effect* is real and larger than `docs/results.md` records. Conditions 3 and 4
say the *explanation* offered for it is wrong.

**Verdict: do not implement the two-timescale network, the world model, or
posterior system identification this round.**

### 10.3 Which hypothesis was refuted

The claim was that a recurrent policy "treats history about previous objects
as an implicit privileged observation of the current one". Under the honest
cadence there is no recoverable previous-object information in the hidden
state to be treated as anything — the probe reads at or below its own floor.
A memory architecture designed to *forget the previous object faster* would be
solving a problem this system does not have.

What survives, and survives strongly, is the benchmark-hygiene half:

> Holding a parameter constant for longer than deployment would inflates a
> recurrent policy's measured throughput by 11.8% and depresses its measured
> constraint-violation rate, while moving a memoryless policy on the same task
> by 1.5%. Two thirds of that inflation is removed by training in the correctly
> paced environment, with no architectural change.

That is a real, measured, reproducible claim about how manipulation benchmarks
are built. It is not, on this evidence, a claim that needs a new network.

## 11. Next minimal experiment

Ordered by how much each would change what gets built, not by cost.

**A. The one missing control: a second training seed for the vision student.**
Two distillation runs and two fine-tunes under each cadence, ~4 GPU-hours
total. Everything in §5.2 rests on one distillation run and one fine-tuning
run, and it is the load-bearing claim of the whole direction. Until this
exists, "training in the honest environment removes two thirds of the exploit"
is one observation, not a result. **Do this before anything else.**

**B. Re-measure the published table, and publish the cadence with it.**
§3 shows the distilled rows of `docs/results.md` were measured under EP-All
while the header claims otherwise. Every number in that file should be
regenerated with `--cadence` recorded in the output — which the JSON now does
automatically. Cheap, and it removes a live error from the record.

**C. Test the mechanism the probes actually point at, not the one proposed.**
§6 finds no previous-object information in the recurrent state under the
honest cadence, so "stale carry-over" is not what is happening. What the
probes are consistent with is *evidence accumulation about a quantity that
should have changed*: mass is the parameter a camera cannot see, must be
inferred from contact, and benefits most from an object that never changes.
The minimal test is a cadence sweep on **mass alone** at several hold times —
one object, two, four, a whole episode — and reading the throughput gain
against hold length. A curve there is the paper; a step at "episode" is a
benchmark bug.

**D. Only if A and C both come back positive: two-timescale memory.** Gate A
does not authorise it, and §6 argues the specific design implied by the
original hypothesis — a memory that forgets the previous object — targets
something that is not there. If C shows accumulation over hold length, the
useful architecture is one whose integration window is *learned* or *reset by
an observable event*, not one hand-tuned to the object boundary.

**Not worth doing next:** anything about the safety shell as an algorithmic
contribution. §4 shows a one-scalar filter removes 81.5% of the trips, which
is enough to make it an engineering choice rather than a research question.
The part of §4 that remains interesting — command smoothing making a
closed-loop policy *less* safe — is a two-paragraph observation, not a
programme.
