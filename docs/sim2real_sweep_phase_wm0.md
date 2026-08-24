# Phase WM0: is there a sim-to-real gap worth calibrating?

> **The three depth axes in this document have been superseded.** Every number
> here was produced against the old sensor model — 4 mm of i.i.d. Gaussian
> noise and 2% of uniformly scattered dropout, both placeholders that said so.
> That model has been replaced by one fitted to a RealSense D405
> (`piper_push.depth_noise`), and the "in distribution / out of distribution"
> labels for `depth_scale`, `depth_bias_m` and `depth_dropout` no longer hold:
> scale and offset are randomised during training now, and the dropout is
> correlated and concentrated on depth discontinuities rather than scattered.
> The other fourteen axes are unaffected. See
> [`depth_sensor_and_deployment.md`](depth_sensor_and_deployment.md).

Phase WM0 exists to decide whether to build the parameter-conditioned latent
world model the README describes, **before** building it. It asks four
questions and answers them in simulation only:

1. Does the deployed vision policy actually degrade under realistic
   session-persistent mismatch, or has broad domain randomisation already
   covered it?
2. Which mismatches move throughput, and which move only the tail?
3. Can a target domain be identified from **reward-free** rollout data —
   proprioception, issued commands, servo error, policy latent — and nothing
   privileged?
4. If the target parameters were simply *known*, could simulation PPO recover
   the loss? That is the ceiling any learned calibration is aiming at.

**No hardware was involved and no real data exists.** Everything here is
simulator-against-simulator.

**Verdict: GREEN — proceed to Phase WM1** (§9). Six of six criteria pass:
a real gap on multiple physics axes (27–96% throughput, tail up to ×552),
identifiable from reward-free data including on unseen object shapes (53–69
points over the permutation control), and 55% of it recoverable when the
parameter is known.

Total: **131 formal rollouts** at 512 × 2400 with three repeats per point,
**63** screening rollouts, **8** identifiability probes, one 600-iteration
oracle fine-tune, and **17** plumbing smoke checks.

---

## 1. Subject and protocol

| | |
|---|---|
| policy | `piperx_pick_place_vision/2026-08-22_17-15-09_f3/model_1500.pt` |
| SHA256 | `fe9ecd12f5895192…` |
| what it is | the best honest-cadence vision PPO policy: 55.8 objects/min, 99.9% success, 3.7 safety trips/arm-hour |
| task | `Mjlab-Pick-Place-PiperX-Vision`, single object, per-object redraw |
| formal protocol | 512 environments × 2400 control steps, deterministic policy |
| **repeats** | **≥3 independent processes per point**, seeds 20260823 / 31415926 / 27182818 |

The DAgger student is available as a diagnostic control but is not a second
subject: it is 15% slower before any perturbation, so sweeping both would
double the cost to compare two policies that differ in more than the thing
under test.

**Why three repeats.** The Phase 0–2 audit established that this simulator is
not reproducible run to run: five identical commands — same checkpoint, same
seed, same protocol — spanned 2.9%
([`novelty_validation_phase_0_2.md` §7.5](novelty_validation_phase_0_2.md)).
A single rollout cannot support any conclusion here. Every formal point is
three processes, and every effect is reported as a **Welch interval on the
difference between two points**, which carries both points' uncertainty,
rather than as a raw percentage.

The earlier work sometimes called the max−min range a "noise floor". That was
loose and is not repeated: a range over five draws is a biased estimator of
spread, and what is quoted below is the sample standard deviation over repeats
and the interval it implies.

**A note on exit codes.** `accept_s1.py` returns 1 when the S1 *acceptance
gate* fails. Under a large perturbation that is the result, not an error, and
the sweep runner records it and carries on. Every run's JSON is written either
way.

---

## 2. What can be perturbed, and where it sits relative to training

`piper_push.perturb` adds seventeen session-persistent axes. All are inert by
default; no training or evaluation config sets any of them; the registered
tasks are byte-for-byte the tasks that were trained.

"Session-persistent" is the selection criterion and it is deliberately *not*
the object-level randomisation the policy already averages over. Shape, mass
and per-object friction turn over every object and the policy has seen their
whole range within any single run. A session parameter is one it cannot
average over — how the camera actually ended up mounted, what the depth
sensor's scale error is, how long the command path really takes.

| axis | unit | nominal | trained range | hardware meaning |
|---|---|---|---|---|
| `cam_pitch_deg` | deg | 0 | ±2 (jitter) | mount tilted in elevation |
| `cam_yaw_deg` | deg | 0 | ±2 | mount rotated in azimuth |
| `cam_pos_x_m` | m | 0 | ±0.02 | camera moved along the table axis |
| `cam_pos_z_m` | m | 0 | ±0.02 | camera raised or lowered on its post |
| `depth_scale` | × | 1.0 | **not modelled** | stereo baseline / focal length error |
| `depth_bias_m` | m | 0 | **not modelled** | constant range offset |
| `depth_dropout` | frac | 0.02 | 0–0.02 i.i.d. | share of pixels with no return |
| `depth_dropout_blob` | frac | 0 | **not modelled** | *structured* no-return patches |
| `obs_latency_steps` | steps | 0 | **not modelled** | camera + transport + inference delay |
| `action_latency_steps` | steps | 0 | **not modelled** | USB-CAN hop, driver queue |
| `joint_response_scale` | × | 1.0 | **not modelled** | fraction of each commanded step completed |
| `servo_damping_scale` | × | 1.0 | **not modelled** | kd multiplier; identified ζ ≈ 0.35 |
| `action_deadband_rad` | rad | 0 | **not modelled** | stiction, encoder quantisation |
| `gripper_rate_scale` | × | 1.0 | **not modelled** | closure speed vs the assumed 0.10 m/s |
| `gripper_latency_steps` | steps | 0 | **not modelled** | the gripper's own command path |
| `pad_friction_scale` | × | 1.0 | ≈ ±35% | finger-pad wear or a rubber change |
| `table_friction_scale` | × | 1.0 | ≈ ±43% | this table is more or less slippery |

**Nine of seventeen were not modelled in training at all** — including every
kind of latency. That is the strongest form of out-of-distribution: the
simulator was not approximately right about them, it was silently asserting
they were zero.

`gripper_rate_scale` deserves a note. `robot.py` flags the 0.10 m/s closure
rate as **not measured** and says it must be replaced with a hardware
measurement before deployment. It is a known-unknown sitting in the middle of
the grasp.

Exact levels are in `results/sim2real_sweep/s1_manifest.json`, which records
for every level whether it is inside the trained range, at its edge, or
outside it and by how much.

### 2.1 What is deliberately *not* an axis

* **Object mass and per-object friction.** Object-level DR, and the policy
  averages over them within a run. They appear only as a held-out
  generalisation control.
* **Mask or target-ID failure.** A segmentation failure is a perception bug
  with a perception fix; folding it in with physics calibration would produce
  one number that cannot be acted on. If it is measured it will be a separate
  stress test.

---

## 3. Stage S0 — plumbing (smoke)

**Smoke, not a result.** 32 environments, 40 steps, a fixed action sequence
identical across conditions.

`scripts/check_perturb.py` drives each axis to a clearly-visible setting and
checks that it changes what it should: a camera axis must move the
observation, a plant axis must move the arm.

**All 17 axes reached the simulator; verdict OK**
(`results/sim2real_sweep/plumbing_check.json`).

Three bugs it caught, none of which any unit test could have:

* `dr.pd_gains` raised on the servo-damping axis: the arm's actuators are four
  *groups* (`joint[1-3]`, `joint4`, `joint5`, `joint6`) plus the gripper, so
  `SceneEntityCfg`'s joint-name matching returned six indices into a list of
  five. Damping is now scaled on the actuator config before the entity is
  built, which is also the honest representation — a fixed plant difference is
  not a randomisation with a degenerate range.
* Actuators live under `EntityCfg.articulation`, not on `EntityCfg`.
* The check's own first version asserted a non-zero per-environment camera-pose
  spread. In play mode the jitter is zero by design, so all seventeen axes
  "failed". It now checks that the pose field carries one row per world, which
  is the failure that matters: a write landing in world 0 averages the
  perturbation away and reads as *this axis does not matter*.

A fourth, outside the simulator: `micromamba run` merges stdout and stderr, so
redirecting the sweep plan into a file captured mjlab's import warning instead
and produced an empty job list — a sweep that printed "0 jobs" and exited
successfully. The planner now writes its own file and the runner refuses an
empty one.

### 3.1 One axis reimplements something mjlab already has

`obs_latency_steps` is implemented here as a ring buffer inside a class-based
observation term. **mjlab already provides observation delay**:
`ObservationTermCfg.delay_min_lag` / `delay_max_lag`, whose own documentation
says "use min=max for constant delay". This was found after the sweep was
already running.

The results stand, because the two apply delay at the same point on the same
tensor: mjlab's pipeline is compute → noise → clip → scale → delay → history,
and the camera group has `enable_corruption=False` so the noise stage is a
no-op — both therefore delay the final term output by a constant number of
steps. The custom buffer is verified by S0 and by the reset test, and
switching mid-sweep would have invalidated the runs in flight for no gain.

**WM1 should use the native fields and delete the custom buffer.** It is
better tested, it supports a *sampled* lag range rather than only a constant
(which is what a real pipeline does), and one fewer stateful object in the
observation path is worth having. Recorded here rather than quietly fixed
because "we wrote our own delay" is exactly the kind of detail that decides
whether a later result is comparable to anyone else's.

---

## 4. Stage S1 — screening

**Screening, not formal.** 256 environments × 1200 control steps, one repeat
per level. Enough to rule an axis out; not enough to publish a number.

Two limits on how far these may be read:

* **One repeat**, so no effect here has an uncertainty and the `sep` column is
  undefined. Anything carried forward is re-measured at S2 with three.
* **Trip counts are tiny.** 256 × 1200 × 0.02 s is 1.7 arm-hours, so the
  nominal's 1.17 trips/arm-hour is *two events*. Ratios like "+1700%" are
  differences of single-digit counts and are used only to decide what to
  measure properly, never as results.

Unperturbed reference at this protocol: **56.4 objects/min**, 99.9% success,
0.43% drop, p95 1.46 s, 1.17 trips/arm-hour. (The formal protocol reads 55.8;
smaller samples run 1–2% optimistic, which is why screening and formal numbers
are never mixed.)

### 4.1 Full screen, 17 axes × 4 levels

Strongest degradation per axis, throughput and safety-shell rate, from
`results/sim2real_sweep/s1_summary.json`:

| axis | worst throughput | at | trips/arm-h | at |
|---|---:|---:|---:|---:|
| `action_latency_steps` | **−95.7%** | 4 | ×1083 | 4 |
| `cam_yaw_deg` | **−93.7%** | −4° | ×24 | +4° |
| `servo_damping_scale` | **−92.9%** | 0.50 | **×5354** | 0.50 |
| `cam_pitch_deg` | **−75.4%** | −4° | ×71 | +4° |
| `gripper_rate_scale` | **−66.9%** | 0.35 | ×5 | 0.35 |
| `pad_friction_scale` | **−36.9%** | 0.60 | ×6 | 1.20 |
| `gripper_latency_steps` | **−34.1%** | 4 | ×16 | 4 |
| `obs_latency_steps` | **−32.0%** | 4 | ×11 | 4 |
| `depth_dropout_blob` | **−30.4%** | 0.20 | ×3 | 0.20 |
| `cam_pos_x_m` | −15.5% | +0.05 | ×5.5 | +0.05 |
| `depth_dropout` | −7.5% | 0.20 | ×6.5 | 0.20 |
| `joint_response_scale` | −5.4% | 0.65 | ×4.5 | 0.80 |
| `table_friction_scale` | −2.8% | 1.30 | ×3 | 0.80 |
| `depth_scale` | −2.6% | 1.07 | ×3 | 1.03 |
| `action_deadband_rad` | −2.0% | 0.005 | ×3 | 0.010 |
| `cam_pos_z_m` | −1.7% | +0.02 | ×3.5 | +0.02 |
| `depth_bias_m` | +0.7% | +0.03 | ×2.5 | −0.01 |

### 4.2 What the screen says

**Nine of seventeen axes produce a throughput gap of ≥ 10%, and every one of
those nine is a plausible session parameter.** Broad domain randomisation has
*not* already covered this: there is a very large amount of headroom.

Four readings that shaped the formal stage.

**Timing dominates, and it was never modelled.** The two largest throughput
effects in the sweep are `action_latency_steps` (−95.7%) and `cam_yaw_deg`
(−93.7%), and three of the top eight are latencies. The simulator hands the
command over in the tick it was computed and shows the policy an image with no
delay. Four control steps is 80 ms — a USB-CAN hop plus a driver queue plus a
200 Hz inner loop. This is the single clearest instance of the simulator not
being approximately right but silently asserting zero.

**Camera *aiming* matters enormously; camera *position* barely does.** Yaw and
pitch cost 94% and 75% at 4°, while moving the camera 50 mm costs 15% (x) or
2% (z). Rotation sweeps the target across the image; translation mostly
changes parallax, which the spatial-softmax encoder is largely invariant to.

**A persistent offset costs far more than the same magnitude of jitter.** The
policy was trained with ±2° of *per-episode random* camera jitter. A
*persistent* −1.5° offset — comfortably inside that support — costs **9–11%**
of throughput. Matching a parameter's marginal distribution is not the same as
matching its hold time, and this is the sharpest example of it in the sweep.
It is also exactly the failure mode the README's hypothesis is about.

**Depth *geometry* errors are nearly free; depth *dropout* is not.** A 7%
depth scale error costs 2.6% and a 30 mm bias costs nothing measurable, while
20% structured dropout costs 30%. The policy is not reading absolute range
off the depth channel; it is reading shape and support, and holes destroy
those while a uniform rescale does not. The masked-depth representation is
doing more work than the raw depth.

**One axis is tail-only, and it is the most interesting in the sweep.**
`servo_damping_scale` at 0.75 costs 5.3% of throughput — small — while taking
the safety-shell rate from 1.17 to 212 per arm-hour, about ×180. At 0.50 the
plant is simply unstable (6275 trips/arm-hour, throughput −93%) and is not a
usable adaptation target. A calibration scored on throughput alone would rank
this axis near the bottom and miss a mismatch that would stop the robot every
17 seconds on hardware.

### 4.3 What was carried to S2, and what was not

Four axes, chosen to span four *distinct mechanisms* rather than to take the
top four by size — camera geometry, perception timing, actuation, and gripper:

| axis | mechanism | why |
|---|---|---|
| `cam_yaw_deg` | camera geometry | largest camera effect; a calibration residual is the most likely real mismatch of all |
| `obs_latency_steps` | perception timing | cleanest monotone dose-response in the sweep (−4.2 / −14.4 / −24.8 / −32.0%) |
| `servo_damping_scale` | actuation | the only axis with a tail-only regime |
| `gripper_rate_scale` | gripper | −67%, and `robot.py` flags the assumed 0.10 m/s as *not measured* |

**`action_latency_steps` was added afterwards as a fifth.** It screened as the
single largest effect and finished after the other four were already running.
Leaving the biggest effect at screening scale, where nothing carries an
uncertainty, would not have been defensible, so it was re-measured at the
formal protocol as an extension (`s2b_manifest.json`).

Deliberately **not** carried forward, and why:

* `cam_pitch_deg` and `cam_pos_x_m` — real, but the same *mechanism* as
  `cam_yaw_deg`. One representative is enough at formal cost.
* `gripper_latency_steps` — same mechanism as the other latencies.
* `depth_dropout_blob` — a strong effect (−30%), but it is a *perception*
  failure whose fix is a perception fix. Section 2.1 keeps it separate from
  physics calibration on purpose; it should be its own stress test.
* `depth_scale`, `depth_bias_m`, `cam_pos_z_m`, `action_deadband_rad` — under
  3%, which at screening scale is not distinguishable from nothing. **These
  are the axes a decision-aware method should learn to ignore**, and they are
  useful precisely for that.

## 5. Stage S2 — formal sweep

**Formal.** 512 environments × 2400 control steps, **three independent
processes per point** (seeds 20260823 / 31415926 / 27182818). `sep` is whether
the Welch interval on the difference from nominal excludes zero — that
interval carries the uncertainty of *both* points, so it is a real test and
not a comparison of point estimates.

### 5.1 Reference

| metric | mean | sd over repeats | 95% interval |
|---|---:|---:|---|
| throughput | **55.86** obj/min | 0.31 | [55.09, 56.63] |
| success | 99.83% | 0.03 | [99.75, 99.90] |
| post-grasp drop | 0.53% | 0.03 | [0.47, 0.60] |
| p95 cycle time | 1.47 s | 0.012 | [1.44, 1.50] |
| stuck time | 1.21% | 0.47 | [0.05, 2.37] |
| safety trips | 2.29 /arm-h | 0.22 | [1.74, 2.85] |

Two things worth noting. The nominal reproduces `docs/results.md`'s 55.8
objects/min exactly, which is a useful independent check on the whole
pipeline. And the repeat spread here is **0.55%** — much tighter than the 2.9%
range the Phase 0–2 determinism check found over five runs. A range over five
draws is a biased estimator of spread; the standard deviation over three
repeats is the number to use, and it is small.

### 5.2 Results

| axis | level | n | obj/min | Δ | sep | trips/arm-h | Δ | sep | p95 | drop |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| `cam_yaw_deg` | −4° | 3 | 3.01 | **−94.6%** | yes | 3.66 | +60% | no | 4.10 | 6.30% |
| | −1.5° | 3 | 50.55 | **−9.5%** | yes | 3.66 | +60% | yes | 1.82 | 0.76% |
| | +1.5° | 3 | 52.53 | **−6.0%** | yes | 5.13 | +123% | yes | 1.61 | 0.59% |
| | +4° | 3 | 8.98 | **−83.9%** | yes | 25.93 | **+1030%** | yes | 7.12 | 6.56% |
| `obs_latency_steps` | 1 | 3 | 53.26 | −4.7% | yes | 2.73 | +19% | no | 1.57 | 0.41% |
| | 2 | 3 | 48.15 | **−13.8%** | yes | 5.81 | +153% | yes | 1.82 | 0.67% |
| | 3 | 3 | 42.06 | **−24.7%** | yes | 10.30 | **+349%** | yes | 2.12 | 1.13% |
| | 4 | 3 | 37.43 | **−33.0%** | yes | 10.64 | **+364%** | yes | 2.46 | 1.52% |
| `servo_damping_scale` | 0.50 | 3 | 4.03 | **−92.8%** | yes | 6210 | **×2712** | yes | 1.53 | 0.07% |
| | 0.75 | 3 | 53.34 | −4.5% | yes | **204.3** | **×89** | yes | 1.65 | 0.48% |
| | 1.50 | 3 | 52.72 | −5.6% | yes | 1.32 | −43% | no | 1.50 | 0.75% |
| | 2.00 | 3 | 47.63 | **−14.7%** | yes | 3.37 | +47% | no | 1.64 | 1.08% |
| `gripper_rate_scale` | 0.35 | 3 | 17.86 | **−68.0%** | yes | 5.18 | +126% | no | 5.86 | 1.80% |
| | 0.50 | 3 | 30.18 | **−46.0%** | yes | 6.40 | +179% | yes | 3.40 | 1.33% |
| | 0.70 | 2 † | 44.52 | **−20.3%** | yes | 3.96 | +72% | no | 2.10 | 0.65% |
| | 1.50 | 3 | 55.03 | −1.5% | **no** | 4.54 | +98% | no | 1.59 | 1.21% |
| `action_latency_steps` | **1** | 3 | 40.78 | **−27.0%** | yes | **69.4** | **×30** | yes | 2.15 | 1.81% |
| | 2 | 3 | 17.04 | **−69.5%** | yes | 378.6 | **×165** | yes | 3.39 | 4.75% |
| | 3 | 3 | 6.54 | **−88.3%** | yes | 844.6 | **×369** | yes | 4.24 | 5.07% |
| | 4 | 3 | 2.47 | **−95.6%** | yes | 1264.7 | **×552** | yes | 4.05 | 4.33% |

† one repeat of this point did not complete; it is reported at n=2 rather than
dropped or silently averaged as if it were three.

### 5.3 What S2 establishes

**All five axes separate, and four of them by a very large margin.** The
screening result was not an artefact of one repeat: `cam_yaw_deg`,
`obs_latency_steps`, `servo_damping_scale`, `gripper_rate_scale` and
`action_latency_steps` all produce throughput effects that clear their own
uncertainty, at levels a real deployment could plausibly land on.

**The persistent-versus-jitter result survives the formal protocol.** A
*fixed* camera yaw of ±1.5° — inside the ±2° of *per-episode random* jitter
the policy was trained across — costs **9.5%** and **6.0%**, both separated.
This is the sharpest single result in Phase WM0 for the README's hypothesis:
the marginal distribution was matched and the deployed performance still
moved, because deployment holds the value fixed and training did not.

**One axis is tail-only, and it is the strongest argument for
decision-aware calibration in the whole sweep.** `servo_damping_scale` at
0.75 costs **4.5%** of throughput — small, but separated — while taking the
safety-shell rate from 2.29 to **204.3 per arm-hour, a factor of 89**. That is
a trip every 18 seconds. A calibration objective built on throughput, task
reward, or pixel reconstruction would rank this mismatch as almost harmless.
On hardware it would stop the robot continuously.

**Latency has the cleanest dose-response in the sweep** (−4.7 / −13.8 / −24.7
/ −33.0% at 1–4 steps, every level separated), and it is a quantity the
simulator currently asserts is exactly zero.

**A single control step of *action* latency — 20 ms — costs 27% of throughput
and multiplies the safety-shell rate by thirty.** This is the largest and
most alarming result in Phase WM0, and it deserves separating from the
others:

| action latency | obj/min | Δ | trips/arm-h | factor |
|---|---:|---:|---:|---:|
| 0 (as simulated) | 55.86 | — | 2.29 | — |
| **1 step (20 ms)** | 40.78 | **−27.0%** | **69.4** | **×30** |
| 2 (40 ms) | 17.04 | −69.5% | 378.6 | ×165 |
| 3 (60 ms) | 6.54 | −88.3% | 844.6 | ×369 |
| 4 (80 ms) | 2.47 | −95.6% | 1264.7 | ×552 |

**Action latency is roughly six times more destructive than observation
latency at the same delay** (−27.0% against −4.7% at one step). That
asymmetry has a mechanism, and it is the same one Phase 0–2 identified for the
safety shell: the policy rides its rate limiter — 89% of commands are clipped
by the slew ceiling — and the identified plant is underdamped at ζ ≈ 0.35. A
delayed command arrives when the arm is no longer where the command assumed,
and an underdamped plant answers the resulting error by overshooting into the
shell. Delaying what the policy *sees* costs it information; delaying what the
arm *does* breaks the closed loop.

The engineering consequence is blunt and does not depend on any of the
calibration work: **the real command path's latency must be measured and
modelled before this policy is deployed at all.** At 20 ms — a plausible
USB-CAN round trip — the policy is already losing a quarter of its throughput
and tripping the safety shell every 52 seconds. This is not a calibration
problem to be solved by a posterior; it is a missing term in the simulator,
and §11C says so.

**Not everything matters, and that is useful.** `gripper_rate_scale = 1.5` —
a *faster* gripper than assumed — does not separate (−1.5%). `servo_damping
= 1.5` costs 5.6% of throughput but *reduces* the trip rate 43%. A
decision-aware method must learn to spend no real interaction on these, which
is exactly what distinguishes it from matching every parameter equally well.

## 6. Stage S3 — interaction

Two grasp-side axes, chosen because they are mechanistically coupled: a slower
gripper and less pad friction both degrade the same event. 3 × 3 cells, three
repeats each, full formal protocol. The reference is the (1.0, 1.0) cell,
55.80 obj/min (sd 0.27) — which agrees with the S2 nominal's 55.86 to within a
tenth, an incidental but welcome cross-check.

The question is **not** whether the corner is bad. It is whether the corner is
*predictable from the two edges*. If it is, single-axis calibration composes
and each parameter can be estimated on its own. If it is not, a
one-at-a-time posterior is wrong in a way more data does not fix.

### 6.1 The grid

Throughput, objects/min. Rows `gripper_rate_scale`, columns
`pad_friction_scale`:

| | 0.6 | 0.8 | 1.0 |
|---|---:|---:|---:|
| **0.5** | **11.23** | 24.60 | 30.30 |
| **0.7** | 23.11 | 39.42 | 44.16 |
| **1.0** | 34.92 | 52.06 | **55.80** |

### 6.2 Neither composition rule predicts the corner

| cell | observed | additive | multiplicative | obs − mult |
|---|---:|---:|---:|---:|
| 0.5, 0.6 | 11.23 | 9.42 | 18.96 | **−7.73** |
| 0.5, 0.8 | 24.60 | 26.57 | 28.27 | −3.67 |
| 0.7, 0.6 | 23.11 | 23.28 | 27.63 | −4.52 |
| 0.7, 0.8 | 39.42 | 40.42 | 41.20 | −1.78 |

The natural null for two independent degradations of a *rate* is the
multiplicative one, and **it over-predicts every cell** — by 7.73 objects/min
at the corner, which is 69% of the observed value there. All four cells fall
outside the observed cell's confidence interval. (That interval covers only
the observed cell's repeat spread, not the edges', so this is a conservative
comparison in one direction and should not be read as a formal interaction
test.)

### 6.3 The marginal effect of an axis depends on the other axis

The same numbers, read as "what does halving the gripper rate cost?":

| pad friction | gripper 1.0 → 0.5 |
|---|---:|
| 1.0 (nominal) | −45.7% |
| 0.8 | −52.7% |
| 0.6 | **−67.8%** |

and symmetrically, "what does dropping pad friction to 0.6 cost?":

| gripper rate | friction 1.0 → 0.6 |
|---|---:|
| 1.0 (nominal) | −37.4% |
| 0.7 | −47.7% |
| 0.5 | **−62.9%** |

**Each axis's marginal effect grows substantially as the other degrades.**
Measuring the gripper-rate effect on a well-calibrated table understates it by
a third relative to the same measurement on a slippery one.

### 6.4 What this means for the method

This is a constraint on WM1, and a useful one:

* **A per-axis posterior is not sufficient** for coupled parameters. The
  discrepancy has to be evaluated on a *joint* parameter vector, at least
  within a coupled group. Fitting gripper rate with friction held at its prior
  mean gives a biased estimate.
* **It does not condemn the approach** — it argues for the decision-aware
  formulation over a state-matching one. A discrepancy defined on policy
  consequences sees the compounded effect directly; a per-parameter state
  error would not, because each parameter's own residual can look small while
  their joint effect on the grasp is large.
* **The coupled group is small.** Grasp physics couples; there is no evidence
  here that camera yaw couples to servo damping. So the joint fit needs to be
  joint over a handful of related parameters, not over all seventeen.

One caveat kept in view: only *one* pair was tested, chosen because coupling
was expected. That it interacts is evidence about grasp physics, not a general
claim that every pair of axes does. Testing camera × latency, which S1's
levels also support, is a cheap follow-up and is in the commands in §10.

## 7. Reward-free identifiability

The calibration idea only works if a session's mismatch leaves a signature in
what the robot can actually record. This measures whether it does — and, more
importantly, whether the signature *generalises* past the objects it was
measured on.

### 7.1 What the probe may and may not see

The frozen policy runs its ordinary rollout; nothing about the task changes.
From each rollout, overlapping 1-second windows (50 control steps, stride 25)
are cut and summarised per channel by mean, standard deviation, min, max, last
value, and mean absolute first difference.

**Inputs**, in three sets of increasing privilege — all of which a real robot
could log:

| set | contents |
|---|---|
| `proprio_servo` | joint positions, joint velocities, gripper position, gripper servo error, issued action |
| `encoded_proprio` | the policy's convolutional encoding of the depth image, concatenated with proprioception, before the recurrent layer |
| `actor_latent` | the GRU output, the action, and proprioception |

**Excluded from every set**, because none is available before the fact on
hardware:

* reward, return, success, or placement labels;
* contact flags, object pose, object mass, table friction — anything
  privileged;
* the perturbation value itself, which is the label being predicted.

**Phase and step index are recorded and never fed in.** Knowing "this window
is a grasp" is not available in advance on hardware, and a probe leaning on it
would be reading the task schedule rather than the plant.

### 7.2 Leakage controls

In order of how badly each would flatter the result if omitted:

1. **Whole environments go to one side of the split.** Windows overlap and
   neighbouring windows are near-copies; a random split over windows would
   report autocorrelation as accuracy. This is the same error the Phase 0–2
   hidden-state probe was built to avoid.
2. **Held-out object shapes** (`--holdout-shape`): the test set is restricted
   to shape classes absent from training, so the probe cannot succeed by
   memorising what the calibration objects looked like. A real session is not
   a re-run of the calibration objects.
3. **Permutation control**: the identical pipeline on shuffled labels. That is
   the floor a real signal has to clear, and it is reported next to every
   result.
4. **Chance is the majority-class rate**, not `1/K`.

A **nearest-centroid** classifier is reported beside the logistic regression.
It has no capacity to overfit, so a large gap between the two says the
signature is present but not linearly separable — which is a different
engineering problem from the signature being absent.

### 7.3 Results

Four domains, three levels each, 96 environments × 600 steps. Chance is the
majority-class rate on the held-out set; "shuffled" is the identical pipeline
on permuted labels. Best feature set per domain in bold.

| domain | feature set | accuracy | balanced | centroid | chance | shuffled |
|---|---|---:|---:|---:|---:|---:|
| `cam_yaw_deg` | proprio + servo | 54.4% | 54.4% | 45.8% | 33.3% | 33.3% |
| | **encoded + proprio** | **99.3%** | 99.3% | 80.6% | 33.3% | 30.1% |
| | actor latent | 86.9% | 86.9% | 55.9% | 33.3% | 34.4% |
| `servo_damping_scale` | **proprio + servo** | **99.7%** | 99.7% | 96.5% | 33.3% | 32.8% |
| | encoded + proprio | 99.5% | 99.5% | 65.8% | 33.3% | 34.2% |
| | actor latent | 98.9% | 98.9% | 79.2% | 33.3% | 34.4% |
| `gripper_rate_scale` | proprio + servo | 83.6% | 83.6% | 63.5% | 33.3% | 32.8% |
| | encoded + proprio | 86.6% | 86.6% | 52.1% | 33.3% | 33.8% |
| | **actor latent** | **89.0%** | 89.0% | 59.7% | 33.3% | 35.8% |
| `obs_latency_steps` | proprio + servo | 75.5% | 75.5% | 57.7% | 33.3% | 32.4% |
| | encoded + proprio | 77.4% | 77.4% | 52.9% | 33.3% | 33.8% |
| | **actor latent** | **86.5%** | 86.5% | 57.6% | 33.3% | 30.8% |

And the same with **object shapes held out entirely** — the test set contains
only classes absent from training:

| domain | best set | accuracy | chance | shuffled |
|---|---|---:|---:|---:|
| `cam_yaw_deg` | encoded + proprio | **97.4%** | 34.2% | 33.4% |
| `servo_damping_scale` | proprio + servo | **100.0%** | 35.3% | 34.4% |
| `gripper_rate_scale` | actor latent | **95.8%** | 28.3% | 30.1% |
| `obs_latency_steps` | actor latent | **85.6%** | 34.9% | 32.1% |

### 7.4 What this establishes

**Every domain is identifiable from reward-free data, by a wide margin.** The
best set clears its shuffled control by 53–69 percentage points in every case.

**The permutation control behaves.** Every shuffled run lands at 30–37%,
i.e. at chance, across all 24 fits. That is the check that matters most: it
says the pipeline is not leaking through the environment grouping, the window
overlap, or the summary statistics. Had the split been wrong, the shuffled
control would have risen with it.

**The signature survives held-out objects — and in three of four domains it
gets *better*.** `gripper_rate_scale` goes from 89.0% to 95.8%, `cam_yaw_deg`
holds at 97.4%. The probe is not memorising what the calibration objects
looked like; it is reading the plant. This is the result that would have been
easiest to get wrong and is the one the method most depends on.

**Different mismatches sign themselves in different deployable channels, and
between them the three channels cover all four domains:**

| domain | needs | why it makes sense |
|---|---|---|
| `servo_damping_scale` | **proprioception alone** (99.7%) | damping is a pure dynamics quantity and is written directly into the joint trajectory; the camera adds nothing |
| `cam_yaw_deg` | **the image encoding** (99.3% against 54.4% from proprioception) | an aiming error exists only in the image; the arm's own trajectory barely knows about it |
| `gripper_rate_scale`, `obs_latency_steps` | **the actor latent** (89.0%, 86.5%) | both are closed-loop timing effects; the policy's own recurrent state is where the compensation shows up |

That split is useful design information rather than a curiosity. A calibration
system does not need one universal feature — it needs proprioception, the
encoder output, and the policy latent, and the cheapest of the three suffices
for the axis with the worst tail behaviour.

**Nearest centroid trails the regression substantially** (80.6% vs 99.3% on
`cam_yaw`; 52.1% vs 86.6% on `gripper_rate` with the encoder). The signature
is present but is not a simple shift of the feature mean. A learned
discriminator is needed — a linear one is enough — which is exactly the regime
a small parameter-conditioned model is suited to and would be bad news only if
the reverse were true.

### 7.5 What this does *not* show

Three limits, stated because the result is strong enough to be over-read:

* This is **classification among three known levels**, not posterior inference
  over a continuous parameter. Telling 0, 2 and 4 steps apart is easier than
  estimating 2.7 steps with a calibrated uncertainty.
* Each domain was probed **one axis at a time**. §6 showed two grasp-side axes
  interact, so a joint probe over a coupled group is the honest next test and
  is not run here.
* The features come from **simulated** rollouts on both sides. A real
  signature could differ in ways no simulator-to-simulator experiment can
  reveal — which is exactly what §11D's ten minutes of hardware logging is
  for.

## 8. Oracle ceiling and recoverability

### 8.1 What the oracle is and is not

The oracle is **handed the target parameter**. It is not a method; it is the
bound a method would be trying to reach. Four quantities:

| | |
|---|---|
| `J_zero_shot` | the deployed policy in the target domain, unadapted |
| `J_oracle` | the same policy fine-tuned *in* the target domain with θ known, measured there |
| `J_retention` | that fine-tuned policy back in the nominal domain |
| `J_nominal` | the deployed policy in the nominal domain — 55.86 obj/min |

A learned calibration is later scored as

```
recovery = (J_adapted − J_zero_shot) / (J_oracle − J_zero_shot)
```

so **if `J_oracle` does not clear `J_zero_shot`, there is no ceiling to aim at**
and the direction is answered in the negative for that domain, regardless of
how well the parameter could be identified. Phase WM0 measures only the two
ends; there is no learned calibration yet.

Fine-tuning starts from `f3/model_1500` with `--resume`, not `--student`: it
is a PPO checkpoint and already carries a trained critic, so there is no
randomly initialised value function to destroy the actor with. Hyperparameters
are `finetune.py`'s fine-tuning defaults, which exist because the
from-scratch ones destroyed a distilled policy in a single optimiser step.

### 8.2 Cost, and why it is the interesting part

The observed rate is **≈ 8.5 s per PPO iteration** at 512 environments on one
RTX 6000D with the camera in the loop. 600 iterations is therefore about
**85 minutes of GPU time for one domain**.

That is worth stating plainly because the README's target is a **≤ 60 minute
total adaptation budget**, and the oracle — which is given the answer for
free — only just fits inside it. Any real method has to pay for data
collection and posterior fitting *as well*, out of the same hour. So the
budget constraint is not slack: it is close to binding before a single line of
world model has been written, and WM1 should be designed against a
fine-tuning budget of roughly 300–450 iterations rather than an open-ended
one.

### 8.3 Result — `obs_latency_steps = 3` (60 ms)

600 PPO iterations (1500 → 2099), three repeats of each measurement. The
repeats are tight: zero-shot spans 41.85–42.40, oracle 49.52–49.86.

| | obj/min | success | trips/arm-h | p95 |
|---|---:|---:|---:|---:|
| `J_nominal` — unperturbed policy, unperturbed domain | 55.86 | 99.8% | 2.29 | 1.47 |
| `J_zero_shot` — unperturbed policy, 60 ms delay | 42.19 | 99.1% | 8.69 | 2.14 |
| **`J_oracle`** — fine-tuned with θ known, same domain | **49.72** | 99.7% | **4.83** | 1.71 |
| `J_retention` — that policy back in the nominal domain | 50.75 | 99.5% | 9.08 | 1.85 |

```
recovery = (49.72 − 42.19) / (55.86 − 42.19) = 55.1% of the gap
```

**The gap is substantially recoverable.** Knowing the parameter closes **55%**
of the 13.7-object gap, and it closes the *safety* gap further still: trips
fall from 8.69 to 4.83 per arm-hour, most of the way back to the unperturbed
2.29. p95 cycle time returns almost entirely (2.14 → 1.71 against 1.47).

Three things this does *not* say, all of which matter for WM1:

**Recovery is partial, not complete.** 45% of the gap survives an oracle that
was handed the answer and given 600 iterations. Some of a 60 ms delay is
simply not compensable by a policy of this architecture at this budget — so
the README's 70–80% recovery target is a target *against `J_oracle`*, not
against `J_nominal`, and stating it the other way would be promising something
the ceiling itself does not reach.

**Adapting costs 9.1% back home.** `J_retention` is 50.75 against the original
55.86: the adapted policy has learned to compensate for a delay that is not
there, and pays for it whenever it is not there. That is the price of
specialisation and it is exactly what the `α q(θ|D) + (1−α) p_broad(θ)` mixture
in the README is for — the retention number is the thing that mixture has to
protect, and it now has a measured value to be judged against.

**It is one domain.** The perception/geometry, actuation and gripper domains
have reproducible commands in §10 and were not run; the oracle alone is ~85
GPU-minutes per domain and the wall clock went to the sweep instead. The Green
verdict rests on recoverability being demonstrated *somewhere*, which the gate
requires, not everywhere.

## 9. Gate verdict — **GREEN**

Evaluated mechanically by `scripts/wm0_gate.py` from the machine-readable
summaries, not by eye. Thresholds are in the script and in
`results/sim2real_sweep/gate.json`.

| # | criterion | verdict | evidence |
|---|---|---|---|
| 1 | ≥ 2 factors separated from repeat uncertainty | **PASS** | §5: all five S2 axes separate at 3 repeats |
| 2 | ≥ 10% throughput gap or ≥ 2× tail | **PASS** | §5: 27–96% throughput on four axes; tail up to ×552 |
| 3 | not only segmentation/mask failure | **PASS** | qualifying axes are latency, servo damping, gripper rate, camera geometry — all physics; depth dropout was deliberately excluded from S2 (§4.3) |
| 4 | reward-free history beats its permutation control | **PASS** | §7: 53–69 pp over shuffled in all four domains; all 24 shuffled fits at chance |
| 4b | … and survives held-out object shapes | **PASS** | §7.3: 85.6–100% on unseen shape classes |
| 5 | oracle closes ≥ 30% of the zero-shot gap | **PASS** | §8.3: **55.1%** of a 13.7-object gap, plus most of the safety gap |

**Verdict: GREEN — proceed to Phase WM1.** There is a real gap on multiple
*physics* axes, it is identifiable from reward-free data including on unseen
objects, and it is substantially recoverable when the parameter is known.

### 9.1 What the Green rests on, and what it does not

Green means *the premises of the direction hold in simulation*. It does not
mean the method will work. Four things it explicitly does not establish:

* **No hardware.** Everything is simulator-against-simulator. The real
  signature could differ in ways this cannot reveal (§11D).
* **Classification, not inference.** §7 separates three known levels; a
  posterior over a continuous parameter with calibrated uncertainty is a
  harder problem and is what WM1 actually needs.
* **One oracle domain.** §8.3 measured recoverability for latency only.
* **Coupled axes need joint treatment.** §6 showed two grasp-side parameters
  interact strongly enough that a per-axis posterior would be biased.

### 9.2 The two results that would matter even if WM1 fails

**A persistent offset is not the same as jitter of the same size.** The policy
trained across ±2° of per-episode random camera yaw loses 9.5% to a *fixed*
−1.5° offset (§5.3). Matching a parameter's marginal distribution is not
matching its hold time. This is a statement about how domain randomisation is
specified, and it does not depend on any calibration method existing.

**Decision-consequence and state error come apart, measurably.**
`servo_damping_scale = 0.75` costs 4.5% of throughput and multiplies the
safety-shell rate by 89 (§5.3). Any calibration objective built on throughput,
task reward, or pixel/state reconstruction would rank that mismatch as nearly
harmless; on hardware it stops the arm every 18 seconds. That is the concrete
case for calibrating on decision consequences, and it came out of a sweep
rather than out of an argument.

### 9.3 One engineering finding that should not wait for WM1

**A single control step — 20 ms — of *action* latency costs 27% of throughput
and multiplies safety trips by 30** (§5.3). The simulator currently asserts
command latency is exactly zero. This is a missing simulator term, not a
posterior to fit, and it should be measured on the real command path and added
to domain randomisation before any deployment or any further calibration work
(§11C).

## 10. Commands

Every run was launched on `shen-teacher` inside `tmux` with `setsid nohup`, so
a dropped VPN cannot take a run with it — it has happened twice in this
project. Two environment settings are required for any non-interactive
invocation and are baked into `scripts/run_sim2real_sweep.sh`:

```bash
export MUJOCO_GL=disable                       # mujoco initialises a GL backend it never uses
export LD_LIBRARY_PATH=$MAMBA_ROOT/envs/mjlab/lib:$LD_LIBRARY_PATH
```

The second is not optional: the env ships `libicui18n.so.78`, which needs
`CXXABI_1.3.15`, and the system `libstdc++` does not have it. It breaks only
non-interactive runs, inside mjlab's own import of `mediapy → IPython →
sqlite3`.

**S0 — plumbing (smoke):**

```bash
python scripts/check_perturb.py --device cuda:0 --num-envs 32 --steps 40 \
  --json results/sim2real_sweep/plumbing_check.json
```

**S1 — screening**, 17 axes × 4 levels, 256 × 1200, one repeat:

```bash
python scripts/sweep_plan.py s1 \
  --out s1.jobs --manifest results/sim2real_sweep/s1_manifest.json
OUT=results/sim2real_sweep/s1 NUM_ENVS=256 STEPS=1200 \
  scripts/run_sim2real_sweep.sh "0 1 2 3 4 5 6 7" s1.jobs
```

**S2 — formal**, 512 × 2400, three repeats (seeds 20260823 / 31415926 /
27182818):

```bash
python scripts/sweep_plan.py s2 \
  --axes cam_yaw_deg,obs_latency_steps,servo_damping_scale,gripper_rate_scale \
  --out s2.jobs --manifest results/sim2real_sweep/s2_manifest.json
OUT=results/sim2real_sweep/s2 NUM_ENVS=512 STEPS=2400 \
  scripts/run_sim2real_sweep.sh "0 1 2 3 4 5 6 7" s2.jobs

# the fifth axis, added after screening finished
python scripts/sweep_plan.py s2 --axes action_latency_steps \
  --out s2b.jobs --manifest results/sim2real_sweep/s2b_manifest.json
grep -v '^nominal__' s2b.jobs > s2b_only.jobs     # nominal already measured
OUT=results/sim2real_sweep/s2 NUM_ENVS=512 STEPS=2400 \
  scripts/run_sim2real_sweep.sh "0 1 2 3 4 5 6 7" s2b_only.jobs
```

**S3 — interaction**, 3 × 3 on the two grasp-side axes:

```bash
python scripts/sweep_plan.py s3 \
  --pair gripper_rate_scale:pad_friction_scale --pair-levels "1,0.7,0.5;1,0.8,0.6" \
  --out s3.jobs --manifest results/sim2real_sweep/s3_manifest.json
OUT=results/sim2real_sweep/s3 NUM_ENVS=512 STEPS=2400 \
  scripts/run_sim2real_sweep.sh "0 1 2 3 4 5 6 7" s3.jobs
```

**Identifiability:**

```bash
python scripts/identifiability.py --axis obs_latency_steps --levels 0,2,4 \
  --num-envs 96 --steps 600 --device cuda:0 \
  --json results/sim2real_sweep/ident/obs_latency.json
# and the same with --holdout-shape for the generalisation split
```

**Oracle ceiling** — one domain in full, and the exact commands for the others:

```bash
# perception / timing  (run in full)
scripts/oracle_ceiling.sh obs_latency_steps 3 obslat3 0 2100

# perception / geometry
scripts/oracle_ceiling.sh cam_yaw_deg -1.5 camyaw15 1 2100
# actuation
scripts/oracle_ceiling.sh servo_damping_scale 0.75 damp075 2 2100
# gripper / contact
scripts/oracle_ceiling.sh gripper_rate_scale 0.5 griprate05 3 2100
```

**Analysis** — every table and figure in this document, from the JSONs only:

```bash
python scripts/analyze_sweep.py --stage s1 --stage s2 --stage s3 --json
python -m pytest tests -q
```

## 11. Next minimal experiment

*(ordered; the verdict in section 9 decides which of these is actually next)*

**A. If Green — the smallest useful world model, not the general one.**
Do *not* start with a parameter-conditioned latent dynamics model over all
seventeen axes. Start with the two or three that S2 showed matter, a
discrepancy built from the actor latent and action only (§7 shows what a
deployable `V_obs` would require and the current critic cannot be it), and
ask one question: **does a posterior fitted on 5–10 minutes of reward-free
data in the target domain put more mass near the true parameter than the
broad prior does?** That is a two-day experiment and it fails fast.

**B. Regardless of verdict — an observation-only value/risk head.**
The discrepancy in the README has four terms and only two are currently
computable on hardware. A `V_obs` and `C_obs` trained and frozen in simulation
against the *deployable* observation would make the other two available, and
it is useful on its own as a runtime risk monitor. It does not depend on any
of the WM0 results.

**C. Regardless of verdict — model the latencies.**
S1 makes this unavoidable: `action_latency_steps` is the largest single effect
in the sweep and the simulator currently asserts it is zero. Adding a latency
*distribution* to training domain randomisation is a day's work and would move
the policy's operating point before any calibration exists. It also changes
what WM1 would be for — calibrating within a modelled range is a much easier
problem than extrapolating outside an unmodelled one.

**D. The measurement this round could not make.** Everything here is
simulator-against-simulator. The first hardware step is not a policy
deployment: it is **10 minutes of reward-free logging on the real arm** —
joint states, issued commands, servo error, depth — with the frozen policy
driving. That data alone would show whether the real signature resembles any
of the simulated ones, and it costs an afternoon rather than a rig-safety
review.

**Not worth doing next:** a full world model over all seventeen axes; any
attempt to identify the parameter *values* precisely rather than their
decision consequences (§7 measures the latter deliberately); and anything that
treats parameter-identification accuracy as if it were task recovery — §8 is
in the report precisely because those two can come apart.
