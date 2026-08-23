# Phase WM0: is there a sim-to-real gap worth calibrating?

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

*(Sections fill in as stages complete; the verdict is section 8.)*

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

*(pending)*

## 6. Stage S3 — interaction

*(pending)*

## 7. Reward-free identifiability

*(pending)*

## 8. Oracle ceiling and recoverability

*(pending)*

## 9. Gate verdict

*(pending)*

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

*(pending)*
