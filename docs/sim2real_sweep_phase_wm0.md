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

*(in progress)*

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

*(pending)*

## 11. Next minimal experiment

*(pending)*
