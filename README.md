# PiPER-X tabletop tidying — vision policy on a D455 + PiPER-X

An AgileX PiPER-X clears objects from a table into a bin, continuously: pick
one up, put it in, the table refills. The deployed policy sees the scene
through a single fixed third-person depth camera and writes six joint position
targets plus a gripper command at 50 Hz. There is no IK, no scripted primitive,
and nothing on the robot that reads object state.

Built on [mjlab](https://pypi.org/project/mjlab/) (MuJoCo-Warp on GPU) with
rsl_rl, so thousands of environments run in parallel.

> The retired cube-pushing task this repository started as, and the IK-driven
> staged push primitive before it, live on the `piperx/cube-policy-v2` and
> `piperx/puck-sustained-push` branches.

---

## What exists today

All of this is built, measured and reproducible in simulation. **It has also
run on hardware**: on 2026-09-01 a trained vision policy picked objects off the
table and placed them in the bin on the PiPER-X, from the calibrated D455
alone. That is a first working run, not a solved problem -- the same day
produced the list of things that are still wrong in
[Lessons from the first hardware runs](#lessons-from-the-first-hardware-runs).

| | |
|---|---|
| Simulation | broad domain randomisation over object shape, size, mass, centre of mass, friction, spawn pose; camera pose jitter; per-object redraw |
| Teacher | privileged-state MLP: object pose, velocity, shape, mass, table friction |
| Student | masked third-person depth, 224×168, three channels (scene depth, target mask, masked depth), spatial-softmax CNN + GRU. No "am I holding it" flag — it gets gripper servo error instead, because nothing on the real arm can measure the flag |
| Training | student-rollout DAgger, then PPO fine-tuning with the critic initialised from the state critic |
| Task | continuous tabletop tidying, single object and three-object clutter |
| Deployment | ONNX and TorchScript export verified to 3e-6 over 8 recurrent steps; explicit hidden state, no stateful ops in the graph |

The depth sensor is no longer a guess. Two placeholder constants in
`camera.py` have been replaced by a model fitted to a RealSense D405 on a
bench, and the pipeline that feeds the policy from a real one is in
[`hardware/deploy/`](hardware/deploy/README.md), checked stage by stage against
the simulator with no hardware attached. What was measured, what it changed,
and what it invalidates:
[`docs/depth_sensor_and_deployment.md`](docs/depth_sensor_and_deployment.md).

Measured results and their caveats are in [`docs/results.md`](docs/results.md);
the audit that re-measured them (and retracts two of its findings) is
`docs/novelty_validation_phase_0_2.md` on the `yf/wm-ra-research` branch. Two
conclusions from that audit constrain everything below:

* **The simulator is not reproducible run to run.** Five identical commands
  span 2.9%. Single rollouts do not support conclusions here; every quoted
  number should be a median over ≥3 independent processes with the spread.
* **A recurrent policy exploits within-episode invariants**, but *not* by
  carrying facts about previous objects — that mechanism was tested three ways
  and refuted.

## Research track (moved to `yf/wm-ra-research`)

The reward-free, decision-aware sim-to-real calibration line (WM0 / WM1 / RA-Sim /
RA-HW) no longer lives on this branch. Its documents, scripts, models, results and
tests are on the branch `yf/wm-ra-research`, frozen at the same commit this branch
was cut from. It stopped on 2026-08-27 under its own pre-registered rules; nothing
below is a hardware result.

Status as the phase documents on that branch actually record it (an earlier
version of this table quoted WM1-A as YELLOW with numbers the document itself
later withdrew):

| phase | verdict | where (on `yf/wm-ra-research`) |
|---|---|---|
| WM0 | GREEN, 6/6 | `docs/sim2real_sweep_phase_wm0.md` |
| WM1-A | **RED** after the 8-seed extension: G3 and G4 fail, recovery 1.02 -> 0.58 | `docs/wm1_latency_vertical_slice.md` |
| WM1-B | **RED**: risk-aware score worse (C2 1.26 [1.07, 1.48]); a shuffled-label control beats the head | `docs/wm1_damping_tail_risk.md` |
| RA-Sim-0 | **RED**: the learned residual is 1.28-1.50x the parametric fit | `docs/ra_sim0_results.md` |
| RA-Sim-1 | **RED** (G1, G5) | `docs/ra_sim1_results.md` |
| RA-HW-0 | **BLOCKED**: no CAN interface on the audit host, 19 items UNKNOWN | `results/ra_hw0/README.md` |

What stays here, because the hardware line imports it: `perturb.py` (opt-in session
mismatch overlay, read by `env_cfg.py` and `hardware/deploy/run.py`), `latency.py`
(the robust task's observation-delay draw), `damping.py` + `prior.py`, and
`hidden_plant.py` + `residual.py` (optional flags of `accept_s1.py` / `finetune.py`,
inert by default). Their docstrings still cite the phase documents by path; read
them on the research branch.

---

## Task

> **Action convention (2026-09-05).** The policy emits u from a Gaussian head
> (`piper_push.squashed`); the action term applies a = tanh(u) and makes a = ±1
> the safe target clip (`piper_push.robot.BOUNDED_ARM_SCALE/OFFSET`).  PPO's
> density is evaluated on the stored u and never on a value recovered from a;
> the `actions` observation, the smoothness penalties, the distillation loss
> and the deploy mapper all work on tanh(u).  Before that,
> nothing bounded a and the arm scales spanned a quarter of the clip: the
> teachers ran the gripper at −28 and joint 4 at −3.5 (`results/audit_20260904/
> gripper_action_*.json`), the smoothness penalties spent a third of their
> weight on a channel the jaw cannot distinguish, and the distillation loss
> regressed on it.  Every checkpoint from before that date -- v3–v9 teachers
> and students, the deployed `d455_v4_final` -- was trained under the old
> convention and evaluates only on the `-V1` task ids
> (`Mjlab-Pick-Place-PiperX{,-Robust,-Vision,-Vision-Robust,-Distill,-Distill-Robust}-V1`);
> the default ids raise at the first step if such a policy is loaded.  Exported
> specs carry an `action_spec` block and the deploy mapper follows it, so v4
> keeps driving the arm it was trained on.

| | |
|---|---|
| Actor observation | proprioception (joint positions and velocities, end-effector pose, gripper opening, pad contacts, gripper servo error, last action) + the 3-channel depth image |
| Critic observation | the above, uncorrupted, plus object pose/velocity/shape and privileged physics — training only |
| Action | 6 joint position targets + gripper, the policy emits u, the action term applies tanh: a = ±1 is the safe target clip on every joint and 0–50 mm on the jaw. Rate-limited to 0.62 × the safety-shell trip speed and interpolated across physics substeps |
| Objects | five shape classes, 25–45 mm wide, 24–90 mm tall, 50–400 g, friction 0.4–1.0, redrawn **per object** |
| Episode | fixed length; success never ends it, so throughput is rewarded directly |

## Setup

```bash
git clone --recurse-submodules <this repo> && cd LivingTwin
git submodule update --init

micromamba create -y -n mjlab -c conda-forge python=3.11 pip
micromamba run -n mjlab pip install "mjlab==1.6.0"
micromamba run -n mjlab pip install -e .
micromamba run -n mjlab list-envs --keyword Pick
```

Two settings any non-interactive run needs, both baked into `scripts/eval.sh`
and `scripts/run_sim2real_sweep.sh`:

```bash
export MUJOCO_GL=disable                       # mujoco initialises a GL backend it never uses
export LD_LIBRARY_PATH=$MAMBA_ROOT/envs/mjlab/lib:$LD_LIBRARY_PATH
```

The second is not optional — the env ships `libicui18n.so.78`, which needs
`CXXABI_1.3.15`, and the system `libstdc++` does not have it.

## Evaluate

```bash
scripts/eval.sh Mjlab-Pick-Place-PiperX-Vision <checkpoint.pt> my_run
```

512 environments × 2400 control steps, deterministic policy, honest per-object
randomisation. Writes metrics, per-environment totals and full provenance
(commit, dirty diff, checkpoint SHA256, library versions) to JSON. Run it at
least three times and quote the spread; see the reproducibility note above.

## Tests

```bash
micromamba run -n mjlab python -m pytest tests -q
```

## Running the policy on the arm

The best run so far is `recordings/v4_stereo_try3` on 2026-09-01: it placed
more objects in the bin than any other, and it ended not because the policy
failed but because the log writer could not keep up. Its arguments, with the
fixes that landed after it:

For an A/B run that changes **only the target carrier to SAM2.1**, keep the
best run's perception and policy recipe: stereo depth, no target lifecycle,
and no held-target reconstruction.  `--no-record-compress` is the sole
operational difference; it prevents the recorder from ending a good run and
does not change the camera tensor.  Start with 20 seconds so one failed reach
does not turn into repeated contact:

```bash
micromamba run -n mjlab python -m hardware.deploy.run \
    --policy hardware/deploy/policies/d455_v4_final \
    --camera d455 --mask depth --policy-device cpu \
    --record recordings/sam21_v4best_try1 --seconds 20 \
    --home-first --command-rate-scale 0.6 \
    --target-tracker sam21 --depth-source stereo \
    --no-record-compress --allow-legacy-action-api
```

Do not add `--target-lifecycle`, `--held-target-radius`, or `--depth-bias` to
that A/B run: each may be useful, but each changes another input or state
transition and makes the comparison inconclusive.  SAM is warmed and reset
before the control loop begins, so CUDA's roughly 245 ms first-anchor cost is
paid while the arm is holding, not on the first real target observation.

The newer full-stack recipe is below.  It is a separate experiment rather
than a reproduction of `v4_stereo_try3`:

```bash
micromamba run -n mjlab python -m hardware.deploy.run \
    --policy hardware/deploy/policies/d455_v4_final \
    --camera d455 --mask depth --policy-device cpu \
    --record recordings/<a fresh directory> \
    --home-first --command-rate-scale 0.6 \
    --target-lifecycle --held-target-radius 0.045 \
    --target-tracker sam21 --depth-source sensor \
    --no-record-compress --allow-legacy-action-api
```

For the lower-latency SAM2.1 recipe, keep ``--depth-source sensor``.  A live
D455 `--no-arm` measurement of the A/B command above (FoundationStereo plus
eager BF16 SAM) ran perception at 14.4 Hz: compute p50/p95 was 67.8/80.1 ms and
capture-to-command age p50/p95/p99 was 118.9/158.8/180.5 ms, with nine stale
holds in ten seconds.  It is below the 200 ms watchdog on most frames, but is
outside the policy's typical 40--80 ms delay range; treat it as a diagnostic
reproduction, not the default.  Do not add ``--sam-vos-optimized``: the
installed torch 2.13/SAM2.1 combination fails on its first propagated frame,
and ``run.py`` refuses the flag; eager SAM is the verified deployment path.

**Before pressing enter.** Keep the physical emergency stop in hand, clear the
workspace, and use a directory that does not exist yet -- `run.py` refuses real
motion without one and refuses to overwrite one that has anything in it. The
run asks for the word `move` on an interactive terminal and will not start
without it.

Then restore the table, because the scene is the largest variable and nothing
in `run.json` records it. Two runs with byte-identical arguments gave very
different results, one having two objects on the table and the other three:

```bash
micromamba run -n mjlab python -m hardware.deploy.scene \
    --like recordings/v4_stereo_try3
```

It reads the table through the same segmenter the policy uses, prints which
object to move and by how much, and exits 0 once the scene matches. It touches
nothing: CAN is opened to read joint angles, because the segmenter subtracts
the arm, and the drives are never enabled.

**What each flag is doing, and what is deliberately absent.**

| | |
|---|---|
| `--home-first` | drives to the training start pose first. Every episode the policy trained on began there; a run that starts wherever the last one stopped hands its GRU an opening it never saw, and three runs in a row went progressively lower because each began where the previous ended |
| `--command-rate-scale 0.6` | scales the commanded joint rate. Occlusion scales with speed -- the arm blocked the camera's line to the object 52% of frames on a fast run against 16% on a slow one |
| `--depth-source stereo` | Fast-FoundationStereo on TensorRT over the raw imagers, 15.3 ms. Optional, and not obviously better: 83.9% fill against the camera's 88.6%, agreeing to 4.4 mm. The best run used it; whether it is *why* is not established |
| no `--min-grasp-height` | the guard now tests the measured pose rather than the commanded setpoint, but any floor near the table will still stop a run that is working. `-0.018` is the value the data supports if one is wanted: below the contact the design allows, above the -25 mm that batted an object off the table |
| no `--min-table-clearance` | omitting it permits light fingertip contact, which the training task allows on purpose |

**Since that run.** The recorder stored every camera frame about twice, which
is what filled its queue and ended it; that is fixed, so the queue should now
keep up with compression on. Leave `--no-record-compress` off unless it fills
again: turning it on frees enough CPU to halve the observation latency, from
the 43 ms the policy was trained for to about 20 ms, which is a control
condition and not a logging one.

**Afterwards.** A review page is written beside the recording on exit, strided
to roughly one picture a second because at full rate it is half a gigabyte. To
look at a grasp properly, re-render a window at every frame:

```bash
micromamba run -n mjlab python -m hardware.deploy.review recordings/<session> \
    --stride 1 --window 25 30 --html detail.html
```


## Lessons from the first hardware runs

2026-09-01. Twenty recorded sessions, 12224 commands to a live arm, and one
policy that finally worked. What follows is what cost the most time, written
down because every one of them looked like something else first.

**A constant beat every hypothesis.** The sim-to-real failure that motivated
three retraining campaigns was `SegmenterCfg.width_range_m`, a 160 mm ceiling
on a component's horizontal extent. As the hand arrives, the target's blob
merges with what `arm_mask` leaves of the gripper and measures **164 mm** --
4 mm over, on three separate recordings (164.0, 164.3, 164.1). Raising the
ceiling took detection from 23% / 40% / 38% to 95% / 100% / 99%. Before
finding it, five plausible mechanisms had been proposed and each refuted by
measurement.

**Measure the conditional you care about, not one that correlates with it.**
Ray-tracing said the arm blocked the camera-to-object line 52% of frames, and
that was true. It was never checked against *which frames actually lost the
target*, and those were different frames: per-stage counting showed `arm_mask`
removed **0%** of the object's pixels and depth dropout **0%**. A correct
measurement of the wrong quantity is still the wrong answer, and it is more
convincing than a guess.

**Guard the robot, not the setpoint.** `--min-grasp-height` compared a floor
against forward kinematics of the *commanded* target. That target leads the
arm and, in normal successful operation, sits below the table **44.7%** of the
time, while the arm itself never goes below 13 mm. It stopped three healthy
runs within seconds, twice with the hand 5-9 cm up, which is also what the
operator saw. The guard now reads measured joint feedback; the setpoint is
still logged, and is not a safety signal.

**When the statistics stop moving, look at a picture.** Detection during the
approach sat at 25-48% and no threshold moved it: footprint ceiling 160 to
350 mm, arm clearance 20 to 65 mm, morphological opening 3 to 0, area floor
150 to 40 px, height floor 20 to 8 mm. One rendered frame explained it -- the
object is a shallow white box about 25 mm tall at 1.2 m, and a detector that
thresholds height above a fitted plane is being asked for a step the sensor
barely resolves.

**A field over a region the arm must work in teaches it to stay out.** A 5 mm
proximity shell around the table once drove the robust teacher to inactivity.
Contact penalties do not have this shape: everything up to touching is free, so
there is no gradient pushing the hand away. The distinction is the difference
between a term that can be used and one that cannot.

**DAgger cannot teach a student to act blind.** Putting the measured mask
dropout into distillation took the behaviour loss from 0.226 to 0.541 and
placements from 2.06 to 0.31; a ramp through PPO held at scale 0.5 (2.50) and
collapsed at 0.75 (0.74) and 1.0 (0.17). The teacher sees the object on every
frame, so on a frame the student is blind the label is not a function of the
student's observation, the loss has an irreducible floor, and its minimiser is
a policy that commits to nothing -- it reached for an object 0.46 times an
episode against 5.94. Reinforcement has no such defect; its critic is
privileged and its objective is return.

**"Same arguments" is not "same conditions".** Two runs with byte-identical
`run.json` gave very different results because one had two objects on the table
and the other three, of a different size, and nothing recorded that.
`hardware/deploy/scene.py` now reads the scene back before a run and says which
object to move and by how much.

**A variable that is assigned and never read cost three runs.**
`_Recorder.write` set `_last_frame` and never compared against it, so a 50 Hz
control loop against a 30 Hz camera stored every frame about twice: 2893 files
for 1488 distinct camera frames, 38% of consecutive pairs byte-identical. The
review page repeated pictures, which reads as a low frame rate; the writer
compressed everything twice, which filled the queue and ended three runs; and
the workaround for that freed enough CPU to halve the observation latency,
moving a control condition nobody meant to move. Fixing one thing moved
another, and only the recording made that visible.

**Compute the simulator's number the way the robot's was computed.**
`scripts/eval_occlusion.py` traces rays against a sphere cover rather than
calling `mj_ray`, which would be more exact and less useful: the number it has
to be compared against was measured on the rig against
`proprio.Kinematics.link_spheres`. When two numbers are compared, the method
must not be one of the variables.

## Layout

```
src/piper_push/
├── robot.py, objects.py, shapes.py, camera.py   scene and randomisation
├── depth_noise.py                               the measured D405, in simulation
├── actions.py                                   rate-limited joint action + plant model
├── perturb.py, latency.py, damping.py           opt-in simulator mismatch (shared with the research branch)
├── models.py, distill.py, checkpoints.py        policy, DAgger, export
└── tasks/pick_place/                            env, MDP terms, PPO config
scripts/
├── accept_s1.py, eval.sh                        the evaluation gate
├── check_cadence.py, check_perturb.py           in-sim plumbing checks
├── eval_endurance.py, eval_occlusion.py         within-episode decay and occlusion metrics
├── probe_hidden.py, trip_phase.py               diagnostics
└── export_obs_spec.py, plot_depth_model.py      what the actor expects; the figure
hardware/
├── depth_bench/                                 choosing and characterising the camera
└── deploy/                                      D405 + PiPER-X -> the trained policy
```
