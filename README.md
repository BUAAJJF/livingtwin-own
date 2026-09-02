# PiPER-X tabletop tidying — vision policy, and sim-to-real calibration

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

## Research Direction: Reward-Free Decision-Aware Sim-to-Real Calibration

### What exists today

All of this is built, measured and reproducible in simulation. **It has now
also run on hardware**: on 2026-09-01 a trained vision policy picked objects
off the table and placed them in the bin on the PiPER-X, from the calibrated
D455 alone. That is a first working run, not a solved problem -- the same day
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

Measured results and their caveats are in
[`docs/results.md`](docs/results.md), and the audit that re-measured them —
including **two findings in that file which it retracts** — is in
[`docs/novelty_validation_phase_0_2.md`](docs/novelty_validation_phase_0_2.md).
Two conclusions from that audit constrain everything below:

* **The simulator is not reproducible run to run.** Five identical commands
  span 2.9%. Single rollouts do not support conclusions here, and every number
  in the new work is a median over ≥3 independent processes with the spread
  quoted.
* **A recurrent policy exploits within-episode invariants**, but *not* by
  carrying facts about previous objects — that mechanism was tested three ways
  and refuted. The remaining direction from that work is benchmark hygiene, not
  a memory architecture, and it is **not** what is proposed here.

### Working hypothesis

> Pixel- or state-level simulator matching may spend real interaction on
> discrepancies that do not change what the deployed policy does. We instead
> investigate **reward-free calibration of simulator parameters according to
> their effect on the frozen policy's latent state, action, deployable
> task-value estimate and predicted tail risk.**

We are *investigating* this; it is not an established result and we have not
run a systematic literature search. The novelty claim, if one survives, is
**not** any of: using a world model; simulator system identification;
sim-to-real adaptation; one-hour adaptation; or collecting real data and
returning to simulation. Each of those is well established.

The narrower combination we think may be new:

1. **decision-aware** discrepancy under *partial visual observation*, rather
   than state- or pixel-reconstruction error;
2. requiring **no real-world reward, success label, demonstration, or ground
   truth physical parameter**;
3. calibrating a simulator **posterior by policy consequence**;
4. posterior-guided simulator fine-tuning that keeps a **broad prior mixed in**,
   so a few minutes of real data cannot collapse the distribution.

### Proposed method

```
  π0  ──  pre-train in broad DR                                    (done)
   │
   ├──  freeze; collect 5–10 min of reward-free target-domain data
   │
   ├──  parameter-conditioned latent dynamics model predicts, given θ:
   │       next policy latent · proprioceptive change
   │       gripper servo error · deployable slip / safety risk
   │
   ├──  decision-aware discrepancy  →  posterior  q(θ | D)
   │
   ├──  p_adapt(θ) = α q(θ | D) + (1 − α) p_broad(θ)
   │
   ├──  fine-tune policy / adapter in the GPU simulator
   │
   └──  redeploy, ≤60 min wall clock end to end
```

Candidate discrepancy:

```text
d_pi = lambda_z * ||phi_pi(o_real_next) - phi_pi(o_sim_next)||^2
     + lambda_a * ||pi(o_real_next)     - pi(o_sim_next)||^2
     + lambda_v * |V_obs(o_real_next)   - V_obs(o_sim_next)|
     + lambda_c * |C_obs(o_real_next)   - C_obs(o_sim_next)|
```

**A constraint that must not be glossed over.** The current critic reads
privileged state — object pose, mass, table friction — so it **cannot** serve
as `V_obs`, which has to be computable from what the robot can actually see.
The first version therefore uses only the actor latent `phi_pi` and the action
`pi`. Any `V_obs` or `C_obs` must be a separate observation-only head, trained
and frozen in simulation, before it can appear in this term. Nothing here
assumes the privileged critic is deployable.

Scope: this calibrates **session-persistent** mismatch only — parameters fixed
for a deployment run and unknown before it starts. It does not revisit the
previous-object / two-timescale-memory idea, which the Phase 0–2 audit
refuted.

### Targets — not results

None of these has been achieved. They are what the direction is aiming at.

| target | value |
|---|---|
| real-world reward, labels, demos | none |
| real interaction | 5–10 min |
| total adaptation wall clock | ≤ 60 min |
| recovery of the zero-shot → oracle-calibrated gap | 70–80% |
| held-out object generalisation | retained |
| reported metrics | throughput, success, post-grasp drop, p95 cycle time, safety events — all of them, not a mean reward |

### Validation route

| phase | question | status |
|---|---|---|
| **WM0** | Is there a gap to recover, is it identifiable from reward-free data, and can simulation recover it given the answer? | **GREEN**, 6/6 — [`docs/sim2real_sweep_phase_wm0.md`](docs/sim2real_sweep_phase_wm0.md) |
| **WM1-A** | One axis end to end: infer 60 ms of observation delay from reward-free history, adapt under the posterior, keep the source domain. | **YELLOW**, 5/6 — [`docs/wm1_latency_vertical_slice.md`](docs/wm1_latency_vertical_slice.md) |
| WM1-B | the same loop on a plant-side axis (servo damping) and a perception-side one (camera pose) | gated on WM1-A |
| WM2 | several axes at once, and a deployable risk head for tail-only mismatch | gated on WM1-B |
| WM3 | hardware | not started |

**WM1-A in one line.** Sixty seconds of reward-free arm time identifies the
domain (1.000 balanced accuracy over five candidates, against three controls at
chance); posterior-guided fine-tuning recovers 102% of what fine-tuning on the
correct parameter achieves while keeping 97% of the source domain; the safety
criterion FAILS -- the trip rate does not reproduce across training seeds; and a three-millisecond ridge baseline with a far worse posterior
recovers just as much. Simulation only — no hardware result, and none claimed.

Phase WM0 exists to decide whether WM1 is worth building. Simulator mismatch
is applied through `piper_push.perturb`, which is inert unless asked for:

```bash
python scripts/check_perturb.py --device cuda:0     # every axis reaches the sim
python scripts/sweep_plan.py s1 > s1.jobs           # the plan, not typed by hand
OUT=results/sim2real_sweep/s1 NUM_ENVS=256 STEPS=1200 \
  scripts/run_sim2real_sweep.sh "0 1 2 3" s1.jobs
python scripts/analyze_sweep.py --stage s1 --json
```

---

## Task

| | |
|---|---|
| Actor observation | proprioception (joint positions and velocities, end-effector pose, gripper opening, pad contacts, gripper servo error, last action) + the 3-channel depth image |
| Critic observation | the above, uncorrupted, plus object pose/velocity/shape and privileged physics — training only |
| Action | 6 joint position targets + gripper, rate-limited to 0.62 × the safety-shell trip speed and interpolated across physics substeps |
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

```bash
micromamba run -n mjlab python -m hardware.deploy.run \
    --policy hardware/deploy/policies/d455_v4_final \
    --camera d455 --mask depth --policy-device cpu \
    --record recordings/<a fresh directory> \
    --home-first --command-rate-scale 0.6 \
    --depth-source stereo
```

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
├── perturb.py                                   opt-in session mismatch (WM0)
├── models.py, distill.py, checkpoints.py        policy, DAgger, export
└── tasks/pick_place/                            env, MDP terms, PPO config
scripts/
├── accept_s1.py, eval.sh                        the evaluation gate
├── check_cadence.py, check_perturb.py           in-sim plumbing checks
├── sweep_plan.py, run_sim2real_sweep.sh         WM0 sweep
├── analyze_sweep.py, analyze_novelty.py         tables and figures, from JSON only
├── probe_hidden.py, trip_phase.py               diagnostics
└── export_obs_spec.py, plot_depth_model.py      what the actor expects; the figure
hardware/
├── depth_bench/                                 choosing and characterising the camera
└── deploy/                                      D405 + PiPER-X -> the trained policy
```
