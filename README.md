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

All of this is built, measured and reproducible in simulation. **None of it has
run on hardware.**

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
chance); posterior-guided fine-tuning recovers 102% of the known-parameter
oracle's throughput gain while keeping 97% of the source domain; the safety half
does not reproduce across training seeds and the criterion fails on its
interval; and a three-millisecond ridge baseline with a far worse posterior
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
