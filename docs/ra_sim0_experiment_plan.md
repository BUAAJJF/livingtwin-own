# Phase RA-Sim-0 — pre-registration

*Written and committed before any result run.  Every constant, split, seed and
threshold below is fixed at the moment of that commit and is not moved
afterwards.  Where a later stage deviates from this document, the deviation is
recorded in `docs/ra_sim0_results.md` under its own heading and the original
text here is left standing.*

Start: 2026-08-25 16:21 UTC.  Budget: 8 hours.  Baseline commit `623e2a9`,
301 tests passing.

---

## 1. The question

> For an actuator mismatch that no existing MuJoCo parameter can express, can a
> residual learned from reward-free target-domain trajectories alone, injected
> into MJWarp, make the augmented simulator more accurate than the best
> parameter calibration — and then train a better target-domain policy?

Two halves, gated separately.  Accuracy is Gate R; control value is Gate C.  A
phase that passes R and fails C has a specific, publishable answer and that
answer is written down rather than hidden.

## 2. Injection point

Settled by measurement in Stage 0, reported in
`docs/residual_injection_audit.md`.  The priority order the phase sets is
*action/activation wrapper > pre-step generalized force > STOP*, and the first
of those is available: `piper_push.actions.RateLimitedJointPositionAction`
already owns a stateful, per-environment, fully batched stage between the
policy's action and the servo's target.  A `command_hooks` tuple was added to
it; a hook is called once per control step with the commanded target and the
action term, returns a corrected target, and owns a `reset(env_ids)`.

Both the hidden target and the residual are hooks.  Neither writes `qpos` or
`qvel`, adds a force, or touches the contact solver.

## 3. The hidden target

`piper_push.hidden_plant`.  Two effects, composed in the order the signal
meets them: the servo's current limit, then the gear train's play.

    v_t = v_{t-1} + beta(|u_t - v_{t-1}|) * (u_t - v_{t-1})
    beta(s) = BETA0 / (1 + KAPPA_j * s / S_REF)
    y_t = min( max( y_{t-1}, v_t - UP_j ), v_t + DOWN_j )

`u` is the commanded position target, `y` what the servo receives, `j` the
joint index.  Both `v` and `y` are per-environment states reset to the fresh
posture at every episode boundary.

| constant | value (joint1 … joint6) |
|---|---|
| `BACKLASH_UP` | 0.012, 0.018, 0.015, 0.008, 0.006, 0.010 rad |
| `BACKLASH_DOWN` | 0.020, 0.009, 0.022, 0.005, 0.011, 0.006 rad |
| `BETA0` | 0.85 |
| `KAPPA` | 1.6, 1.6, 1.6, 0.9, 0.9, 0.9 |
| `S_REF` | 0.04 rad |

The gripper's command path is not touched.

**Why it is out of reach of the existing axes.**  `perturb.AXES` offers a
whole-step transport delay, one constant fraction of each commanded step
(`joint_response_scale`), one symmetric *rate* deadband, and a damping
multiplier.  Backlash is a *position* hysteresis whose sign depends on the
direction of travel and whose two flanks differ per joint; the rate deadband
holds a slow ramp forever and backlash lets it through with an offset, so they
are not the same function.  `beta` depends on the size of the step, and
`joint_response_scale` is that fraction held constant — a single value cannot
be right at 0.005 rad and at 0.5 rad at once, which `tests/test_hidden_plant.py`
asserts numerically.

**Oracle discipline.**  `HiddenPlant.state` exists for one consumer: the oracle
arm of Stage 5/6, which installs the target itself.  Nothing else reads it.
The residual's feature builder takes exactly four tensors and there is no
argument through which a hidden quantity could arrive.

## 4. What the residual may see

`piper_push.residual.build_features(q, qd, u, u_prev)` — measured arm joint
positions and velocities, the position target the controller commanded this
step and the previous one, and their difference `q - u`, which is the servo
error a deployed controller reports.  30 numbers.

Forbidden and structurally unreachable: the target's effective command `y`, the
plant states `v` and `y_{t-1}`, reward, success, the safety-shell label, the
object's pose, the privileged critic.

Model: 4-member ensemble of one-layer GRU (hidden 64) with a zero-initialised
linear head and a `tanh` bound of **DELTA_MAX = 0.05 rad**.  18,822 parameters
per member, 75,288 for the ensemble.  Identity at initialisation, so
"residual off" is the nominal simulator exactly and not a second code path.

## 5. Data

One control step is 20 ms.  A **budget of B seconds** means `50 * B` control
steps per environment, following the convention Phase WM1 used; parallel
environments are independent sessions of that length, not extra time.

Budgets reported: **10 s / 30 s / 60 s / 180 s / 300 s**, nested inside one
300 s collection so that the smaller budgets are genuine prefixes.

Composition of every target-domain split, as the phase specifies:

| share | content |
|---|---|
| 70% | the frozen vision policy's natural rollout |
| 20% | the same policy with a small pre-registered action perturbation, `N(0, 0.05)` on the normalised action, clipped to ±0.15 |
| 10% | a scripted no-load probe: slow multi-frequency sweeps on the arm plus gripper open/close, object out of reach |

Splits, each a separate generation seed and never a re-slice of another:

| split | seed | shapes | envs × steps | used for |
|---|---|---|---|---|
| `train` | 7101, 7102 | classes 0–2 | 64 × 15000 | fitting the residual |
| `val` | 7201 | classes 0–2 | 64 × 15000 | every hyper-parameter and the parameter calibration |
| `valh` | 7301 | classes 3–4 | 64 × 15000 | held-out **shape** check |
| `test` | 7401 | classes 3–4 | 64 × 15000 | Gate R, and nothing else |
| `test_amp` | 7402 | classes 3–4 | 64 × 15000 | held-out **action amplitude / reversal** check; perturbation `N(0, 0.12)` clipped to ±0.35, which is outside anything in `train` |
| `nominal` | 7501 | classes 0–2 | 64 × 15000 | the surrogate only; collected in the **nominal** simulator |

The frozen policy is
`logs/rsl_rl/piperx_pick_place_vision/2026-08-22_17-15-09_f3/model_1500.pt`,
the same checkpoint Phases WM1-A and WM1-B used.  Its SHA-256 is recorded in
every result file.

Data are collected in `Mjlab-Pick-Place-PiperX-Vision`.  Model accuracy is
measured in `Mjlab-Pick-Place-PiperX`, whose physics, actions, events and
scene are the same object — `vision=True` adds a sensor, an observation group
and a camera-pose event and changes nothing the arm does.  That claim is
checked, not assumed.

## 6. How accuracy is measured

Teacher-forced replay in the **real** simulator.  For a recorded target
trajectory and a candidate simulator:

* write the recorded physical state (arm `q`/`qdot`, object root state) into
  the candidate every `P` control steps;
* step the candidate with the recorded action stream;
* compare the candidate's `q`/`qdot` against the recording.

`P = 1` gives the one-step error, `P = 10` and `P = 25` the multi-step ones.
The command-path state (slew memory, delay pipeline, hook states) is *not*
re-synchronised: it is a deterministic function of the action history, which
both simulators see identically, and its divergence is part of what is being
measured.  Every candidate is treated identically.

Writing state between steps is a re-synchronisation for a teacher-forced
evaluation, not an injection mechanism.  No candidate simulator uses a
post-step state overwrite to produce its dynamics.

Reported per candidate: one-step / 10-step / 25-step NRMS on `q` and `qdot`,
joint-wise error, error inside the reversal and deadband regions, rollout
divergence time, ensemble coverage, throughput and memory.

Normalisation: NRMS is the RMS error divided by the RMS of the *observed*
per-step change in that quantity over the same segments, so 1.0 means "as
wrong as predicting no motion".

## 7. Candidate simulators

| name | what it is |
|---|---|
| `nominal` | the simulator as shipped |
| `param_1d` | the single best axis of `perturb`, refit |
| `param_joint` | the best joint fit over four axes at once |
| `broad_dr` | the best single predictor drawn from a broad randomisation |
| `residual` | nominal physics + the learned residual hook |
| `oracle` | the hidden target installed; an upper bound, not a method |

Calibration searches, pre-registered:

    action_latency_steps  in {0, 1, 2, 3}
    joint_response_scale  in [0.30, 1.00]
    action_deadband_rad   in [0.000, 0.030]
    servo_damping_scale   in {0.75, 1.0, 1.25, 1.5}
    lowpass_hz            in {off, 40, 20, 10, 5}
    gripper_rate_scale    in [0.5, 1.5]

scored by one-step NRMS on `val` and on nothing else.  The hidden target's
formula is not in the search space.

## 8. Gates

### Gate P — the parameters really are not enough

All four, or STOP:

1. the best parameter fit still shows significant structural error on
   held-out data;
2. that error varies with state, action magnitude or direction reversal;
3. the oracle is clearly better than the best parameter fit;
4. neither result is explained by leakage or by mismatched initial states.

### Gate R — the residual really is more accurate

All of, against `param_joint`, on `test` and `test_amp`, in real MJWarp:

* one-step error down ≥ **30%**;
* 10-step error down ≥ **25%**;
* 25-step error down ≥ **20%**;
* improvement on both the held-out shape split and the held-out action split;
* no constraint violation, state jump or non-physical energy;
* 95% intervals support the improvement, over ≥ 3 residual training seeds.

Fail ⇒ STOP; no policy training.

### Gate C — accuracy becomes control

From one frozen checkpoint, same PPO config, steps and seed bank.  Arms:
zero-shot, source refit, `param_joint`, broad DR, `residual`, oracle.
Screening at 2 training seeds; expansion to 8 only if the residual shows any
improving trend.  ≥ 3 evaluations of 512 envs × 2400 steps per checkpoint.
Counts clustered on the training seed, negative binomial with a
cluster-robust check beside it.

At least one of:

* target throughput up ≥ **5%** with safety no worse;
* trip-rate ratio ≤ **0.70** with throughput loss ≤ 5%;

and all of: source retention loss ≤ 5%; multi-seed intervals support the
improvement; the improvement is not simply slowing down; the gap to the oracle
narrows visibly; adaptation wall-clock ≤ 60 minutes.

## 9. Discipline

* All new features default off; the audit asserts that an installed-but-inert
  hook reproduces the baseline exactly for ≥ 8 steps.
* No `git add -A`; only the files each stage names.
* Raw tensors stay out of git; a manifest with shapes, seeds and SHA-256 goes
  in.
* GPUs 4–7 only, `CUDA_VISIBLE_DEVICES` set explicitly per job.
* No threshold, target constant or baseline is changed after a result is seen.
* Negative results are reported.  A stage that does not run is marked *not
  executed*, and one stopped by a gate is marked *stopped by Gate X*.
