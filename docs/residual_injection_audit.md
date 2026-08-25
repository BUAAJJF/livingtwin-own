# Stage 0 — where a residual can actually enter MJWarp in this repository

*Phase RA-Sim-0, 2026-08-25. Measured at 256 environments in
`Mjlab-Pick-Place-PiperX`, 60 control steps of a fixed scripted command stream
with reversals in it. `scripts/ra_sim0_audit.py`; raw numbers in
`results/ra_sim0/injection_audit.json`.*

**Verdict: L2, the action/command wrapper.** The phase's priority order is
*action/activation wrapper > pre-step generalized force > STOP*, and the first
of those is available, batched, per-environment and already reset-aware.

---

## The finding that had to come first

**MJWarp is not bit-reproducible across two builds of the same environment at
the same seed.** Two builds that differ in nothing at all, driven by the same
command stream, disagree by

| | mean \|Δq\| | max \|Δq\| |
|---|---|---|
| after 8 control steps | 3.6 × 10⁻⁷ rad | 1.7 × 10⁻³ rad |
| after 60 control steps | 7.6 × 10⁻³ rad | 1.37 rad |

The solver's reductions are order-dependent on the GPU, and a contact-rich
scene amplifies the difference: over sixty steps the *maximum* divergence
reaches a radian, which is a different grasp. So "an installed-but-inert hook
changes nothing" cannot be tested as bitwise equality against a second build,
and the maximum is not a statistic two identical builds agree about. Every
comparison below is a **ratio against that floor**, on the mean over 92k
elements, which is stable.

This is not a detail. It sets the noise floor under the whole phase, and it is
why the accuracy gate in Stage 5 re-anchors a candidate on the recorded state
every 1 or 25 steps rather than letting a rollout run free.

## The five layers, tested

| # | layer | reachable | batched | per-env reset | adopted |
|---|---|---|---|---|---|
| L1 | observation post-processing | yes | yes | yes | no |
| L2 | stateful wrapper between action and `ctrl` | **yes** | **yes** | **yes** | **yes** |
| L3 | actuator activation / command | yes | yes | n/a | no — subsumed by L2 |
| L4 | pre-step generalized force | yes | yes | n/a | no |
| L5 | per-environment residual hidden state | **yes** | **yes** | **yes** | **yes** |

**L1** already exists: `piper_push.perturb.PerturbedCameraScene` is a class
observation term wrapped in mjlab's `DelayBuffer`. Not used, for a reason that
is about the question rather than the mechanism — an observation filter cannot
change what the arm does, and the mismatch under test is in the command path.

**L2** is `piper_push.actions.RateLimitedJointPositionAction`. It already owned
a stateful stage between the policy's action and the servo's target — the
delay pipeline, the deadband and the response scale of Phase WM0 — with a
`reset(env_ids)` that flushes them per environment. It gained a
`command_hooks` tuple: a hook is called once per control step with the
commanded target, returns a corrected one, and owns its own `reset(env_ids)`.
Empty by default.

**L3** is reachable through `Entity.write_ctrl_to_sim`, and for a position
actuator `ctrl` *is* the target this task's action term already writes. Using
it would mean reimplementing the substep ramp and the encoder bias somewhere
else; L2 keeps them in one place.

**L4** is reachable through `Entity.write_external_wrench_to_sim`, which takes
a batched `[num_envs, num_bodies, 3]` force and torque that MuJoCo adds before
it integrates. It was tested and accepted. Not adopted, because the mismatch
being modelled is a *command* mismatch: a servo that does not travel the whole
commanded step is not a servo with a mystery force on it, and expressing it as
one would need the residual to invent the arm's inertia.

**No post-step state overwrite is used anywhere.** The phase says that if the
only available injection were a post-step overwrite of `qpos`/`qvel`, the
answer is STOP. It is not the only one available, and it is not used. Stage 5's
evaluation *does* write state between control steps — to re-anchor a
teacher-forced comparison, identically for every candidate — which is a
different thing from producing dynamics that way.

## What the tests actually returned

| test | result | floor | verdict |
|---|---|---|---|
| inert hook vs baseline, 8 steps | mean \|Δq\| 4.9 × 10⁻⁸ | 3.6 × 10⁻⁷ | **0.14× the floor** |
| inert hook vs baseline, 60 steps | mean \|Δq\| 7.607 × 10⁻³ | 7.614 × 10⁻³ | **1.00× the floor** |
| residual ensemble at identity init, 8 steps | mean \|Δq\| 3.7 × 10⁻⁷ | 3.6 × 10⁻⁷ | **1.03× the floor** |
| hidden structural target, 8 steps | mean \|Δq\| 4.4 × 10⁻² | 3.6 × 10⁻⁷ | **122,572× the floor** |
| per-environment state isolation | untouched drift 0.0, reset lands on the fresh posture to 0.0 | — | exact |

The identity-initialised residual is worth a sentence. Its head is
zero-initialised and its bound is a `tanh`, so it returns exactly zero and the
augmented simulator is the nominal one — which is what lets "residual off" be
a *control* rather than a second code path. The measured 1.03× says the
ensemble's arithmetic does not perturb the rollout even at the level the GPU's
own non-determinism sits at.

## Throughput

| configuration | env-steps/s at 256 envs |
|---|---|
| baseline | 6,972 |
| + 4-member GRU residual ensemble (75,288 parameters) | 5,873 |
| + hidden structural target | 13,236 |

The residual costs **15.8%** of throughput. That is a real cost and it is
carried into every training estimate in this phase.

The hidden target's figure is *not* a hook overhead and must not be read as
one: it moves the arm differently, and a lagged, backlashed arm makes fewer
and simpler contacts for the solver, so it runs nearly twice as fast. Any
"speedup" from a hook that changes the trajectory is a statement about
contacts, not about the hook.

## Constraints the phase imposed, and where each is met

* *Torch/Warp batched only* — a hook receives `[num_envs, num_joints]` and
  returns the same; there is no Python loop over environments anywhere in
  `hidden_plant.py` or `residual.py`.
* *No per-environment CPU callback in the 512-environment loop* — none exists.
* *No post-step `qpos`/`qvel` overwrite* — none, see above.
* *No contact-solver modification* — none; `MujocoCfg` is untouched.
* *Residual off must match the original simulator for ≥8 steps within
  numerical tolerance* — measured at 1.03× the tolerance the simulator itself
  sets.
* *Per-environment reset with no cross-talk* — measured exact, both for the
  hidden plant's two states and for the ensemble's four hidden vectors.
* *Throughput overhead recorded* — 15.8%.
