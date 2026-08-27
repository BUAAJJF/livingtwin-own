# Phase RA-HW-0 — pre-registration

*Design only. No motor was enabled, no command was sent, no emergency stop was
released, and no controller parameter was written. The gate between this
document and a moving arm is a human sentence, and no code in this repository
can produce it.*

Started 2026-08-27 14:01 UTC from commit `c3d9c45` (383 tests). Phases
RA-Sim-0 and RA-Sim-1 stand at **RED** and nothing here reopens them.

---

## 1. The question

> Does the real PiPER-X's actuator error contain structure that the best
> parameter model cannot explain, and that a stable stateful actuator model
> might?

Two simulated phases have already narrowed it. RA-Sim-0 showed that a small
additive command residual is the wrong parameterisation for accumulated lag
and that its bound binds at every setting. RA-Sim-1 showed that a stateful
chasing recursion recovers most of the multi-step gap at no compute cost and
with better stability — and still loses at one step, because reproducing an
instantaneous response needs a command move larger than any physically
arguable actuator rate.

Both of those were measured against a mismatch **this project invented**. The
question that decides whether any of it matters is whether the real arm has
that shape, and nothing in simulation can answer it.

## 2. Hardware capability audit — what is known and what is not

`scripts/ra_hw0_audit.py`, run offline and with `--hardware`.
`results/ra_hw0/audit.json`.

**No arm is attached to this machine.** `can0` does not exist; `ip -br link`
shows no CAN interface of any kind. The kernel modules are loaded (`can`,
`can_raw`, `can_dev`, `gs_usb` — the driver for the candleLight-class USB-CAN
adapter AgileX ships), so the host has been set up for one and the adapter is
unplugged. `lsusb` shows no CAN adapter. The `--hardware` path was exercised
and refused: *"can0 is not present; interfaces are none"*.

So every device-side fact below is **UNKNOWN**, and Gate H-A is BLOCKED.

### Known, from the installed SDK

`piper_sdk` **0.6.2**, `python-can` 4.6.1.

| what | value | source |
|---|---|---|
| joint limits | j1 ±2.6179, j2 [0, 3.14], j3 [−2.967, 0], j4 ±1.745, j5 ±1.22, j6 ±2.09439 rad | `piper_param_manager` |
| gripper range | 0–0.07 m | same |
| joint feedback / command unit | 0.001 degree | `GetArmJointMsgs`, `JointCtrl` |
| motor speed | 0.001 rad/s | `GetArmHighSpdInfoMsgs` |
| motor current | 0.001 A | same |
| torque/effort | "converted using a fixed coefficient" | same |
| driver voltage | 0.1 V | `GetArmLowSpdInfoMsgs` |
| driver and motor temperature | 1 °C | same |
| device angle limits | 0.1 degree | `GetAllMotorAngleLimitMaxSpd` |
| device max acceleration | 0.001 rad/s² | `GetAllMotorMaxAccLimit` |
| control modes | standby / CAN / ethernet / wifi / offline-trajectory; MOVE P·J·L·C·M·CPV; position-velocity or MIT | `MotionCtrl_2` |
| what this repo uses | `MotionCtrl_2(0x01, 0x01, 100, 0x00)` — CAN mode, MOVE J, 100% speed rate, position-velocity | `hardware/deploy/robot.py` |
| per-motor fault bits | voltage_too_low, motor_overheating, driver_overcurrent, driver_overheating, collision_status, driver_error_status, driver_enable_status, stall_status | `foc_status` |
| arm-level status | normal / e-stop / no-solution / singularity / **target angle over limit** / joint comms / brake not released / collision / teach overspeed / joint state abnormal | `GetArmStatus` |
| per-joint error bits | angle-limit and communication status, per joint | `err_code` |
| timestamps | every accessor returns `time_stamp` and `Hz`; `time_stamp` is python-can's frame timestamp, i.e. the **host kernel's** receive time | traced through `piper_protocol_v2` |
| **device clock** | **not exposed** — `device_timestamp_s` is UNKNOWN and is recorded as `null` | — |

**A correction to this repository.** `hardware/deploy/robot.py` differentiates
position to get velocity, on the grounds that the drive's own speed unit "is
not pinned down" by the documentation. In SDK 0.6.2 it is: `motor_speed` is in
0.001 rad/s. RA-HW-0 records **both** — the device's speed and the
differentiated one — and lets H0 decide which is better behaved rather than
choosing in advance.

### UNKNOWN, and each one blocks motion

Model, serial, firmware; the arm's own angle limits, maximum joint speed and
maximum acceleration; telemetry rate and jitter; the still-arm current
baseline; temperature and bus-voltage baselines and their maxima; the
undervoltage threshold; what the arm does when CAN is unplugged; whether
releasing the hardware e-stop resumes motion; the configured gripper jaw gap;
what power-on, enable and fault-reset do; and **the positive rotation sense of
each joint on the physical arm**, which nothing in the SDK or this repository
establishes.

### The joint-5 finding

`piper_push.robot.SAFE_TARGET_CLIP` gives joint 5 ±1.3090 rad. The vendor's
limit is ±1.2200. **The envelope the policy was trained inside is 0.089 rad
(5.1°) wider than the arm allows, at each end.** A trained policy can command
an angle the drive refuses and reports as `arm_status = 0x04`. Joints 1, 2, 3
and 6 exceed the vendor table by 0.0001–0.0016 rad, which is that table's
rounding; joint 4 is 0.19 rad tighter than the vendor's, deliberately.

Every bound RA-HW-0 uses is the **intersection** of the two. The finding is
recorded here rather than fixed in `piper_push.robot`, because changing the
trained envelope is a simulator change and this phase does not make one.

## 3. Authorisation

Motion requires the human sentence

    APPROVE RA-HW-0 MOTION

and, independently, all of Gate H-A. `configs/ra_hw0_safety_limits.json`
carries `motion_authorised: false` and lists every UNKNOWN row; nothing in
this repository writes that field. `scripts/ra_hw0_collect.py` demands four
separate keys before it will open CAN — `--hardware`, an audit that READ an
arm, a limits table with no unknowns, and `--i-have-approval` — and the
hardware backend itself is **not implemented in this phase**, so no version of
this repository that predates the approval can move an arm at all.

## 4. Staged collection

Each stage is a gate on the next; none may be skipped.

### H0 — static telemetry, no command

The arm sits in the manufacturer's safe state. Nothing is transmitted;
`--stage H0` and a command stream are mutually exclusive and the collector
asserts it. 30 s minimum, and it measures the things every later threshold is
derived from: timestamp jitter, telemetry noise, dropped and duplicated
frames, still-arm `q`/`q̇`, current, temperature, fault state, and the
relationship between the host clock and message arrival.

An unknown *limit* blocks motion; it does not block observation. H0 is the
stage that makes those limits knowable, so it records an unknown threshold as
a finding and keeps recording — and every stage that transmits uses the strict
rule instead. That difference is one argument in `check_state` and it is the
difference between a stage that can and cannot hurt anybody.

### H1 — one joint, least energy

One joint moves; the others hold. **No segment exceeds 5 s**, each ends
stationary, and a human confirms before the next.

Joint order, by how much energy a mistake puts in the room rather than by how
interesting the joint is: **joint5, joint4, joint6, joint3, joint2, joint1**.
Joint 5 is the wrist pitch — least mass, shortest lever, no cable twist. The
joints that matter most for identification are the ones that carry the arm,
and they are last.

Start posture: `joint2 = +1.05`, `joint3 = −1.05`, everything else `0`. The
closest joint is **1.05 rad (60°) from its nearest limit**. It is deliberately
not the policy's home pose, which is a task posture a few centimetres above
the table.

The four segments, previewed and envelope-checked in
`results/ra_hw0/preview.json`:

| segment | what | duration | amplitude | peak commanded speed | Cartesian travel |
|---|---|---|---|---|---|
| H1-A hold | the pose the arm is already in | 3.00 s | 0 | 0 | 0 mm |
| H1-B step | ramp to +0.02 rad, hold, ramp back | 3.80 s | 0.0200 rad | 0.05 rad/s | 3.5 mm |
| H1-C triangle | two one-sided cycles to +0.05 rad | 4.00 s | 0.0500 rad | 0.05 rad/s | 8.9 mm |
| H1-D reversal | +0.05 → −0.05 → 0, crossing twice | 4.50 s | 0.0500 rad | 0.05 rad/s | 17.7 mm |

Lowest moving link throughout: **126.8 mm above the table plane** (`link1`).

**H1-B is a ramp, not a step.** A true single-period step of 0.02 rad asks for
1.0 rad/s — twenty times what H1 commands and three times its own stop
threshold — and `trajectories.check` refused it, which is how it was found.
A true step is the more informative experiment and it belongs in H2, where the
speed cap is re-derived rather than inherited.

0.05 rad/s is 1/39th of joint 1's trained command rate limit and 1/63rd of its
safety-shell trip speed. The measured stop threshold is 0.35 rad/s: seven times
what H1 commands, and 9–11% of the trip speeds, so noise cannot trip it and a
runaway cannot hide under it.

### H2 — single-joint system identification

Only after every H1 joint has passed. Multi-amplitude steps (including true
single-period ones, with the speed cap re-derived), multi-speed triangles, a
low-frequency chirp or multisine, deliberately asymmetric forward/reverse
tests, and repeats from several safe start postures. Amplitudes, speeds and
frequencies are pre-registered against the arm's *own* reported limits, which
H0 will have read, and every one gets a simulated envelope preview.

### H3 — multi-joint and policy-shaped commands

Only after H2's data quality and safety gates. Low-amplitude multi-joint
combinations; offline policy command segments **projected into the envelope**;
reversal-dense but low-energy commands. No object contact, no approach to the
table, no gripper closure. Raw policy output is never executed: it is clipped
offline, collision-checked and previewed first.

## 5. Recorded fields

Every sample carries all of these; a field the arm cannot supply is `null` and
stays in the record, because a log whose columns depend on what happened
cannot be compared with another one.

`host_monotonic_s`, `host_wall_s`, `device_timestamp_s`, `session_id`,
`trial_id`, `trajectory_id`, `sample_index`, `controller_mode`,
`control_rate_hz`, `command_requested_rad`,
`command_after_safety_filter_rad`, `q_rad`, `qdot_rad_s`, `servo_error_rad`,
`joint_current_a`, `joint_effort`, `motor_speed_rad_s`, `driver_temp_c`,
`motor_temp_c`, `bus_voltage_v`, `fault_flags`, `watchdog_state`,
`gripper_state`, `operator_event`, `telemetry_gap_s`, `msg_hz`.

The session's `.meta.json` carries the start and end reasons, whether a human
stopped it, the thresholds in force, the start state, the units, the joint
order, the git commit and the config hash.

**Raw logs are append-only.** A session id is used once; the recorder refuses
to open an existing raw file. Derived velocity, filtering and alignment go in
a separate `.derived.json` written by `scripts/ra_hw0_validate.py`, because a
raw log that has been improved is a raw log nobody can check.

## 6. Splits and leakage

Pre-registered before collection: train, validation and test split by **whole
trajectory**, never by slicing one continuous trajectory across splits. Test
carries start postures, amplitudes, frequencies and reversal combinations that
training does not, and at least one cross-session test taken after a power
cycle. The final sealed test is opened once, after the model and config are
frozen and hashed. Every stop, anomaly and failed trial is kept: no joint,
direction or trajectory is dropped for producing an inconvenient result. A
small pilot checks acquisition quality first and never counts as test data.

## 7. Analysis

Same real commands, same initial states, comparing: nominal MuJoCo; RA-Sim-0's
best parameter fit; RA-Sim-1's stable stateful actuator; a parameter model
**recalibrated on the real training split**; and, only if the evidence
supports it, a stateful model retrained on real data.

Metrics: command-to-motion latency; 1-, 10- and 25-step `q`/`q̇` NRMS;
forward-versus-reverse asymmetry; reversal and deadband error; dependence on
amplitude, speed and posture; residual autocorrelation; frequency response;
cross-session stability; per-joint error; prediction-interval coverage.

The unobservable effective command is never substituted by the commanded
value as a label. It is not measurable on this hardware and the analysis is
built so that it is not needed.

## 8. Stop conditions

Operator e-stop; watchdog or heartbeat anomaly; a timestamp gap over 0.10 s;
motion in a direction the command did not ask for; any of `q`, `q̇`, `q̈`,
current, temperature or servo error over its threshold; approach to a joint or
workspace boundary; anything wrong with the base, cables or surroundings; any
fault code; a failed recorder; or any state or unit that cannot be confirmed.

Each one stops transmission immediately and puts the arm in the
manufacturer's safe state. **No automatic fault clearing, no retry, no
advance to the next trajectory.** The stop machine is one-way; resuming means
a human runs the tool again.

## 9. Gates

**H-A — first motion permitted.** Hardware identity, units, joint order and
control mode confirmed; e-stop and watchdog verified on this arm; the
per-joint absolute limit table complete with no UNKNOWN; H0 passed; every
trajectory passed its simulated envelope check; the dry-run suite green; and
the human sentence given. Otherwise the tools stay read-only.
**Current verdict: BLOCKED** — no arm audited, 19 UNKNOWN items, 16 blocking
limit rows.

**H-D — the data is usable.** Consistent timestamps, units and joint order; no
unexplained dropped frames or control-period anomalies; reproducible start;
signal above the measurement noise; the command *after* the safety filter
recorded, not only the request; no continuous trajectory split across train
and test.

**H-P — the parameter model is insufficient.** All of: the best calibration
still leaves significant structural error on held-out real data; the error
depends on direction reversal, amplitude, frequency or posture; the structure
repeats across joints and across sessions; and it is not an artefact of time
alignment, units, initial state or a logging bug. If the parameter model
already explains most of the error, the actuator-residual line stops there.

## 10. This round's stopping point

Design, code, dry runs and the checklist. **No motion.** The hardware backend
is deliberately absent so that no version of this repository predating the
approval can command an arm.
