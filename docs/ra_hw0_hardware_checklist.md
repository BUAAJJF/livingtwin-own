# RA-HW-0 — hardware safety checklist

**This is an operator's document.** Every line is checked by a person, on the
day, in this order. Nothing in this repository can tick any of them, and no
software gate substitutes for one.

**Status: no arm has been audited.** `can0` does not exist on this machine and
no PiPER has been read, so §D below is entirely unfilled and motion is
blocked. See `docs/ra_hw0_experiment_plan.md` §2 for what is UNKNOWN.

---

## A. Before power

- [ ] The arm's base is bolted to the bench, and the bolts have been checked
      by hand today. A PiPER's own mass will move an unbolted base during a
      reversal.
- [ ] The gripper is **empty** and the arm is otherwise unloaded.
- [ ] Nothing is inside the swept envelope: no tools, no cups, no cables, no
      camera tripod, no laptop.
- [ ] No part of any cable can enter the envelope as the arm moves. Route and
      strain-relieve before power, not after.
- [ ] The emergency stop is within the operator's reach **from where the
      operator will actually be standing**, not from where the laptop is.
- [ ] A second person knows a run is happening, or the door is shut.
- [ ] The table surface below the arm is clear, so a fall damages one thing.

## B. Emergency stop and watchdog — verified, not assumed

- [ ] The **hardware** e-stop has been pressed, with the arm powered and
      enabled but stationary, and the arm was observed to lose motion. This is
      done by the manufacturer's procedure and it is done **before** the first
      trajectory, every session.
- [ ] Releasing the hardware e-stop is confirmed **not** to resume motion on
      its own. If it does on this arm, that fact is written here and the
      session does not proceed until a procedure exists that makes it safe.
- [ ] The **software** e-stop path is known: `MotionCtrl_1(emergency_stop=0x01)`
      on CAN ID 0x150, with `0x02` as *resume*. RA-HW-0 never sends `0x02`.
- [ ] Comms-loss behaviour has been observed deliberately: with the arm
      enabled and stationary, the CAN cable is pulled and what the arm does is
      written down. **Until this has been done once, on this arm, the answer
      is UNKNOWN and no trajectory runs.**
- [ ] The host watchdog has been tested in dry run: `scripts/ra_hw0_collect.py`
      stops on a telemetry gap over 100 ms, and the test suite covers it.
- [ ] The data recorder has been confirmed writing; a session whose log dies
      is a session that stops.

## C. Software state

- [ ] `python scripts/ra_hw0_audit.py --hardware` has been run and its report
      says `device.status == "READ"`.
- [ ] `configs/ra_hw0_safety_limits.json` has **no** `UNKNOWN` row.
- [ ] `python scripts/ra_hw0_replay.py` has been run **today**, its preview is
      `OK`, and the operator has read the printed envelope for every segment.
- [ ] The full test suite is green.
- [ ] The gripper is not commanded by anything in this phase, and the
      collector records that as a deliberate field rather than an omission.
- [ ] `git status` is clean for everything this session will use.

## D. The arm itself — **all UNKNOWN until an audit reads one**

- [ ] Model and degrees of freedom: ______________
- [ ] Serial number: ______________
- [ ] Firmware version (`GetPiperFirmwareVersion`): ______________
- [ ] Configured gripper jaw gap (`max_range_config`): ______ mm
      *The SDK's default is 70 mm and the simulator assumes 100 mm; on a 70 mm
      arm the top 30% of the trained gripper command does nothing.*
- [ ] Device-reported per-joint angle limits, max speed
      (`GetAllMotorAngleLimitMaxSpd`): ______________
- [ ] Device-reported max acceleration (`GetAllMotorMaxAccLimit`):
      ______________
- [ ] Telemetry rate and jitter measured over ≥30 s: ______ Hz, p99 gap
      ______ ms
- [ ] Still-arm current baseline per joint: ______________ A
- [ ] Driver and motor temperature at rest: ______ °C / ______ °C
- [ ] Bus voltage at rest: ______ V
- [ ] **Positive rotation sense of each joint, verified against the physical
      arm.** Nothing in the SDK or this repository establishes it, and a sign
      error sends the first trajectory the wrong way.

## E. Per-joint absolute stop thresholds

Units are radians and radians per second. The absolute bounds are the
**intersection** of the vendor's parameter table (`piper_sdk` 0.6.2) and the
envelope the policy was trained inside (`piper_push.robot.SAFE_TARGET_CLIP`);
the H1 envelope is a ±0.20 rad window around the start posture, kept 0.15 rad
clear of every absolute bound.

| joint | absolute min | absolute max | H1 envelope | start | stop \|q̇\| | stop \|q̈\| | stop tracking error |
|---|---|---|---|---|---|---|---|
| joint1 | −2.6179 (−150.0°) | +2.6179 (+150.0°) | [−0.2000, +0.2000] | +0.00 | 0.35 rad/s | 8.0 rad/s² | 0.05 rad |
| joint2 | +0.0000 (0.0°) | +3.1400 (+179.9°) | [+0.8500, +1.2500] | +1.05 | 0.35 rad/s | 8.0 rad/s² | 0.05 rad |
| joint3 | −2.9670 (−170.0°) | +0.0000 (0.0°) | [−1.2500, −0.8500] | −1.05 | 0.35 rad/s | 8.0 rad/s² | 0.05 rad |
| joint4 | −1.5533 (−89.0°) | +1.5533 (+89.0°) | [−0.2000, +0.2000] | +0.00 | 0.35 rad/s | 8.0 rad/s² | 0.05 rad |
| joint5 | −1.2200 (−69.9°) | +1.2200 (+69.9°) | [−0.2000, +0.2000] | +0.00 | 0.35 rad/s | 8.0 rad/s² | 0.05 rad |
| joint6 | −2.0944 (−120.0°) | +2.0944 (+120.0°) | [−0.2000, +0.2000] | +0.00 | 0.35 rad/s | 8.0 rad/s² | 0.05 rad |

**A finding worth reading twice.** The simulator's joint-5 clip is ±1.3090 rad
and the vendor's limit is ±1.2200 — **the trained envelope is 0.089 rad (5.1°)
wider than the arm allows at each end**. A policy trained in it can command an
angle the drive refuses and reports as `arm_status = 0x04`. Joints 1, 2, 3 and
6 exceed the vendor table by 0.0001–0.0016 rad, which is that table's own
rounding. Joint 4's simulated clip is 0.19 rad *tighter* than the vendor's,
deliberately. Every bound above is the intersection, so nothing in RA-HW-0
commands past either.

## F. Non-joint stop thresholds

| quantity | threshold | source |
|---|---|---|
| telemetry gap | 0.10 s (five control periods) | derived |
| host command watchdog | 0.10 s | derived |
| driver temperature | **UNKNOWN** | blocks motion |
| motor temperature | **UNKNOWN** | blocks motion |
| joint current | **UNKNOWN** | blocks motion — set from H0's measured baseline plus a documented margin |
| bus undervoltage | **UNKNOWN** | blocks motion |
| any `foc_status` fault bit except `driver_enable_status` | stop | vendor |
| any `arm_status` other than `0x00 normal` | stop | vendor |
| any per-joint `err_code` angle-limit or communication bit | stop | vendor |

## G. During a run

- [ ] The operator's hand is near the e-stop for the whole segment and the
      operator is watching the **arm**, not the terminal.
- [ ] Before each segment the terminal has printed its duration, the joints it
      moves, the commanded amplitude and peak speed, and its envelope. If it
      has not, the segment does not run.
- [ ] No segment exceeds **5 s** in H1.
- [ ] After every segment the arm is stationary and a human types the
      confirmation for the next one. There is no automatic advance.

## H. After a stop

- [ ] **Nothing is cleared, retried or resumed automatically**, and no code in
      this phase can. A stop ends the session.
- [ ] The reason is read off the session's `.meta.json` before anything else
      happens.
- [ ] If the stop was a fault, the arm is inspected physically before power is
      cycled.
- [ ] `DisableArm` is understood: it removes holding torque **immediately**
      and the arm falls. It is used only with the arm physically supported or
      parked, and never as an ordinary software close. RA-HW-0's tools call
      `disconnect()`, never `close()`.

## I. Interface behaviours that can move the arm

Written here because they are surprising, and because two of them are in this
repository rather than in the vendor's SDK.

| entry point | what it does |
|---|---|
| `hardware/deploy/robot.py: PiperArm.close(disable=False)` | calls `hold()`, which calls `command()` — **closing the client transmits a joint command**. RA-HW-0 uses `disconnect()`. |
| `PiperArm.close(disable=True)` / `DisableArm(7)` | removes holding torque; observed on this rig as the arm falling when the calibration GUI exited |
| `MotionCtrl_1(grag_teach_ctrl=0x01)` | enters drag-teach; the arm becomes back-driveable and can fall |
| `MotionCtrl_1(track_ctrl=0x02)` | continues a stored trajectory — motion with no new position command |
| `MotionCtrl_1(emergency_stop=0x02)` | **resume** from e-stop |
| `MotionCtrl_2(ctrl_mode=0x07)` | offline trajectory mode |
| `EnableArm(7)` | energises the drives |
| `ConnectPort(piper_init=True)` | sends three **enquiry** frames only (max angle/speed, max acceleration, firmware). No motion — this is the one on the list that is safe, and it is on the list so that nobody has to wonder. |
