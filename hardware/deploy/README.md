# Deploying the vision policy on a RealSense D405

The policy in `src/piper_push` sees a 224×168 depth image from a 52° camera at a
fixed place in the robot's base frame, plus 36 numbers of proprioception, and
writes six joint targets and a gripper command at 50 Hz. This directory turns a
D405 and a PiPER-X into exactly that.

Nothing here has run against the arm. There is no PiPER and no CAN interface on
the machine it was written on. What *has* run is everything that can be checked
against the simulator, which is most of it — see **What is verified** below.

---

## The five things that have to be right

| | | where |
|---|---|---|
| the image | the D405's 848×480 at 87° resampled into the policy's 224×168 at 52°, from wherever the camera actually is | `rectify.py` |
| the mask | which pixels are the object being fetched, with no segmentation buffer to read | `mask.py` |
| the numbers | 36 proprioceptive values in the order the actor was trained on | `proprio.py` |
| the command | policy output → joint target, through the same scaling, clipping and slew limit as training | `robot.py` |
| where the camera is | eye-to-hand calibration, because everything above depends on it | `calibrate.py` |

---

## The environment

Everything here runs in the `mjlab` environment, because it imports
`piper_push` for the robot model, the action scaling and the camera geometry —
sharing those rather than restating them is most of what makes the pipeline
agree with the simulator.

That environment did not have the camera driver. The bench in
`hardware/depth_bench/` was run under the base conda python, so
`pyrealsense2` was installed there and not here, and the first thing that
touches a camera raised `ModuleNotFoundError` several layers down. It is
installed now:

```bash
micromamba run -n mjlab pip install --no-deps pyrealsense2
```

`--no-deps` on purpose: the wheel is self-contained and nothing else in the
environment should move for it. `piper_sdk` is the same story and is not
installed, because there is no arm on this machine to talk to.

## Bring-up, in order

Every command is in the `mjlab` environment; see above for why, and for the
one package that had to be added to it.

```bash
M="micromamba run -n mjlab python"

# 0. what the actor expects.  Run once per checkpoint; everything reads it.
$M scripts/export_obs_spec.py

# 1. check the stack against the simulator.  No hardware needed, and this is
#    the check to run first and after every change.
$M -m hardware.deploy.selftest

# 2. export the policy.  Convert the distillation checkpoint to an actor
#    first, then export ON THE DEPLOYMENT TASK -- see the warning below.
$M scripts/student_to_actor.py <distill/model_1499.pt> /tmp/actor.pt
$M scripts/check_export.py /tmp/actor.pt \
    --task Mjlab-Pick-Place-PiperX-Vision --out /tmp/vision_policy
#    --checkpoint is what proves the graph is the policy that was trained.
$M -m hardware.deploy.selftest --policy /tmp/vision_policy \
    --checkpoint /tmp/actor.pt

# 3. dry run -- no camera, no arm -- then a replayed session, which is the
#    only thing that measures the real cost of the loop without hardware.
$M -m hardware.deploy.run --policy /tmp/vision_policy --dry-run --seconds 10
$M -m hardware.deploy.simrecord --frames 240 --out recordings/sim
$M -m hardware.deploy.run --policy /tmp/vision_policy --replay recordings/sim \
    --allow-nominal --seconds 20

# 3b. the arm's CAN link.  Needs root and the system's can-utils, so it is
#     not something this pipeline can do for itself:
#       sudo apt install can-utils
#       sudo bash $(python -c "import piper_sdk,os;print(os.path.dirname(piper_sdk.__file__))")/can_activate.sh can0 1000000
#       ip -details link show can0     # must say UP and bitrate 1000000
#     Then confirm the arm is actually talking before trusting anything else:
$M -m hardware.deploy.jointcheck --joint 1 --dry-run   # the script works
$M -c "from hardware.deploy import robot; a=robot.PiperArm(); a.connect(); \
       print(a.read()); print('jaw gap', a.check_gripper_range())"

# 4. mount the camera near the nominal pose, then measure where it really is.
#    The board goes ON THE GRIPPER -- see "Calibration" below for why, and for
#    what to pass if it is not the sheet in hardware/depth_bench/targets.
#    Either the guided page -- preflight, live coverage, solve, save:
$M -m hardware.deploy.calibgui                # http://127.0.0.1:8771
#    or the same steps from the prompt:
$M -m hardware.deploy.calibrate --preview     # does the detector see the board?
$M -m hardware.deploy.calibrate --collect     # vary the ORIENTATION
$M -m hardware.deploy.calibrate --solve
$M scripts/rig_to_sim.py                      # what the simulator now gets wrong

# 5. the camera, still no motion.  Watch the mask find things on the real table.
$M -m hardware.deploy.run --policy /tmp/vision_policy --no-arm --seconds 30 \
    --record recordings/look

# 6. and if the depth segmenter struggles on the real objects, train the
#    appearance model on what it did manage, and use that instead.
$M -m hardware.deploy.autolabel recordings/look --out hardware/deploy/yolo/data
$M -m hardware.deploy.train_yolo --epochs 60
$M -m hardware.deploy.run --policy /tmp/vision_policy --no-arm --mask yolo

# 7. the arm moves for the first time -- one joint, five degrees, and a
#    number.  This is what stands between a units error and the table.
$M -m hardware.deploy.jointcheck --joint 1
$M -m hardware.deploy.jointcheck --joint 2      # ... and so on

# 8. the policy drives.  E-stop in reach, workspace clear.
$M -m hardware.deploy.run --policy /tmp/vision_policy --seconds 30
```

Read the two summary lines every run prints. The first is the control loop and
it must show **0 overruns**; the second is the perception thread and its rate
is the observation's age. If either looks wrong, nothing downstream is worth
interpreting.

---

## What is verified, and how

`selftest.py` adds a second camera to the simulated scene with the D405's
resolution and field of view, at the place the D405 is meant to be mounted, and
feeds its depth image to the pipeline as if it had come off the sensor. Latest
run:

| stage | result |
|---|---|
| resampling 848×480 → 224×168 | 0.40 mm median, 1.24 mm at p95, over 35.6k near pixels; fill 1.0000 |
| camera model and extrinsic | no systematic offset: median −0.16 mm |
| observation assembly | 0.00025 median, 0.00150 at p99, in normalised units |
| target mask carried through the resampling | IoU 0.918 against the segmentation buffer |
| target mask, depth segmenter, measured sensor | recall 0.84–0.96 of the simulator's target pixels |
| tracker confirmation | one sighting is refused; three in five frames is accepted |
| proprioception, driven joints | exact to 2.2e-7 — joint_pos, joint_vel, ee_pose, gripper, actions, squeeze |
| proprioception, mimicked finger | 2.8e-4 against the simulator's equality constraint |
| action mapping | 2.6e-7 rad over 20 random actions against the trained command path |
| exported policy | runs in the loop and returns 7 actions |
| the two paths' hole rates | simulator 7.3%, robot 4.2% — 1.75x, the simulator harsher, which is the safe direction |
| the two paths' noise | 6.1 mm reconstructed against 6.5 mm applied directly, at 0.70 m |
| the exported graph is the trained policy | 1.9e-6 against a policy built on the deployment task |
| the image matters to this policy | blanking the camera moves the action by 0.67 |
| the policy cannot tell the two paths apart | 0.0147 between paths, 0.0122 within the simulator's own noise — 2% of what the image is worth |

And on a replayed session, which is the only thing that measures the loop
itself:

| | |
|---|---|
| control loop | 1.5 ms median, 1.8 ms at p95, 6.9 ms worst, **0 overruns** in 999 steps |
| perception thread | 21.5 ms median (46.6 Hz), 27.8 ms at p95 |
| observation freshness | 2 stale frames in 999 steps |

Seven of those numbers were wrong the first time and each one is a bug that
would have been invisible on the robot, or nearly so:

* **the intrinsics.** Feeding the real D405's principal point to a rendered
  image threw every ray by up to 8 mrad and put 9 mm of error in the resampled
  depth. `rectify.mujoco_K` exists because of it.
* **the slew limiter's starting point.** The trained action term seeds its
  previous target from *where the arm actually is*, not from the nominal pose.
  Starting from the nominal instead disagreed by 0.68 rad on joint 1 — on the
  robot, a lunge on the first command of every run.
* **the arm exclusion.** Spheres at each body origin leave a 300 mm link
  uncovered, and the segmenter reported the forearm as a 3000-pixel object and
  reached for it. It now uses every geom with MuJoCo's own bounding radius.
* **the thread pools.** numpy and onnxruntime size their pools to the machine.
  On 24 cores that is a loop doing under a millisecond of arithmetic and
  waiting fourteen for it: 14.8 ms median and 44 overruns in six seconds,
  against 1.5 ms and none with the pools pinned to one thread.
* **onnxruntime's own thread pool.** `OMP_NUM_THREADS` does not govern it —
  it runs an Eigen pool sized to the core count and busy-waits between calls.
  Uncapped, it starved the vision thread from 22 ms a frame to 878, a 40×
  slowdown in the only stage that costs anything. The control loop's own
  median moved from 1.0 ms to 1.5, so **nothing in the loop's timing showed
  it**; it was visible only as 397 stale frames in 648 steps. Fixed in
  `policy.Policy` with `intra_op_num_threads = 2` and spinning off.
* **the replay reader.** It decoded a 500 kB npz on every call and the vision
  thread called it in a tight loop, so 200 ms of file system was being
  reported as the cost of the pipeline. It now serves frames on the camera's
  own 30 Hz clock and decodes each one once, which is also what the camera
  does.
* **doing the vision inline.** Segmenting a quarter of a million points and
  re-rendering them costs 35 ms, and the control period is 20. The first
  version ran it in the loop and the overrun guard stopped it after 25
  consecutive misses — correctly. It now runs on its own thread at the
  camera's rate, which is where it belongs: there is no new information
  between frames.

What the selftest cannot check: the calibration itself, the RealSense driver,
the CAN units, and whether a real D405 pointed at a real table produces
something like a rendered one.

---

## Calibration

**The board goes on the gripper, not on the table.** This is eye-to-hand: the
camera is bolted to the world and cannot see itself, so the only way to relate
it to the base frame is to show it something whose base-frame pose is already
known, and the only such thing is the arm. A board lying on the table has an
unknown pose in both frames and constrains nothing. If the board that cannot
be gripped is the precision one, print
`hardware/depth_bench/targets/target_a4.pdf` and grip that instead — the
precision board still earns its keep on the table, for the depth checks in
`hardware/depth_bench`, which are about range and not about pose.

**Size it before buying it.** Hand-eye error is set by how many pixels the
pattern spans and how many corners it has, and a target that is too small
cannot be rescued by collecting more poses. Measured synthetically — this arm's
workspace, the D405's intrinsics, 0.2 px of corner-localisation noise, error in
the recovered `T_base_cam`, median of 40 runs:

| target | corners | span | 12 poses | 30 poses |
|---|---|---|---|---|
| one 40 mm ArUco | 4 | 37 px | **122 mm** | **87 mm** |
| one 80 mm ArUco | 4 | 73 px | 11 mm | — |
| ChArUco 5×5, 25 mm | 16 | 67 px | 8.1 mm | 5.2 mm |
| ChArUco 5×5, 33 mm (the A4 sheet) | 16 | 88 px | 6.1 mm | 3.7 mm |
| ChArUco 9×7, 25 mm | 48 | 137 px | 2.0 mm | 1.4 mm |
| ChArUco 12×9, 25 mm | 88 | 193 px | 1.1 mm | — |

Two things to read off it. **A single marker is not a calibration target**, at
any pose count: four coplanar corners spanning 25 pixels leave the planar-pose
ambiguity nearly degenerate, so the error does not average down — 8 poses and
30 poses are both about 100 mm, and `--solve` refuses both. And **`--collect`
is cheap, so collect more than eight**: at fixed board size the error falls
roughly as 1/√n, which is the only free variable left once the board is bought.

Detection is not the limit and should not be confused with it: a 40 mm marker
at 0.70 m is 26 px on a side and decodes 100% of the time. It is found
perfectly and located uselessly.

**Motion-capture spheres**, measured the same way — the same arm poses, the
same 0.2 px of feature noise, only the target geometry changed:

| target | 12 poses | 30 poses |
|---|---|---|
| 3 spheres, 80 mm, coplanar | 536 mm | 404 mm |
| 3 spheres, 80 mm, 20 mm relief | 614 mm | 469 mm |
| 4 spheres, 80 mm, 30 mm relief | 10.5 mm | 8.4 mm |
| 5 spheres, 120 mm, 40 mm relief | 10.9 mm | 6.2 mm |
| 8 spheres, 150 mm, 50 mm relief | **3.4 mm** | **2.8 mm** |
| the A4 ChArUco sheet, 16 corners | 4.6 mm | 4.1 mm |

The interesting row is the last two. A well-spread **non-coplanar** cluster
beats the flat board, and it should: a plane is the worst-conditioned case for
PnP, and 50 mm of relief removes the ambiguity the board has to live with. So
the idea is sound and it is not sound in its usual form — three spheres is not
a pose, it is a P3P problem with up to four solutions, and it comes out
hundreds of millimetres wrong.

Three practical reasons it still loses here, none of them geometric. **This
camera has no IR projector** (`hardware/depth_bench/README.md`), so a
retroreflective sphere has no co-axial illuminator to retro-reflect to and is
simply a white ball — and a matte white surface is the one this bench measured
as the camera's worst case. **Spheres have no identity**, so correspondence has
to be inferred from the geometry and a wrong permutation is a wrong pose with
no symptom, where every ArUco marker states which one it is. And **the cluster
has to be measured**: the whole argument above assumes the ball positions are
known to a fraction of a millimetre, and a hand-built cluster whose model is
wrong is a uniformly wrong ruler — self-consistent, small residual, wrong
answer, exactly the failure the square size has.

**Say which board it is.** The defaults describe the printed A4 sheet. Anything
else needs its numbers, and the failure mode for getting them wrong is not a
bad answer, it is `board not found` at every pose with nothing saying which of
the five numbers is wrong. `--preview` is there to be run first, with the board
in view, until it prints corners.

```bash
# a bought 12x9 ChArUco with 25 mm squares
$M -m hardware.deploy.calibrate --preview --squares 12x9 --square-mm 25

# a plain checkerboard -- SQUARES is INNER CORNERS, one fewer each way
$M -m hardware.deploy.calibrate --preview --board-kind checker \
    --squares 11x8 --square-mm 25
```

Three ways a board goes wrong quietly, all of them handled and none of them
detectable by looking at the image:

* **The square size scales the answer.** Solving a 25 mm board as a 33 mm one
  multiplies the camera's distance by 1.32 and leaves the residual small,
  because the residual measures self-consistency and a uniformly wrong ruler is
  perfectly self-consistent. The board description is therefore stored in
  `calib_poses.json` with the poses, and `--solve` uses that one.
* **A checkerboard has a 180° symmetry.** Rotated half a turn it is the same
  image, so the detector's corner order can flip between poses; hand-eye fed a
  mixture solves for a camera that is not there. `solve` detects and undoes it
  using the fact that `A` and `B` in `AX = XB` are conjugate and so must rotate
  by the same angle — which needs no solution and therefore is not circular.
  It prints how many poses it flipped. ChArUco does not have the problem.
* **ChArUco origin conventions changed in OpenCV 4.6.** A board numbered the
  old way still detects perfectly and puts its origin at a different corner,
  which moves the answer by the width of the board and looks exactly like a
  mounting error. `--legacy` if the residual is fine and the camera lands a
  board-width from where it obviously is.

**Or drive it from the page.** `python -m hardware.deploy.calibgui` serves
`http://127.0.0.1:8771` and walks the same four steps, which exists because the
CLI asks the operator to track four things at once and shows one of them. It
gates the record button on the three that are checkable — the board is
detected, the view has stopped moving, and this pose is actually different from
the ones already recorded — and draws the fourth, the rotation spread, as a
disc of where each recorded pose faces. Spreading those dots *is* the task, and
a picture of where they are not beats a number that says 24°.

Its preflight also answers the question that is expensive to get wrong: **is the
board on the gripper at all.** It runs forward kinematics on the live joint
angles, puts the detected board into the base frame through the *nominal*
extrinsic, and reports the separation. On the rig this was written against it
read 411 mm — the board was propped on the table, where it constrains nothing,
and every one of the thirty poses would have been wasted before `--solve` said
so.

**Then take it back to the simulator.** `scripts/rig_to_sim.py` reads
`rig.json` and states the calibration as the axes `piper_push.perturb` already
defines, against the ranges training randomised. It prints the
`accept_s1.py` flags that replay the measured rig in simulation — run that
against the nominal too, and the difference is what this mount costs — and, if
anything is outside the trained envelope, the two edits that would close it and
which one is the right kind of decision. Three quantities it can only name:
the camera's roll, its lateral offset, and the table's tilt. `SessionMismatchCfg`
has no term for any of them.

## The mask, and why there are two of them

In simulation this channel is exact. On the robot nothing knows which pixels
are the object, and the channel still has to be filled — this is where the
deployment actually differs from the training environment, and
`pick_place.mdp.CameraScene` now corrupts the simulated mask to match what can
be delivered: it drops wherever the depth dropped, and its boundary is a pixel
out.

**`DepthSegmenter`** needs no training. Fit the table plane, keep what stands
above it, subtract the arm and the bin, label the connected components. It runs
on the sensor's grid halved — 424×240, where a 40 mm object is still 12 pixels
across, a third more than the 9 it will have on the policy's, and a quarter of
the cost: 13.9 ms against 37.9. The *depth* is never decimated; that is the
measurement.

Three things in it are not obvious and all three come from the bench
measurements:

* **the table plane is fitted every frame, not taken from the calibration.**
  The camera's range bias is a per-unit unknown — measured at −14 mm at 0.7 m
  on this one — and a bias that size against a fixed height threshold either
  turns the whole table into objects or hides the short ones.
* **the tracker confirms across frames.** A blob of correlated depth noise two
  standard deviations high looks exactly like a short object; requiring three
  sightings in the last five frames removes the ones that move. It costs about
  100 ms of latency once, when the scene changes.
* **the height threshold follows the noise.** What confirmation *cannot* remove
  is the third of the sensor's error that does not change between frames: a
  static bump is a phantom in the same place every time, and only its height
  distinguishes it. So the threshold is 3.4 times the table's own measured
  scatter, not a constant. A constant was wrong in both directions — 20 mm took
  the segmenter from 8.6 instances per frame to about 2 under the fitted sensor
  model, and then rejected a real object outright in a scene with no noise at
  all, because 20 mm is most of the height of the shortest thing the task
  uses.

**`YoloSegmenter`** is for the case the first one cannot handle at all. The
bench measured the D405's fill rate on a blank white surface at 88% on average
and 42% in the worst shot, and a depth segmenter has nothing to segment where
there is no depth. Colour is unaffected by that failure.

Nobody labels anything. `autolabel.py` runs the depth segmenter over recorded
sessions, keeps only the frames where it was confident — high fill, instance
confirmed, the instance's own pixels mostly valid — and writes its output as
ground truth. The model is trained on the easy frames and asked to generalise to
the hard ones, which is the right way round: the hard frames are hard because
the depth is missing, not because the object looks different.

```bash
python -m hardware.deploy.run --policy … --no-arm --seconds 120 --record recordings/table
python -m hardware.deploy.autolabel recordings/table
python -m hardware.deploy.train_yolo --epochs 60
python -m hardware.deploy.run --policy … --mask yolo
```

`simrecord.py` writes a session in the same format out of the simulator, so the
whole labelling and training path runs with no hardware. It is a check on the
plumbing and **not** a training set: the scene renders untextured and unshaded,
so a model fitted to those images has learned what MuJoCo looks like.

---

## Files

```
config.py       every number, and which of three places it came from
rectify.py      D405 point cloud re-rendered through the policy's camera
mask.py         table-plane segmentation, tracking, and the YOLO backend
obs.py          the three channels, matching mdp.CameraScene line for line
proprio.py      the 36 numbers, with FK from the simulator's own robot model
policy.py       the exported ONNX actor, hidden state carried by hand
robot.py        action mapping, a PiPER CAN backend, and a dry-run stand-in
sensor.py       the D405 on its own thread, at the settings the bench measured
calibrate.py    eye-to-hand, any board, residuals reported, refuses to guess
calibgui.py     the same calibration, guided, in a browser -- + calibgui.html
run.py          the 50 Hz loop, and what it does when something is wrong
jointcheck.py   one joint, five degrees: the CAN units, before anything else
selftest.py     all of the above, against the simulator, nothing plugged in
simrecord.py    a synthetic session, for exercising the labelling path
autolabel.py    depth segmenter -> YOLO dataset, no hand labels
train_yolo.py   fine-tune YOLO26-seg on it
```

## Export the policy on the task it will be deployed on

`scripts/check_export.py` exports a graph and compares it against the policy it
built **from the same task config**. That catches a broken export. It cannot
catch the wrong one.

Exported with `--task Mjlab-Pick-Place-PiperX-Distill`, the graph came out
disagreeing with the trained policy by **4.35** on actions of magnitude 1–5 —
a different network entirely — and `check_export.py` printed `OK`, because it
had compared that network to itself. Everything downstream looked plausible:
the graph loaded, took the right input shapes, returned seven numbers, and ran
in 0.7 ms. Its actions were near-constant at ±0.09, which is the only sign
there was, and it takes knowing the right magnitude to notice.

So: convert to an actor checkpoint, export on `Mjlab-Pick-Place-PiperX-Vision`,
and run `selftest.py --checkpoint`, which loads the checkpoint into a policy
built on the deployment task and compares that against the graph as
`deploy.policy` actually feeds it. Wrong task: 3.78. Right task: 1.9e-6.

## One thing that will bite whoever installs this

`cv2.calibrateHandEye` does not exist in OpenCV 5. The `CALIB_HAND_EYE_*`
constants are still exported, so a check for them passes and the call then
raises `AttributeError`. Ultralytics pulls OpenCV 5 in, so any environment with
the YOLO backend has it. `calibrate.calibrate_hand_eye` is Park and Martin's
method written out — twenty-five lines, tested against a known pose to under a
millimetre in `tests/test_deploy.py` — rather than a pin on `opencv-python<5`
that would then fight ultralytics.

## Known gaps

* `robot.PiperArm` has still never driven an arm, but it is no longer
  unverified. `piper_sdk` 0.6.2 is installed, and
  `tests/test_deploy.py` now builds every CAN message the command path emits —
  same classes, same arguments — and lets the SDK's own constructors validate
  them. That found one real bug on its first run: the gripper effort was sent
  as `int(GRIPPER_FORCE_N * 1000)` = 10000, for a field documented and
  validated as 0–5000, so **every gripper command would have raised inside the
  control loop**. The cause was a unit *kind* error, not a scale one — the
  simulator's 10 N is a force on a prismatic finger joint and the CAN field is
  a torque in 0.001 N·m, and the two are not convertible without a lever arm
  the vendor does not publish. `GRIPPER_TORQUE_NM` is now its own deployment
  constant at 1.5 N·m, which is a starting point and not a measurement.
* `JointCtrl` has **no** range validation of its own, unlike the gripper
  message. `PiperArm.command` therefore re-clips to `SAFE_TARGET_CLIP` at the
  CAN boundary even though `ActionMapper` already did — it is the last six
  comparisons before the drives, and a units error upstream is a number they
  will try to achieve.
* **The gripper's jaw gap has to be configured.** The simulator's is 100 mm
  (`piper_push.robot.GRIPPER_OPEN_M`, measured); the SDK's
  `GripperTeachingPendantParamConfig` defaults `max_range_config` to 70. On a
  70 mm arm the top 30% of the policy's gripper command does nothing and the
  opening it reads back never exceeds 70 — the policy commands a gap it never
  observes, and no log names that. `PiperArm.check_gripper_range()` reads back
  what the arm is set to.
* `pad_contact` is reconstructed from the gripper drive's load, because the two
  contact sensors it reads in simulation do not exist. Both channels carry the
  same bit. The threshold (`ProprioBuilder.contact_effort`) is a guess until
  someone squeezes something and reads the number.
* The observation is one vision period old — about 22 ms, or 1.1 control
  steps. That is inside the 2–4 steps `piper_push.perturb` calls the hardware
  range for camera-and-inference delay, so it is a delay the policy has been
  evaluated against, but nobody has measured this particular distribution.
* `--mask yolo` has been run end to end only on synthetic data.
* **The depth segmenter has a size limit.** Over 24 fresh scenes under the
  measured sensor it found the object in 22, with a median IoU of 0.59; the
  two it missed were 8 and 78 pixels of the policy's image and the smallest it
  found was 88. So the limit sits somewhere between 78 and 88 pixels — a
  9×9 patch — and below it the thresholds that reject noise blobs start
  rejecting objects too. That is the sensor, not the algorithm: lowering the
  thresholds brings the phantoms back, and a phantom is worse than a miss
  because the arm reaches for it. Closing it is what the YOLO backend is for.
  Re-measure with `selftest.py --mask-sweep 24` after any change to `mask.py`.
