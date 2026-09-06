# Deploying the vision policy on a RealSense D455 + PiPER-X

The mask policy in `src/piper_push` sees a 224×168 depth image from a 52°
camera at a fixed place in the robot's base frame, plus 36 numbers of
proprioception, and writes six joint targets and a gripper command at 50 Hz.
This directory turns a D455 (a D405 until 2026-08-25) and a PiPER-X into
exactly that.  The point-cloud policies of branch `yf/pc` reuse the same loop
through `run.py --obs pc` (`pc_perception.py`, `pc_obs.py`); their shadow
runner is `pc_run.py` and their runbook is `docs/pc_shadow_runbook.md`.

This stack has driven the arm: on 2026-09-01 the `d455_v4_final` policy
picked objects off the table and placed them in the bin from the calibrated
D455 alone (`recordings/v4_stereo_try3`; the command is in the root README).
The text below was written before that and describes the checks that made it
possible; where a number is quoted it is the D405-era measurement unless it
says otherwise.

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
$M -m hardware.deploy.run --policy /tmp/vision_policy --replay recordings/v4_stereo_try3 \
    --seconds 20                                # a recorded D455 session, dry arm

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
# If launching through `micromamba run` directly, attach its streams so the
# ~30 s model import and each hardware startup stage are visible immediately:
micromamba run -a "" -n mjlab python -u -m hardware.deploy.calibgui
#    or the same steps from the prompt:
$M -m hardware.deploy.calibrate --preview     # does the detector see the board?
$M -m hardware.deploy.calibrate --collect     # vary the ORIENTATION
$M -m hardware.deploy.calibrate --solve
$M scripts/rig_to_sim.py                      # what the simulator now gets wrong

# 5. the camera, still no motion.  Watch the mask find things on the real table.
$M -m hardware.deploy.run --policy /tmp/vision_policy --no-arm --seconds 30 \
    --record recordings/look

# 6. the appearance backend (--mask yolo / fused) loads the weights under
#    hardware/deploy/yolo_d455/.  The labelling and training tools that made
#    them are in git history before 2026-09-06; the best run used --mask depth.

# 7. the arm moves for the first time -- one joint, five degrees, and a
#    number.  This is what stands between a units error and the table.
$M -m hardware.deploy.jointcheck --joint 1
$M -m hardware.deploy.jointcheck --joint 2      # ... and so on

# 8. the policy drives.  E-stop in reach, workspace clear.
$M -m hardware.deploy.run --policy /tmp/vision_policy --seconds 30
```

The policy command intentionally omits `--min-table-clearance`. Light
fingertip/table contact is allowed: the real setup has no table-force signal,
so a binary simulator contact is neither a training termination nor a hardware
deployment condition. Joint-speed, malformed-action, stale-observation and
communication guards remain active. `--min-table-clearance` is still available
as an explicitly requested conservative calibrated-geometry stop.

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

For the real rig the recommended target is now
[`calib_compact_v2.pdf`](../depth_bench/targets/calib_compact_v2.pdf): a
180x152 mm finished sheet with a 168x140 mm, 6x5 ChArUco pattern, 28 mm
squares, 22 mm `DICT_4X4_50` markers and 20 interpolated corners.  It is much
smaller than A4, has more constraints than the old 5x5 board, and its shorter
4x4 code is easier to decode at 30--50 cm.  Print **100% / actual size** on
matte stock, retain the 6 mm white border, verify a 28 mm square with calipers,
and bond it flat to a rigid plate.  Its machine-readable description is the
adjacent `calib_compact_v2.json`.

For an inkjet printer prefer the separately named
[`calib_compact_white_v2.pdf`](../depth_bench/targets/calib_compact_white_v2.pdf).
It is the exact geometric inverse, reducing measured black coverage from 63%
to 23%, and its adjacent JSON sets `inverted: true` so the detector enables
white-marker decoding.  Do not use the normal JSON with the white print: board
polarity is stored as part of pose-file identity specifically to prevent that
mix-up.

The ready-to-print variant is
[`calib_compact_white_v2_cut.pdf`](../depth_bench/targets/calib_compact_white_v2_cut.pdf):
an A4 page with a 0.20 mm crop rectangle around the exact 180x152 mm finished
target.  Print the A4 page at 100% and cut through the centre of that line; the
6 mm white quiet border remains inside the cut on every side.

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

The GUI's calibration stream is deliberately different from deployment's
depth-aligned preview.  It opens depth plus the D405's **raw, unwarped left Y8
imager at 1280x720**.  That imager defines the depth optical frame, so no
colour-to-depth warp or extra extrinsic is needed; deployment stays at its
characterised 848x480 mode.  A recorded pose is the median of matching corner
IDs across at least eight settled frames, not one lucky PnP result.  The final
closed-form Park--Martin result is then jointly refined against every saved
corner pixel with a robust loss, sharing one camera pose and one rigid
board-to-gripper pose across the session.

Start a fresh, independent session for the compact board (the old pose file is
left untouched):

```bash
micromamba run -a "" -n mjlab python -u -m hardware.deploy.calibgui \
  --board hardware/depth_bench/targets/calib_compact_white_v2.json \
  --poses hardware/deploy/calib_poses_compact_white_v2.json
```

The assisted workflow after the manual bootstrap is:

1. Move and record five visibly different poses by hand, including at least
   15° of rotation spread.
2. The page computes a rough, navigation-only extrinsic.  This does not relax
   the final solver: it still refuses below eight poses and 30°.
3. Using the current board detection, the rough extrinsic and the robot's own
   MuJoCo kinematics, it searches nearby joint poses.  Candidates outside the
   D405 image, below 22 projected marker pixels at 1280 width, too edge-on,
   inside a joint-limit margin, too similar to existing samples, or introducing
   a new model self-collision are rejected.
4. The chosen pose appears as a cyan predicted board outline over the live
   grayscale image, with joint angles and grasp-site position in the page.
   Clicking **move arm to the previewed target** is the confirmation: the arm
   enables, follows a 0.22 rad/s rest-to-rest joint trajectory, and stops on a
   feedback tracking error.  **stop and hold** cancels an in-progress stream
   and commands the measured pose.  Once the multi-frame fusion gate turns
   green, record the pose and the next target is generated.

An existing `rig.json` can seed guidance immediately; as soon as the current
session has enough bootstrap poses, its rough solve supersedes that old result.
Automatic motion checks the model and the camera view, not the physical room:
the operator must still keep the real swept volume clear and confirm every
move from the page.

The page also has a separate **Table calibration · loose checkerboard** card.
It is configured for the rig's 11x8-inner-corner checkerboard with 25 mm
squares.  Lay that loose board completely flat, wait for the magenta overlay
and eight-frame stillness gate, record it, then move it at least 50 mm to a new
part of the table.  Three placements are the minimum and six spread across the
working area are recommended.  These observations never enter the hand-eye
pose file and never command the arm: their known metric corner planes are
transformed through the solved extrinsic and jointly fitted.  **fit and write
table to rig.json** reports table height at the robot-base origin, tilt,
cross-placement spread and x/y coverage.  A thick backing measures its top
surface, so subtract that thickness or use the printed sheet directly if the
physical tabletop height is required.

Exiting the GUI **holds the measured joint position and disconnects CAN; it
does not disable the drives**.  On the real PiPER, `DisableArm(7)` removes
holding torque immediately and the arm can fall.  Intentional drive disable is
available only as the explicit `PiperArm.close(disable=True)` API and must be
done with the arm physically supported or safely parked.

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

Nobody labelled anything: the depth segmenter ran over recorded sessions and
its confident frames -- high fill, instance confirmed, the instance's own
pixels mostly valid -- became the ground truth (90 accepted D455 frames; mask
mAP50 0.931).  The labelling and training tools (`autolabel.py`,
`train_yolo.py`, `collect_yolo_gui.py`, `simrecord.py`) were retired on
2026-09-06 and are in git history; the trained weights stay under
`yolo_d455/` and `YoloSegmenter` still loads them.  Pure YOLO was never the
deployed mode; the runs that worked used `--mask depth`.

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
yolo_backend.py the YOLO26-seg appearance backend (weights under yolo_d455/)
sam_tracker.py  SAM2.1 as a causal tracker for the target (+ sam2_predictor, rgbmap)
lifecycle.py, target_mask.py   which instance is the target, and what the policy is shown
stereo.py       Fast-FoundationStereo depth from the raw imagers (TensorRT)
scene.py        does the table match a run you want to repeat
review.py, logview.py, graspview.py   look at a recording, frame by frame
gripcal.py, mit.py, sysid.py   the gripper drive, the MIT command boundary, plant identification
pc_obs.py, pc_perception.py, pc_run.py   the point-cloud policies (branch yf/pc)
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
* **`--mask fused` still misses the frame period, and the network is not why.**
  The campaign notes record the D455 segmentation model at 4.5 ms p50 under
  CUDA PyTorch and recommend fused depth + YOLO on the robot. Measured as a
  *stage*, through this loop, on a replayed D455 session, fused cost 127.8 ms
  a frame and took the control loop from zero overruns to 23.

  Almost all of it was `arm_mask`, at 47 ms of a 20 ms budget -- see below.
  With that fixed:

  | `--mask` | perception | control loop |
  |---|---|---|
  | `depth`, `--device cpu` | 20.1 ms, 49.8 Hz | 0 overruns |
  | `depth`, `--device cuda` | **19.6 ms, 51.1 Hz** | **0 overruns** |
  | `fused`, both on cuda | 55.9 ms, 17.9 Hz | 52 overruns |

  So `depth` is the default and it is comfortably ahead of the camera. Fused is
  still 3x the frame period, and the forward pass is 8.1 ms of it. Where the
  rest goes, per frame, measured:

  | | |
  |---|---|
  | the network | 8.1 ms |
  | `fit_table_plane` | 4.3 ms |
  | `arm_mask` | 2.1 ms |
  | unprojection, workspace, bin, the index scatter | 2.2 ms |
  | placing each detection against the depth | ~6 ms |

  And `FusedSegmenter` runs **all of that twice** -- `DepthSegmenter` at
  decimate 2 and `YoloSegmenter` at decimate 1 each unproject the frame, fit
  the plane, and subtract the arm and the bin for themselves. Sharing one
  geometry pass between them is worth about 7.5 ms and is the obvious next
  thing; until it is done, deploy with `depth`.

* **`arm_mask` was 47 ms a frame, and it is the reason none of the above
  fitted.** It subtracts the robot from the point cloud using the sphere cover
  of its own geometry, and it already had a per-sphere bounding box in front of
  the squared-distance test. But a per-sphere box is still spheres x points:
  24 boxes over the 129k workspace points, each allocating three `(n, 3)`
  boolean temporaries, single-threaded because the loop pins the thread pools.
  The distance test was never the cost; proving that the table is not the robot
  was.

  Two changes, both exact -- the output is bit-identical over 30 frames at
  randomised arm poses:

  * one box around **every** sphere first, in a single column-wise pass. A
    point inside sphere *i* satisfies `p > centre_i - radius_i >= lo`, so
    nothing that the per-sphere tests would have dropped is discarded. It
    removes 84% of the workspace.
  * the per-sphere test narrows one axis at a time, carrying indices instead of
    masks, so the second and third comparisons never see what the first ruled
    out.

  **47.0 ms to 2.43 ms, 19.3x.** `DepthSegmenter` went 19.4 to 8.4 ms,
  `YoloSegmenter` 65.5 to 22.9, and the depth path from 31.9 ms a frame to
  20.1. Re-measure with `--replay` after any change to it; the standalone
  segmenter benchmarks are about 2x optimistic against the threaded loop.

* **`--camera` did not choose the YOLO weights.** `--camera d455` with the
  default weights loaded `yolo/best.pt`, the D405 model, and nothing said so —
  the rig file has a serial cross-check and this had nothing. The weights now
  follow the camera and a model belonging to the other one is refused.
* **The depth segmenter has a size limit.** Over 24 fresh scenes under the
  measured sensor it found the object in 22, with a median IoU of 0.59; the
  two it missed were 8 and 78 pixels of the policy's image and the smallest it
  found was 88. So the limit sits somewhere between 78 and 88 pixels — a
  9×9 patch — and below it the thresholds that reject noise blobs start
  rejecting objects too. That is the sensor, not the algorithm: lowering the
  thresholds brings the phantoms back, and a phantom is worse than a miss
  because the arm reaches for it. Closing it is what the YOLO backend is for.
  Re-measure with `selftest.py --mask-sweep 24` after any change to `mask.py`.
