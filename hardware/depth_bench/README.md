# Depth bench — choosing the camera the policy will actually see through

Three candidates, one seat: **RealSense D405**, **ZED X**, **Manifold Tech
Odin 1**. This directory is how the choice gets made on evidence rather than on
datasheets, and it exists because of a comment in `src/piper_push/camera.py`:

```python
# Depth sensor realism.  These are PLACEHOLDERS: a real depth camera's noise
# grows with distance and it drops out entirely on dark, specular and thin
# surfaces, and neither is a Gaussian.  Measure the actual sensor against a
# known plane and put the fitted model here before S5 -- guessing this is the
# single largest sim2real risk in the vision stage.
DEPTH_NOISE_M = 0.004
DEPTH_DROPOUT = 0.02
```

So the bench has two jobs, and the second is the one that matters:

1. rank the three cameras on the same measurements;
2. produce the noise and dropout model that replaces those two constants, for
   whichever one wins.

**Both are done for the D405.** `model/fit_noise.py` turns the captures in
`results/` into `model/d405_noise.json`, `src/piper_push/depth_noise.py`
implements it, and the two constants above are gone. What was measured and what
it changed:
[`docs/depth_sensor_and_deployment.md`](../../docs/depth_sensor_and_deployment.md).
The pipeline that runs the policy on that camera is in
[`hardware/deploy/`](../deploy/README.md).

## The target

`targets/target_a4.pdf` — one A4 sheet carrying three regions.

| region | what it is | what it answers |
|---|---|---|
| `charuco` | 5×5 ChArUco, 33 mm squares, `DICT_5X5_100` | fixes the sheet's pose; also the well-textured best case |
| `white` | 60 mm blank paper square | textureless surface — where passive stereo has nothing to match |
| `black` | 60 mm solid black square | dark surface — the infrared-absorbing failure mode |

The patches sit at known offsets from the board origin, so one ChArUco pose
supplies the ground-truth plane for all three regions and the comparison
between them carries no per-region alignment error.

**Printing.** 100% / actual size, no fit-to-page, no borderless scaling. Matte
paper, not glossy — a glossy sheet returns specular highlights and measures the
lamp instead of the camera. Then check the 150 mm bar with a ruler; if it is
not 150 mm the printer scaled it and every distance the bench reports is wrong
by that factor. Tape it flat to something rigid: a curled sheet is no longer a
plane, and the plane is the entire reference.

Regenerate with `python targets/make_board.py`.

## Two ways to collect

### Live, both cameras at once — `live.py`

```bash
python live.py            # then open http://127.0.0.1:8770
```

Press start, carry the cameras around the printed target together, press stop.
Shots are taken automatically and the comparison is built when you stop.

This is the one to use when the question is *which camera*, because of what it
does with the confounds:

* **A shot is a pair.** Every camera is snapshotted at the same instant on the
  same scene, so the pair shares the lamp, the table, the paper and the
  operator, and differs only in the sensor. Characterising one camera, then
  unplugging it and characterising the next, lets all of that drift between
  them and charges the difference to the camera.
* **The trigger is novelty, not a timer.** A shot fires only when the viewpoint
  is far from *every* shot already taken — more than `--d-dist` metres or
  `--d-angle` degrees away. A timer rewards standing still and fills the
  session with one viewpoint measured fifty times, which looks like a lot of
  data and constrains nothing. Novelty spreads the captures over distance and
  angle without anyone planning them.
* **Nothing is captured while anything moves.** Motion blur and rolling-shutter
  smear are not sensor noise, but they are indistinguishable from it once
  they are in the numbers. A shot requires the whole 24-frame ring buffer to
  have been still, so every measured frame predates the trigger and none of
  them contains the approach.

The page shows, per camera, the live view with the three regions drawn on it,
the target's distance and tilt, and a meter for how close the current viewpoint
is to firing. `--save-raw` also writes the frame stacks (~40 MB per camera per
shot). Results land in `results/live/<timestamp>/`.

If the page will not load, check for an HTTP proxy: this machine has
`http_proxy=127.0.0.1:7890` set, which intercepts `127.0.0.1:8770` unless
localhost is in the bypass list.

### One careful capture — `measure.py`


```bash
python measure.py --backend d405 --preview              # frame it, then look at view.png
python measure.py --backend d405 --label 20cm --frames 60
python measure.py --backend d405 --label 35cm --frames 60
python measure.py --backend d405 --label 50cm --frames 60
python measure.py --backend d405 --label 70cm --frames 60
python analyze.py --backend d405
```

The labels are names, not measurements. The distance in every result comes from
the ChArUco pose — move the sheet, and the bench records where it actually was.
Asking someone to place a sheet at exactly 400 mm and then trusting the number
they typed is how a bias measurement becomes a measurement of the tape measure.

Distances should bracket **0.70 m**, which is how far the simulator camera sits
from what it aims at (`CAMERA_POS` to `CAMERA_AIM` in `camera.py`). Vary the
tilt a little between captures too, but keep it under about 30°: stereo noise
grows with obliquity, and a comparison across cameras is only fair at similar
tilt. Each result records its tilt so this can be checked afterwards.

Keep the scene still — these are static-scene statistics, and a hand in frame
during the temporal measurement shows up as sensor noise.

## What comes out

Per region, per capture:

| | |
|---|---|
| `fill` | fraction of pixels with a measurement within 5 cm of the sheet. This is `1 - DEPTH_DROPOUT`. |
| `stable_fill` | fraction valid in *every* frame. A pixel that flickers is worse for a recurrent policy at 50 Hz than one that is honestly always missing, and `fill` alone cannot tell them apart. |
| `bias_m` | median signed error against the ChArUco plane. Systematic, survives averaging, and is what a wrong baseline or a wrong depth unit looks like. |
| `spatial_rms_m` | RMS error with the bias removed — the flatness of a flat thing. The closest analogue to `DEPTH_NOISE_M`. |
| `temporal_std_m` | per-pixel standard deviation over the frame stack. Distinct from `spatial_rms_m`: fixed-pattern error is invisible here and dominant there. |
| `p95_abs_m` | the tail, because a sim-to-real story told in means is how a policy meets an edge it never saw. |

`analyze.py` fits `sigma_z = a·z²` across captures — the stereo form, since
depth error follows from disparity error — and extrapolates to 0.70 m. It
reports the implied `sigma_disparity`, which should stay roughly constant
across distances if the sensor is behaving and the model is the right one.

### The reference plane has its own error

`bias_m` is measured against a pose, and the pose is not exact. Each result
carries `plane_uncertainty_m`, estimated as `distance × reproj_rms_px / board_span_px`,
which sets the floor under any bias number. On the synthetic capture in
`selftest.py`, where the true pose is known, it predicts 0.79 mm against an
actual pose error of 0.53 mm. **A bias smaller than that figure is not a
measurement.** `spatial_rms_m`, `temporal_std_m` and `fill` do not depend on the
pose in the same way and are good well below it.

Fill the frame with the sheet and keep `reproj_rms_px` low if the bias is what
you care about.

## Trusting the bench

```bash
python selftest.py
```

Two stages, both without hardware.

The first renders the printed target onto a plane at a chosen pose, injects a
known bias, a known noise and known per-region dropout, and asserts they come
back. This is not a formality: the board frame's handedness, whether depth is Z
or range, and whether the region rectangles land on the patches or beside them
are all silent failures that otherwise produce confident, plausible, wrong
numbers. It also checks that `plane_uncertainty_m` actually covers the pose
error it describes.

The second drives `live.py`'s session logic with **two** synthetic cameras of
deliberately different quality: it checks that four viewpoints produce four
shots, that repeating a viewpoint produces none, and that the report ranks the
quieter camera as quieter. The second camera does not exist yet — the ZED X is
away and the Odin 1 has no backend — so without this the paired-comparison
path, which is the entire point of the live viewer, would first run on the day
the hardware arrives.

## Adding a camera

`capture/` backends have one obligation, in `capture/__init__.py`: return a
stack of depth frames, a grayscale image **on the depth grid**, and that grid's
intrinsics. `metrics.py` has never heard of RealSense. Depth is never resampled
to meet the grayscale — the colour image is warped into the depth frame instead,
because the depth samples are the measurement.

| backend | status |
|---|---|
| `d405` | working |
| `odin1` | working, except `bias` — see below |
| `zedx` | stub — unit has a hardware fault, retest in a week |

## The Odin 1, and what it cost the contract

This sensor is why `Capture` carries a ray table instead of an intrinsics
matrix, and the change was not optional. Four things had to be measured
because the vendor does not document them, and one of them contradicts what is
documented.

**Its depth grid is not a pinhole projection.** Fitting `(fx, fy, cx, cy)` to
its measured ray directions leaves a 17-pixel residual on a 256×192 grid. So
the geometry is carried as a measured ray table: over 45 frames each pixel's
direction varies by 1.4e-08, which is to say it is a device constant and
measuring it is exact. It is cached in `capture/odin1_raytable.npz` and
coverage only improves — a pixel resolved in any session keeps its direction.
Pixels the sensor has never returned are filled from a degree-7 surface, good
to about 0.6 px, so that a pixel returning nothing during a measurement can
still be told whether it was aimed at the target. Without that, dropout on
this sensor would not be measurable at all.

**Raw depth is in millimetres.** The data sheet says `float32 x // X axis, in
meters`, but that documents the SLAM cloud message and the raw DTOF stream is
a different path. The check that settles it needs no documentation: the
ChArUco distance is true metric, fixed by 33 mm printed squares, and puts the
sheet at 721 mm where the lidar's own plane fit lands at 698.

**The board is found in the colour image, not the depth image.** At 256×192
over 120° a 25 mm marker is under two pixels across. The 1600×1296 colour
camera is a `FishPoly` fisheye, so it is rectified to a pinhole first and the
pose is carried into the lidar frame by `T_dg`.

**The factory extrinsic is not usable as shipped, and `bias` is withheld until
it is calibrated.** The vendor documents that `Tcl` maps lidar to camera and
that the camera frame is OpenCV's, but never states the lidar frame's axis
convention or whether the dTOF grid is rotated relative to the colour sensor.
Both had to be found by searching the 24 signed axis permutations against the
device, scored by the angle between the ChArUco plane's normal and a plane
fitted to the lidar's own points — a test only the rotation can affect. One
wins by 28°, and it says the dTOF grid is rotated 180° about the optical axis.
Deriving this instead of measuring it gave a different answer that the device
disagreed with by 86°.

A single board pose cannot pin the continuous part, so:

```bash
python calibrate_odin1.py       # move the sheet, and change its TILT
```

Until that has run, `evaluate` compares this sensor against a plane fitted to
its own points and reports no bias. `fill`, `spatial_rms` and `temporal_std`
never cross the extrinsic and are unaffected; `bias` does, and a wrong bias is
worse than no bias, because it is the number that decides whether a sensor is
trusted. The calibration solves for rotation, translation **and a constant
depth bias together**, and refuses to write a result whose tilt spread is
under 25° — on fronto-parallel views a translation along the line of sight and
a depth bias are the same residual, and a fit that cannot tell them apart will
move the sensor's bias into the extrinsic and then measure a bias of zero.

## Two things about the Odin 1's colour camera

**Its exposure is locked, not automatic.** Under mains lighting an auto-exposing
10 Hz rolling shutter beats against the 100 Hz flicker and lays moving
horizontal bands across the frame. That is not the scene changing, but it is
indistinguishable from it: the live viewer's stillness gate measures
inter-frame difference, and the banding held that gate shut about 80% of the
time, so no shot could ever be taken. Locking the exposure to a whole number
of flicker periods — 20 ms at 50 Hz — integrates the same light every frame and
the bands go away. Measured on the device, the motion metric's median fell
from 31.3 grey levels to 4.1.

The gain is then trimmed automatically to reach a target brightness, because
the right value depends on the room. `--exposure 0` restores auto exposure and
the flicker with it. Locking also makes a capture reproducible, which the
measurement wants anyway: an auto-exposing camera is one whose noise you
measured under conditions you did not record.

`lidar_set_ae_param` is not in the Python bindings and is declared by hand in
`capture/odin1.py`.

**The depth panel is turned for display.** The dTOF array is stored bottom-up
relative to the colour sensor, so the two panels show the same scene mirrored
and cannot be checked against each other by eye — which is the one job that
view has. `view.display_transform` derives the turn from the extrinsic rather
than hardcoding it, applies it *after* the region outlines are drawn so they
travel with the image, and labels the panel. The geometry never sees it:
masks are computed through the ray table and are unaffected.

## The stillness gate

A shot needs every camera to have been still for a whole ring buffer. "Still"
is measured as the 98th percentile of the blurred, row-median-subtracted
difference between consecutive greyscale frames, against a threshold of a few
times that camera's own noise floor — the 10th percentile of its recent motion
values.

Every part of that is there because a simpler version failed. A *fixed* grey
level threshold cannot work: the two cameras' floors here are 36 and 1, and the
original threshold of 3 made "still" unreachable for one and trivial for the
other. A *mean* rather than a percentile dilutes a hand in one corner into the
noise of everything that did not move. And the row-median subtraction removes
what is left of the rolling-shutter banding after the exposure lock.

## Both cameras on one USB controller

`lsusb -t` puts the D405 and the Odin 1 on the same xHCI controller here. The
D405 can stop delivering frames once the lidar starts streaming, and it does
not recover by itself — `wait_for_frames` times out for ever, including after
the offending process exits. `live.py` issues a USB hardware reset and
reopens the stream after 25 consecutive failures, which makes a session
survive it. The actual fix is to move one of them to a different controller;
buses 2, 6 and 8 are separate root hubs with free ports.

## Known concerns going in — D405

Two, both of which the protocol above is shaped around.

**It has no infrared projector.** `rs-enumerate-devices -o` lists no Laser Power
and no Emitter Enabled, unlike every other D400. It is passive stereo and
matches on whatever texture the scene already has. The blank white patch is
there for this: on a projector-equipped camera it is unremarkable, and on this
one it is the thing that decides. A bare table, a plain-coloured object, and a
matte white bin are exactly the surfaces the tidying task is made of.

**Its rated range is roughly 7–50 cm.** The simulator camera is at 0.70 m. That
is outside the specification, so the `a·z²` extrapolation to 0.70 m is an
extrapolation someone has to look at, and captures should be taken at and beyond
0.70 m rather than relying on the fit. If the D405 wins on quality but only
inside 0.5 m, the honest conclusion is a camera move in `camera.py`, not a
quiet acceptance of the fit.
