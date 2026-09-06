# Depth bench — choosing the camera the policy will actually see through

The camera was chosen on this bench: the **RealSense D405** first, then the
**D455** the rig now carries.  The ZED X and Odin 1 candidates and their
backends are in git history before 2026-09-06.  The bench exists because of a
comment in `src/piper_push/camera.py`:

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
[`docs/history/depth_sensor_and_deployment.md`](../../docs/history/depth_sensor_and_deployment.md).
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
quieter camera as quieter, so the paired-comparison path runs without a second
physical camera.

## Adding a camera

`capture/` backends have one obligation, in `capture/__init__.py`: return a
stack of depth frames, a grayscale image **on the depth grid**, and that grid's
intrinsics. `metrics.py` has never heard of RealSense. Depth is never resampled
to meet the grayscale — the colour image is warped into the depth frame instead,
because the depth samples are the measurement.

| backend | status |
|---|---|
| `d405` | working |

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
