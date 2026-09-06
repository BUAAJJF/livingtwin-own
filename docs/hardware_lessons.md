# Lessons from the first hardware runs

2026-09-01. Twenty recorded sessions, 12224 commands to a live arm, and one
policy that finally worked (`recordings/v4_stereo_try3`, the `d455_v4_final`
mask policy; its command is in the root README). What follows is what cost the
most time, written down because every one of them looked like something else
first.

**A constant beat every hypothesis.** The sim-to-real failure that motivated
three retraining campaigns was `SegmenterCfg.width_range_m`, a 160 mm ceiling
on a component's horizontal extent. As the hand arrives, the target's blob
merges with what `arm_mask` leaves of the gripper and measures **164 mm** --
4 mm over, on three separate recordings (164.0, 164.3, 164.1). Raising the
ceiling took detection from 23% / 40% / 38% to 95% / 100% / 99%. Before
finding it, five plausible mechanisms had been proposed and each refuted by
measurement. The same constant bound again at 200 mm a day later (206.6 mm
merged blob); a second, different mechanism (`arm_mask` deleting 82% of the
object's pixels at close range) was found on another frame of the same
session. Several failure modes coexist; one frame's mechanism is not the
session's.

**Measure the conditional you care about, not one that correlates with it.**
Ray-tracing said the arm blocked the camera-to-object line 52% of frames, and
that was true. It was never checked against *which frames actually lost the
target*, and those were different frames: per-stage counting showed `arm_mask`
removed **0%** of the object's pixels and depth dropout **0%** on those
recordings. A correct measurement of the wrong quantity is still the wrong
answer, and it is more convincing than a guess.

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

**Start where training started.** Without `--home-first` each run begins where
the last one stopped, and three in a row went progressively lower
(263 → 67 → 53 mm). Every episode the policy trained on began at the home pose.

**Speed is occlusion.** The arm blocked the camera's line to the object 52% of
frames on a fast run against 16% on a slow one; `--command-rate-scale 0.6` is
a perception setting as much as a safety one.
