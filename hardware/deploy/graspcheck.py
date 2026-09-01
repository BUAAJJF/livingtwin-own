"""Put the object where it should be grasped, and measure what the stack thinks.

Everything the deployment does downstream rests on two numbers agreeing: where
the segmenter says the object is, and where forward kinematics says the grasp
site is.  If the object is physically sitting in the right place to be grasped,
those two are the same point, and any difference between them is the stack's
systematic error -- measured, not inferred.

That matters because inference has already been wrong here.  A run showed the
object landing on one finger pad rather than between the pads, and decomposing
the error into the gripper's own frame gave -37 mm along the jaw axis on the
arm against +0.1 mm in simulation.  That says the error is real and on the
robot, and says nothing about which stage produces it.  This measures it at the
one moment when the right answer is known by construction.

    # arm holding still, object placed between the fingers where a grasp would
    # hold it, gripper open around it
    python -m hardware.deploy.graspcheck

Nothing is commanded.  CAN is opened to read joint angles for the kinematics
and closed again; the drives are never enabled and never receive a target.

Two readings are printed and both matter:

* **arm subtracted** is what the deployment sees.  An object held between the
  fingers is inside the arm's sphere cover and the segmenter is *supposed* to
  drop it -- "no longer a thing on the table to be found" -- so this usually
  finds nothing, and that is the correct answer to a different question.
* **arm kept** is the measurement.  It is the same segmentation with the robot
  left in, which is what makes the object visible while it is in the hand.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

from . import config, mask, proprio, rectify


def measure(reader, rig, kin, seg, arm_spheres, frames: int,
            settle_frames: int = 4):
    """Segment several frames and return the instances seen, most recent first."""
    tracker = mask.TargetTracker()
    seen = set()
    out = []
    deadline = time.time() + max(4.0, frames / 8.0)
    while len(out) < frames and time.time() < deadline:
        f = reader.latest()
        if f is None or f.index in seen:
            time.sleep(0.01)
            continue
        seen.add(f.index)
        s = seg(f.depth, rgb=f.gray, arm=arm_spheres)
        label = tracker.update(s, kin.site_pos)
        out.append((s, label))
    del settle_frames
    return out


def report(rows, kin, label_name: str) -> dict | None:
    site = kin.site_pos.copy()
    R = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    picks = []
    for s, label in rows:
        # The nearest instance to the hand, whether or not the tracker had
        # confirmed it: this is a static scene held on purpose, and three
        # frames of confirmation is a rule for a moving one.
        if not s.instances:
            continue
        c = np.stack([i.centroid_base for i in s.instances])
        j = int(np.argmin(np.linalg.norm(c - site, axis=1)))
        picks.append((s.instances[j], label))
    if not picks:
        print(f"  {label_name}: no instance in any frame")
        return None

    cent = np.stack([p[0].centroid_base for p in picks])
    tops = np.array([p[0].top_z for p in picks]) * 1000
    npx = np.array([p[0].n_px for p in picks])
    obj = np.median(cent, axis=0)
    err_base = site - obj
    err_grip = R.T @ err_base

    print(f"  {label_name}: {len(picks)} frames with an instance")
    print(f"    object centroid (base)  {np.round(obj * 1000, 1).tolist()} mm")
    print(f"    frame-to-frame spread   "
          f"{np.round(cent.std(axis=0) * 1000, 1).tolist()} mm")
    print(f"    height above table      {np.median(tops):.1f} mm")
    print(f"    size                    {int(np.median(npx))} px")
    print(f"    grasp site (base)       {np.round(site * 1000, 1).tolist()} mm")
    print()
    print(f"    site - object, base     "
          f"{np.round(err_base * 1000, 1).tolist()} mm  "
          f"(|{np.linalg.norm(err_base) * 1000:.1f}|)")
    print(f"    the same, gripper frame  x {err_grip[0] * 1000:+6.1f}   "
          f"y {err_grip[1] * 1000:+6.1f}   z {err_grip[2] * 1000:+6.1f} mm")
    print(f"      x = across the jaws, y = the closing direction, "
          f"z = the approach")
    return {
        "n": len(picks),
        "object_base_m": obj.tolist(),
        "spread_m": cent.std(axis=0).tolist(),
        "top_mm": float(np.median(tops)),
        "n_px": int(np.median(npx)),
        "site_base_m": site.tolist(),
        "error_base_m": err_base.tolist(),
        "error_gripper_m": err_grip.tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", choices=("d405", "d455"), default="d455")
    p.add_argument("--rig-file", default=None)
    p.add_argument("--serial", default=None)
    p.add_argument("--can", default=config.CAN_INTERFACE)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--arm-clearance-mm", type=float, default=20.0,
                   help="the deployment's value, for the 'arm subtracted' "
                        "reading")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    suffix = "" if a.camera == "d405" else f"_{a.camera}"
    rig_path = (pathlib.Path(a.rig_file) if a.rig_file
                else pathlib.Path(config.RIG_FILE).with_name(f"rig{suffix}.json"))
    rig = config.Rig.load(rig_path)

    from . import robot, sensor
    from .run import _wait_for_feedback

    print("Place the object between the fingers, where a successful grasp "
          "would hold it.")
    print("The arm is read only -- the drives are not enabled and nothing is "
          "commanded.")
    input("press enter when it is in position: ")

    reader = sensor.Reader(serial=a.serial or rig.serial, backend=a.camera)
    arm = robot.PiperArm(a.can)
    try:
        reader.wait_for_first()
        if rig.serial and reader.serial and str(rig.serial) != str(reader.serial):
            raise SystemExit(f"{rig_path} belongs to camera {rig.serial}, "
                             f"connected {reader.serial}")
        rig.K = reader.K
        arm.connect()
        st = _wait_for_feedback(arm)
        kin = proprio.Kinematics()
        kin.update(np.array([*st.q, st.gripper, -st.gripper]))
        reproj = rectify.Reprojector(rig, device="cpu")

        print()
        print("joints (deg): " + " ".join(f"{x:+.1f}" for x in np.degrees(st.q)))
        print(f"jaw gap {st.gripper * 2000:.1f} mm      "
              f"grasp site {np.round(kin.site_pos * 1000, 1).tolist()} mm")
        r = math.hypot(kin.site_pos[0], kin.site_pos[1])
        print(f"site r {r:.3f} m   azimuth "
              f"{math.degrees(math.atan2(kin.site_pos[1], kin.site_pos[0])):.1f} deg")
        print()

        out = {}
        spheres = kin.link_spheres()
        cfg = mask.SegmenterCfg()
        seg = mask.DepthSegmenter(rig, reproj, cfg=cfg)
        rows = measure(reader, rig, kin, seg, spheres, a.frames)
        out["arm_subtracted"] = report(rows, kin, "arm subtracted (deployment)")
        print()
        seg2 = mask.DepthSegmenter(rig, reproj, cfg=cfg)
        rows2 = measure(reader, rig, kin, seg2, None, a.frames)
        out["arm_kept"] = report(rows2, kin, "arm kept (the measurement)")
    finally:
        reader.close()
        try:
            arm.close()
        except Exception:
            pass

    m = out.get("arm_kept")
    if m:
        e = np.asarray(m["error_gripper_m"]) * 1000
        print()
        print("Reading it: the object is where you put it, so a stack with no "
              "systematic error")
        print("reports all three components near zero.  On the arm the closing "
              "axis (y) came out")
        print("-37 mm during a policy run, against +0.1 mm in simulation; if "
              "that number shows up")
        print("here too, it is in the perception or the kinematics and not in "
              "the policy.")
        print(f"    measured now: x {e[0]:+.1f}   y {e[1]:+.1f}   "
              f"z {e[2]:+.1f} mm")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
