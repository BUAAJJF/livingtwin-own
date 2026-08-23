#!/usr/bin/env python3
"""Check the measurement against a depth image whose answer is known.

The bench's whole claim is that it recovers a sensor's bias, noise and dropout
from a picture of a sheet of paper.  That claim is testable without a sensor:
render the printed target onto a plane at a pose we choose, add a bias and a
noise we choose, delete pixels at rates we choose, and see whether the numbers
come back.

This is not a formality.  The board frame's handedness, whether depth is Z or
range, and whether the region rectangles land on the patches or beside them are
all silent failures -- each one produces confident, plausible, wrong numbers on
real data.  Here they produce a failed assertion.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics as M
from capture import Capture, rays_from_K

W, H = 848, 480
K = np.array([[430.0, 0, 424.0], [0, 430.0, 240.0], [0, 0, 1.0]])
DIST = np.zeros(5)

TRUE_BIAS_M = 0.0032
TRUE_SIGMA_M = 0.0011
TRUE_DROP = {"charuco": 0.03, "white": 0.55, "black": 0.30}


def render(spec, distance=0.45, tilt_deg=8.0, seed=0,
           bias=None, sigma=None, drop=None):
  """A synthetic capture of the printed sheet at a known pose."""
  bias = TRUE_BIAS_M if bias is None else bias
  sigma = TRUE_SIGMA_M if sigma is None else sigma
  drop = TRUE_DROP if drop is None else drop
  page = cv2.imread(str(HERE / "targets" / "target_a4.png"), cv2.IMREAD_GRAYSCALE)
  ppm = spec["px_per_mm"] * 1000.0  # pixels per metre of paper

  rvec = np.array([np.radians(tilt_deg), np.radians(tilt_deg * 0.4), 0.03])
  R, _ = cv2.Rodrigues(rvec)
  centre = np.array([0.0825, 0.1385, 0.0])  # sheet centre in the board frame
  tvec = (np.array([0.0, 0.0, distance]) - R @ centre).reshape(3, 1)
  pose = {"rvec": rvec, "tvec": tvec, "R": R,
          "normal": R[:, 2] / np.linalg.norm(R[:, 2])}

  rays = rays_from_K(K, (H, W))
  gt = M.plane_depth(rays, pose)

  # Camera ray -> plane point -> board coordinates -> paper pixel.
  u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
  d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
  p_cam = d * gt[..., None]
  p_board = np.einsum("ij,hwj->hwi", R.T, p_cam - tvec.reshape(1, 1, 3))
  px = (p_board[..., 0] + 0.0225) * ppm  # board origin sits 22.5 mm into the page
  py = (p_board[..., 1] + 0.0100) * ppm  # ... and 10 mm down
  # Anti-alias before sampling.  The page is 20 px/mm and lands on roughly one
  # image pixel per millimetre, so nearest-neighbour sampling aliases the marker
  # edges, which biases ChArUco corner refinement, which tilts the ground-truth
  # plane, which shows up as a bias that differs between the left and right
  # patches.  That is a rendering artefact and it would mask a real one.
  scale = min(1.0, 2.0 * (K[0, 0] / distance) / ppm)  # 2x the sampling density
  small = cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
  gray = np.full((H, W), 60, np.uint8)  # off-sheet: a dim background
  sampled = cv2.remap(small, (px * scale).astype(np.float32),
                      (py * scale).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=60)
  inside = (px >= 0) & (px < page.shape[1]) & (py >= 0) & (py < page.shape[0])
  gray[inside] = sampled[inside]

  rng = np.random.default_rng(seed)
  n = 24
  depth = np.where(np.isfinite(gt), gt + bias, 0.0)[None].repeat(n, 0)
  depth += rng.normal(0.0, sigma, depth.shape)

  _, p_board = M.board_coords(rays, pose)
  for name, rate in drop.items():
    mask = M.region_mask(p_board, spec["regions"][name])
    hit = rng.random(depth.shape) < rate
    depth[hit & mask[None]] = 0.0
  depth[~np.isfinite(gt)[None].repeat(n, 0)] = 0.0

  return Capture(depth=depth.astype(np.float32), gray=gray, K=K, dist=DIST,
                 meta={"backend": "synthetic", "n_frames": n}), pose


def main() -> None:
  board, spec = M.load_target(HERE / "targets" / "target_a4.json")
  cap, truth = render(spec)

  res = M.evaluate(cap, board, spec)
  print(f"pose: {res['pose']['distance_m'] * 1000:.1f} mm "
        f"(true {abs(float(truth['normal'] @ truth['tvec'].ravel())) * 1000:.1f}), "
        f"tilt {res['pose']['tilt_deg']:.2f} deg, "
        f"{res['pose']['n_corners']} corners, "
        f"reproj {res['pose']['reproj_rms_px']:.3f} px")

  fails = []

  def check(name, got, want, tol, unit=""):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {name:34s} {got:9.4f} vs {want:9.4f} "
          f"(tol {tol:g}){unit}")
    if not ok:
      fails.append(name)

  true_dist = abs(float(truth["normal"] @ truth["tvec"].ravel()))
  unc = res["pose"]["plane_uncertainty_m"]
  check("pose distance (m)", res["pose"]["distance_m"], true_dist, 2e-3)
  check("reprojection rms (px)", res["pose"]["reproj_rms_px"], 0.0, 1.0)

  # The reported uncertainty has to actually cover the error it is there to
  # describe, or it is decoration.  It is an estimate, so it is allowed to be
  # up to 2x optimistic and no more.
  pose_err = abs(res["pose"]["distance_m"] - true_dist)
  print(f"  plane uncertainty {unc * 1000:.2f} mm, actual pose error "
        f"{pose_err * 1000:.2f} mm")
  if pose_err > 2 * unc:
    fails.append("plane_uncertainty underestimates the pose error")

  for name, rate in TRUE_DROP.items():
    m = res["regions"][name]
    print(f" region {name} ({m['n_pixels']} px, "
          f"{m['temporal_n_pixels']} usable for temporal):")
    check(f"{name} fill", m["fill"], 1.0 - rate, 0.03)
    # Bias is only knowable to the accuracy of the reference plane.  Quoting a
    # tighter tolerance here would make the test pass on a broken pose and fail
    # on a working one.
    check(f"{name} bias (m)", m["bias_m"], TRUE_BIAS_M, max(3e-4, 2 * unc))
    check(f"{name} spatial rms (m)", m["spatial_rms_m"], TRUE_SIGMA_M, 2e-4)
    check(f"{name} temporal std (m)", m["temporal_std_m"], TRUE_SIGMA_M, 3e-4)

  cv2.imwrite(str(HERE / "results" / "selftest_view.png"),
              __import__("measure").annotate(cap, board, spec))
  print(f"\n-> {HERE / 'results' / 'selftest_view.png'}")
  print("\n--- live session ---")
  fails += check_live_session()

  print("\n--- odin1 extrinsic calibration ---")
  fails += check_odin1_calibration()

  if fails:
    raise SystemExit(f"\n{len(fails)} check(s) failed: {', '.join(fails)}")
  print("\nall checks passed")



# --------------------------------------------------------------------------
# the live session


def check_live_session() -> list[str]:
  """Drive live.py's session logic with two synthetic cameras.

  The second camera does not exist yet -- the ZED X is away with a fault and
  the Odin 1 has no backend -- so the paired-comparison path, which is the
  entire point of the live viewer, would otherwise ship untested and first run
  on the day the hardware arrives.  Here it runs now, against two synthetic
  sensors with deliberately different noise and dropout, and the report has to
  rank them the right way round.
  """
  import argparse as _argparse
  import threading

  import live

  board, spec = M.load_target(HERE / "targets" / "target_a4.json")
  fails: list[str] = []

  class FakeWorker:
    """Only what Session actually touches, which is a short list."""

    def __init__(self, name, a_per_m, drop):
      self.name, self.a, self.drop = name, a_per_m, drop
      self.lock = threading.Lock()
      self.pose = None
      self.still_frames = live.RING
      self.motion = 0.0
      self.error = None
      self._cap = None

    def place(self, distance, tilt, seed):
      self._cap, _ = render(spec, distance=distance, tilt_deg=tilt, seed=seed,
                            bias=0.0015, sigma=self.a * distance**2,
                            drop=self.drop)
      self._cap.meta.update({"model": self.name, "fx_px": 430.0,
                             "stereo_baseline_m": 0.018})
      with self.lock:
        self.pose = M.detect_pose(self._cap.gray, self._cap.K, self._cap.dist,
                                  board)

    def snapshot(self):
      return self._cap

    def state(self):
      return {"error": None}

    def preview(self, width=400):
      ok, buf = cv2.imencode(".jpg", self._cap.gray)
      return buf.tobytes()

  good = FakeWorker("GOOD", 0.004, {"charuco": 0.02, "white": 0.10, "black": 0.08})
  poor = FakeWorker("POOR", 0.012, {"charuco": 0.05, "white": 0.70, "black": 0.40})
  workers = [good, poor]

  args = _argparse.Namespace(d_dist=0.06, d_angle=12.0, still=3.0,
                             min_interval=0.0, save_raw=False)
  outdir = HERE / "results" / "selftest_live"
  session = live.Session(workers, board, spec, args, outdir)
  session.start()

  plan = [(0.30, 5.0), (0.45, 12.0), (0.60, 8.0), (0.75, 15.0)]
  for i, (dist, tilt) in enumerate(plan):
    for w in workers:
      w.place(dist, tilt, seed=i)
    session.tick()
    for _ in range(200):  # _fire runs on its own thread
      if not session.busy:
        break
      time.sleep(0.05)
  n = len(session.shots)
  print(f"  live: {n} shot(s) from {len(plan)} viewpoints")
  if n != len(plan):
    fails.append(f"expected {len(plan)} shots, fired {n}")

  # The same viewpoint again must not produce a second shot.
  for w in workers:
    w.place(*plan[-1], seed=len(plan) - 1)
  session.tick()
  for _ in range(200):
    if not session.busy:
      break
    time.sleep(0.05)
  if len(session.shots) != n:
    fails.append("a repeated viewpoint fired a duplicate shot")
  else:
    print("  live: a repeated viewpoint correctly did not fire")

  rep = session.stop()
  print(f"  live: report over {rep['n_shots']} shots, cameras {rep['cameras']}")
  for cam in ("GOOD", "POOR"):
    m = rep["summary"][cam]["charuco"]
    print(f"    {cam:5s} a={m['a_per_m']:.5f}  sigma@0.70m="
          f"{m['sigma_at_0.70m'] * 1000:5.2f} mm  "
          f"white fill={rep['summary'][cam]['white']['fill_mean'] * 100:.1f}%")

  g, p = rep["summary"]["GOOD"], rep["summary"]["POOR"]
  for name, got, want, tol in [
    ("GOOD a", g["charuco"]["a_per_m"], 0.004, 0.0008),
    ("POOR a", p["charuco"]["a_per_m"], 0.012, 0.0020),
    ("GOOD white fill", g["white"]["fill_mean"], 0.90, 0.03),
    ("POOR white fill", p["white"]["fill_mean"], 0.30, 0.03),
  ]:
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {name:20s} {got:8.5f} vs {want:8.5f}")
    if not ok:
      fails.append(name)
  if not (g["charuco"]["a_per_m"] < p["charuco"]["a_per_m"]):
    fails.append("the report ranks the noisier camera as quieter")
  if not (outdir / "comparison.png").exists():
    fails.append("no comparison figure was written")
  else:
    print(f"  live: figure -> {outdir / 'comparison.png'}")
  return fails


# --------------------------------------------------------------------------
# the Odin 1 extrinsic calibration


def check_odin1_calibration() -> list[str]:
  """Recover a known extrinsic correction and a known depth bias.

  The transform this refines is a *reflection* -- the Odin 1's dTOF frame has
  the opposite handedness to its colour camera, per the vendor's own driver --
  so the fit has to work on a matrix with det -1.  Every search written here
  before assumed a rotation, which is exactly how the answer stayed out of
  reach for so long; this checks that the code that replaced them can start
  from the vendor value, apply a proper correction on top of it, and come back
  with the truth.

  The board is rendered against a background 1.6 m away rather than as a plane
  filling the frame.  An earlier version of this test filled the frame, and
  then every candidate scored perfectly because whatever it pointed at was
  still that plane -- which is also why the real procedure says to hold the
  sheet up in the air rather than lay it on the desk.
  """
  import calibrate_odin1 as C
  from capture import rays_from_K

  board, spec = M.load_target(HERE / "targets" / "target_a4.json")
  fails: list[str] = []

  FLIP = np.diag([1.0, -1.0, 1.0])                    # det -1, as on the device
  R_true = cv2.Rodrigues(np.array([0.03, -0.02, 0.015]))[0] @ FLIP
  t_true = np.array([-0.0495, 0.005, 0.011])
  T_true = np.eye(4)
  T_true[:3, :3], T_true[:3, 3] = R_true, t_true
  BIAS = 0.004

  rays = rays_from_K(K, (H, W))
  shots = []
  for i, (dist, tilt) in enumerate([(0.35, 6.0), (0.45, 28.0), (0.55, 15.0),
                                    (0.65, 38.0), (0.50, 45.0), (0.40, 20.0)]):
    cap, _ = render(spec, distance=dist, tilt_deg=tilt, seed=i,
                    bias=0.0, sigma=0.0008, drop={})
    pose_c = M.detect_pose(cap.gray, K, DIST, board)
    pose_d = M.transform_pose(pose_c, T_true)
    gt, pb = M.board_coords(rays, pose_d)
    sheet = ((pb[..., 0] > -0.02) & (pb[..., 0] < 0.19) &
             (pb[..., 1] > -0.01) & (pb[..., 1] < 0.29) & np.isfinite(gt))
    shots.append({"pose": pose_c,
                  "depth": np.nan_to_num(np.where(sheet, gt + BIAS, 1.6), nan=0.0)})

  # Start from the structural factor alone: the vendor value with none of the
  # per-unit correction, which is what the real script starts from.
  T0 = np.eye(4)
  T0[:3, :3] = FLIP
  T0[:3, 3] = t_true + np.array([0.004, -0.003, 0.005])

  T, bias, rms, n = C.solve(shots, rays, T0, spec)
  ang = np.degrees(np.arccos(np.clip(
    (np.trace(T[:3, :3].T @ R_true) - 1) / 2, -1, 1)))
  dt = float(np.linalg.norm(T[:3, 3] - t_true))
  print(f"  odin1-cal: recovered to {ang:.2f} deg, translation off by "
        f"{dt * 1000:.1f} mm, bias {bias * 1000:+.2f} mm (true {BIAS * 1000:+.2f}), "
        f"residual {rms * 1000:.2f} mm on {n} points")
  if np.linalg.det(T[:3, :3]) > 0:
    fails.append("the fit lost the reflection (det became positive)")
  for name, got, want, tol in [("rotation (deg)", ang, 0.0, 1.5),
                               ("translation (m)", dt, 0.0, 0.010),
                               ("recovered bias (m)", bias, BIAS, 0.0015)]:
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {name:22s} {got:9.5f} vs {want:9.5f}")
    if not ok:
      fails.append(name)
  return fails


if __name__ == "__main__":
  main()
