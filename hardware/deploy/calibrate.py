"""Where the camera is, measured rather than assumed.

Everything else in this directory is checked against the simulator by
``selftest.py``.  This is not, and cannot be: it is the one measurement that
connects the two, and if it is wrong every downstream number is wrong in the
same consistent way and nothing complains.  So it is written to be checked
against itself instead -- residuals reported per pose, the worst pose named,
and a refusal to write the result when the poses do not span enough of the
workspace to constrain it.

The setup is eye-to-hand: the camera is fixed to the world and the board is
carried by the gripper.  ``calibrate_hand_eye`` below solves ``AX = XB`` for
the camera's pose in the robot base frame, given the arm's forward kinematics
at each pose and the board's pose in the camera at each pose.  It is written
out rather than called from OpenCV because OpenCV 5 removed the binding.

Two things are easy to get backwards and both are silent.

*Eye-to-hand is not eye-in-hand.*  For a camera mounted on the arm, the
routine is called with the gripper pose in the base frame and returns the
camera in the gripper frame.  For a camera fixed to the world -- this one --
the same routine is called with the **inverse** poses, base-in-gripper, and
returns the camera in the base frame.  Passing the un-inverted poses produces a
transform that looks plausible and is wrong by the whole arm.

*The board pose has a sign.*  ``solvePnP`` returns the board in the camera, and
hand-eye wants exactly that, not its inverse.

The table plane is measured at the same time, from the depth image, because
``mask.py`` needs it and because it is a free check: a table that comes out
tilted by more than a degree in the base frame means the calibration is wrong,
the robot is not bolted down, or the table is not the table.

    python -m hardware.deploy.calibrate --collect     # move the arm, press enter
    python -m hardware.deploy.calibrate --solve
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import cv2
import numpy as np

from . import config

HERE = pathlib.Path(__file__).resolve().parent
POSES_FILE = HERE / "calib_poses.json"

BOARD_SQUARES = (5, 5)
BOARD_SQUARE_M = 0.033
BOARD_MARKER_M = 0.025
BOARD_DICT = "DICT_5X5_100"
"""The board ``hardware/depth_bench/targets/make_board.py`` prints, and the
default only because it is the one this repository can produce.  Any board
works -- see ``Board`` -- and the numbers here are not privileged, they are
just what you get if you print the sheet in ``hardware/depth_bench/targets``
rather than buying one."""

MIN_POSES = 8
MIN_ROT_SPAN_DEG = 30.0
"""Hand-eye is only determined by rotation.  Poses that differ by translation
alone leave the rotation unconstrained and the solver returns something anyway;
this is the guard that makes that a refusal rather than a silent answer."""


@dataclasses.dataclass(frozen=True)
class Board:
  """Which board is on the gripper, because it is not necessarily this one.

  The rig originally hard-coded the printed A4 sheet, which is fine until
  somebody buys a board -- and a bought board is the better instrument: glass
  or aluminium instead of paper on cardboard, and a square size held to
  micrometres instead of to whatever the printer did.  A calibration is only
  as good as its ruler, so the ruler has to be describable.

  Two kinds, and the difference is not cosmetic.

  ``charuco`` carries ArUco markers in the white squares, so every detection
  is anchored to marker *identities*.  The board's origin is therefore the
  same physical corner in every frame, at any orientation, and it survives the
  board being half out of view.

  ``checker`` is a plain checkerboard and has a symmetry: rotating it 180
  degrees about its own normal maps the pattern onto itself, so the corner
  ordering a detector returns can flip between poses.  Nothing in a single
  frame can tell which one happened -- the image is identical -- and hand-eye
  fed a mixture of the two returns nonsense with no obvious symptom beyond a
  large residual.  ``solve`` undoes it (see ``_unflip``), but the honest
  summary is that ChArUco does not have the problem and a checkerboard needs
  the fix to be trusted.

  ``legacy`` is the third trap and the quietest.  OpenCV changed which corner
  a ChArUco board starts numbering from in 4.6.  A board printed or bought
  against the old convention still detects perfectly against the new one; its
  origin is simply somewhere else, which moves the answer by the size of the
  board and looks like a mounting error.  If the residual is fine but the
  camera lands a board-width away from where it obviously is, this is why.
  """

  kind: str = "charuco"
  squares: tuple[int, int] = BOARD_SQUARES
  """ChArUco: squares across and down.  Checkerboard: *inner corners*, which
  is one less than the squares in each direction and the most common way to
  get a checkerboard wrong."""
  square_m: float = BOARD_SQUARE_M
  marker_m: float = BOARD_MARKER_M
  dictionary: str = BOARD_DICT
  legacy: bool = False
  inverted: bool = False
  """Ink-saving white-marker inverse print; part of board identity."""
  min_corners: int = 6

  def __post_init__(self):
    if self.kind not in ("charuco", "checker"):
      raise ValueError(f"unknown board kind {self.kind!r}")
    if self.kind == "charuco" and not (0 < self.marker_m < self.square_m):
      raise ValueError(
        f"marker {self.marker_m * 1000:.1f} mm must be smaller than the "
        f"square {self.square_m * 1000:.1f} mm")

  # -- construction ---------------------------------------------------------

  @classmethod
  def from_dict(cls, d: dict) -> "Board":
    f = {k.name for k in dataclasses.fields(cls)}
    kw = {k: v for k, v in d.items() if k in f}
    if "squares" in kw:
      kw["squares"] = tuple(int(x) for x in kw["squares"])
    return cls(**kw)

  @classmethod
  def load(cls, path) -> "Board":
    """From a JSON file.

    Reads ``hardware/depth_bench/targets/target_a4.json`` as well as this
    module's own format, because that file already describes a board and
    having two spellings of the same sheet is how they drift apart.
    """
    d = json.loads(pathlib.Path(path).read_text())
    if "squares_x" in d:              # the depth bench's target description
      d = {"kind": "charuco",
           "squares": (d["squares_x"], d["squares_y"]),
           "square_m": d["square_m"], "marker_m": d["marker_m"],
           "dictionary": d.get("dictionary", BOARD_DICT),
           "min_corners": d.get("min_corners", 6),
           "inverted": d.get("inverted", False),
           "legacy": d.get("legacy", False)}
    return cls.from_dict(d)

  def to_dict(self) -> dict:
    return dataclasses.asdict(self)

  def n_corners(self) -> int:
    nx, ny = self.squares
    return (nx - 1) * (ny - 1) if self.kind == "charuco" else nx * ny

  def suggested_poses(self) -> int:
    """How many poses this board needs, from the measurement in the README.

    Not a formula.  The table there was produced by projecting each pattern
    through this camera with 0.2 px of corner noise and solving; what it shows
    is that corner count dominates, and that a 16-corner board wants two to
    three times the poses a 48-corner one does to reach the same 4 mm gate.
    Rounded to the three cases that table actually covers, because
    interpolating between six synthetic points would dress a lookup up as a
    model.
    """
    n = self.n_corners()
    if n < 8:
      return 999            # a single marker; no pose count rescues it
    return 30 if n < 24 else (16 if n < 48 else 12)

  def describe(self) -> str:
    n = f"{self.squares[0]}x{self.squares[1]}"
    if self.kind == "charuco":
      return (f"ChArUco {n}, {self.square_m * 1000:.1f} mm squares, "
              f"{self.marker_m * 1000:.1f} mm markers, {self.dictionary}"
              + (", inverted white-marker print" if self.inverted else "")
              + (", legacy origin" if self.legacy else ""))
    return (f"checkerboard {n} inner corners, "
            f"{self.square_m * 1000:.1f} mm squares")

  # -- detection ------------------------------------------------------------

  def _charuco(self):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
    b = cv2.aruco.CharucoBoard(self.squares, self.square_m, self.marker_m, d)
    if self.legacy:
      b.setLegacyPattern(True)
    return b

  def object_points(self) -> np.ndarray:
    """Checkerboard corners in the board frame, z = 0."""
    nx, ny = self.squares
    g = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2).astype(np.float64)
    return np.concatenate([g * self.square_m, np.zeros((nx * ny, 1))], axis=1)

  def detect(self, gray: np.ndarray):
    """``(object_points, image_points)`` in the board and image frames, or
    None if the board is not there."""
    if self.kind == "charuco":
      board = self._charuco()
      params = cv2.aruco.DetectorParameters()
      params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
      params.detectInvertedMarker = bool(self.inverted)
      det = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(),
                                      params)
      corners, ids, _, _ = det.detectBoard(gray)
      if ids is None or len(ids) < self.min_corners:
        return None
      return board.matchImagePoints(corners, ids)

    ok, corners = cv2.findChessboardCornersSB(
      gray, self.squares, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
      return None
    return self.object_points().reshape(-1, 1, 3).astype(np.float32), corners


DEFAULT_BOARD = Board()


def board_image(frame) -> np.ndarray:
  """The image to find the board in.

  The left infrared imager when the reader was opened for it, and the
  depth-aligned colour frame otherwise.  They are not interchangeable and the
  difference is not subtle: ``frame.gray`` is warped into the depth grid, so it
  is black wherever the stereo matcher returned nothing and displaced wherever
  the depth is noisy.  Both happen at depth discontinuities, a ChArUco board is
  nothing but discontinuities, and the board is carried by the arm -- which
  casts the largest alignment shadow in the frame.

  Measured on this rig: the aligned image drops out over whole regions of the
  scene and carries a black halo around the arm; the infrared one has neither.
  """
  ir = getattr(frame, "ir", None)
  return frame.gray if ir is None else ir


def make_board(board: Board = DEFAULT_BOARD):
  """The OpenCV ChArUco object, for callers that want to draw it."""
  return board._charuco()


def detect_board(gray: np.ndarray, K: np.ndarray, dist: np.ndarray,
                 board: Board = DEFAULT_BOARD):
  """Board pose in the camera frame, or None.

  Corner refinement is on.  The bench measured what it buys on this board: the
  pose error fell from 0.92 mm to 0.71 mm, and the calibration residual is the
  thing this whole file is trying to keep small.
  """
  found = board.detect(gray)
  if found is None:
    return None
  obj, img = found
  ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
  if not ok:
    return None
  proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
  rms = float(np.sqrt(((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2)
                      .sum(1).mean()))
  return {"rvec": rvec, "tvec": tvec, "n_corners": int(len(img)),
          "reproj_rms_px": rms,
          "object_points": obj.reshape(-1, 3),
          "image_points": img.reshape(-1, 2)}


def fuse_detections(detections: list[dict], K: np.ndarray,
                    dist: np.ndarray, board: Board = DEFAULT_BOARD,
                    min_frames: int = 8,
                    min_fraction: float = 0.5) -> dict | None:
  """Fuse repeated, settled detections in image space before PnP.

  Averaging already-solved 6-D poses is both awkward and biased.  ChArUco IDs
  give a better option: group the same physical corner across frames, take its
  coordinate-wise median, and solve PnP once from those robust image points.
  A corner must appear in at least half the valid frames, so one accidental
  decode cannot enter the calibration simply because it has a plausible ID.
  """
  valid = [d for d in detections
           if d is not None and d.get("object_points") is not None
           and d.get("image_points") is not None]
  if len(valid) < int(min_frames):
    return None
  grouped: dict[tuple[float, float, float], list[np.ndarray]] = {}
  for d in valid:
    obj = np.asarray(d["object_points"], dtype=np.float64).reshape(-1, 3)
    img = np.asarray(d["image_points"], dtype=np.float64).reshape(-1, 2)
    for p, uv in zip(obj, img):
      key = tuple(np.round(p, 7).tolist())
      grouped.setdefault(key, []).append(uv)
  required = max(3, int(np.ceil(len(valid) * float(min_fraction))))
  fused = [(k, np.median(np.stack(v), axis=0), np.stack(v))
           for k, v in grouped.items() if len(v) >= required]
  if len(fused) < board.min_corners:
    return None
  fused.sort(key=lambda x: x[0])
  obj = np.asarray([x[0] for x in fused], dtype=np.float64)
  img = np.asarray([x[1] for x in fused], dtype=np.float64)
  ok, rvec, tvec = cv2.solvePnP(obj, img, np.asarray(K, dtype=np.float64),
                                np.asarray(dist, dtype=np.float64),
                                flags=cv2.SOLVEPNP_ITERATIVE)
  if not ok:
    return None
  proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
  reproj = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
  spreads = np.concatenate([
    np.linalg.norm(samples - median, axis=1)
    for _, median, samples in fused
  ])
  return {
    "rvec": rvec,
    "tvec": tvec,
    "n_corners": int(len(img)),
    "reproj_rms_px": float(np.sqrt(np.mean(reproj ** 2))),
    "object_points": obj,
    "image_points": img,
    "fusion_frames": len(valid),
    "corner_spread_px": float(np.sqrt(np.mean(spreads ** 2))),
  }


def _rt(rvec, tvec) -> np.ndarray:
  T = np.eye(4)
  T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
  T[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
  return T


def _angle_between(Ra, Rb) -> float:
  c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
  return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def calibrate_hand_eye(R_a, t_a, R_b, t_b, min_angle_deg: float = 5.0):
  """Solve ``A X = X B`` for X, by Park and Martin's method.

  Written out rather than called, because ``cv2.calibrateHandEye`` does not
  exist in OpenCV 5 -- the constants ``CALIB_HAND_EYE_*`` are still exported
  and the function is gone -- and this environment has OpenCV 5 because
  ultralytics pulled it in.  Twenty-five lines is a better dependency than a
  binding that can disappear under the one measurement everything else is
  built on.  ``tests/test_deploy.py`` checks it recovers a known pose to under
  a millimetre and a twentieth of a degree.

  Rotation first, in closed form: with ``alpha = log(R_A)`` and
  ``beta = log(R_B)`` as axis-angle vectors, ``R_X`` is the orthogonal matrix
  closest to the one that maps every beta onto its alpha, which is
  ``(M^T M)^{-1/2} M^T`` for ``M = sum beta alpha^T``.  Then translation, as a
  least-squares solve of ``(R_A - I) t_X = R_X t_B - t_A`` stacked over pairs.

  Pairs that barely rotate are dropped: their alpha and beta are noise with a
  direction, and they pull the sum towards it.
  """
  n = len(R_a)
  M = np.zeros((3, 3))
  rows, rhs = [], []
  eps = np.radians(min_angle_deg)
  for i in range(n):
    for j in range(i + 1, n):
      Ta = _compose(R_a[j], t_a[j], inv=True) @ _compose(R_a[i], t_a[i])
      Tb = _compose(R_b[j], t_b[j]) @ _compose(R_b[i], t_b[i], inv=True)
      alpha = cv2.Rodrigues(Ta[:3, :3])[0].ravel()
      beta = cv2.Rodrigues(Tb[:3, :3])[0].ravel()
      if np.linalg.norm(alpha) < eps or np.linalg.norm(beta) < eps:
        continue
      M += np.outer(beta, alpha)
      rows.append((Ta, Tb))
  if not rows:
    raise RuntimeError(
      "no pose pair rotates by more than "
      f"{min_angle_deg} degrees; hand-eye is undetermined"
    )

  w, V = np.linalg.eigh(M.T @ M)
  R_x = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
  # Nearest rotation, in case the square root left it a hair off orthogonal.
  U, _, Vt = np.linalg.svd(R_x)
  R_x = U @ Vt
  if np.linalg.det(R_x) < 0:
    U[:, -1] *= -1
    R_x = U @ Vt

  A, b = [], []
  for Ta, Tb in rows:
    A.append(Ta[:3, :3] - np.eye(3))
    rhs = R_x @ Tb[:3, 3] - Ta[:3, 3]
    b.append(rhs)
  t_x, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
  return R_x, t_x.reshape(3, 1)


def _compose(R, t, inv: bool = False) -> np.ndarray:
  T = np.eye(4)
  T[:3, :3] = np.asarray(R, dtype=np.float64)
  T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
  return np.linalg.inv(T) if inv else T


def _flip_matrix(board: Board) -> np.ndarray:
  """The 180-degree rotation about the board normal, as a board-frame pose.

  Maps corner ``(x, y)`` to ``(W - x, H - y)``, which for a checkerboard is
  the identity on the *image* and a different answer for the pose.
  """
  nx, ny = board.squares
  n = (nx - 1, ny - 1) if board.kind == "checker" else (nx, ny)
  F = np.eye(4)
  F[0, 0] = F[1, 1] = -1.0
  F[0, 3] = n[0] * board.square_m
  F[1, 3] = n[1] * board.square_m
  return F


def _unflip(records: list[dict], board: Board, gripper_site: str):
  """Undo the checkerboard's 180-degree corner-ordering ambiguity.

  Done without the hand-eye solution, which is the point: the solution is what
  the flips would corrupt, so deciding them from it would be circular.

  What is used instead is an invariant of ``A X = X B``.  ``A`` and ``B`` are
  conjugate -- ``A = X B X^-1`` -- and conjugate rotations have the *same
  angle*, whatever ``X`` is.  So for every pair of poses the gripper's rotation
  angle and the board's rotation angle must agree, and a pose whose corners
  came back rotated by 180 degrees disagrees loudly.  Each pose is assigned the
  orientation that agrees best with the ones already decided.

  A globally consistent flip is not corrected and does not need to be: it names
  the opposite corner of the board as the origin, and hand-eye absorbs that
  into the constant board-in-gripper pose it never reports.
  """
  from .proprio import Kinematics

  kin = Kinematics(site_name=gripper_site)
  T_bg, T_cb = [], []
  for r in records:
    kin.update(np.asarray(r["joint_pos"], dtype=np.float64))
    T = np.eye(4)
    T[:3, :3] = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    T[:3, 3] = kin.data.site_xpos[kin.site_id]
    T_bg.append(T)
    T_cb.append(_rt(r["rvec"], r["tvec"]))

  F = _flip_matrix(board)
  flipped = [False] * len(records)

  def angle(T):
    return abs(float(np.linalg.norm(cv2.Rodrigues(T[:3, :3])[0])))

  for j in range(1, len(records)):
    cost = [0.0, 0.0]
    for i in range(j):
      Ci = T_cb[i] @ F if flipped[i] else T_cb[i]
      a = angle(np.linalg.inv(T_bg[j]) @ T_bg[i])
      for k, Cj in enumerate((T_cb[j], T_cb[j] @ F)):
        cost[k] += abs(a - angle(Cj @ np.linalg.inv(Ci)))
    if cost[1] < cost[0]:
      flipped[j] = True

  out = []
  for r, f in zip(records, flipped):
    if not f:
      out.append(r)
      continue
    T = _rt(r["rvec"], r["tvec"]) @ F
    r = dict(r)
    r["rvec"] = cv2.Rodrigues(T[:3, :3])[0].ravel().tolist()
    r["tvec"] = T[:3, 3].tolist()
    out.append(r)
  return out, sum(flipped)


def _transform_mean(Ts: list[np.ndarray]) -> np.ndarray:
  """A small SE(3) mean suitable for a rigid-mount initial guess."""
  M = np.sum([T[:3, :3] for T in Ts], axis=0)
  U, _, Vt = np.linalg.svd(M)
  R = U @ Vt
  if np.linalg.det(R) < 0:
    U[:, -1] *= -1
    R = U @ Vt
  out = np.eye(4)
  out[:3, :3] = R
  out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
  return out


def _pack_transform(T: np.ndarray) -> np.ndarray:
  return np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]


def _unpack_transform(x: np.ndarray) -> np.ndarray:
  T = np.eye(4)
  T[:3, :3] = cv2.Rodrigues(np.asarray(x[:3], dtype=np.float64))[0]
  T[:3, 3] = np.asarray(x[3:6], dtype=np.float64)
  return T


def refine_reprojection(records: list[dict], T_base_cam: np.ndarray,
                        K: np.ndarray | None = None,
                        dist: np.ndarray | None = None,
                        gripper_site: str = "grasp_site") -> dict | None:
  """Jointly refine camera and rigid board mount from every fused corner.

  The closed-form hand-eye result remains the initial guess.  Unlike a
  pose-level AX=XB solve, this stage does not compress every image to one PnP
  transform: it minimises every observed corner's pixel error with a robust
  loss while sharing one camera pose and one board-to-gripper transform.
  """
  observed = [r for r in records
              if r.get("object_points") is not None
              and r.get("image_points") is not None]
  if len(observed) < 4:
    return None
  if K is None:
    K = observed[0].get("camera_K")
  if dist is None:
    dist = observed[0].get("camera_dist", [0.0] * 5)
  if K is None:
    return None
  K = np.asarray(K, dtype=np.float64).reshape(3, 3)
  dist = np.asarray(dist, dtype=np.float64).reshape(-1)

  from scipy.optimize import least_squares
  from .proprio import Kinematics

  kin = Kinematics(site_name=gripper_site)
  samples = []
  mounts = []
  for r in observed:
    kin.update(np.asarray(r["joint_pos"], dtype=np.float64))
    T_bg = np.eye(4)
    T_bg[:3, :3] = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    T_bg[:3, 3] = kin.data.site_xpos[kin.site_id]
    obj = np.asarray(r["object_points"], dtype=np.float64).reshape(-1, 3)
    img = np.asarray(r["image_points"], dtype=np.float64).reshape(-1, 2)
    if len(obj) != len(img) or len(obj) < 4:
      continue
    samples.append((T_bg, obj, img))
    mounts.append(np.linalg.inv(T_bg) @ T_base_cam
                  @ _rt(r["rvec"], r["tvec"]))
  if len(samples) < 4:
    return None
  T_gripper_board = _transform_mean(mounts)
  x0 = np.r_[_pack_transform(T_base_cam),
             _pack_transform(T_gripper_board)]

  def residual(x):
    T_bc = _unpack_transform(x[:6])
    T_gb = _unpack_transform(x[6:])
    T_cam_base = np.linalg.inv(T_bc)
    out = []
    for T_bg, obj, img in samples:
      T_cb = T_cam_base @ T_bg @ T_gb
      rvec = cv2.Rodrigues(T_cb[:3, :3])[0]
      uv, _ = cv2.projectPoints(obj, rvec, T_cb[:3, 3], K, dist)
      e = uv.reshape(-1, 2) - img
      xyz = obj @ T_cb[:3, :3].T + T_cb[:3, 3]
      if np.any(xyz[:, 2] <= 0.03):
        e += np.maximum(0.03 - xyz[:, 2:3], 0.0) * 1000.0
      out.append(e.ravel())
    return np.concatenate(out)

  before = residual(x0)
  fit = least_squares(residual, x0, loss="soft_l1", f_scale=1.0,
                      x_scale="jac", max_nfev=300)
  after = residual(fit.x)
  rms0 = float(np.sqrt(np.mean(before ** 2)))
  rms = float(np.sqrt(np.mean(after ** 2)))
  if (not fit.success or not np.isfinite(rms)
      or rms > max(rms0 * 1.05, rms0 + 0.05)):
    return None
  per_pose = []
  offset = 0
  for _, obj, _ in samples:
    n = len(obj) * 2
    per_pose.append(float(np.sqrt(np.mean(after[offset:offset + n] ** 2))))
    offset += n
  return {
    "T_base_cam": _unpack_transform(fit.x[:6]),
    "T_gripper_board": _unpack_transform(fit.x[6:]),
    "reprojection_rms_px": rms,
    "reprojection_before_px": rms0,
    "per_pose_reprojection_px": per_pose,
    "n_observations": int(sum(len(x[1]) for x in samples)),
    "optimizer_nfev": int(fit.nfev),
  }


def solve(records: list[dict], gripper_site: str = "grasp_site",
          board: Board | None = None,
          min_rotation_span_deg: float = MIN_ROT_SPAN_DEG,
          K: np.ndarray | None = None,
          dist: np.ndarray | None = None,
          refine: bool = True):
  """Camera pose in the base frame, plus a residual per pose.

  The residual is the thing to read.  ``calibrateHandEye`` will return a
  transform for any input; what says whether to believe it is that the
  board's pose in the base frame, computed through the solution, lands in the
  same place from every viewpoint.  It is a fixed thing held in a moving hand,
  so its pose in the *gripper* frame is constant, and the spread of that is the
  error.

  ``min_rotation_span_deg`` is explicit so the GUI may request a rough result
  for view planning after its manual bootstrap.  Saving and the CLI leave it
  at ``MIN_ROT_SPAN_DEG``; lowering it does not lower the calibration gate.
  """
  from .proprio import Kinematics

  n_flipped = 0
  if board is not None and board.kind == "checker":
    records, n_flipped = _unflip(records, board, gripper_site)

  kin = Kinematics(site_name=gripper_site)
  R_bg, t_bg, R_cb, t_cb = [], [], [], []
  for r in records:
    kin.update(np.asarray(r["joint_pos"], dtype=np.float64))
    T_bg = np.eye(4)
    T_bg[:3, :3] = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    T_bg[:3, 3] = kin.data.site_xpos[kin.site_id]
    # Eye-to-hand: hand the solver the inverse poses and it returns the camera
    # in the base frame instead of the camera in the gripper frame.
    T_gb = np.linalg.inv(T_bg)
    R_bg.append(T_gb[:3, :3])
    t_bg.append(T_gb[:3, 3])
    T_cb_i = _rt(r["rvec"], r["tvec"])
    R_cb.append(T_cb_i[:3, :3])
    t_cb.append(T_cb_i[:3, 3])

  spans = [_angle_between(R_bg[i], R_bg[j])
           for i in range(len(R_bg)) for j in range(i + 1, len(R_bg))]
  rot_span = max(spans) if spans else 0.0
  if rot_span < min_rotation_span_deg:
    # Return rather than raise, so the caller can print the number and say what
    # to do about it.  There is nothing to solve: with no rotation the equation
    # is satisfied by any X, and a solver handed this returns one.
    return {"T_base_cam": None, "rot_span_deg": rot_span,
            "n_poses": len(records), "residual_mm": float("nan"),
            "worst_mm": float("nan"), "worst_pose": -1, "per_pose_mm": [],
            "n_flipped": n_flipped}

  R, t = calibrate_hand_eye(R_bg, t_bg, R_cb, t_cb)
  T_base_cam = np.eye(4)
  T_base_cam[:3, :3] = R
  T_base_cam[:3, 3] = np.asarray(t).ravel()

  refined = (refine_reprojection(records, T_base_cam, K, dist, gripper_site)
             if refine and (board is None or board.kind == "charuco") else None)
  if refined is not None:
    T_base_cam = refined["T_base_cam"]

  # The board in the gripper frame, from every pose.  Constant if the solution
  # is right.
  in_gripper = []
  for r, Rc, tc in zip(records, R_cb, t_cb):
    kin.update(np.asarray(r["joint_pos"], dtype=np.float64))
    T_bg = np.eye(4)
    T_bg[:3, :3] = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    T_bg[:3, 3] = kin.data.site_xpos[kin.site_id]
    T_cb_i = np.eye(4)
    T_cb_i[:3, :3], T_cb_i[:3, 3] = Rc, tc
    in_gripper.append(np.linalg.inv(T_bg) @ T_base_cam @ T_cb_i)

  origins = np.stack([T[:3, 3] for T in in_gripper])
  centre = origins.mean(axis=0)
  per_pose_mm = np.linalg.norm(origins - centre, axis=1) * 1000
  result = {
    "T_base_cam": T_base_cam,
    "residual_mm": float(np.sqrt((per_pose_mm ** 2).mean())),
    "worst_mm": float(per_pose_mm.max()),
    "worst_pose": int(per_pose_mm.argmax()),
    "per_pose_mm": per_pose_mm.tolist(),
    "rot_span_deg": rot_span,
    "n_poses": len(records),
    "n_flipped": n_flipped,
  }
  if refined is not None:
    result.update({
      "joint_refined": True,
      "reprojection_rms_px": refined["reprojection_rms_px"],
      "reprojection_before_px": refined["reprojection_before_px"],
      "per_pose_reprojection_px": refined["per_pose_reprojection_px"],
      "n_corner_observations": refined["n_observations"],
      "optimizer_nfev": refined["optimizer_nfev"],
      "T_gripper_board": refined["T_gripper_board"],
    })
  else:
    result["joint_refined"] = False
  return result


def fit_table(depth: np.ndarray, T_base_cam: np.ndarray, K: np.ndarray) -> dict:
  """Height and tilt of the table in the base frame, from one depth frame."""
  from . import rectify

  rig = config.Rig(T_base_cam=T_base_cam, K=K)
  reproj = rectify.Reprojector(rig)
  pts = reproj.points_base(depth, rig)
  (xlo, xhi), (ylo, yhi), _ = config.WORKSPACE
  sel = ((pts[:, 0] > xlo) & (pts[:, 0] < xhi)
         & (pts[:, 1] > ylo) & (pts[:, 1] < yhi)
         & (np.abs(pts[:, 2] - config.TABLE_Z_M) < 0.08))
  if sel.sum() < 5000:
    raise RuntimeError(
      f"only {int(sel.sum())} points near the expected table height.  Either "
      "the extrinsic is wrong or the camera is not looking at the table."
    )
  q = pts[sel]
  w = np.ones(q.shape[0])
  for _ in range(4):
    mu = (q * w[:, None]).sum(0) / w.sum()
    cov = ((q - mu) * w[:, None]).T @ (q - mu) / w.sum()
    n = np.linalg.eigh(cov)[1][:, 0]
    if n[2] < 0:
      n = -n
    r = (q - mu) @ n
    s = 1.4826 * np.median(np.abs(r)) + 1e-4
    w = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
  return {
    "table_z": float(mu[2]),
    "tilt_deg": float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))),
    "flatness_mm": float(1.4826 * np.median(np.abs((q - mu) @ n)) * 1000),
    "n_points": int(sel.sum()),
  }


# ---------------------------------------------------------------------------


def load_poses(path=POSES_FILE) -> tuple[Board, list[dict]]:
  """The recorded poses and the board they were recorded against.

  The board is stored with them because it is not deducible from them and
  getting it wrong is silent: solving a 25 mm board's poses as a 33 mm one
  scales the whole answer by 1.32 and reports a residual that is still small,
  because the residual measures self-consistency and a uniformly wrong ruler
  is perfectly self-consistent.

  A bare list is the old format, from before there was more than one board.
  """
  d = json.loads(pathlib.Path(path).read_text())
  if isinstance(d, list):
    return DEFAULT_BOARD, d
  return Board.from_dict(d.get("board", {})), d["poses"]


def save_poses(board: Board, records: list[dict], path=POSES_FILE) -> None:
  pathlib.Path(path).write_text(json.dumps(
    {"board": board.to_dict(), "poses": records}, indent=2))


def board_from_args(args) -> Board:
  b = Board.load(args.board) if args.board else Board()
  kw = {}
  if args.board_kind:
    kw["kind"] = args.board_kind
  if args.squares:
    kw["squares"] = tuple(int(x) for x in args.squares.lower().split("x"))
  if args.square_mm:
    kw["square_mm"] = args.square_mm
  if args.marker_mm:
    kw["marker_mm"] = args.marker_mm
  if args.dict:
    kw["dictionary"] = args.dict
  if args.legacy:
    kw["legacy"] = True
  if "square_mm" in kw:
    kw["square_m"] = kw.pop("square_mm") / 1000.0
  if "marker_mm" in kw:
    kw["marker_m"] = kw.pop("marker_mm") / 1000.0
  # A ChArUco board given a square size and no marker size: the ratio on every
  # board this repository has seen is 0.75, and guessing it wrong is caught by
  # the detector immediately rather than quietly.
  if kw.get("square_m") and "marker_m" not in kw and \
     kw.get("kind", b.kind) == "charuco":
    kw["marker_m"] = round(kw["square_m"] * 0.75, 5)
  return dataclasses.replace(b, **kw)


def preview(args) -> int:
  """The board, the camera, and nothing else -- run this before the arm.

  Collecting eight poses only to find out at ``--solve`` that the dictionary
  was wrong is eight poses of wasted time, and the failure at collection time
  is a bare "board not found" that does not say which of the five numbers is
  the wrong one.  This prints what the detector sees, continuously, so the
  board description can be fixed while looking at it.
  """
  from . import sensor

  board = board_from_args(args)
  print(f"board: {board.describe()}\n")
  reader = sensor.Reader(serial=args.serial, width=1280, height=720,
                         gray_source="left_ir")
  first = reader.wait_for_first()
  if getattr(first, "ir", None) is None:
    print("WARNING: no infrared stream; falling back to the depth-aligned "
          "colour frame, which is holed wherever the depth is.\n")
  try:
    for i in range(args.frames):
      frame = reader.latest()
      if frame is None:
        print("no frame")
        continue
      pose = detect_board(board_image(frame), reader.K, reader.dist, board)
      if pose is None:
        print(f"[{i:3d}] not found")
      else:
        t = np.asarray(pose["tvec"]).ravel()
        print(f"[{i:3d}] {pose['n_corners']:3d} corners  "
              f"{pose['reproj_rms_px']:.2f} px  "
              f"range {np.linalg.norm(t) * 1000:.0f} mm")
      time.sleep(0.2)
  finally:
    reader.close()
  return 0


def collect(args) -> int:
  """Walk through poses by hand, saving one record each time.

  Deliberately manual.  An automatic sweep needs a trusted extrinsic to know
  where to point the board, which is the thing being measured, and a scripted
  arm carrying a printed board towards a fixed camera is a way to put a hole in
  a printed board.
  """
  from . import robot, sensor

  reader = sensor.Reader(serial=args.serial, width=1280, height=720,
                         gray_source="left_ir")
  reader.wait_for_first()
  # Connected but deliberately not enabled: a disabled PiPER is back-drivable,
  # so the poses are set by moving the arm with a hand, and the drives still
  # report their angles.  Enabling it here would mean jogging a robot towards a
  # fixed camera with a board on the end, one pose at a time.
  arm = robot.PiperArm(args.can) if not args.dry_run else None
  if arm is not None:
    arm.connect()

  board = board_from_args(args)
  records = []
  if POSES_FILE.exists():
    stored, records = load_poses()
    if stored != board and records:
      print(f"REFUSING to append: {len(records)} pose(s) were recorded "
            f"against\n  {stored.describe()}\nand this run is using\n  "
            f"{board.describe()}\nPoses from two boards cannot be solved "
            f"together.  Delete {POSES_FILE.name} to start over.")
      reader.close()
      if arm is not None:
        arm.close()
      return 1
  print(f"board: {board.describe()}, {board.n_corners()} corners")
  want = board.suggested_poses()
  if want > 100:
    print("This target has four corners.  Measured, a single marker gives a "
          "camera position\nwrong by ~100 mm and does not improve with more "
          "poses -- see the table in\nREADME.md.  --solve will refuse it.  "
          "Use a ChArUco board.")
  else:
    print(f"Plan on about {want} poses for this board (README.md has the "
          f"measurement).\n{MIN_POSES} is only the minimum the solver will "
          f"accept, not the number that makes it good.")
  print(f"{len(records)} pose(s) already recorded.  Move the arm so the board "
        "is fully visible, then press enter.  'q' to stop.")
  print("The arm is connected but NOT enabled -- move it by hand.")
  print("Vary the ORIENTATION, not just the position: hand-eye is determined "
        f"by rotation and this needs at least {MIN_ROT_SPAN_DEG:.0f} degrees "
        "of spread.")
  try:
    while True:
      if input(f"[{len(records)}] > ").strip().lower() == "q":
        break
      frame = reader.latest()
      if frame is None:
        print("  no frame")
        continue
      pose = detect_board(board_image(frame), reader.K, reader.dist, board)
      if pose is None:
        print("  board not found -- move it into view, add light, or check "
              "the board description with --preview")
        continue
      if arm is None:
        print(f"  --dry-run: {pose['n_corners']} corners, "
              f"{pose['reproj_rms_px']:.2f} px, no arm to read, not recorded")
        continue
      st = arm.read()
      records.append({
        "joint_pos": [*st.q.tolist(), st.gripper, -st.gripper],
        "rvec": np.asarray(pose["rvec"]).ravel().tolist(),
        "tvec": np.asarray(pose["tvec"]).ravel().tolist(),
        "n_corners": pose["n_corners"],
        "reproj_rms_px": pose["reproj_rms_px"],
        "object_points": np.asarray(pose["object_points"]).tolist(),
        "image_points": np.asarray(pose["image_points"]).tolist(),
        "camera_K": reader.K.tolist(),
        "camera_dist": reader.dist.tolist(),
        "image_size": list(reader.meta["resolution"]),
        "image_source": reader.meta.get("gray_source"),
      })
      save_poses(board, records)
      print(f"  recorded: {pose['n_corners']} corners, "
            f"reprojection {pose['reproj_rms_px']:.2f} px")
  finally:
    reader.close()
    if arm is not None:
      arm.close()
  return 0


def run_solve(args) -> int:
  if not POSES_FILE.exists():
    print(f"no poses at {POSES_FILE}; run --collect first")
    return 1
  board, records = load_poses()
  if len(records) < MIN_POSES:
    print(f"{len(records)} poses is not enough; {MIN_POSES} is the minimum")
    return 1

  out = solve(records, board=board)
  print(f"board            {board.describe()}")
  print(f"poses            {out['n_poses']}")
  if out.get("n_flipped"):
    print(f"corner order     {out['n_flipped']} pose(s) came back rotated by "
          "180 deg and were corrected (checkerboard symmetry)")
  print(f"rotation spread  {out['rot_span_deg']:.1f} deg")
  if out["T_base_cam"] is None:
    print(f"\nREFUSING to solve: the poses span {out['rot_span_deg']:.1f} "
          f"degrees of rotation and hand-eye needs {MIN_ROT_SPAN_DEG:.0f}.  "
          "With no rotation the equation is satisfied by any answer.  Collect "
          "more poses with the board TILTED differently, not just moved.")
    return 1
  print(f"residual         {out['residual_mm']:.2f} mm rms, "
        f"{out['worst_mm']:.2f} mm worst (pose {out['worst_pose']})")
  if out.get("joint_refined"):
    print(f"corner refine    {out['reprojection_before_px']:.3f} -> "
          f"{out['reprojection_rms_px']:.3f} px over "
          f"{out['n_corner_observations']} fused corners")
  else:
    print("corner refine    unavailable (legacy poses have no saved corners)")
  T = out["T_base_cam"]
  print(f"camera position  {np.round(T[:3, 3], 4).tolist()} m")
  nominal = config.sim_camera_extrinsic()
  d_pos = float(np.linalg.norm(T[:3, 3] - nominal[:3, 3])) * 1000
  d_rot = _angle_between(T[:3, :3], nominal[:3, :3])
  print(f"vs the simulator {d_pos:.1f} mm and {d_rot:.2f} deg away")

  if out["rot_span_deg"] < MIN_ROT_SPAN_DEG:
    print(f"\nREFUSING to write: the poses span {out['rot_span_deg']:.1f} "
          f"degrees of rotation and hand-eye needs {MIN_ROT_SPAN_DEG:.0f}.  "
          "Collect more with the board tilted differently, not just moved.")
    return 1
  if out["residual_mm"] > args.max_residual_mm:
    print(f"\nREFUSING to write: {out['residual_mm']:.2f} mm residual is over "
          f"the {args.max_residual_mm:.1f} mm limit.  Pose "
          f"{out['worst_pose']} is the worst; drop it and try again, or "
          "check the board is rigid on the gripper.")
    return 1
  # The camera pose jitter the policy was trained with is 20 mm and 2 degrees,
  # so a calibration further out than that is outside what it has seen.
  if d_pos > 25.0 or d_rot > 3.0:
    print("\nWARNING: the mount is outside the randomisation envelope the "
          "policy was trained with (20 mm, 2 deg).  The pipeline resamples "
          "into the simulator's camera so the image will still be right, but "
          "the parallax is not something it has seen.  Consider moving the "
          "mount.")

  rig = config.Rig(T_base_cam=T, residual_mm=out["residual_mm"])
  if args.table:
    from . import sensor
    reader = sensor.Reader(serial=args.serial)
    frame = reader.wait_for_first()
    reader.close()
    rig.K = reader.K
    table = fit_table(frame.depth, T, reader.K)
    rig.table_tilt_deg = table["tilt_deg"]
    rig.table_flatness_mm = table["flatness_mm"]
    print(f"table            z = {table['table_z'] * 1000:+.1f} mm, tilt "
          f"{table['tilt_deg']:.2f} deg, flatness "
          f"{table['flatness_mm']:.1f} mm over {table['n_points']} points")
    if table["tilt_deg"] > 1.5:
      print("WARNING: the table is more than 1.5 degrees off level in the "
            "base frame.  Either it is, or the calibration is wrong.")
    rig.table_z = table["table_z"]
  rig.save()
  print(f"\nwrote {config.RIG_FILE}")
  return 0


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--collect", action="store_true")
  p.add_argument("--solve", action="store_true")
  p.add_argument("--preview", action="store_true",
                 help="detect the board and print what was found, no arm.  "
                      "Run this first, with the board in the gripper, to "
                      "confirm the description below matches the board.")
  g = p.add_argument_group(
    "the board", "Defaults describe the sheet in hardware/depth_bench/"
    "targets.  A bought board needs its own numbers, and they are recorded "
    "with the poses so --solve cannot use different ones.")
  g.add_argument("--board", default=None, help="JSON file describing it")
  g.add_argument("--board-kind", choices=("charuco", "checker"), default=None)
  g.add_argument("--squares", default=None,
                 help="'5x5'.  ChArUco: squares.  Checkerboard: INNER "
                      "corners, one fewer than the squares each way.")
  g.add_argument("--square-mm", type=float, default=None)
  g.add_argument("--marker-mm", type=float, default=None,
                 help="ChArUco only; defaults to 0.75 x the square")
  g.add_argument("--dict", default=None,
                 help="ChArUco only, e.g. DICT_5X5_100 or DICT_4X4_50")
  g.add_argument("--legacy", action="store_true",
                 help="ChArUco board numbered by the pre-OpenCV-4.6 "
                      "convention; see Board")
  p.add_argument("--frames", type=int, default=100,
                 help="--preview only: how many to report before stopping")
  p.add_argument("--table", action="store_true", default=True,
                 help="also measure the table plane (default on)")
  p.add_argument("--no-table", dest="table", action="store_false")
  p.add_argument("--serial", default=None)
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--dry-run", action="store_true")
  p.add_argument("--max-residual-mm", type=float, default=4.0)
  a = p.parse_args()
  if a.preview:
    return preview(a)
  if a.collect:
    return collect(a)
  if a.solve:
    return run_solve(a)
  p.print_help()
  return 1


if __name__ == "__main__":
  sys.exit(main())
