"""Manifold Tech Odin 1 backend.

This device does not work the way the other two do, and three of its
properties were established by measurement here rather than taken from the
vendor's documentation -- one of them contradicts it.

**Depth arrives in millimetres, not metres.**  The SDK notes say metres.  They
are wrong, and they are wrong in a way that only a real device reveals: the
claim had been checked against synthetic buffers.  Pointed at a desk 39 cm
away this sensor returns 390.  Guarding against a future firmware that changes
its mind, ``Stream`` sanity-checks the median depth on the first frame and
refuses to run if it is not in a plausible tabletop range.

**Its depth grid is not a pinhole projection.**  Fitting (fx, fy, cx, cy) to
the measured ray directions leaves a 17-pixel residual on a 256x192 grid, and
the implied field of view is 120 x 101 degrees.  So the bench carries a **ray
table** instead, measured from the sensor's own XYZ output: over 45 frames the
per-pixel direction varies by 1.4e-08, which is to say the geometry is a fixed
device constant and measuring it is exact.  The ``depth`` channel the SDK also
offers is all zeros on this unit and is not used; ``z``/``xyz`` are.

**The board has to be found in the colour image.**  At 256x192 over 120
degrees a 25 mm ArUco marker is under two pixels across -- there is no
detector that reads that.  The 1600x1296 colour camera is a ``FishPoly``
fisheye, not a pinhole, so it is undistorted to a pinhole first (verified by
straight edges coming out straight), the board is found there, and the pose is
carried into the lidar frame by the factory extrinsic.

That extrinsic needed one derivation.  ``calib.yaml`` gives ``Tcl``, camera
from *lidar* frame, but the XYZ stream is in a camera-convention frame with +Z
forward, and ``Tcl`` expects +X forward.  Composing ``Tcl`` with the fixed
permutation (x, y, z) -> (z, -x, -y) makes the rotation come out within a
degree of identity, which is what two sensors bolted side by side looking the
same way must give; projecting the lidar cloud through it reproduces the
colour scene, and dropping the permutation produces garbage.

**What this costs the bias number.**  ``bias`` on this camera is measured
across that extrinsic, so its error adds to ``plane_uncertainty_m``, which only
knows about the fit in the image it saw.  ``fill``, ``spatial_rms`` and
``temporal_std`` do not cross the extrinsic and are unaffected.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import Capture

NAME = "odin1"
USB_ID = ("2207", "0019")
DTOF_W, DTOF_H = 256, 192

RAYS_CACHE = Path(__file__).resolve().parent / "odin1_raytable.npz"

_PERM = np.eye(4)
_PERM[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
"""dTOF output frame -> the 'lidar' frame Tcl is written against."""


def add_args(ap) -> None:
  ap.add_argument("--odin-rate", type=int, default=2, choices=(0, 1, 2),
                  help="0=10Hz 1=14.5Hz 2=29Hz")
  ap.add_argument("--conf-min", type=int, default=30,
                  help="vendor-recommended confidence gate (30-35); "
                       "this is a pipeline choice and it moves `fill`")
  ap.add_argument("--undistort-f", type=float, default=620.0,
                  help="focal length of the pinhole the colour is rectified to")
  ap.add_argument("--undistort-size", default="1280x1024")
  ap.add_argument("--rebuild-rays", action="store_true")


def available() -> list[dict]:
  """Look for the device in sysfs rather than opening it.

  Opening takes about eleven seconds and claims the USB interface; the live
  viewer enumerates every backend at startup and must not pay that, or hold
  the device, for a camera it may not use.
  """
  out = []
  for d in Path("/sys/bus/usb/devices").glob("*"):
    try:
      if (d / "idVendor").read_text().strip() != USB_ID[0]:
        continue
      if (d / "idProduct").read_text().strip() != USB_ID[1]:
        continue
      serial = (d / "serial").read_text().strip() if (d / "serial").exists() else ""
      out.append({"backend": NAME, "serial": serial,
                  "model": "Odin 1", "usb": str(d.name),
                  "product": (d / "product").read_text().strip()})
    except OSError:
      continue
  return out


class FishPoly:
  """The vendor's fisheye polynomial colour model, from ``calib.yaml``.

  r(theta) = theta + k2 theta^2 + ... + k7 theta^7, then an affine map with a
  skew term.  The convention is not documented; it is the one under which
  ``theta`` at the stated 120 degree maximum incident angle lands on the image
  edge (u = 1573 of 1600) and under which straight edges rectify straight.
  """

  def __init__(self, cam: dict):
    self.k = [float(cam[f"k{i}"]) for i in range(2, 8)]
    self.A11, self.A12, self.A22 = (float(cam["A11"]), float(cam["A12"]),
                                    float(cam["A22"]))
    self.u0, self.v0 = float(cam["u0"]), float(cam["v0"])
    self.width, self.height = int(cam["image_width"]), int(cam["image_height"])

  def project(self, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    rho = np.linalg.norm(P[..., :2], axis=-1)
    th = np.arctan2(rho, P[..., 2])
    r = th.copy()
    for i, k in enumerate(self.k):
      r = r + k * th ** (i + 2)
    s = np.where(rho > 1e-12, r / np.where(rho > 1e-12, rho, 1.0), 0.0)
    mx, my = P[..., 0] * s, P[..., 1] * s
    return np.stack([self.A11 * mx + self.A12 * my + self.u0,
                     self.A22 * my + self.v0], axis=-1)


def _parse_calib(path: Path) -> tuple[np.ndarray, FishPoly]:
  import yaml

  d = yaml.safe_load(Path(path).read_text())
  Tcl = np.array(d["Tcl_0"], float).reshape(4, 4)
  if d["cam_0"]["cam_model"] != "FishPoly":
    raise RuntimeError(f"unexpected colour model {d['cam_0']['cam_model']!r}; "
                       "the rectification here assumes FishPoly")
  return Tcl, FishPoly(d["cam_0"])


def _fill_holes(rays: np.ndarray, have: np.ndarray) -> np.ndarray:
  """Complete the ray table where the sensor never returned anything.

  Measured directions are kept exactly -- they are good to 1e-8 and a fit is
  not.  The gaps are filled from a degree-7 surface, which reproduces the
  measured field to about 0.6 px; those pixels exist so that a pixel which
  returns nothing during a measurement can still be told whether it was aimed
  at the target, which is the difference between dropout being measured and
  dropout being invisible.
  """
  h, w, _ = rays.shape
  u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
  un, vn = (u - w / 2) / w, (v - h / 2) / h
  cols = [(un ** i) * (vn ** j) for i in range(8) for j in range(8 - i)]
  A = np.stack(cols, -1)
  out = rays.copy()
  for k in range(2):
    c, *_ = np.linalg.lstsq(A[have], rays[..., k][have], rcond=None)
    out[..., k] = np.where(have, rays[..., k], A @ c)
  out[..., 2] = 1.0
  return out


def _ray_table(read_frames, args, serial: str) -> tuple[np.ndarray, dict]:
  """Load, extend and cache the sensor's fixed ray geometry."""
  cached, have = None, None
  if RAYS_CACHE.exists() and not args.rebuild_rays:
    z = np.load(RAYS_CACHE)
    if str(z["serial"]) == serial:
      cached, have = z["rays"], z["have"]

  xyz, zz = read_frames(60)
  ok = zz > 0.05
  with np.errstate(invalid="ignore", divide="ignore"):
    r = np.where(ok[..., None], xyz / np.where(zz[..., None] > 0, zz[..., None], 1.0),
                 np.nan)
  fresh = np.nanmean(r, axis=0) if r.shape[0] else np.full((DTOF_H, DTOF_W, 3), np.nan)
  fresh_have = np.isfinite(fresh[..., 0])

  if cached is None:
    rays, have = fresh, fresh_have
  else:
    # Coverage only ever improves: a pixel resolved in any past session keeps
    # its measured direction, and this session fills in whatever it can.
    rays = np.where(have[..., None], cached, fresh)
    have = have | fresh_have
  np.savez_compressed(RAYS_CACHE, rays=rays, have=have, serial=serial)
  info = {"rays_measured_frac": float(have.mean()),
          "rays_cache": str(RAYS_CACHE)}
  return _fill_holes(np.nan_to_num(rays, nan=0.0), have), info


class Stream:
  def __init__(self, args, serial: str | None = None):
    import odin1 as sdk

    self._sdk = sdk
    self.dev = sdk.Odin1()
    try:
      self.dev.wait_for_device(timeout=15.0)
    except Exception:
      # The unit streams with an empty device_info block; the attach callback
      # having fired is what matters and open() below is the real check.
      pass
    self.dev.open()
    self.dev.set_mode("raw")
    self.dev.set_depth_rate(args.odin_rate)
    self.dev.start_stream(sdk.LIDAR_DT_RAW_DTOF)
    self.dev.start_stream(sdk.LIDAR_DT_RAW_RGB)

    with tempfile.TemporaryDirectory() as td:
      self.dev.save_calibration(td)
      Tcl, self.cam = _parse_calib(Path(td) / "calib.yaml")
    T_cam_dtof = Tcl @ _PERM
    self.T_dg = np.linalg.inv(T_cam_dtof)

    w, h = (int(x) for x in args.undistort_size.split("x"))
    f = args.undistort_f
    self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    self.dist = np.zeros(5)
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    d = np.stack([(uu - w / 2) / f, (vv - h / 2) / f, np.ones_like(uu)], -1)
    src = self.cam.project(d)
    self._mapx = src[..., 0].astype(np.float32)
    self._mapy = src[..., 1].astype(np.float32)

    self._conf_min = args.conf_min
    self._last_gray = None
    dev_serial = (available() or [{}])[0].get("serial", "")
    self.rays, ray_info = _ray_table(self._raw_frames, args, dev_serial)

    depth0, _ = self.read()
    med = float(np.median(depth0[depth0 > 0])) if (depth0 > 0).any() else 0.0
    if not 0.05 < med < 20.0:
      raise RuntimeError(
        f"median depth {med:.3f} m is not plausible -- the SDK's depth units "
        "may have changed. This backend divides the raw z by 1000 because this "
        "device reports millimetres, contrary to the SDK notes."
      )

    self.meta = {
      "backend": NAME,
      "model": "Manifold Tech Odin 1",
      "serial": dev_serial,
      "sdk": sdk.SDK_VERSION,
      "resolution": [DTOF_W, DTOF_H],
      "fps": {0: 10, 1: 14.5, 2: 29}[args.odin_rate],
      "depth_units_m": 1e-3,
      "conf_min": args.conf_min,
      "emitter": "SPAD dTOF, active illumination",
      "colour_model": "FishPoly rectified to a pinhole",
      "colour_resolution": [w, h],
      "colour_fx_px": f,
      "geometry": "measured ray table (the depth grid is not a pinhole)",
      "cross_frame_pose": True,
      **ray_info,
    }

  def _raw_frames(self, n: int) -> tuple[np.ndarray, np.ndarray]:
    xyz, zz = [], []
    for _ in range(n):
      d = self.dev.read_depth(timeout=15.0)
      xyz.append(np.asarray(d.xyz, np.float64) / 1000.0)
      zz.append(np.asarray(d.z, np.float64) / 1000.0)
    return np.stack(xyz), np.stack(zz)

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    d, c = self.dev.read_frame(timeout=15.0)
    z = np.asarray(d.z, np.float32) / 1000.0
    conf = np.asarray(d.confidence)
    depth = np.where((z > 0) & (conf >= self._conf_min), z, 0.0).astype(np.float32)
    if c is not None:
      und = cv2.remap(np.asarray(c.image), self._mapx, self._mapy,
                      cv2.INTER_LINEAR, borderValue=(0, 0, 0))
      self._last_gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    if self._last_gray is None:
      raise RuntimeError("no colour frame yet; cannot locate the target")
    return depth, self._last_gray

  def close(self) -> None:
    try:
      self.dev.shutdown()
    except Exception:
      pass


def grab(args, n_frames: int, warmup: int = 15) -> Capture:
  s = Stream(args)
  try:
    for _ in range(warmup):
      s.read()
    depths, gray = [], None
    for _ in range(n_frames):
      d, gray = s.read()
      depths.append(d)
    return Capture(depth=np.stack(depths), gray=gray, K=s.K, dist=s.dist,
                   meta=dict(s.meta, n_frames=n_frames),
                   rays=s.rays, T_dg=s.T_dg)
  finally:
    s.close()
