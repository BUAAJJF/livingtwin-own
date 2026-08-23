"""Fit the depth-noise model the simulator will use, from the bench captures.

The simulator needs four separate things and the bench measures all four, but
only if they are kept apart.  Collapsing them into one Gaussian -- which is
what ``camera.DEPTH_NOISE_M`` did -- gets the magnitude roughly right and the
*character* completely wrong, and the character is what a convolutional policy
keys on.

  1. how the error grows with distance          sigma = a * z^2   (stereo)
  2. how much of it changes between frames      temporal vs static
  3. how far it is correlated across the image  correlation length
  4. where the sensor returns nothing at all    holes, and whether they flicker

The distinction in (2) is the one that matters most and it is not intuitive.
Passive stereo matches on the texture that is there; if the texture does not
move, the disparity error does not move either.  So most of the error is a
*fixed pattern* that a policy sees as real geometry, and only a small part is
the twinkle that per-pixel Gaussian noise models.

Reads:  results/live/<session>/report.json    (many distances, three surfaces)
        results/d405/desk_v2/frames.npz       (40 raw frames, one scene)
Writes: model/d405_noise.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def fit_quadratic(z, sigma):
  """sigma = a * z^2 through the origin, which is the stereo prediction.

  Not a free intercept: an intercept buys fit quality by absorbing whatever the
  target was doing at the near end, and then the simulator extrapolates it to
  0.7 m where it is no longer true.  The residual is reported so the shape can
  be checked rather than assumed.
  """
  z = np.asarray(z, float)
  sigma = np.asarray(sigma, float)
  ok = np.isfinite(z) & np.isfinite(sigma) & (z > 0)
  z, sigma = z[ok], sigma[ok]
  if z.size < 2:
    return float("nan"), float("nan")
  a = float((z**2 @ sigma) / (z**2 @ z**2))
  resid = sigma - a * z**2
  return a, float(np.sqrt(np.mean(resid**2)))


def fit_linear(x, y):
  """y = m*x + c, reported so a depth bias can be told from a scale error."""
  x = np.asarray(x, float)
  y = np.asarray(y, float)
  ok = np.isfinite(x) & np.isfinite(y)
  x, y = x[ok], y[ok]
  if x.size < 2:
    return float("nan"), float("nan"), float("nan")
  A = np.stack([x, np.ones_like(x)], axis=1)
  (m, c), *_ = np.linalg.lstsq(A, y, rcond=None)
  resid = y - (m * x + c)
  return float(m), float(c), float(np.sqrt(np.mean(resid**2)))


def from_session(report: dict) -> dict:
  """Distance dependence, per surface class, from the paired live session."""
  out = {}
  for cam, regions in report["summary"].items():
    if cam.lower() != "d405":
      continue
    raw = report["raw"][cam]
    for region, s in regions.items():
      z = np.asarray(raw[region]["z"], float)
      rms = np.asarray(raw[region]["rms"], float)
      bias = np.asarray(raw[region].get("bias", []), float)
      a, resid = fit_quadratic(z, rms)
      rec = {
        "n": int(s["n"]),
        "z_range_m": [float(z.min()), float(z.max())],
        "sigma_a_per_m": a,
        "sigma_fit_resid_m": resid,
        "sigma_at_0.70m": a * 0.70**2,
        "fill_mean": s["fill_mean"],
        "fill_min": s["fill_min"],
        "temporal_median_m": s["temporal_median_m"],
      }
      if bias.size == z.size and np.isfinite(bias).any():
        m, c, r = fit_linear(z, bias)
        rec["bias_slope"] = m           # dimensionless: a range scale error
        rec["bias_offset_m"] = c        # a constant offset
        rec["bias_fit_resid_m"] = r
        rec["bias_median_m"] = float(np.nanmedian(bias))
      out[region] = rec
  return out


def _plane_residual(depth, rays, mask):
  """Distance from each valid sample to the best plane through the patch.

  Fitted in 3-D rather than on the depth image, because a plane tilted away
  from the camera is not a plane in z(u, v) and the tilt would land in the
  residual as noise.
  """
  pts = rays[mask] * depth[mask][:, None]
  c = pts.mean(0)
  _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
  n = vt[-1]
  return (pts - c) @ n


def from_frames(path: pathlib.Path, patch: int = 48, min_valid: float = 0.995) -> dict:
  """Split the error into the part that moves and the part that does not.

  Works on small patches of one static scene: within a patch the surface is
  close enough to planar that what is left after removing a plane is the
  sensor.  Doing it per patch rather than per image also keeps the answer local
  in distance, so the z-dependence does not leak into the spread.

  Everything here is an RMS, never a median, because the live session's
  ``spatial_rms`` is an RMS and the two numbers have to be comparable.  On a
  heavy-tailed error the median is roughly half the RMS, which is enough to
  reverse the conclusion about which component dominates.

  The static term is what is left of the N-frame mean once the temporal term's
  own contribution to that mean -- which is ``sigma_t / sqrt(N)``, not zero --
  has been taken out in quadrature.
  """
  z = np.load(path, allow_pickle=True)
  depth = z["depth"].astype(np.float64)       # (N, H, W)
  rays = z["rays"]
  n_frames, H, W = depth.shape
  valid = depth > 0

  stack = np.where(valid, depth, np.nan)
  with np.errstate(all="ignore"):
    mu = np.nanmean(stack, axis=0)
    temporal = np.nanstd(stack, axis=0, ddof=1)
  keep = valid.mean(0) >= min_valid

  rows, recs = [], []
  for y0 in range(0, H - patch, patch // 2):
    for x0 in range(0, W - patch, patch // 2):
      sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
      m = keep[sl]
      if m.mean() < 0.98:
        continue
      zc = float(np.nanmedian(mu[sl][m]))
      if not (0.25 < zc < 1.1):
        continue
      # Reject patches straddling a depth edge: the plane fit would be
      # meaningless and the residual would be the step, not the sensor.
      if np.nanpercentile(mu[sl][m], 98) - np.nanpercentile(mu[sl][m], 2) > 0.05:
        continue
      res = _plane_residual(mu[sl], rays[sl], m)
      if np.sqrt(np.mean(res**2)) > 0.02:      # not actually a planar patch
        continue
      rows.append((zc,
                   float(np.sqrt(np.mean(res**2))),
                   float(np.sqrt(np.nanmean(temporal[sl][m] ** 2)))))
      recs.append((y0, x0))

  rows = np.asarray(rows)
  a_mean, r_mean = fit_quadratic(rows[:, 0], rows[:, 1])   # of the N-frame mean
  a_tp, r_tp = fit_quadratic(rows[:, 0], rows[:, 2])       # per frame
  a_static = float(np.sqrt(max(a_mean**2 - a_tp**2 / n_frames, 0.0)))

  # Correlation length, for the two components separately: they are drawn
  # differently in the simulator and there is no reason they should match.
  # Measured on a patch four times the longest lag looked at, because removing
  # a plane from a patch forces the autocorrelation negative at lags
  # comparable to the patch itself -- which is what a first attempt on 24 px
  # patches produced, and it is an artefact of the fit, not the sensor.
  lags = np.arange(0, patch // 4 + 1)

  def _profile(get):
    acc, n_acc = np.zeros(lags.size), 0
    for (y0, x0) in recs:
      sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
      m = keep[sl]
      if m.mean() < 1.0:
        continue
      img = get(sl, m)
      if img is None:
        continue
      img = img - np.nanmean(img)
      v = float(np.nanmean(img * img))
      if not np.isfinite(v) or v <= 0:
        continue
      acc += np.asarray([float(np.nanmean(img[:, : patch - k] * img[:, k:]) / v)
                         for k in lags])
      n_acc += 1
    return acc / max(n_acc, 1), n_acc

  def _static(sl, m):
    img = np.full((patch, patch), np.nan)
    img[m] = _plane_residual(mu[sl], rays[sl], m)
    return img

  def _twinkle(sl, m):
    # One frame minus the mean: the static pattern cancels and what is left is
    # only what changed, so no plane has to be removed and no artefact is
    # introduced by removing one.
    img = np.full((patch, patch), np.nan)
    d = stack[n_frames // 2][sl] - mu[sl]
    img[m] = d[m]
    return img if np.isfinite(img[m]).all() else None

  prof_s, n_s = _profile(_static)
  prof_t, n_t = _profile(_twinkle)

  def _corr_len(prof):
    below = np.where(prof < np.exp(-1.0))[0]
    return float(below[0]) if below.size else float(lags[-1])

  # Temporal correlation at lag 1, which decides whether the twinkle can be
  # redrawn independently every control step or has to be carried over.
  res_t = stack - mu[None]
  sel = keep & (mu > 0.25) & (mu < 1.1)
  with np.errstate(all="ignore"):
    lag1 = float(np.nanmean(res_t[:-1][:, sel] * res_t[1:][:, sel])
                 / np.nanmean(res_t[:, sel] ** 2))

  # Holes: steady or flickering.  A pixel that is never valid is a surface the
  # sensor cannot see; one that is valid half the time is the failure mode a
  # recurrent policy actually has to survive, and the two want different
  # treatment in the simulator.
  frac = valid.mean(0)
  near = (mu > 0.25) & (mu < 1.1)
  near = near | (valid.sum(0) == 0)          # never-valid pixels have no mu
  return {
    "source": str(path),
    "n_frames": int(n_frames),
    "n_patches": int(rows.shape[0]),
    "patch_px": patch,
    "mean_image_a_per_m": a_mean,
    "temporal_a_per_m": a_tp,
    "temporal_fit_resid_m": r_tp,
    "static_a_per_m": a_static,
    "static_share_of_variance": float(a_static**2 / (a_static**2 + a_tp**2)),
    "corr_len_static_px": _corr_len(prof_s),
    "corr_len_temporal_px": _corr_len(prof_t),
    "corr_profile_static": [float(x) for x in prof_s],
    "corr_profile_temporal": [float(x) for x in prof_t],
    "corr_n_patches": [n_s, n_t],
    "temporal_lag1_corr": lag1,
    "hole_never_valid": float((frac[near] == 0).mean()),
    "hole_always_valid": float((frac[near] == 1).mean()),
    "hole_flickering": float(((frac[near] > 0) & (frac[near] < 1)).mean()),
  }


def from_edges(path: pathlib.Path) -> dict:
  """How the sensor fails at a depth discontinuity, which is where it matters.

  A 25-45 mm object at 0.7 m is almost entirely silhouette: its outline is what
  says how wide and how tall it is, and passive stereo has nothing to match
  across an occlusion boundary.  So the interesting question is not the average
  fill rate, it is the fill rate as a function of the local depth step.

  The gradient is expressed per radian rather than per pixel so the number
  transfers to the policy's own 224x168 grid, where a pixel subtends 2.7x more
  angle and the same physical edge therefore looks 2.7x steeper.
  """
  z = np.load(path, allow_pickle=True)
  depth = z["depth"].astype(np.float64)
  K = z["K"]
  valid = depth > 0
  with np.errstate(all="ignore"):
    mu = np.nanmean(np.where(valid, depth, np.nan), axis=0)
  frac = valid.mean(0)

  gy, gx = np.gradient(np.nan_to_num(mu, nan=0.0))
  f_px_per_rad = float(0.5 * (K[0, 0] + K[1, 1]))
  ok = np.isfinite(mu) & (mu > 0.25) & (mu < 1.1)
  with np.errstate(all="ignore"):
    g = np.where(ok, np.hypot(gx, gy) / np.maximum(mu, 1e-6) * f_px_per_rad, np.nan)

  bins = [0.0, 0.9, 2.2, 4.3, 8.6, 21.5, 43.1, np.inf]
  rows = []
  for lo, hi in zip(bins[:-1], bins[1:]):
    sel = ok & (g >= lo) & (g < hi)
    if sel.sum() < 200:
      continue
    rows.append({
      "grad_lo_per_rad": lo,
      "grad_hi_per_rad": float(hi) if np.isfinite(hi) else None,
      "n": int(sel.sum()),
      "fill": float(frac[sel].mean()),
      "flicker": float(((frac[sel] > 0) & (frac[sel] < 1)).mean()),
    })

  # p_valid = p_flat / (1 + (g / g50)**n), fitted on the bin centres.  Two
  # parameters because a third would be fitted to seven points, and both of
  # these are randomised in the simulator anyway -- the shape is what is being
  # transferred, not the exact value.
  gc = np.array([0.5 * (r["grad_lo_per_rad"] + (r["grad_hi_per_rad"] or 86.0))
                 for r in rows])
  pv = np.array([r["fill"] for r in rows])
  p_flat = float(pv[0])
  best = (np.inf, 1.0, 1.0)
  for g50 in np.geomspace(1.0, 200.0, 200):
    for n in np.linspace(0.3, 2.0, 60):
      pred = p_flat / (1.0 + (gc / g50) ** n)
      e = float(np.mean((pred - pv) ** 2))
      if e < best[0]:
        best = (e, g50, n)
  rms, g50, n = best
  return {
    "f_px_per_rad": f_px_per_rad,
    "bins": rows,
    "p_flat": p_flat,
    "g50_per_rad": float(g50),
    "exponent": float(n),
    "fit_rms": float(np.sqrt(rms)),
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--session", default=None,
                  help="results/live/<stamp>; default is the newest")
  ap.add_argument("--frames", default=str(RESULTS / "d405" / "desk_v2" / "frames.npz"))
  ap.add_argument("--out", default=str(HERE / "d405_noise.json"))
  a = ap.parse_args()

  sessions = sorted((RESULTS / "live").glob("*/report.json"))
  if not sessions:
    raise SystemExit("no live sessions under results/live")
  report_path = (pathlib.Path(a.session) / "report.json") if a.session else sessions[-1]
  report = json.loads(report_path.read_text())

  model = {
    "camera": "Intel RealSense D405",
    "session": str(report_path.parent),
    "distance": from_session(report),
    "structure": from_frames(pathlib.Path(a.frames)),
    "edges": from_edges(pathlib.Path(a.frames)),
  }
  pathlib.Path(a.out).write_text(json.dumps(model, indent=2))
  print(json.dumps(model, indent=2))


if __name__ == "__main__":
  main()
