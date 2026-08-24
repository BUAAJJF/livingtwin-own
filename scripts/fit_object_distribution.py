"""Place the objects you actually bought inside the distribution that was trained.

Run it after filling in ``hardware/objects/measured.json``.  It answers three
questions and nothing else:

  where does each object land?    its width, height and aspect as percentiles of
                                  the sampled distribution, and whether it is
                                  outside a hard limit -- which is a returns
                                  question, not a training question.
  is the mass range wrong?        the trained range is 50-400 g drawn
                                  independently of size, which implies densities
                                  no material has.  Real objects will say how
                                  wrong.
  which fix?                      widen ``OBJECT_MASS_RANGE``, or sample density
                                  and compute mass from the composed volume.
                                  The measured spread decides it, and the rule
                                  is printed with the numbers behind it.

The reference distribution is not a model of the sampler, it is the sampler:
``shapes._compose`` is called directly, so this cannot drift from what training
draws.

    micromamba run -n mjlab python scripts/fit_object_distribution.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

# mjlab first: importing it runs the entry-point scan that imports
# ``piper_push.tasks``, which imports ``shapes``.  Reaching for ``shapes``
# before that starts the chain from inside itself and the task package fails to
# register -- with a traceback on stderr and a working script, which is the
# worst of both.
import mjlab.tasks  # noqa: F401
import torch

from piper_push import objects, shapes

HERE = pathlib.Path(__file__).resolve().parents[1]
MEASURED = HERE / "hardware" / "objects" / "measured.json"

CLASSES = {name: i for i, name in enumerate(objects.SHAPE_CLASSES)}


def reference(n: int, device: str, seed: int) -> dict[str, torch.Tensor]:
  """Sample the training distribution and reduce it to the quantities compared."""
  g = torch.Generator(device=device).manual_seed(seed)
  size, pos, half, cls = shapes._compose(n, torch.device(device), g, 1.0)
  wide = 2.0 * torch.maximum(half[:, 0], half[:, 1])
  narrow = 2.0 * torch.minimum(half[:, 0], half[:, 1])
  height = 2.0 * half[:, 2]
  return {
    "cls": cls,
    "wide": wide, "narrow": narrow, "height": height,
    "mean_width": 0.5 * (wide + narrow),
    "aspect": height / (0.5 * (wide + narrow)),
    "anisotropy": wide / narrow,
    "bbox_vol": wide * narrow * height,
    "solid_vol": solid_volume(size),
  }


def solid_volume(size: torch.Tensor) -> torch.Tensor:
  """The composed volume, by the same rule ``_mass_properties`` uses.

  Part 1 is the cylinder and its third size entry is unused; a part that is
  switched off is a millimetre cube and contributes nothing.
  """
  cyl = torch.zeros(size.shape[0], 3, dtype=torch.bool, device=size.device)
  cyl[:, 1] = True
  ext = torch.where(
    cyl.unsqueeze(-1),
    torch.stack((size[..., 0], size[..., 0], size[..., 1]), dim=-1),
    size,
  )
  vol = torch.where(
    cyl,
    math.pi * size[..., 0] ** 2 * 2.0 * size[..., 1],
    8.0 * size[..., 0] * size[..., 1] * size[..., 2],
  )
  vol = torch.where(ext.max(dim=-1).values <= objects.COLLAPSED_HALF * 1.5,
                    torch.zeros_like(vol), vol)
  return vol.sum(dim=1)


def nominal_solid_volume(cls: str, long_m: float, short_m: float,
                         height_m: float) -> float:
  """What the simulator's composition of this class fills, in m^3.

  The measured object will not fill it exactly -- a real jar is hollow, a real
  bracket has a radius on its corner -- and that difference is precisely the
  thing being measured here.  The point of comparison has to be the sim's own
  geometry, because that is what the density parameter would feed.
  """
  hx, hy, hz = long_m / 2, short_m / 2, height_m / 2
  mean_half = 0.5 * (hx + hy)
  if cls == "box":
    return 8 * hx * hy * hz
  if cls == "cylinder":
    return math.pi * mean_half ** 2 * 2 * hz
  if cls == "stepped":
    return 8 * hx * hy * (hz * 0.6) + 8 * (hx * 0.55) * (hy * 0.55) * (hz * 0.4)
  if cls == "l_shape":
    return 8 * (hx * 0.5) * hy * hz + 8 * (hx * 0.5) * hy * (hz * 0.45)
  if cls == "capped":
    r = min(hx, hy) * 0.75
    return 8 * hx * hy * (hz * 0.55) + math.pi * r ** 2 * 2 * (hz * 0.45)
  raise ValueError(f"unknown shape class {cls!r}")


def pct(ref: torch.Tensor, value: float) -> float:
  return 100.0 * float((ref < value).float().mean())


def quantiles(x: torch.Tensor, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> list[float]:
  return [float(torch.quantile(x.float(), q)) for q in qs]


def check_limits(o: dict) -> list[str]:
  bad = []
  long_mm, short_mm = o["long_mm"], o["short_mm"]
  h, m = o["height_mm"], o["mass_g"]
  if long_mm > 50:
    bad.append(f"{long_mm:.0f} mm wide, over the 50 mm the pads can hold")
  if h < 24:
    bad.append(f"{h:.0f} mm tall, under the 24 mm the fingertip needs")
  if h > 90:
    bad.append(f"{h:.0f} mm tall, over the 90 mm that stays upright")
  if long_mm / max(short_mm, 1e-6) > 1.4:
    bad.append(f"plan ratio {long_mm / short_mm:.2f}, over the sampler's 1.4")
  if m > 600:
    bad.append(f"{m:.0f} g, over the gripper's ~600 g")
  elif m > 400:
    bad.append(f"{m:.0f} g, over the trained 400 g (holdable, not trained)")
  return bad


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--measured", default=str(MEASURED))
  p.add_argument("--samples", type=int, default=200_000)
  p.add_argument("--device", default="cpu")
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()

  ref = reference(a.samples, a.device, a.seed)

  doc = json.loads(pathlib.Path(a.measured).read_text())
  need = ("long_mm", "short_mm", "height_mm", "mass_g")
  found, blank = [], []
  for o in doc["objects"]:
    (found if all(o.get(k) is not None for k in need) else blank).append(o)
  if not found:
    print(f"nothing filled in yet in {a.measured}.\n"
          f"Weigh the objects first -- every row needs "
          f"{', '.join(need)}.")
    return 1
  if blank:
    print(f"note: {len(blank)} of {len(doc['objects'])} rows are still blank "
          f"(ids {[o['id'] for o in blank]}); reporting on the rest.\n")

  print(f"reference: {a.samples:,} draws from shapes._compose at variety=1.0\n")

  # ---- where each object lands ------------------------------------------
  print(f"{'#':>2} {'class':9s} {'name':14s} {'L×S×H mm':>16s} {'g':>6s} "
        f"{'W%':>4s} {'H%':>4s} {'A%':>4s} {'ρ_solid':>8s} {'ρ_bbox':>7s}")
  rows = []
  for o in found:
    cls = o["class"]
    L, S, H, m = (float(o["long_mm"]), float(o["short_mm"]),
                  float(o["height_mm"]), float(o["mass_g"]))
    mean_w = 0.5 * (L + S)
    aspect = H / mean_w
    vs = nominal_solid_volume(cls, L / 1000, S / 1000, H / 1000)
    rho_s = (m / 1000) / vs / 1000            # g/cm^3
    rho_b = (m / 1000) / (L * S * H / 1e9) / 1000
    rows.append(dict(o, L=L, S=S, H=H, m=m, aspect=aspect,
                     rho_solid=rho_s, rho_bbox=rho_b, vol=vs))
    print(f"{o['id']:>2} {cls:9s} {(o.get('name') or '-')[:14]:14s} "
          f"{f'{L:.0f}×{S:.0f}×{H:.0f}':>16s} {m:6.0f} "
          f"{pct(ref['wide'] * 1000, L):4.0f} "
          f"{pct(ref['height'] * 1000, H):4.0f} "
          f"{pct(ref['aspect'], aspect):4.0f} "
          f"{rho_s:8.2f} {rho_b:7.2f}")
  print("  W/H/A are percentiles of the trained distribution; ρ in g/cm³, "
        "solid against\n  the sim's own composition of that class.")

  # ---- anything that should go back --------------------------------------
  problems = [(r, check_limits(r)) for r in rows]
  problems = [(r, b) for r, b in problems if b]
  if problems:
    print("\noutside a hard limit:")
    for r, b in problems:
      for line in b:
        print(f"  #{r['id']} {(r.get('name') or r['class'])}: {line}")
  else:
    print("\nevery object is inside the hard limits.")

  # ---- coverage of the marginals ------------------------------------------
  print("\ncoverage")
  for key, label, scale in (("wide", "width  ", 1000.0),
                            ("height", "height ", 1000.0),
                            ("aspect", "aspect ", 1.0)):
    r5, r25, r50, r75, r95 = quantiles(ref[key] * scale)
    have = sorted(r["L"] if key == "wide" else
                  r["H"] if key == "height" else r["aspect"] for r in rows)
    pcts = sorted(pct(ref[key] * scale, v) for v in have)
    gap = max([pcts[0]] + [b - a_ for a_, b in zip(pcts, pcts[1:])]
              + [100 - pcts[-1]])
    print(f"  {label} trained p5-p95 {r5:6.1f} - {r95:6.1f}   "
          f"set {have[0]:6.1f} - {have[-1]:6.1f}   "
          f"percentiles {pcts[0]:.0f}-{pcts[-1]:.0f}, "
          f"largest gap {gap:.0f} points")
  seen = {r["class"] for r in rows}
  missing = [c for c in objects.SHAPE_CLASSES if c not in seen]
  if missing:
    print(f"  no object for shape class(es): {', '.join(missing)} -- the policy "
          f"was trained on them\n  and nothing on the table will test them.")

  # ---- the mass question --------------------------------------------------
  masses = torch.tensor([r["m"] / 1000 for r in rows])
  rho = torch.tensor([r["rho_solid"] for r in rows])
  lo, hi = objects.OBJECT_MASS_RANGE
  inside = float(((masses >= lo) & (masses <= hi)).float().mean())

  print(f"\nmass, against the trained OBJECT_MASS_RANGE = "
        f"({lo:.2f}, {hi:.2f}) kg")
  print(f"  measured  {float(masses.min()) * 1000:.0f} - "
        f"{float(masses.max()) * 1000:.0f} g "
        f"(median {float(masses.median()) * 1000:.0f})")
  print(f"  {100 * inside:.0f}% of the set is inside the trained range")

  rho_lo, rho_hi = float(rho.min()), float(rho.max())
  cv_m = float(masses.std() / masses.mean()) if len(rows) > 1 else 0.0
  cv_r = float(rho.std() / rho.mean()) if len(rows) > 1 else 0.0
  print(f"  measured solid density {rho_lo:.2f} - {rho_hi:.2f} g/cm³ "
        f"(median {float(rho.median()):.2f})")
  print(f"  spread: mass CV {cv_m:.2f}, density CV {cv_r:.2f}")

  # What the trained sampler implies, for contrast.
  implied = (0.5 * (lo + hi)) / ref["bbox_vol"] / 1000
  q = quantiles(implied)
  print(f"  the trained sampler's implied bbox density, at its own mean mass: "
        f"p5 {q[0]:.1f}  p50 {q[2]:.1f}  p95 {q[4]:.1f} g/cm³")

  # ---- the recommendation --------------------------------------------------
  print("\n" + "=" * 70)
  m_lo = max(0.01, math.floor(float(masses.min()) * 1000 * 0.8 / 10) * 10 / 1000)
  m_hi = min(0.60, math.ceil(float(masses.max()) * 1000 * 1.25 / 10) * 10 / 1000)

  if cv_r < 0.8 * cv_m and len(rows) >= 4:
    verdict = "sample density, not mass"
    why = (f"the objects vary more in mass (CV {cv_m:.2f}) than in density "
           f"(CV {cv_r:.2f}),\nwhich is what a set of real objects does: mass "
           f"is size times material, and only\nthe material is free.  Drawing "
           f"mass independently of size cannot reproduce that.")
  else:
    verdict = "widen OBJECT_MASS_RANGE"
    why = (f"density is as scattered as mass here (CV {cv_r:.2f} against "
           f"{cv_m:.2f}), so a density\nparameter would buy no structure.  "
           f"Move the range and retrain.")
  print(f"recommendation: {verdict}\n\n{why}\n")

  print(f"option 1 -- src/piper_push/objects.py")
  print(f"    OBJECT_MASS_RANGE = ({m_lo:.2f}, {m_hi:.2f})"
        f"    # was ({lo:.2f}, {hi:.2f})")
  covered = float(((masses >= m_lo) & (masses <= m_hi)).float().mean())
  print(f"    covers {100 * covered:.0f}% of the set, with margin either side.")

  d_lo = max(0.2, math.floor(rho_lo * 0.8 * 10) / 10)
  d_hi = math.ceil(rho_hi * 1.25 * 10) / 10
  implied_mass = torch.tensor([d_lo, d_hi]).mean() * ref["solid_vol"] * 1000
  mm = quantiles(implied_mass * 1000)
  over = 100.0 * float((implied_mass * 1000 > 400).float().mean())
  print(f"\noption 2 -- draw density and compute mass from the composed volume")
  print(f"    OBJECT_DENSITY_RANGE = ({d_lo:.1f}, {d_hi:.1f})   # g/cm³")
  print(f"    mass = density * solid_volume, clipped to the gripper's 600 g")
  print(f"    over the sampled shapes that gives, at mean density: "
        f"p5 {mm[0]:.0f}  p50 {mm[2]:.0f}  p95 {mm[4]:.0f} g")
  print(f"    {over:.1f}% of draws would exceed 400 g")
  print(f"    shapes.py line ~341 is the one place mass is drawn; it already "
        f"has the\n    composed volume in hand at _mass_properties.")
  print("=" * 70)
  print("\nEither way it is one change and one retrain.  Do not do both.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
