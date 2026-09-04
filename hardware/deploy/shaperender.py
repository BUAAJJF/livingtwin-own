"""Render the recorded depth as the shape image a segmenter could be run on.

``DepthSegmenter`` already works on shape and nothing else: it reprojects the
depth into the base frame, fits the table, and thresholds the height above it.
What it does *not* do is group those heights the way an instance model would --
it uses connected components, which is why two touching objects come back as
one and a noisy object comes back as three.

That suggests an input this repository has never tried: give an instance model
the height field instead of the camera's grayscale.  It keeps the property the
task actually wants -- no colour, no texture, nothing that changes when the
lighting or the object does -- while replacing connected components with
something that understands objects.  This writes that image so a model in
another environment can be pointed at it.

The rendering is a decision, not a formatting step, and it is deliberately the
same quantity the depth backend thresholds:

* **height above the fitted table plane**, clipped to 0-120 mm, so the objects
  (24-90 mm) use most of the range and the arm saturates rather than
  compressing everything else into the bottom of the scale.
* the plane is **fitted per frame**, as ``DepthSegmenter`` does, because the
  camera's range bias is a per-unit unknown and a fixed plane turns the whole
  table into objects or hides the short ones.
* **turbo**, matching ``review.py``'s depth view, so a person comparing the two
  is looking at the same colours.
* where the depth dropped there is no height, and those pixels are **black**.
  Painting them as table would be inventing a measurement.

    python -m hardware.deploy.shaperender recordings/session --out DIR
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import cv2
import numpy as np

from . import mask, segbench

CEILING_M = 0.120
"""Top of the colour scale.  Above the 110 mm ``max_top_z_m`` a component may
have and still be an object, so anything the task cares about is inside the
ramp and the arm is off the end of it."""


def render(height_full: np.ndarray, valid: np.ndarray,
           style: str = "turbo") -> np.ndarray:
  """The height field as a picture.

  Two styles, and the difference is not cosmetic.  ``turbo`` maps height
  straight to colour, which is the readable choice for a person and measured
  to be a very poor one for SAM3: run on it the model returned 0.79 detections
  a frame against 6.67 on the grayscale, and segmenter recall collapsed from
  0.86 to 0.07.  A colourmapped scalar field is nothing like the photographs
  the model was trained on.

  ``relief`` shades the same field as a surface lit from above and to the left,
  which is what a photograph of those objects would look like with the colour
  and texture removed.  It carries exactly the same information -- there is no
  appearance in it -- and it presents it as an image rather than as a chart.
  """
  if style == "turbo":
    v = np.clip(height_full / CEILING_M, 0.0, 1.0)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = (0, 0, 0)
    return img

  # Lambertian shading of the height field.  The gradient is taken on a lightly
  # blurred copy so the sensor's per-pixel scatter does not become texture the
  # model then tries to segment.
  h = np.where(valid, height_full, 0.0).astype(np.float32)
  h = cv2.GaussianBlur(h, (0, 0), 1.6)
  # Exaggerated: a 30 mm object over a 0.7 m scene is a very shallow relief,
  # and unexaggerated it shades to a flat grey.
  gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) * 60.0
  gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) * 60.0
  n = np.dstack([-gx, -gy, np.ones_like(h)])
  n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
  light = np.array([-0.45, -0.55, 0.70], np.float32)
  light /= np.linalg.norm(light)
  lam = np.clip((n * light).sum(axis=2), 0.0, 1.0)
  # Height also lifts the base tone, so a tall thing reads as tall and not only
  # as an edge -- the shading alone makes a flat top identical to the table.
  base = 0.30 + 0.45 * np.clip(h / CEILING_M, 0.0, 1.0)
  grey = np.clip(0.45 * base + 0.75 * lam * base + 0.10, 0.0, 1.0)
  img = (grey * 255).astype(np.uint8)
  out = np.dstack([img, img, img])
  out[~valid] = (0, 0, 0)
  return out


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("session", type=pathlib.Path)
  p.add_argument("--out", type=pathlib.Path, required=True)
  p.add_argument("--style", choices=("turbo", "relief"),
                 default="turbo")
  p.add_argument("--quality", type=int, default=92)
  p.add_argument("--limit", type=int, default=None)
  a = p.parse_args()

  rig, reproj, meta, files = segbench._session_setup(a.session)
  seg = mask.DepthSegmenter(rig, reproj, decimate=1)
  """Full resolution here, unlike the depth backend's own halved grid: this is
  an image for a network that resizes it anyway, and halving it would throw
  away the separation between two touching objects that the whole exercise is
  meant to test."""
  a.out.mkdir(parents=True, exist_ok=True)
  if a.limit:
    files = files[:a.limit]

  for k, f in enumerate(files):
    key = pathlib.Path(f).stem
    z = np.load(f)
    depth = z["depth"].astype(np.float32) / 10000.0
    d = seg.reproj.source(depth)
    h, w = d.shape
    pts = seg.reproj.points_base(d, rig)
    src = seg.reproj.last_src
    in_box = mask.workspace_mask(pts)
    height, _, _ = mask.fit_table_plane(pts, in_box, rig.table_z,
                                        seg.cfg.plane_fit_points)
    full = np.zeros(h * w, dtype=np.float32)
    valid = np.zeros(h * w, dtype=bool)
    full[src] = height
    valid[src] = True
    img = render(full.reshape(h, w), valid.reshape(h, w), a.style)
    cv2.imwrite(str(a.out / f"{key}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, int(a.quality)])
    if k % 400 == 0:
      print(f"{k + 1}/{len(files)}", flush=True)
  print(f"wrote {len(files)} shape images to {a.out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
