#!/usr/bin/env python3
"""Generate the A4 target the depth bench measures against.

One sheet has to answer three different questions, so it carries three regions
rather than being a plain calibration board:

  * a **ChArUco board**, which fixes the pose of the sheet.  The plane depth is
    compared against is then measured, not assumed to be the table -- and the
    markers survive the board being half out of frame, which a bare
    checkerboard does not.  It is also the well-textured best case for any
    stereo camera.
  * a **blank white patch**.  This is the one that decides things for the
    D405, which has no infrared projector and matches on scene texture alone;
    a textureless white surface is where passive stereo has nothing to match.
    A projector-equipped camera should barely notice it.
  * a **solid black patch**, the dark-surface case.  ``camera.py`` currently
    models dropout as one uniform probability over the whole image, and the
    gap between these two patches is the measurement that says how wrong that
    is.

The patches sit at known offsets from the board origin, so a single ChArUco
pose gives the ground-truth plane for all three regions at once and the
comparison between them is free of any per-region alignment error.

Geometry is laid out in millimetres at an exact integer pixels-per-millimetre
scale, so a printer told "100%, no fit-to-page" reproduces the dimensions the
scripts assume.  The 150 mm bar is there to catch the printer that scales
anyway.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PX_PER_MM = 20  # 508 dpi -- integer millimetres, comfortably above print res
A4_MM = (210.0, 297.0)

DICT_NAME = "DICT_5X5_100"
SQUARES_X, SQUARES_Y = 5, 5
SQUARE_MM = 33.0
MARKER_MM = 25.0
BOARD_X0, BOARD_Y0 = 22.5, 10.0

PATCH_MM = 60.0
PATCH_Y0 = 183.0
PATCH_X0 = {"white": 35.0, "black": 115.0}
PATCH_MARGIN_MM = 8.0
"""How far the measured region is eroded inside the printed patch.  Covers the
outline, the print edge, and a couple of pixels of ChArUco pose error."""

BAR_MM = 150.0
BAR_Y = 256.0

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def mm(v: float) -> int:
  return int(round(v * PX_PER_MM))


def build() -> tuple[Image.Image, dict]:
  dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
  board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_MM / 1000.0, MARKER_MM / 1000.0, dictionary
  )
  board_px = (mm(SQUARES_X * SQUARE_MM), mm(SQUARES_Y * SQUARE_MM))
  # borderBits=1: each marker's quiet zone is the white checker square it sits in.
  board_img = board.generateImage(board_px, marginSize=0, borderBits=1)

  page = Image.new("L", (mm(A4_MM[0]), mm(A4_MM[1])), 255)
  page.paste(Image.fromarray(board_img), (mm(BOARD_X0), mm(BOARD_Y0)))

  draw = ImageDraw.Draw(page)
  title = ImageFont.truetype(FONT_BOLD, mm(4.5))
  body = ImageFont.truetype(FONT, mm(3.2))
  small = ImageFont.truetype(FONT, mm(2.6))

  patches = {}
  for name, x0 in PATCH_X0.items():
    box = [mm(x0), mm(PATCH_Y0), mm(x0 + PATCH_MM), mm(PATCH_Y0 + PATCH_MM)]
    if name == "black":
      draw.rectangle(box, fill=0)
    else:
      # Outline only -- the interior must stay bare paper, or it is no longer
      # a textureless target.
      draw.rectangle(box, outline=0, width=mm(0.4))
    draw.text((mm(x0 + PATCH_MM / 2), mm(PATCH_Y0 + PATCH_MM + 2)),
              {"white": "WHITE - no texture", "black": "BLACK - dark surface"}[name],
              font=small, fill=0, anchor="ma")

    # In the ChArUco board frame: origin at the board's top-left corner, x
    # right, y down, metres, z = 0.  Verified against board.getObjPoints().
    m = PATCH_MARGIN_MM
    patches[name] = {
      "x_min_m": (x0 + m - BOARD_X0) / 1000.0,
      "x_max_m": (x0 + PATCH_MM - m - BOARD_X0) / 1000.0,
      "y_min_m": (PATCH_Y0 + m - BOARD_Y0) / 1000.0,
      "y_max_m": (PATCH_Y0 + PATCH_MM - m - BOARD_Y0) / 1000.0,
    }

  bar_x = mm((A4_MM[0] - BAR_MM) / 2)
  by = mm(BAR_Y)
  draw.line([(bar_x, by), (bar_x + mm(BAR_MM), by)], fill=0, width=mm(0.5))
  for t in range(0, int(BAR_MM) + 1, 10):
    tx = bar_x + mm(t)
    draw.line([(tx, by), (tx, by - (mm(3.5) if t % 50 == 0 else mm(2.0)))],
              fill=0, width=mm(0.4))
    if t % 50 == 0:
      draw.text((tx, by + mm(1.2)), f"{t}", font=small, fill=0, anchor="ma")
  cx = page.width // 2
  draw.text((cx, mm(263)), "measure this bar: it must be exactly 150 mm",
            font=body, fill=0, anchor="ma")
  draw.text((cx, mm(269)), "LivingTwin depth bench - A4 target",
            font=title, fill=0, anchor="ma")
  draw.text((cx, mm(276)),
            f"{DICT_NAME}  {SQUARES_X}x{SQUARES_Y}  square {SQUARE_MM:g} mm  "
            f"marker {MARKER_MM:g} mm  patch {PATCH_MM:g} mm",
            font=body, fill=0, anchor="ma")
  draw.text((cx, mm(281)),
            "Print at 100% / actual size - no fit-to-page, no borderless scaling. "
            "Matte paper, not glossy. Tape flat to a rigid board.",
            font=small, fill=0, anchor="ma")

  # Board-frame extent of the ChArUco squares themselves, for the textured ROI.
  meta = {
    "dictionary": DICT_NAME,
    "squares_x": SQUARES_X,
    "squares_y": SQUARES_Y,
    "square_m": SQUARE_MM / 1000.0,
    "marker_m": MARKER_MM / 1000.0,
    "page": "A4",
    "px_per_mm": PX_PER_MM,
    "scale_bar_mm": BAR_MM,
    "regions": {
      "charuco": {
        "x_min_m": SQUARE_MM / 2000.0,
        "x_max_m": (SQUARES_X * SQUARE_MM - SQUARE_MM / 2) / 1000.0,
        "y_min_m": SQUARE_MM / 2000.0,
        "y_max_m": (SQUARES_Y * SQUARE_MM - SQUARE_MM / 2) / 1000.0,
      },
      **patches,
    },
  }
  return page, meta


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, default=Path(__file__).parent)
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  page, meta = build()
  dpi = PX_PER_MM * 25.4
  page.save(args.out / "target_a4.pdf", "PDF", resolution=dpi)
  page.save(args.out / "target_a4.png", dpi=(dpi, dpi))
  (args.out / "target_a4.json").write_text(json.dumps(meta, indent=2) + "\n")
  print(f"{args.out/'target_a4.pdf'}  ({page.width}x{page.height} px @ {dpi:.1f} dpi)")
  for k, v in meta["regions"].items():
    w = (v["x_max_m"] - v["x_min_m"]) * 1000
    h = (v["y_max_m"] - v["y_min_m"]) * 1000
    print(f"  region {k:8s} {w:5.1f} x {h:5.1f} mm")


if __name__ == "__main__":
  main()
