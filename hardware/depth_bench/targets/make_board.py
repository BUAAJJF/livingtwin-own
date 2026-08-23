#!/usr/bin/env python3
"""Generate the A4 ChArUco target the depth bench measures against.

The board does three jobs at once, which is why it is a ChArUco and not a
plain checkerboard or a plain white card:

  * the markers give an unambiguous pose even when the board is partly out of
    frame, so the plane the depth is compared against is *measured*, not
    assumed to be the table;
  * the white squares are the clean surface the noise and bias numbers come
    from -- printer black absorbs infrared and would otherwise contaminate them;
  * the black squares are the dark-surface stress case, measured separately,
    because dropout on dark objects is a failure mode the simulator currently
    models as a single uniform probability.

Geometry is laid out in millimetres at an exact integer pixels-per-millimetre
scale, so a printer told "100%, no fit-to-page" reproduces the dimensions the
scripts assume.  The 150 mm bar on the sheet is there to catch the printer
that scales anyway.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PX_PER_MM = 20  # 508 dpi -- integer mm, comfortably above print resolution
A4_MM = (210, 297)

SQUARES_X, SQUARES_Y = 5, 7
SQUARE_MM = 33.0
MARKER_MM = 25.0
DICT_NAME = "DICT_5X5_100"

BOARD_TOP_MM = 20.0
BAR_MM = 150.0

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
  # borderBits=1: the marker quiet zone is the white checker square itself.
  img = board.generateImage(board_px, marginSize=0, borderBits=1)

  page = Image.new("L", (mm(A4_MM[0]), mm(A4_MM[1])), 255)
  x0 = (page.width - board_px[0]) // 2
  y0 = mm(BOARD_TOP_MM)
  page.paste(Image.fromarray(img), (x0, y0))

  draw = ImageDraw.Draw(page)
  title = ImageFont.truetype(FONT_BOLD, mm(4.5))
  body = ImageFont.truetype(FONT, mm(3.2))
  small = ImageFont.truetype(FONT, mm(2.6))

  # Scale bar: 150 mm, ticks every 10 mm, taller ticks every 50 mm.
  bar_y = y0 + board_px[1] + mm(14)
  bar_x = (page.width - mm(BAR_MM)) // 2
  draw.line([(bar_x, bar_y), (bar_x + mm(BAR_MM), bar_y)], fill=0, width=mm(0.5))
  for t in range(0, int(BAR_MM) + 1, 10):
    h = mm(3.5) if t % 50 == 0 else mm(2.0)
    tx = bar_x + mm(t)
    draw.line([(tx, bar_y), (tx, bar_y - h)], fill=0, width=mm(0.4))
    if t % 50 == 0:
      draw.text((tx, bar_y + mm(1.5)), f"{t}", font=small, fill=0, anchor="ma")
  draw.text(
    (page.width // 2, bar_y + mm(6.5)),
    "measure this bar: it must be exactly 150 mm",
    font=body, fill=0, anchor="ma",
  )

  ty = bar_y + mm(13)
  draw.text((page.width // 2, ty), "LivingTwin depth bench - ChArUco A4",
            font=title, fill=0, anchor="ma")
  spec = (f"{DICT_NAME}  {SQUARES_X}x{SQUARES_Y} squares  "
          f"square {SQUARE_MM:g} mm  marker {MARKER_MM:g} mm")
  draw.text((page.width // 2, ty + mm(6.5)), spec, font=body, fill=0, anchor="ma")
  draw.text(
    (page.width // 2, ty + mm(11.5)),
    "Print at 100% / actual size - no fit-to-page, no borderless scaling. "
    "Matte paper, not glossy.",
    font=small, fill=0, anchor="ma",
  )

  meta = {
    "dictionary": DICT_NAME,
    "squares_x": SQUARES_X,
    "squares_y": SQUARES_Y,
    "square_m": SQUARE_MM / 1000.0,
    "marker_m": MARKER_MM / 1000.0,
    "page": "A4",
    "px_per_mm": PX_PER_MM,
    "scale_bar_mm": BAR_MM,
  }
  return page, meta


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, default=Path(__file__).parent)
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  page, meta = build()
  dpi = PX_PER_MM * 25.4
  pdf = args.out / "charuco_a4.pdf"
  png = args.out / "charuco_a4.png"
  page.save(pdf, "PDF", resolution=dpi)
  page.save(png, dpi=(dpi, dpi))
  (args.out / "charuco_a4.json").write_text(json.dumps(meta, indent=2) + "\n")
  print(f"{pdf}  ({page.width}x{page.height} px @ {dpi:.1f} dpi)")
  print(f"{png}")
  print(json.dumps(meta))


if __name__ == "__main__":
  main()
