#!/usr/bin/env python3
"""Generate the compact high-detectability hand-eye calibration board.

The 180 x 152 mm finished target is smaller than A4 and only slightly wider
than the old 165 mm square pattern.  Its 6x5 ChArUco layout provides 20
sub-pixel chess corners instead of 16, while DICT_4X4_50 and 22 mm markers put
more camera pixels in every code module than the old 5x5/25 mm markers.

Print at 100% / actual size on matte stock and bond it flat to a rigid plate.
Do not laminate it with glossy film.  The white 6 mm surround is part of the
finished target and should not be trimmed away.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PX_PER_MM = 20                 # 508 dpi
PAGE_MM = (180.0, 152.0)
MARGIN_MM = 6.0
SQUARES = (6, 5)
SQUARE_MM = 28.0
MARKER_MM = 22.0
DICT_NAME = "DICT_4X4_50"


def mm(x: float) -> int:
  return int(round(float(x) * PX_PER_MM))


def build(inverted: bool = False) -> tuple[Image.Image, dict]:
  dictionary = cv2.aruco.getPredefinedDictionary(
    getattr(cv2.aruco, DICT_NAME))
  board = cv2.aruco.CharucoBoard(
    SQUARES, SQUARE_MM / 1000.0, MARKER_MM / 1000.0, dictionary)
  size = (mm(SQUARES[0] * SQUARE_MM), mm(SQUARES[1] * SQUARE_MM))
  pattern = board.generateImage(size, marginSize=0, borderBits=1)
  if inverted:
    pattern = 255 - pattern
  page = Image.new("L", (mm(PAGE_MM[0]), mm(PAGE_MM[1])), 255)
  x = (page.width - pattern.shape[1]) // 2
  y = (page.height - pattern.shape[0]) // 2
  page.paste(Image.fromarray(pattern), (x, y))
  assert abs(x / PX_PER_MM - MARGIN_MM) < 0.01
  assert abs(y / PX_PER_MM - MARGIN_MM) < 0.01
  meta = {
    "kind": "charuco",
    "dictionary": DICT_NAME,
    "squares_x": SQUARES[0],
    "squares_y": SQUARES[1],
    "square_m": SQUARE_MM / 1000.0,
    "marker_m": MARKER_MM / 1000.0,
    "min_corners": 8,
    "inverted": bool(inverted),
    "page_mm": list(PAGE_MM),
    "pattern_mm": [SQUARES[0] * SQUARE_MM, SQUARES[1] * SQUARE_MM],
    "margin_mm": MARGIN_MM,
    "px_per_mm": PX_PER_MM,
    "print": ("100% / actual size; matte; keep the 6 mm white border; "
              + ("white-marker inverted print" if inverted
                 else "standard black-marker print")),
  }
  return page, meta


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, default=Path(__file__).parent)
  ap.add_argument("--inverted", action="store_true",
                  help="ink-saving white-marker board for inkjet printing")
  ap.add_argument("--stem", default=None)
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  page, meta = build(inverted=args.inverted)
  dpi = PX_PER_MM * 25.4
  name = args.stem or ("calib_compact_white_v2" if args.inverted
                       else "calib_compact_v2")
  stem = args.out / name
  page.save(stem.with_suffix(".pdf"), "PDF", resolution=dpi)
  page.save(stem.with_suffix(".png"), dpi=(dpi, dpi))
  stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
  print(f"{stem.with_suffix('.pdf')}  {PAGE_MM[0]:g} x {PAGE_MM[1]:g} mm")
  print(f"pattern {SQUARES[0] * SQUARE_MM:g} x "
        f"{SQUARES[1] * SQUARE_MM:g} mm; {DICT_NAME}")


if __name__ == "__main__":
  main()
