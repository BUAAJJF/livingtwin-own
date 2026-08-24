"""Fine-tune YOLO26-seg on the labels the depth segmenter produced.

Small on purpose.  ``yolo26n-seg`` is the smallest of the family and it is
still far more model than this needs: one class, one fixed viewpoint, one
table, objects that are 25-45 mm blocks.  The reason to reach for a network at
all is not that the problem is hard, it is that the depth segmenter's failure
mode -- no depth on untextured surfaces -- is exactly where a colour model has
no trouble, and a bigger model would spend inference time the 20 ms control
period does not have.

The augmentation is deliberately narrow.  The camera does not move, the table
does not move, and the lighting is whatever is in the room.  Mosaic, large
scale jitter and flips would all be teaching invariances this deployment does
not need and cannot use, at the cost of the capacity that goes into them.  What
is left on is brightness and a little translation, which is what actually
varies between one session and the next.

    python -m hardware.deploy.train_yolo --epochs 60
    python -m hardware.deploy.train_yolo --check      # what it learned, in numbers
"""

from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
YOLO_DIR = HERE / "yolo"


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--data", default=str(YOLO_DIR / "data" / "data.yaml"))
  p.add_argument("--model", default="yolo26n-seg.pt")
  p.add_argument("--epochs", type=int, default=60)
  p.add_argument("--imgsz", type=int, default=832,
                 help="close to the sensor's 848 wide, so the network sees the "
                      "object at the 25 pixels it actually covers rather than "
                      "at the 9 a 640 crop would leave")
  p.add_argument("--batch", type=int, default=8)
  p.add_argument("--device", default="0")
  p.add_argument("--check", action="store_true",
                 help="validate the existing weights instead of training")
  a = p.parse_args()

  from ultralytics import YOLO

  data = pathlib.Path(a.data)
  if not data.exists():
    print(f"no dataset at {data}.  Record a session with "
          "`run.py --record`, then label it with `autolabel.py`.")
    return 1

  if a.check:
    weights = YOLO_DIR / "best.pt"
    if not weights.exists():
      print(f"no weights at {weights}")
      return 1
    m = YOLO(str(weights))
    r = m.val(data=str(data), imgsz=a.imgsz, device=a.device)
    print(f"mask mAP50 {r.seg.map50:.3f}  mAP50-95 {r.seg.map:.3f}")
    # The number that matters is not mAP.  It is whether the model finds the
    # object on the frames the depth segmenter could not label, and no
    # validation set built from the frames it *could* label can measure that.
    # Compare the two on a recording with a white object in it before trusting
    # this.
    print("mAP here is measured against the depth segmenter's own output, so "
          "it says how well the student copied the teacher -- not whether it "
          "beats it where the teacher fails, which is the point.")
    return 0

  m = YOLO(a.model)
  m.train(
    data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
    device=a.device, project=str(YOLO_DIR), name="train", exist_ok=True,
    # A fixed camera on a fixed table: none of the usual augmentation applies.
    mosaic=0.0, mixup=0.0, copy_paste=0.0,
    fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
    scale=0.15, translate=0.06,
    hsv_h=0.0, hsv_s=0.2, hsv_v=0.4,
  )
  best = YOLO_DIR / "train" / "weights" / "best.pt"
  if best.exists():
    (YOLO_DIR / "best.pt").write_bytes(best.read_bytes())
    print(f"copied {best} -> {YOLO_DIR / 'best.pt'}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
