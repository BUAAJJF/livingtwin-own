"""One stored picture per camera frame, not one per control step.

``_Recorder.write`` assigned ``_last_frame`` and never compared against it, so
at 50 Hz control against a 30 Hz camera every frame was written between one and
two extra times.  Measured on ``recordings/v4_stereo_repro_scene2``: 2893 files
for 1488 distinct camera frames, 38% of consecutive pairs byte-identical in
depth.

Nothing failed loudly.  What it did instead was make the review page repeat
pictures -- so a session that ran at the camera's full 33.4 ms cadence reads as
a low frame rate when you scrub it -- double the writer thread's compression
load, which is what filled the queue and ended three runs, and send the
workaround for that (``--no-record-compress``) on to halve the observation
latency, moving a control condition by accident.

The sibling path, ``event``, has always deduplicated correctly.  These tests
pin both, because the failure mode is a comparison that quietly is not made.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from hardware.deploy import run as run_mod


class _Frame:
  def __init__(self, index: int) -> None:
    self.index = index
    self.depth = np.zeros((4, 4), np.float32)
    self.gray = np.zeros((4, 4), np.uint8)


def _recorder(tmp_path):
  """A recorder whose worker thread is stopped, so the queue can be read."""
  rec = run_mod._Recorder(str(tmp_path / "session"), queue_size=256)
  rec._thread.join(0)          # the worker is a daemon; do not race it
  return rec


def _drain(rec):
  items = []
  while not rec._queue.empty():
    items.append(rec._queue.get_nowait())
  return items


def _fb(q=0.0):
  return types.SimpleNamespace(
    position=np.zeros(8), velocity=np.zeros(8), target=np.zeros(8),
    gripper_effort=0.0)


def test_one_picture_per_camera_frame_not_per_control_step(tmp_path):
  rec = _recorder(tmp_path)
  depth, gray = np.zeros((4, 4), np.float32), np.zeros((4, 4), np.uint8)
  # 50 Hz control against a 30 Hz camera: five steps, three camera frames.
  for idx in (7, 7, 8, 8, 9):
    rec.write(depth, gray, _fb(), np.zeros(7), 1,
              extra={"frame_index": idx})
  images = _drain(rec)
  assert len(images) == 3, "one image per distinct frame_index"
  assert [r["i"] for r, _, _ in images] == [0, 1, 2]
  # Every control step is still recorded -- the feedback is a 50 Hz fact.
  assert len(rec.control) == 5
  assert sum("frame_file" in r for r in rec.control) == 3


def test_a_step_without_a_frame_index_is_still_stored(tmp_path):
  """Absent is not the same as unchanged.

  ``--dry-run`` and the replay reader can hand the loop a step with no camera
  frame behind it.  Treating a missing index as "same as last time" would drop
  those silently, which is the opposite failure and just as quiet.
  """
  rec = _recorder(tmp_path)
  depth, gray = np.zeros((4, 4), np.float32), np.zeros((4, 4), np.uint8)
  rec.write(depth, gray, _fb(), np.zeros(7), 1, extra={"frame_index": 3})
  rec.write(depth, gray, _fb(), np.zeros(7), 1, extra=None)
  rec.write(depth, gray, _fb(), np.zeros(7), 1, extra={"frame_index": 3})
  images = _drain(rec)
  assert len(images) == 2, "no index stores; a repeated index does not"


def test_the_hold_path_agrees_with_the_command_path(tmp_path):
  """``event`` and ``write`` must not disagree about what a new frame is.

  They share ``_last_frame``.  A run that alternates between commanding and
  holding -- which is every run that loses its target -- passes control back
  and forth between them, and if only one of them advances the counter the
  other stores duplicates.
  """
  rec = _recorder(tmp_path)
  depth, gray = np.zeros((4, 4), np.float32), np.zeros((4, 4), np.uint8)
  rec.write(depth, gray, _fb(), np.zeros(7), 1, extra={"frame_index": 11})
  rec.event(_fb(), np.zeros(7), 0, "hold_no_target",
            {"frame_index": 11}, frame=_Frame(11))
  rec.write(depth, gray, _fb(), np.zeros(7), 1, extra={"frame_index": 11})
  rec.event(_fb(), np.zeros(7), 0, "hold_no_target",
            {"frame_index": 12}, frame=_Frame(12))
  images = _drain(rec)
  assert len(images) == 2, "frames 11 and 12, once each"


def test_visual_perception_arrays_stay_out_of_json_and_are_saved(tmp_path):
  rec = run_mod._Recorder(str(tmp_path / "recorded"), queue_size=8,
                          compress=False)
  frame = _Frame(4)
  frame.rgb = np.full((4, 4, 3), 33, np.uint8)
  frame.ir = np.full((4, 4), 11, np.uint8)
  frame.ir_right = np.full((4, 4), 22, np.uint8)
  frame.detection_labels = np.array(
    [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]], np.int32)
  frame.source_mask = frame.detection_labels > 0
  frame.sam_raw_mask = frame.detection_labels > 0
  frame.sam_rgb_mask = frame.detection_labels > 0
  frame.detection_rgb_labels = frame.detection_labels
  frame.policy_depth = np.full((4, 4), 0.8, np.float32)
  frame.policy_mask = np.ones((2, 3), bool)
  frame.detections = [{"label": 2, "n_px": 4, "top_z": 0.03,
                       "centroid_base": [0.1, 0.2, 0.03],
                       "bbox": [1, 1, 2, 2]}]
  frame.mask_state = "tracking"
  frame.sensor_meta = {"depth_frame_number": 101,
                       "color_frame_number": 101}
  rec.write(frame.depth, frame.gray, _fb(), np.zeros(7), 2,
            extra={"frame_index": 4}, frame=frame)
  rec.close()

  meta = json.loads((rec.dir / "meta.json").read_text())
  control = json.loads((rec.dir / "control.json").read_text())
  assert "_frame_arrays" not in meta[0]
  assert "_frame_arrays" not in control[0]
  assert meta[0]["mask_state"] == "tracking"
  assert meta[0]["sensor"]["color_frame_number"] == 101
  with np.load(rec.dir / "000000.npz") as saved:
    assert int(saved["schema_version"]) == 3
    assert set(("rgb", "ir_left", "ir_right",
                "detection_labels_shape", "detection_labels_run_start",
                "detection_labels_run_length", "detection_labels_run_value",
                "detection_rgb_labels_shape", "sam_rgb_mask_bits",
                "sam_rgb_mask_shape", "source_mask_bits",
                "source_mask_shape")).issubset(saved.files)
    assert not set(("sensor_depth", "sam_raw_mask", "policy_mask")) \
      & set(saved.files)
    source = np.unpackbits(saved["source_mask_bits"], count=16,
                           bitorder="little").reshape(4, 4)
    assert source.sum() == 4
    assert saved["detection_labels_run_value"].tolist() == [2, 2]
