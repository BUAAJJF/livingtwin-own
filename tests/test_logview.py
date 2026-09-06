import json
import pathlib

import cv2
import numpy as np
import pytest

from hardware.deploy import logview


def _recording(root: pathlib.Path, name="session", n=3):
  session = root / name
  session.mkdir()
  meta = []
  for i in range(n):
    np.savez(session / f"{i:06d}.npz",
             gray=np.full((12, 16), 30 + i, np.uint8),
             depth=np.full((12, 16), 6500 + i, np.uint16))
    meta.append({
      "i": i, "frame_file": f"{i:06d}.npz", "frame_index": 10 + 2 * i,
      "frame_stamp": 100.0 + 0.05 * i, "event": "command",
      "joint_pos": [0, 0, 0, 0, 0, 0, 0.04, -0.04],
      "joint_vel": [0] * 8, "target": [0] * 8, "action": [0] * 7,
      "observation_age_s": 0.08, "perception_publish_age_s": 0.02,
    })
  (session / "meta.json").write_text(json.dumps(meta))
  (session / "run.json").write_text(json.dumps({
    "stop_reason": "time_limit", "args": {"target_tracker": "sam21"},
    "perception": {"frames": n, "elapsed_s": 0.1, "actual_hz": 20.0},
  }))
  return session


def test_catalog_and_manifest_keep_every_recorded_frame(tmp_path):
  _recording(tmp_path)
  catalog = logview.list_sessions(tmp_path)
  assert [x["name"] for x in catalog] == ["session"]
  manifest = logview.load_session(tmp_path, "session")
  assert len(manifest["frames"]) == 3
  assert manifest["duration_s"] == pytest.approx(0.1)
  assert manifest["fps"] == pytest.approx(20.0)
  assert [f["frame_index"] for f in manifest["frames"]] == [10, 12, 14]


def test_render_frame_is_an_eight_pane_jpeg(tmp_path):
  _recording(tmp_path)
  manifest = logview.load_session(tmp_path, "session")
  encoded = logview.render_frame(tmp_path, manifest, 1)
  image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
  assert image is not None
  assert image.shape[1] == 64
  assert image.shape[0] == 2 * 12 + 46


def test_compact_raw_and_detection_schema_renders_without_derived_arrays(
    tmp_path):
  from hardware.deploy import run

  session = tmp_path / "compact"
  session.mkdir()
  labels = np.zeros((12, 16), np.uint8)
  labels[3:8, 5:11] = 2
  arrays = {}
  run._encode_sparse_labels(arrays, "detection_labels", labels)
  run._encode_binary_mask(arrays, "sam_rgb_mask", labels > 0)
  run._encode_binary_mask(arrays, "source_mask", labels > 0)
  np.savez(session / "000000.npz", schema_version=np.uint8(3),
           depth=np.full((12, 16), 6500, np.uint16),
           rgb=np.full((12, 16, 3), 80, np.uint8),
           ir_left=np.full((12, 16), 90, np.uint8),
           ir_right=np.full((12, 16), 100, np.uint8), **arrays)
  (session / "meta.json").write_text(json.dumps([{
    "i": 0, "frame_file": "000000.npz", "frame_stamp": 1.0,
    "event": "command", "label": 2, "mask_state": "tracking",
  }]))
  (session / "run.json").write_text(json.dumps({
    "args": {"depth_source": "stereo", "target_tracker": "sam21"},
  }))

  manifest = logview.load_session(tmp_path, "compact")
  assert manifest["visuals"]["exact"]
  encoded = logview.render_frame(tmp_path, manifest, 0)
  image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
  assert image.shape == (2 * 12 + 46, 4 * 16, 3)


def test_session_name_cannot_escape_recording_root(tmp_path):
  _recording(tmp_path)
  with pytest.raises(ValueError, match="invalid recording name"):
    logview.load_session(tmp_path, "../session")


def test_sessions_without_frames_are_not_offered(tmp_path):
  empty = tmp_path / "empty"
  empty.mkdir()
  (empty / "meta.json").write_text("[]")
  assert logview.list_sessions(tmp_path) == []
