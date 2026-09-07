"""The log viewer renders a point-cloud session as the cloud the policy saw, not as mask tiles."""
import json
import pathlib

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _session(tmp_path: pathlib.Path) -> pathlib.Path:
  from hardware.deploy import config
  s = tmp_path / "pc_motion_test"
  s.mkdir()
  rig = config.Rig.load(pathlib.Path(config.RIG_FILE).with_name("rig_d455.json"))
  K = np.asarray(rig.K, dtype=np.float64)
  T = np.asarray(rig.T_base_cam, dtype=np.float64)
  # A flat table on the calibrated plane plus a 40 mm block in the sector: the
  # depth of each pixel is where its ray meets those surfaces, so the rebuilt
  # cloud has to put the block above the cut and the table below it.
  h, w = config.D405_HEIGHT, config.D405_WIDTH
  u, v = np.meshgrid(np.arange(w), np.arange(h))
  rays_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=np.float64)], -1)
  R, t = T[:3, :3], T[:3, 3]
  d_base = rays_cam @ R.T                     # ray directions in the base frame
  n = np.asarray(rig.table_normal_base if rig.table_normal_base is not None else (0, 0, 1.0), dtype=np.float64)
  n = n / np.linalg.norm(n)
  p0 = np.array([0.0, 0.0, rig.table_z])
  def hit(height):
    denom = d_base @ n
    lam = ((p0 + n * height - t) @ n) / np.where(np.abs(denom) < 1e-6, np.nan, denom)
    return lam
  lam_table = hit(0.0)
  lam_block = hit(0.040)
  xy_block = t[None, None, :2] + lam_block[..., None] * d_base[..., :2]
  centre = np.array([0.0, 0.35])           # inside the +90 degree sector, away from the bin
  on_block = (np.abs(xy_block - centre) < 0.02).all(-1)
  lam = np.where(on_block, lam_block, lam_table)
  depth = np.where(np.isfinite(lam) & (lam > 0), lam * rays_cam[..., 2], 0.0).astype(np.float32)
  rgb = np.zeros((h, w, 3), np.uint8)
  frames = [("000000.npz", depth), ("000001.npz", np.zeros_like(depth))]
  meta = []
  for i, (name, d) in enumerate(frames):
    np.savez(s / name, schema_version=np.asarray(3, dtype=np.uint8), depth=(d * 10000).astype(np.uint16), rgb=rgb)
    meta.append({"i": i, "frame_file": name, "frame_stamp": 100.0 + i / 30, "event": "command" if i == 0 else "hold_no_target",
                 "label": 0, "joint_pos": [1.6, 1.7, -1.2, 1.1, 0.0, 0.0, 0.045, -0.045],
                 "joint_vel": [0.0] * 8, "gripper_effort": 0.1 * i, "mask_state": "pc:P1BZ6",
                 "detections": [{"workspace_points": 100, "object_points": 50, "table_empty": False}]})
  (s / "meta.json").write_text(json.dumps(meta))
  (s / "run.json").write_text(json.dumps({"args": {"obs": "pc", "camera": "d455", "cloud_height_min": 0.004,
                                                    "policy": "/nonexistent/bundle", "depth_source": "sensor"}}))
  return s


def test_pc_session_renders_the_rebuilt_cloud(tmp_path):
  from hardware.deploy import logview

  s = _session(tmp_path)
  store = logview.Store(tmp_path, cache_frames=4)
  manifest = store.session(s.name)
  assert logview._is_pc_session(manifest)
  replay = logview.CloudReplay(s, manifest)
  assert replay.route == "P1BZ6" and replay.cut == pytest.approx(0.004)
  with np.load(s / "000000.npz") as z:
    an = replay.analyse(np.asarray(z["depth"], np.float32) / 10000.0, manifest["frames"][0]["joint_pos"])
  # The block survives the cut; the table does not.  Its points sit ~40 mm up.
  assert an["count"] > 200, an["count"]
  loose = an["h"][~an["arm"] & ~an["bin"]]
  assert loose.size and np.median(loose) == pytest.approx(0.040, abs=0.004)
  assert an["object"] > 0

  jpeg = store.frame(s.name, 0)
  img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
  assert img is not None and img.shape[1] == 4 * 848 and img.shape[0] > 480 + 300
  # An empty depth (a frame with no points at all) still renders.
  assert len(store.frame(s.name, 1)) > 1000
