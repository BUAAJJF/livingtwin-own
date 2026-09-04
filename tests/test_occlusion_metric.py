"""The occlusion metric, which was wrong twice and had no test either time.

Both faults produced a plausible number, which is why they survived: a sphere
cover fat enough to swallow a held object, and an index into
``geom_pos_w`` that used global geom ids where the array is in the robot's own
geom order.  Neither raises, neither looks odd in a JSON blob, and together
they said the carrying phase was 97% blocked where the renderer sees 6%.

So these test the thing that catches that class of fault: that the metric
reads the pixel it claims to read.  A projection sign error, an off-by-one in
the buffer lookup, or a camera pose taken from the wrong place all show up as
"the segmentation said X at the point where I put X and the metric disagreed".
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from piper_push import camera as sim_camera  # noqa: E402
from eval_occlusion import (ARM, FINGERS, GRIPPER_BASE, INSET,  # noqa: E402
                            OTHER, VISIBLE, quat_to_mat, sample_box,
                            sample_status)

W, H = sim_camera.WIDTH, sim_camera.HEIGHT
F = 0.5 * H / np.tan(np.deg2rad(sim_camera.FOVY_DEG) / 2.0)
TARGET_GEOM, FINGER_GEOM, BACKDROP = 7, 11, 3
GEOM_TYPE = 5           # mujoco.mjtObj.mjOBJ_GEOM


def _scene(seg_ids, seg_types=None, obj=(0.0, 0.0, -1.0), half=(0.03, 0.03, 0.03),
           cam=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
  """The smallest stand-ins ``sample_status`` will accept.

  Camera at the origin with an identity quaternion looks down its own -z, so
  an object at (0, 0, -1) is a metre straight ahead and lands dead centre.
  That makes every expected pixel computable by hand, which is the point.
  """
  ids = torch.as_tensor(seg_ids, dtype=torch.long).reshape(1, H, W)
  typ = (torch.full_like(ids, GEOM_TYPE) if seg_types is None
         else torch.as_tensor(seg_types, dtype=torch.long).reshape(1, H, W))
  sensor = types.SimpleNamespace(
    camera_idx=0,
    data=types.SimpleNamespace(segmentation=torch.stack([ids, typ], dim=-1)))
  cmd = types.SimpleNamespace(
    _object_pos_local=lambda: torch.tensor([obj], dtype=torch.float32),
    object_half_size=torch.tensor([half], dtype=torch.float32),
    target_geom_ids=torch.tensor([[TARGET_GEOM]], dtype=torch.long))
  model = types.SimpleNamespace(
    cam_pos=torch.tensor([[cam]], dtype=torch.float32),
    cam_quat=torch.tensor([[quat]], dtype=torch.float32))
  gmap = torch.full((32,), OTHER, dtype=torch.long)
  gmap[FINGER_GEOM] = FINGERS
  gmap[TARGET_GEOM] = OTHER          # only the target-id test may mark it visible
  return sensor, cmd, model, gmap


def _offsets(inset=INSET):
  return torch.tensor(sample_box(inset=inset), dtype=torch.float32)


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------

def test_sample_box_is_fifteen_points_and_the_inset_shrinks_them():
  full = sample_box(inset=1.0)
  assert full.shape == (15, 3)
  assert len({tuple(p) for p in full}) == 15
  assert np.abs(full).max() == pytest.approx(1.0)
  assert np.allclose(sample_box(inset=0.5), full * 0.5)


def test_the_inset_keeps_the_points_inside_the_box():
  """The corners are the reason it exists.

  At inset 1.0 the eight corners sit exactly on the silhouette edge and
  rounding to a pixel throws about half of them off the object, where they
  read as blocked by scenery -- 33 points of apparent visibility on a frame
  with nothing in the way.
  """
  assert np.abs(sample_box()).max() < 1.0


# ---------------------------------------------------------------------------
# does it read the pixel it says it reads
# ---------------------------------------------------------------------------

def test_a_frame_that_is_all_target_is_fully_visible():
  s, c, m, g = _scene(np.full((H, W), TARGET_GEOM))
  assert (sample_status(s, c, m, 0, g, _offsets()) == VISIBLE).all()


def test_a_frame_that_is_all_finger_is_attributed_to_the_fingers():
  s, c, m, g = _scene(np.full((H, W), FINGER_GEOM))
  st = sample_status(s, c, m, 0, g, _offsets())
  assert (st == FINGERS).all()
  assert not (st == VISIBLE).any()


def test_a_non_geom_pixel_is_not_credited_to_any_body():
  """Segmentation carries a type alongside the id and ids repeat across types.

  Reading the id without checking the type would let a site or a light whose
  id happens to equal a finger's be charged to the gripper.
  """
  s, c, m, g = _scene(np.full((H, W), FINGER_GEOM),
                      seg_types=np.full((H, W), GEOM_TYPE + 1))
  assert (sample_status(s, c, m, 0, g, _offsets()) == OTHER).all()


def test_the_lookup_is_per_point_and_not_one_pixel_for_all_of_them():
  """Half the image target, half finger: the split must show up in the result.

  A metric that projected once and reused the answer -- or that transposed
  (u, v) -- passes every uniform-image test above and fails this one.
  """
  ids = np.full((H, W), FINGER_GEOM)
  ids[:, W // 2:] = TARGET_GEOM
  s, c, m, g = _scene(ids)
  st = sample_status(s, c, m, 0, g, _offsets())
  assert (st == VISIBLE).any() and (st == FINGERS).any()


# ---------------------------------------------------------------------------
# the projection's signs, each one independently
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift,axis,expect", [
  ((0.2, 0.0, -1.0), "u", "greater"),    # +x in the world is right in the image
  ((-0.2, 0.0, -1.0), "u", "less"),
  ((0.0, 0.2, -1.0), "v", "less"),       # +y is UP, and up is a smaller row
  ((0.0, -0.2, -1.0), "v", "greater"),
])
def test_the_object_lands_on_the_side_of_the_image_it_should(shift, axis, expect):
  """A sign error here would put the metric's window on the wrong half.

  Checked by making only that half of the image the target: if the projection
  disagrees, every sample reads as something else.
  """
  ids = np.full((H, W), BACKDROP)
  if axis == "u":
    ids[:, W // 2:] = TARGET_GEOM if expect == "greater" else BACKDROP
    ids[:, :W // 2] = TARGET_GEOM if expect == "less" else BACKDROP
  else:
    ids[:H // 2, :] = TARGET_GEOM if expect == "less" else BACKDROP
    ids[H // 2:, :] = TARGET_GEOM if expect == "greater" else BACKDROP
  s, c, m, g = _scene(ids, obj=shift, half=(0.01, 0.01, 0.01))
  assert (sample_status(s, c, m, 0, g, _offsets()) == VISIBLE).all()


def test_the_camera_pose_comes_from_the_model_not_from_the_nominal_constant():
  """Domain randomisation moves the camera; the projection has to follow it.

  Same object, same picture, camera translated sideways.  If the projection
  used ``camera.CAMERA_POS`` the two would agree, which is the bug.
  """
  ids = np.full((H, W), BACKDROP)
  ids[:, W // 2:] = TARGET_GEOM

  def status(cam):
    s, c, m, g = _scene(ids, obj=(0.0, 0.0, -1.0), half=(0.01, 0.01, 0.01),
                        cam=cam)
    return sample_status(s, c, m, 0, g, _offsets())

  # Straight ahead the object sits on the seam; slid left, the whole of it is
  # in the target half.  Only a projection that reads the pose can tell them
  # apart.
  assert not torch.equal(status((0.0, 0.0, 0.0)), status((-0.5, 0.0, 0.0))), \
    "projection ignored the camera pose"


def test_quat_to_mat_matches_a_known_quarter_turn():
  q = torch.tensor([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]],
                   dtype=torch.float32)          # +90 deg about z
  got = quat_to_mat(q)[0].numpy()
  want = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  assert np.allclose(got, want, atol=1e-6)


# ---------------------------------------------------------------------------
# the cover that caused the first fault, pinned so it cannot be reused blind
# ---------------------------------------------------------------------------

def test_the_group_labels_are_distinct():
  """``sample_status`` returns these packed into one tensor; an accidental
  collision would silently merge two bodies' contributions."""
  assert len({VISIBLE, FINGERS, GRIPPER_BASE, ARM, OTHER}) == 5
