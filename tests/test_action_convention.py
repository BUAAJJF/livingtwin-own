"""The bounded action convention: a = +-1 is the safe clip, and an unbounded policy is refused."""

from __future__ import annotations

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import numpy as np
import pytest
import torch

from piper_push import robot as piper


def test_bounded_scale_and_offset_span_exactly_the_safe_clip():
  for j, (lo, hi) in piper.SAFE_TARGET_CLIP.items():
    s, o = piper.BOUNDED_ARM_SCALE[j], piper.BOUNDED_ARM_OFFSET[j]
    assert o - s == pytest.approx(lo) and o + s == pytest.approx(hi)


def test_action_spec_records_both_conventions():
  b = piper.action_spec("bounded")
  assert b["squashed"] and len(b["scale"]) == 7 and b["joints"][-1] == "gripper_joint1"
  assert np.allclose(np.array(b["offset"][:6]) - np.array(b["scale"][:6]), b["clip_lo"][:6])
  home = {j: piper.PICK_HOME_KEYFRAME.joint_pos.get(j, 0.0) for j in piper.ARM_JOINT_ORDER}
  v1 = piper.action_spec("v1", home)
  assert not v1["squashed"]
  assert v1["scale"][:6] == [piper.PICK_ARM_SCALE[j] for j in piper.ARM_JOINT_ORDER]
  assert v1["offset"][3] == pytest.approx(home["joint4"])
  with pytest.raises(ValueError):
    piper.action_spec("v1")
  with pytest.raises(ValueError):
    piper.action_spec("v3")


def test_default_ids_are_bounded_and_v1_ids_are_not():
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  for tid, bounded in (("Mjlab-Pick-Place-PiperX", True), ("Mjlab-Pick-Place-PiperX-V1", False),
                       ("Mjlab-Pick-Place-PiperX-Vision-Robust", True),
                       ("Mjlab-Pick-Place-PiperX-Vision-Robust-V1", False),
                       ("Mjlab-Pick-Place-PiperX-Distill-Robust", True),
                       ("Mjlab-Pick-Place-PiperX-Distill-Robust-V1", False)):
    cfg = load_env_cfg(tid)
    arm, grip = cfg.actions["arm"], cfg.actions["gripper"]
    assert arm.bounded is bounded and grip.bounded is bounded
    assert arm.use_default_offset is (not bounded)
    rl = load_rl_cfg(tid)
    head = getattr(rl, "actor", None) or rl.student
    squashed = "Squashed" in head.distribution_cfg["class_name"]
    assert squashed is bounded, tid
    if bounded:
      assert arm.scale == piper.BOUNDED_ARM_SCALE and arm.offset == piper.BOUNDED_ARM_OFFSET
    else:
      assert arm.scale == piper.PICK_ARM_SCALE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="builds a MuJoCo-Warp environment")
def test_bounded_term_refuses_an_unbounded_action_and_maps_unit_to_the_clip():
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  cfg = load_env_cfg("Mjlab-Pick-Place-PiperX", play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  try:
    env.reset()
    a = torch.zeros(2, 7, device="cuda:0")
    a[:, 6] = 1.0
    env.step(a)  # in range: fine
    term = env.action_manager.get_term("arm")
    # +-1 on the arm is the safe clip, before the slew limiter has its say.
    hi = torch.ones(2, 6, device="cuda:0")
    target = hi * term.scale + term.offset
    assert torch.allclose(target[0], torch.tensor([b for _, b in piper.SAFE_TARGET_CLIP.values()],
                                                  device="cuda:0"), atol=1e-6)
    a[:, 0] = 3.0
    with pytest.raises(ValueError, match="-V1"):
      env.step(a)
  finally:
    env.close()


def test_deploy_mapper_reads_the_convention_from_the_spec():
  """A spec without action_spec is a pre-2026-09-05 export and gets v1; one
  with the bounded block gets a = +-1 at the safe clip.  The v4 policy on the
  rig is the former and must keep driving the same robot."""
  import json
  import pathlib

  from hardware.deploy import proprio, robot

  legacy = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())
  legacy.pop("action_spec", None)
  m1 = robot.ActionMapper(legacy)
  assert m1.convention == "v1"
  names = legacy["joint_names"]
  j4 = legacy["default_joint_pos"][names.index("joint4")]
  assert m1.offset[3] == pytest.approx(j4) and m1.scale[3] == pytest.approx(piper.PICK_ARM_SCALE["joint4"])

  bounded = dict(legacy, action_spec=piper.action_spec("bounded"))
  m2 = robot.ActionMapper(bounded)
  assert m2.convention == "bounded"
  m2.reset(np.zeros(7))
  m2.max_step = np.full(7, 1e9)  # take the slew limiter out of the picture
  hi = m2(np.ones(7))
  assert hi[:6] == pytest.approx([b for _, b in piper.SAFE_TARGET_CLIP.values()])
  assert hi[6] == pytest.approx(piper.GRIPPER_OPEN_M)
  lo = m2(-np.ones(7))
  assert lo[:6] == pytest.approx([a for a, _ in piper.SAFE_TARGET_CLIP.values()])

  bad = dict(legacy, action_spec=dict(piper.action_spec("bounded"), joints=["x"] * 7))
  with pytest.raises(ValueError, match="action_spec joints"):
    robot.ActionMapper(bad)
