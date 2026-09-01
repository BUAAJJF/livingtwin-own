"""The pyAgxArm transport: MIT impedance that this arm actually executes.

``robot.PiperArm`` speaks ``piper_sdk``, whose ``JointMitCtrl`` this arm accepts
and ignores -- 2569 frames at a joint commanded +-3 degrees moved it 0.0000
degrees (2026-08-31).  ``pyAgxArm`` drives the same joints through a
firmware-tiered driver and does move them.  Two differences that matter and are
invisible from the outside:

* **the per-joint torque scale.**  ``t_ff`` is divided by ``joint_torque_b =
  (4.0, 2.5, 4.0, 1.2, 0.8, 1.0)`` before an 8-bit +-8 encode, so the real limit
  is +-(32, 20, 32, 9.6, 6.4, 8) N.m.  ``piper_sdk`` has no such concept and
  would send 4x the requested torque on J1.
* **the mode message.**  ``set_motion_mode('mit')`` is a V188/V189 encoding, not
  ``MotionCtrl_2(..., 0xAD)``.

Feedback is 200 Hz (measured: median interval 5.00 ms, p95 5.03), and that -- not
the CAN bus and not Python -- is what sets the useful control rate.  The drives
do not export velocity, so it is differenced here, the way the deployment must
also do it.
"""

from __future__ import annotations

import dataclasses
import sys
import time

import numpy as np

SITE_PACKAGES = "/home/yunfan/miniconda3/envs/piper_mjlab/lib/python3.11/site-packages"

# Per-joint feedforward torque limits, N.m, from pyAgxArm's joint_torque_b for
# piper_x.  Written here so a caller can size a command without importing the
# SDK, and cross-checked against it at connect time.
TORQUE_LIMIT_NM = np.array([32.0, 20.0, 32.0, 9.6, 6.4, 8.0])
KP_RANGE = (0.0, 500.0)
KD_RANGE = (-5.0, 5.0)
FEEDBACK_HZ = 200.0

# Yoyo's identified gains, used as the starting impedance.  They are what this
# identification is here to confirm or replace, not an answer.
START_KP = np.array([25.0, 25.0, 25.0, 12.0, 12.0, 8.0])
START_KD = np.array([1.3, 1.3, 1.3, 0.8, 0.6, 0.4])


def _import_sdk():
  if SITE_PACKAGES not in sys.path:
    sys.path.append(SITE_PACKAGES)
  from pyAgxArm import (AgxArmFactory, ArmModel, PiperFW,  # noqa: F401
                        create_agx_arm_config)
  return AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config


@dataclasses.dataclass
class Sample:
  t: float
  q: np.ndarray
  """(6,) rad, as the drives report it."""
  dq: np.ndarray
  """(6,) rad/s, differenced -- the drives do not export velocity."""
  torque: np.ndarray
  """(6,) N.m from the motor state registers."""
  stamp: float
  """The SDK's own timestamp for the joint message, which advances at 200 Hz
  whether or not the host read it.  Use this, not ``t``, to tell a fresh
  sample from a re-read of the same one."""


class AgxArm:
  """Connect, enable, stream MIT, and always hand the arm back."""

  def __init__(self, can: str = "can0", tier: str = "V189") -> None:
    factory, model, fw, mkcfg = _import_sdk()
    self._cfg = mkcfg(robot=model.PIPER_X, firmeware_version=getattr(fw, tier),
                      interface="socketcan", channel=can)
    self._robot = factory.create_arm(self._cfg)
    self.connected = False
    self._prev: tuple[float, np.ndarray] | None = None
    self._in_mit = False

  # -- lifecycle -----------------------------------------------------------

  def connect(self, timeout_s: float = 5.0) -> None:
    self._robot.connect()
    time.sleep(0.4)
    b = self._cfg.get("joint_torque_b")
    limit = np.asarray(b, dtype=float) * 8.0
    if not np.allclose(limit, TORQUE_LIMIT_NM):
      raise RuntimeError(
        f"this arm's torque limits are {limit.tolist()}, not the "
        f"{TORQUE_LIMIT_NM.tolist()} this module was written against")
    self.connected = True

  def enable(self, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not self._robot.enable():
      if time.monotonic() > deadline:
        raise RuntimeError("arm did not enable")
      time.sleep(0.01)
    if not all(self._robot.get_joints_enable_status_list()):
      raise RuntimeError("a joint drive is still disabled")

  def release(self) -> None:
    """Position mode, holding where the arm is.  Every exit path calls this."""
    try:
      q = self.read().q
      self._robot.set_motion_mode(self._robot.OPTIONS.MOTION_MODE.J)
      self._robot.move_js(q.tolist())
    finally:
      self._in_mit = False

  # -- feedback ------------------------------------------------------------

  def read(self) -> Sample:
    fb = self._robot.get_joint_angles()
    q = np.asarray(fb.msg, dtype=np.float64)
    torque = np.array([float(self._robot.get_motor_states(i).msg.torque)
                       for i in range(1, 7)])
    now = time.monotonic()
    dq = np.zeros(6)
    if self._prev is not None:
      t_prev, q_prev = self._prev
      dt = now - t_prev
      if dt > 1e-4:
        dq = (q - q_prev) / dt
    self._prev = (now, q.copy())
    return Sample(t=now, q=q, dq=dq, torque=torque, stamp=float(fb.timestamp))

  def feedback_hz(self) -> float:
    return float(self._robot.get_joint_angles().hz)

  # -- command -------------------------------------------------------------

  def move_mit(self, q_ref, dq_ref, kp, kd, tau_ff) -> dict:
    """One impedance frame per joint, every field clamped before it is sent."""
    q = np.asarray(q_ref, dtype=np.float64).reshape(6)
    dq = np.clip(np.asarray(dq_ref, dtype=np.float64).reshape(6), -45.0, 45.0)
    kp = np.clip(np.asarray(kp, dtype=np.float64).reshape(6), *KP_RANGE)
    kd = np.clip(np.asarray(kd, dtype=np.float64).reshape(6), *KD_RANGE)
    t = np.clip(np.asarray(tau_ff, dtype=np.float64).reshape(6),
                -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)
    if not (np.isfinite(q).all() and np.isfinite(t).all()):
      raise ValueError("non-finite MIT command")
    for j in range(6):
      self._robot.move_mit(joint_index=j + 1, p_des=float(q[j]),
                           v_des=float(dq[j]), kp=float(kp[j]),
                           kd=float(kd[j]), t_ff=float(t[j]))
    self._in_mit = True
    return {"q_ref": q, "dq_ref": dq, "kp": kp, "kd": kd, "tau_ff": t}

  def move_js(self, q) -> None:
    self._robot.set_motion_mode(self._robot.OPTIONS.MOTION_MODE.J)
    self._robot.move_js(np.asarray(q, dtype=np.float64).reshape(6).tolist())
