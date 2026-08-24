"""Run the exported policy, carrying its hidden state by hand.

The actor is recurrent, and the export deliberately does not hide that: the
graph takes the hidden state in and hands it back out, with no stateful
operator anywhere in it.  ``scripts/check_export.py`` verifies that stepping it
this way tracks the training-time module to 3e-6 over eight steps, which is the
only reason it can be trusted here.

Carrying the state by hand is also what makes a reset meaningful.  When the
task restarts -- the object is placed, the operator intervenes, the arm is
re-homed -- the policy has to forget, and with the state inside the graph the
only way to do that would be to reload it.
"""

from __future__ import annotations

import pathlib

import numpy as np


class Policy:
  """ONNX Runtime, CPU or CUDA, with the recurrent state as an explicit tensor.

  Input names are read from the graph rather than assumed.  The export writes
  them as ``(flat observations, one tensor per image group..., hidden)`` and the
  order of the image groups follows the actor's own ``obs_groups_2d`` -- there
  is one here, but reading the names keeps this file correct if that changes.
  """

  def __init__(self, path: str | pathlib.Path, providers: list[str] | None = None,
               threads: int = 2):
    import onnxruntime as ort

    path = pathlib.Path(path)
    if path.is_dir():
      path = path / "policy.onnx"

    # Two threads, and no spinning between calls.  This network is small and
    # runs 50 times a second; left to itself ONNX Runtime opens one intra-op
    # thread per core and busy-waits between inferences, and on this machine
    # that starved the vision thread from 22 ms a frame to 878 -- a 40x
    # slowdown in the stage that actually costs something, caused by the stage
    # that does not.  The control loop's own median barely moved, so nothing in
    # its timing would have shown it.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, int(threads))
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    self.sess = ort.InferenceSession(
      str(path), opts,
      providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    self.path = str(path)
    inputs = self.sess.get_inputs()
    self.input_names = [i.name for i in inputs]
    if len(self.input_names) < 3:
      raise RuntimeError(
        f"expected (obs, image..., hidden), got {self.input_names}. "
        "This looks like a non-recurrent export; the deployed policy is an "
        "RNN and stepping it without its state gives a different policy."
      )
    def _fixed(shape):
      return tuple(1 if not isinstance(d, int) else d for d in shape)

    # The graph's own declared widths, checked against every vector fed to it.
    # A proprioception vector one element short does not raise anywhere -- ONNX
    # Runtime would reject it, but only if the rank were wrong -- and the
    # failure it produces is a policy that acts confidently on garbage.
    self.flat_width = int(_fixed(inputs[0].shape)[-1])
    self.image_shapes = [_fixed(i.shape)[1:] for i in inputs[1:-1]]

    shape = inputs[-1].shape
    self.hidden_shape = tuple(
      1 if not isinstance(d, int) else d for d in shape
    )
    self.hidden = np.zeros(self.hidden_shape, dtype=np.float32)

  def reset(self) -> None:
    self.hidden = np.zeros(self.hidden_shape, dtype=np.float32)

  def __call__(self, proprio: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """Args:
      proprio: ``(36,)``.
      camera: ``(3, H, W)``.
    Returns the action, ``(7,)``.
    """
    if proprio.size != self.flat_width:
      raise ValueError(
        f"the policy expects {self.flat_width} proprioception values and got "
        f"{proprio.size}.  Re-run scripts/export_obs_spec.py against the task "
        "this checkpoint was trained on."
      )
    if tuple(camera.shape) != tuple(self.image_shapes[0]):
      raise ValueError(
        f"the policy expects a {self.image_shapes[0]} image and got "
        f"{tuple(camera.shape)}"
      )
    feed = {
      self.input_names[0]: proprio.reshape(1, -1).astype(np.float32),
      self.input_names[1]: camera[None].astype(np.float32),
      self.input_names[-1]: self.hidden,
    }
    action, self.hidden = self.sess.run(None, feed)
    return np.asarray(action, dtype=np.float64).reshape(-1)
