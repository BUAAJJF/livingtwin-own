"""A policy that both sees and remembers.

rsl_rl ships a CNN model and an RNN model and no way to have both: ``CNNModel``
encodes 2D observation groups and hands the result to an MLP, ``RNNModel`` runs
1D observations through a recurrent layer and hands *that* to an MLP, and each
overrides the same two hooks to do it.  The vision stage needs both, because
taking the grasp flag away is what makes the task partially observed and the
camera is what replaced the object's state.

Both parents override ``get_latent`` and ``_get_latent_dim`` and nothing else,
so composing them is a matter of running the recurrent layer over what the CNN
produced rather than reimplementing either.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import RNN, HiddenState
from tensordict import TensorDict


class SpatialSoftmaxRecurrentModel(SpatialSoftmaxCNNModel):
  """Spatial-softmax convolutional encoder followed by a GRU or LSTM."""

  is_recurrent: bool = True

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict,
    cnns: nn.ModuleDict | None = None,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    rnn_type: str = "gru",
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
  ) -> None:
    # The parent asks for the latent width while it is still constructing, so
    # this has to be set before the parent runs, exactly as RNNModel does.
    self.latent_dim = rnn_hidden_dim
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      cnn_cfg,
      cnns,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
    )
    # Measure the encoder's output rather than trusting the declared width.
    # ``cnn_latent_dim`` is the sum of what each encoder reports, and the
    # spatial-softmax encoder reports 64 while producing 128, so building the
    # recurrent layer from the declaration fails on the first forward pass with
    # a shape error a hundred frames into training.
    with torch.no_grad():
      # The model is still on the CPU here; the observations it was handed are
      # already on the GPU.  Probe on whichever device the weights are on.
      device = next(self.cnns.parameters()).device
      rnn_input_dim = int(self._encode(obs.to(device)).shape[-1])
    # The recurrent layer sees the proprioception and the image encoding
    # together, not the image alone: "am I holding it" is a fact about both.
    self.rnn = RNN(rnn_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

  def _encode(self, obs: TensorDict) -> torch.Tensor:
    """Proprioception and the encoded image, before the recurrent layer.

    Recurrent training replays whole sequences, so the images arrive as
    (T, B, C, H, W) and a convolution only takes four dimensions.  The leading
    axes are folded away for the encoder and put back afterwards, because the
    RNN needs the time axis it was given.
    """
    lead = obs[self.obs_groups_2d[0]].shape[:-3]
    parts = []
    for group in self.obs_groups_2d:
      x = obs[group]
      parts.append(self.cnns[group](x.reshape(-1, *x.shape[-3:])).reshape(*lead, -1))
    latent = torch.cat(parts, dim=-1)
    if self.obs_groups:
      # MLPModel's, explicitly.  ``super(SpatialSoftmaxCNNModel, self)`` reads
      # like "the non-CNN part" and resolves to CNNModel in this MRO, which
      # feeds the raw five-dimensional observation straight back into the
      # convolutions.
      latent = torch.cat([MLPModel.get_latent(self, obs), latent], dim=-1)
    return latent

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    return self.rnn(self._encode(obs), masks, hidden_state).squeeze(0)

  def reset(
    self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None
  ) -> None:
    self.rnn.reset(dones, hidden_state)

  def get_hidden_state(self) -> HiddenState:
    return self.rnn.hidden_state  # type: ignore[return-value]

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    self.rnn.detach_hidden_state(dones)

  def _get_latent_dim(self) -> int:
    return self.latent_dim

  def as_jit(self) -> nn.Module:
    raise NotImplementedError(
      "Export for the recurrent vision policy is not written yet; it needs the "
      "hidden state threaded through, which the CNN and RNN exporters each do "
      "on their own terms."
    )

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    raise NotImplementedError(
      "Export for the recurrent vision policy is not written yet; it needs the "
      "hidden state threaded through, which the CNN and RNN exporters each do "
      "on their own terms."
    )
