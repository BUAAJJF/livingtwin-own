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

import copy

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
    return _TorchSpatialSoftmaxRecurrentModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxSpatialSoftmaxRecurrentModel(self, verbose)


def _parts(model: SpatialSoftmaxRecurrentModel) -> tuple:
  """The pieces both exporters need, detached from the training wrappers.

  ``model.rnn`` is rsl_rl's wrapper, which owns a hidden state and masks it on
  done; an exported policy has one environment and resets when it is told to,
  so the exporters take the ``nn.GRU`` underneath instead.
  """
  if not isinstance(model.rnn.rnn, nn.GRU):
    raise NotImplementedError(
      f"Export is written for GRU, not {type(model.rnn.rnn).__name__}. An "
      "LSTM needs the cell state carried alongside the hidden state, through "
      "the signature, the dummy inputs and the names."
    )
  distribution = (
    model.distribution.as_deterministic_output_module()
    if model.distribution is not None
    else nn.Identity()
  )
  return (
    copy.deepcopy(model.obs_normalizer),
    nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d]),
    copy.deepcopy(model.rnn.rnn),
    copy.deepcopy(model.mlp),
    distribution,
  )


class _TorchSpatialSoftmaxRecurrentModel(nn.Module):
  """The exported policy for TorchScript: one environment, its own memory."""

  def __init__(self, model: SpatialSoftmaxRecurrentModel) -> None:
    super().__init__()
    (
      self.obs_normalizer,
      self.cnns,
      self.rnn,
      self.mlp,
      self.deterministic_output,
    ) = _parts(model)
    self.rnn.cpu()
    self.register_buffer(
      "hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
    )

  def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
    latent = self.obs_normalizer(obs_1d)
    for i, cnn in enumerate(self.cnns):
      latent = torch.cat([latent, cnn(obs_2d[i])], dim=-1)
    x, h = self.rnn(latent.unsqueeze(0), self.hidden_state)
    self.hidden_state[:] = h  # type: ignore[index]
    return self.deterministic_output(self.mlp(x.squeeze(0)))

  @torch.jit.export
  def reset(self) -> None:
    self.hidden_state[:] = 0.0  # type: ignore[index]


class _OnnxSpatialSoftmaxRecurrentModel(nn.Module):
  """The exported policy for ONNX: the memory is an input and an output.

  ONNX graphs are pure, so the hidden state cannot live in a buffer the way the
  TorchScript export keeps it.  Whatever runs this on the robot owns the state:
  feed zeros on the first control step and each step's ``h_out`` back in as the
  next step's ``h_in``, and zero it again whenever the task restarts.
  """

  is_recurrent: bool = True

  def __init__(self, model: SpatialSoftmaxRecurrentModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    (
      self.obs_normalizer,
      self.cnns,
      self.rnn,
      self.mlp,
      self.deterministic_output,
    ) = _parts(model)
    self.obs_groups_2d = model.obs_groups_2d
    self.obs_dims_2d = model.obs_dims_2d
    self.obs_channels_2d = model.obs_channels_2d
    self.obs_dim_1d = model.obs_dim
    self.hidden_size = self.rnn.hidden_size
    self.num_layers = self.rnn.num_layers

  def forward(self, obs_1d: torch.Tensor, *args: torch.Tensor):
    obs_2d, h_in = args[:-1], args[-1]
    latent = self.obs_normalizer(obs_1d)
    for i, cnn in enumerate(self.cnns):
      latent = torch.cat([latent, cnn(obs_2d[i])], dim=-1)
    x, h = self.rnn(latent.unsqueeze(0), h_in)
    return self.deterministic_output(self.mlp(x.squeeze(0))), h

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    images = tuple(
      torch.zeros(1, self.obs_channels_2d[i], *self.obs_dims_2d[i])
      for i in range(len(self.obs_groups_2d))
    )
    return (
      torch.zeros(1, self.obs_dim_1d),
      *images,
      torch.zeros(self.num_layers, 1, self.hidden_size),
    )

  @property
  def input_names(self) -> list[str]:
    return ["obs", *self.obs_groups_2d, "h_in"]

  @property
  def output_names(self) -> list[str]:
    return ["actions", "h_out"]
