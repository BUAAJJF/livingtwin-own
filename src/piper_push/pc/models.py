"""A recurrent policy over 1D observations, point sets and images.

``SetRecurrentModel`` is to ``piper_push.models.SpatialSoftmaxRecurrentModel``
what a point set is to an image: 1D groups go through rsl_rl's normaliser as in
``MLPModel``; every other group is handed to the encoder named for it in
``cnn_cfg`` (the field mjlab's model config already carries, reused here as
the encoder spec so that no runner code has to change); the concatenation
feeds a GRU, whose state is the policy's memory; the MLP head emits the
pre-squash ``u``.  ``cnn_cfg`` is ``{group: {"type": ..., ...}}`` --
see ``piper_push.pc.encoders.build_encoder``.

Groups are told apart by rank, as rsl_rl does: (B, C) is 1D, (B, N, C) a point
set, (B, C, H, W) an image.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import RNN, HiddenState
from tensordict import TensorDict

from piper_push.pc.encoders import build_encoder


class SetRecurrentModel(MLPModel):
  is_recurrent: bool = True

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    rnn_type: str = "gru",
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
  ) -> None:
    self.latent_dim = int(rnn_hidden_dim)
    self.obs_groups_nd: list[str] = []
    self.obs_shapes_nd: list[tuple[int, ...]] = []
    super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                     obs_normalization, distribution_cfg)
    if not self.obs_groups_nd:
      raise ValueError("SetRecurrentModel needs at least one point-set or image group")
    self.encoder_spec = {g: dict(cnn_cfg[g]) for g in self.obs_groups_nd}
    self.encoders = nn.ModuleDict({
      g: build_encoder(self.encoder_spec[g], shape)
      for g, shape in zip(self.obs_groups_nd, self.obs_shapes_nd)})
    with torch.no_grad():
      device = next(self.mlp.parameters()).device
      self.encoders.to(device)
      rnn_input_dim = int(self._encode(obs.to(device)).shape[-1])
    self.rnn = RNN(rnn_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

  def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str):
    groups_1d, dim_1d = [], 0
    for g in obs_groups[obs_set]:
      shape = tuple(obs[g].shape)
      if len(shape) == 2:
        groups_1d.append(g)
        dim_1d += shape[-1]
      elif len(shape) in (3, 4):
        self.obs_groups_nd.append(g)
        self.obs_shapes_nd.append(shape[1:])
      else:
        raise ValueError(f"unsupported observation shape for {g}: {shape}")
    return groups_1d, dim_1d

  def _get_latent_dim(self) -> int:
    return self.latent_dim

  @property
  def obs_groups_2d(self) -> list[str]:
    """The encoded groups under the name rsl_rl's CNN model uses, so that
    scripts/check_export.py and the deployment feed the export by the same
    list whether a group is an image or a point set."""
    return list(self.obs_groups_nd)

  def _encode(self, obs: TensorDict) -> torch.Tensor:
    parts = []
    lead = None
    for g, shape in zip(self.obs_groups_nd, self.obs_shapes_nd):
      x = obs[g]
      lead = x.shape[:-len(shape)]
      parts.append(self.encoders[g](x.reshape(-1, *shape)).reshape(*lead, -1))
    if self.obs_groups:
      parts.insert(0, MLPModel.get_latent(self, obs))
    return torch.cat(parts, dim=-1)

  def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None,
                 hidden_state: HiddenState = None) -> torch.Tensor:
    return self.rnn(self._encode(obs), masks, hidden_state).squeeze(0)

  def recon_loss(self, obs: TensorDict) -> torch.Tensor:
    """Sum of the encoders' masked-reconstruction losses on this batch (0 when none has one)."""
    total = None
    for g, shape in zip(self.obs_groups_nd, self.obs_shapes_nd):
      enc = self.encoders[g]
      if hasattr(enc, "recon_loss") and getattr(enc, "recon", False):
        x = obs[g].reshape(-1, *shape)
        l = enc.recon_loss(x)
        total = l if total is None else total + l
    if total is None:
      first = obs[self.obs_groups_nd[0]]
      return first.new_zeros(())
    return total

  def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
    self.rnn.reset(dones, hidden_state)

  def get_hidden_state(self) -> HiddenState:
    return self.rnn.hidden_state  # type: ignore[return-value]

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    self.rnn.detach_hidden_state(dones)

  def as_jit(self) -> nn.Module:
    return _TorchSetRecurrentModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxSetRecurrentModel(self, verbose)


def _parts(model: SetRecurrentModel) -> tuple:
  if not isinstance(model.rnn.rnn, nn.GRU):
    raise NotImplementedError("export is written for GRU")
  distribution = (model.distribution.as_deterministic_output_module()
                  if model.distribution is not None else nn.Identity())
  return (copy.deepcopy(model.obs_normalizer),
          nn.ModuleList([copy.deepcopy(model.encoders[g]) for g in model.obs_groups_nd]),
          copy.deepcopy(model.rnn.rnn), copy.deepcopy(model.mlp), distribution)


class _TorchSetRecurrentModel(nn.Module):
  """One environment, its own memory (TorchScript)."""

  def __init__(self, model: SetRecurrentModel) -> None:
    super().__init__()
    (self.obs_normalizer, self.encoders, self.rnn, self.mlp, self.deterministic_output) = _parts(model)
    self.rnn.cpu()
    self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

  def forward(self, obs_1d: torch.Tensor, obs_nd: list[torch.Tensor]) -> torch.Tensor:
    latent = self.obs_normalizer(obs_1d)
    for i, enc in enumerate(self.encoders):
      latent = torch.cat([latent, enc(obs_nd[i])], dim=-1)
    x, h = self.rnn(latent.unsqueeze(0), self.hidden_state)
    self.hidden_state[:] = h  # type: ignore[index]
    return self.deterministic_output(self.mlp(x.squeeze(0)))

  @torch.jit.export
  def reset(self) -> None:
    self.hidden_state[:] = 0.0  # type: ignore[index]


class _OnnxSetRecurrentModel(nn.Module):
  """The memory is an input and an output (ONNX)."""

  is_recurrent: bool = True

  def __init__(self, model: SetRecurrentModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    (self.obs_normalizer, self.encoders, self.rnn, self.mlp, self.deterministic_output) = _parts(model)
    self.obs_groups_nd = list(model.obs_groups_nd)
    self.obs_shapes_nd = list(model.obs_shapes_nd)
    self.obs_dim_1d = model.obs_dim
    self.hidden_size = self.rnn.hidden_size
    self.num_layers = self.rnn.num_layers

  def forward(self, obs_1d: torch.Tensor, *args: torch.Tensor):
    obs_nd, h_in = args[:-1], args[-1]
    latent = self.obs_normalizer(obs_1d)
    for i, enc in enumerate(self.encoders):
      latent = torch.cat([latent, enc(obs_nd[i])], dim=-1)
    x, h = self.rnn(latent.unsqueeze(0), h_in)
    return self.deterministic_output(self.mlp(x.squeeze(0))), h

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    return (torch.zeros(1, self.obs_dim_1d),
            *[torch.zeros(1, *s) for s in self.obs_shapes_nd],
            torch.zeros(self.num_layers, 1, self.hidden_size))

  @property
  def input_names(self) -> list[str]:
    return ["obs", *self.obs_groups_nd, "h_in"]

  @property
  def output_names(self) -> list[str]:
    return ["actions", "h_out"]
