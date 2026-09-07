"""Encoders for point sets and depth images.  What each one actually is, stated once.

``PointNetEncoder``      shared per-point MLP 4-64-128-256, LayerNorm, masked max-pool,
                         then a linear to ``out_dim``.  The P1a encoder.

``PointPatchEncoder``    farthest-point centres, k-nearest patches, a per-patch
                         mini-PointNet embedding plus a centre embedding, a
                         2-layer transformer over the patch tokens, mean+max
                         pooling.  This is the encoder of PointPatchRL
                         (Gyenes et al., 2024) WITHOUT its masked-reconstruction
                         auxiliary objective: the timebox did not cover the
                         official implementation, so P1b is "point-patch
                         transformer, no reconstruction loss", not PointPatchRL.
                         Columns beyond the fourth (x, y, z, valid) are per-point
                         features and enter the patch mini-PointNet beside the
                         local coordinates; with four columns the module is
                         parameter-for-parameter the first-generation one.

``DepthResNetLite``      a BasicBlock ResNet-18 layout (2-2-2-2 blocks, widths
                         32-64-128-256, GroupNorm, 2-channel input), random
                         initialisation, global average pool.  P0 asked for
                         DeFM (Depth Foundation Model, leggedrobotics/defm)
                         features in front of a ResNet-18-class head; DeFM could
                         not be obtained on this machine (the external checkout
                         is refused by the environment policy), so P0 runs this
                         encoder and is reported as "ResNet18-lite from scratch",
                         never as DeFM.

``SetMLPEncoder``        the top-K grasp candidates (P2): per-candidate MLP and
                         masked max-pool, the same shape as PointNet on a
                         19-dimensional point.

GroupNorm and LayerNorm rather than BatchNorm everywhere: the same module runs
on 512-environment rollouts and on one environment at deployment, and a
batch statistic would make those two different networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims: tuple[int, ...]) -> nn.Sequential:
  layers: list[nn.Module] = []
  for i in range(len(dims) - 1):
    layers += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]), nn.ELU()]
  return nn.Sequential(*layers)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Max over the point axis of ``x`` (B, N, C) where ``mask`` (B, N) is set; 0 if none.

  ``-1e30`` stands in for -inf, written as a literal: TorchScript has neither
  ``torch.finfo`` nor module-level constants.
  """
  y = torch.where(mask.unsqueeze(-1), x, torch.full_like(x, -1.0e30)).amax(dim=1)
  any_ = mask.any(dim=1, keepdim=True)
  return torch.where(any_, y, torch.zeros_like(y))


class PointNetEncoder(nn.Module):
  def __init__(self, in_dim: int = 4, dims: tuple[int, ...] = (64, 128, 256), out_dim: int = 256) -> None:
    super().__init__()
    self.in_dim = in_dim
    self.net = _mlp((in_dim, *dims))
    self.head = nn.Linear(dims[-1], out_dim)
    self.output_dim = out_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    mask = x[..., 3] > 0.5 if x.shape[-1] >= 4 else torch.ones_like(x[..., 0], dtype=torch.bool)
    return self.head(masked_max(self.net(x), mask))


class SetMLPEncoder(PointNetEncoder):
  """Same pooling, a different notion of ``valid``: the last feature is the flag."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    mask = x[..., -1] > 0.5
    return self.head(masked_max(self.net(x), mask))


def farthest_points(xyz: torch.Tensor, valid: torch.Tensor, n_groups: int) -> torch.Tensor:
  """Indices (B, G) of farthest-point-sampled centres among the valid points."""
  b, n, _ = xyz.shape
  dev = xyz.device
  dist = torch.full((b, n), 1.0e30, device=dev)
  dist = torch.where(valid, dist, torch.full_like(dist, -1.0))
  first = valid.float().argmax(dim=1)
  idx = torch.empty(b, n_groups, dtype=torch.long, device=dev)
  cur = first
  ar = torch.arange(b, device=dev)
  for g in range(n_groups):
    idx[:, g] = cur
    centre = xyz[ar, cur].unsqueeze(1)
    d = ((xyz - centre) ** 2).sum(-1)
    d = torch.where(valid, d, torch.full_like(d, -1.0))
    dist = torch.minimum(dist, d)
    cur = dist.argmax(dim=1)
  return idx


class PointPatchEncoder(nn.Module):
  def __init__(self, in_dim: int = 4, n_groups: int = 32, group_size: int = 16, dim: int = 128,
               n_layers: int = 2, n_heads: int = 4, out_dim: int = 256,
               recon: bool = False, mask_ratio: float = 0.4) -> None:
    super().__init__()
    self.n_groups, self.group_size, self.dim = n_groups, group_size, dim
    self.feat_dim = max(0, int(in_dim) - 4)
    self.patch = _mlp((3 + self.feat_dim, 64, dim))
    # PointPatchRL's masked-reconstruction objective (Gyenes et al., 2024), the
    # piece the first generation left out: a fraction of the patch tokens is
    # replaced by a learned mask token before the transformer and a small
    # decoder has to reproduce each masked patch's local points (Chamfer).
    # Lives in ``recon_loss``, never in ``forward``, so the exported graph is
    # untouched and the policy always encodes the unmasked cloud.
    self.recon = bool(recon)
    self.mask_ratio = float(mask_ratio)
    if self.recon:
      self.mask_token = nn.Parameter(torch.zeros(dim))
      self.decoder = nn.Sequential(nn.Linear(dim, 128), nn.ELU(), nn.Linear(128, group_size * 3))
    self.centre = nn.Linear(3, dim)
    layer = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=2 * dim, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
    self.encoder = nn.TransformerEncoder(layer, n_layers)
    self.norm = nn.LayerNorm(dim)
    self.head = nn.Linear(2 * dim, out_dim)
    self.output_dim = out_dim

  def _patches(self, x: torch.Tensor):
    xyz = x[..., :3]
    valid = x[..., 3] > 0.5
    with torch.no_grad():
      cidx = farthest_points(xyz, valid, self.n_groups)                    # (B, G)
      centres = torch.gather(xyz, 1, cidx.unsqueeze(-1).expand(-1, -1, 3))  # (B, G, 3)
      d = torch.cdist(centres, xyz)                                         # (B, G, N)
      d = torch.where(valid.unsqueeze(1), d, torch.full_like(d, 1.0e30))
      nidx = torch.topk(d, self.group_size, dim=-1, largest=False)[1]      # (B, G, K)
    gathered = torch.gather(xyz.unsqueeze(1).expand(-1, self.n_groups, -1, -1), 2,
                            nidx.unsqueeze(-1).expand(-1, -1, -1, 3))        # (B, G, K, 3)
    local = gathered - centres.unsqueeze(2)
    if self.feat_dim > 0:
      feats = x[..., 4:4 + self.feat_dim]
      gf = torch.gather(feats.unsqueeze(1).expand(-1, self.n_groups, -1, -1), 2,
                        nidx.unsqueeze(-1).expand(-1, -1, -1, self.feat_dim))  # (B, G, K, F)
      local = torch.cat([local, gf], dim=-1)
    return local, centres, valid.any(dim=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    local, centres, frame_valid = self._patches(x)
    tokens = self.patch(local).amax(dim=2) + self.centre(centres)           # (B, G, D)
    tokens = self.norm(self.encoder(tokens))
    pooled = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
    out = self.head(pooled)
    return torch.where(frame_valid.unsqueeze(-1), out, torch.zeros_like(out))

  @torch.jit.unused
  def recon_loss(self, x: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance of the decoded masked patches to the true ones, in metres.

    A fraction ``mask_ratio`` of the patch tokens is replaced by the mask token
    (the centre embedding is kept, as in PointPatchRL); the decoder predicts
    the masked patch's ``group_size`` local points.  Frames with no valid
    point contribute nothing.
    """
    if not self.recon:
      return x.new_zeros(())
    local, centres, frame_valid = self._patches(x)
    b, g = centres.shape[:2]
    tokens = self.patch(local).amax(dim=2) + self.centre(centres)
    m = torch.rand(b, g, device=x.device) < self.mask_ratio
    m = m & frame_valid.unsqueeze(1)
    if not bool(m.any()):
      return x.new_zeros(())
    masked_in = torch.where(m.unsqueeze(-1), self.mask_token + self.centre(centres), tokens)
    enc = self.norm(self.encoder(masked_in))
    pred = self.decoder(enc[m]).view(-1, self.group_size, 3)
    target = local[m][..., :3]
    d = torch.cdist(pred, target)                                            # (M, K, K)
    return 0.5 * (d.amin(dim=2).mean() + d.amin(dim=1).mean())


class _Basic(nn.Module):
  def __init__(self, cin: int, cout: int, stride: int) -> None:
    super().__init__()
    g = 8
    self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
    self.n1 = nn.GroupNorm(g, cout)
    self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
    self.n2 = nn.GroupNorm(g, cout)
    self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(g, cout))
                 if (stride != 1 or cin != cout) else nn.Identity())

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = F.elu(self.n1(self.c1(x)))
    y = self.n2(self.c2(y))
    return F.elu(y + self.skip(x))


class DepthResNetLite(nn.Module):
  """BasicBlock 2-2-2-2, widths 32-64-128-256, stem stride 2 + maxpool: a ResNet-18 layout at half width."""

  def __init__(self, in_channels: int = 2, widths: tuple[int, ...] = (32, 64, 128, 256), out_dim: int = 256) -> None:
    super().__init__()
    self.stem = nn.Sequential(nn.Conv2d(in_channels, widths[0], 5, 2, 2, bias=False),
                              nn.GroupNorm(8, widths[0]), nn.ELU(), nn.MaxPool2d(3, 2, 1))
    blocks: list[nn.Module] = []
    cin = widths[0]
    for i, w in enumerate(widths):
      stride = 1 if i == 0 else 2
      blocks += [_Basic(cin, w, stride), _Basic(w, w, 1)]
      cin = w
    self.blocks = nn.Sequential(*blocks)
    self.head = nn.Linear(widths[-1], out_dim)
    self.output_dim = out_dim
    self.structure = ("ResNet18-lite: BasicBlock 2-2-2-2, widths 32-64-128-256, GroupNorm, "
                      "random init.  NOT DeFM.")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = self.blocks(self.stem(x))
    return self.head(y.mean(dim=(-2, -1)))


def build_encoder(spec: dict, shape: tuple[int, ...]) -> nn.Module:
  """``spec["type"]`` -> module, given the per-sample observation shape."""
  kind = spec.get("type")
  out_dim = int(spec.get("out_dim", 256))
  if kind == "pointnet":
    return PointNetEncoder(in_dim=int(shape[-1]), out_dim=out_dim)
  if kind == "pointpatch":
    return PointPatchEncoder(in_dim=int(shape[-1]), n_groups=int(spec.get("n_groups", 32)),
                             group_size=int(spec.get("group_size", 16)), dim=int(spec.get("dim", 128)),
                             n_layers=int(spec.get("n_layers", 2)), out_dim=out_dim,
                             recon=bool(spec.get("recon", False)), mask_ratio=float(spec.get("mask_ratio", 0.4)))
  if kind == "setmlp":
    return SetMLPEncoder(in_dim=int(shape[-1]), dims=(64, 128, 128), out_dim=out_dim)
  if kind == "depthresnet":
    return DepthResNetLite(in_channels=int(shape[0]), out_dim=out_dim)
  raise ValueError(f"unknown encoder type {kind!r}")
