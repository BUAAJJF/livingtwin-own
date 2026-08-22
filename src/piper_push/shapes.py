"""Compose a rigid object out of the parts the shared topology provides.

MuJoCo-Warp compiles one model for every environment, so a geom's type is
fixed for the whole run.  The object body therefore carries a box, a cylinder
and a second box at all times, and a shape class is expressed by sizing the
parts it uses and shrinking the ones it does not.

Mass and inertia are written from the composition rather than left at whatever
the compiler produced.  ``dr.body_mass`` changes mass alone -- mjlab warns
about this in its own docstring -- so the previous distribution ran a 50 g
object and a 400 g object, and every size in between, on one inertia tensor
belonging to the default box.  Rotation during a 1.4 m/s release is decided by
exactly that tensor, so it could not be left stale.
"""

from __future__ import annotations

import math

import mujoco
import torch
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from piper_push import objects


def _quat_from_matrix(R: torch.Tensor) -> torch.Tensor:
  """Batched (N, 3, 3) rotation matrices to (N, 4) w-first quaternions."""
  m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
  trace = m00 + m11 + m22
  q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

  big = trace > 0
  s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-12)) * 2.0
  q[big, 0] = 0.25 * s[big]
  q[big, 1] = (R[big, 2, 1] - R[big, 1, 2]) / s[big]
  q[big, 2] = (R[big, 0, 2] - R[big, 2, 0]) / s[big]
  q[big, 3] = (R[big, 1, 0] - R[big, 0, 1]) / s[big]

  # The three degenerate branches, each keyed on which diagonal term dominates.
  rest = ~big
  for axis in range(3):
    j, k = (axis + 1) % 3, (axis + 2) % 3
    diag = R[:, axis, axis]
    other = torch.maximum(R[:, j, j], R[:, k, k])
    sel = rest & (diag >= other)
    if not bool(sel.any()):
      continue
    s = torch.sqrt(
      torch.clamp(1.0 + R[sel, axis, axis] - R[sel, j, j] - R[sel, k, k], min=1e-12)
    ) * 2.0
    q[sel, 0] = (R[sel, k, j] - R[sel, j, k]) / s
    q[sel, 1 + axis] = 0.25 * s
    q[sel, 1 + j] = (R[sel, j, axis] + R[sel, axis, j]) / s
    q[sel, 1 + k] = (R[sel, k, axis] + R[sel, axis, k]) / s
    rest = rest & ~sel

  return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _compose(
  n: int,
  device: torch.device,
  generator: torch.Generator | None,
  variety: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Sample a shape per environment.

  Returns ``(size, pos, half, cls)``: part sizes and offsets shaped
  (n, 3 parts, 3), the bounding half-extents (n, 3), and the class index (n,).

  ``variety`` in [0, 1] interpolates from a single fixed box towards the full
  distribution, so the curriculum can be opened gradually without a second
  code path.
  """

  def u(lo, hi, shape=(n,)):
    r = torch.rand(shape, device=device, generator=generator)
    return lo + (hi - lo) * r

  # Width and aspect are sampled directly.  Drawing the three half-extents
  # independently -- the obvious thing, and what the first distribution did --
  # makes a flat object require height at its floor and width at its ceiling
  # simultaneously, which happened in 0.4% of draws.
  lo_w, hi_w = objects.OBJECT_WIDTH_RANGE
  mid_w = 0.5 * (lo_w + hi_w)
  width = u(mid_w + (lo_w - mid_w) * variety, mid_w + (hi_w - mid_w) * variety)

  lo_a, hi_a = objects.OBJECT_ASPECT_RANGE
  aspect = u(1.0 + (lo_a - 1.0) * variety, 1.0 + (hi_a - 1.0) * variety)

  lo_n, hi_n = objects.OBJECT_ANISOTROPY_RANGE
  anis = u(1.0 + (lo_n - 1.0) * variety, 1.0 + (hi_n - 1.0) * variety)

  mean_half = width / 2.0
  root = torch.sqrt(anis)
  hx = mean_half * root
  hy = mean_half / root
  hz = torch.clamp(
    aspect * mean_half,
    min=objects.OBJECT_HEIGHT_FLOOR / 2.0,
    max=objects.OBJECT_MAX_HALF_HEIGHT,
  )
  # Scale the plan back rather than clipping one axis: clipping would turn the
  # widest bars into squares and quietly delete the anisotropy that was drawn.
  shrink = (objects.OBJECT_MAX_HALF_WIDTH / torch.maximum(hx, hy)).clamp(max=1.0)
  hx = hx * shrink
  hy = hy * shrink
  mean_half = 0.5 * (hx + hy)

  weights = torch.tensor(objects.SHAPE_WEIGHTS, device=device)
  cls = torch.multinomial(weights.expand(n, -1), 1, generator=generator).squeeze(-1)
  # At variety 0 every environment is the plain box the smoke test debugs on.
  if variety < 1.0:
    keep = torch.rand(n, device=device, generator=generator) < variety
    cls = torch.where(keep, cls, torch.zeros_like(cls))

  size = torch.zeros(n, 3, 3, device=device)
  pos = torch.zeros(n, 3, 3, device=device)
  collapsed = objects.COLLAPSED_HALF
  size[:] = collapsed

  is_box = cls == 0
  is_cyl = cls == 1
  is_step = cls == 2
  is_ell = cls == 3
  is_cap = cls == 4

  # 0 box: one geom carrying the whole bounding box.
  core = torch.stack((hx, hy, hz), dim=-1)
  size[is_box, 0] = core[is_box]

  # 1 cylinder: radius from the mean half-width, so the bounding box is square
  # in plan whatever anisotropy was drawn.
  size[is_cyl, 0] = collapsed
  size[is_cyl, 1, 0] = mean_half[is_cyl]
  size[is_cyl, 1, 1] = hz[is_cyl]
  size[is_cyl, 1, 2] = 0.0

  # 2 stepped: a tall base with a narrower block on top.
  size[is_step, 0] = torch.stack((hx, hy, hz * 0.6), dim=-1)[is_step]
  pos[is_step, 0, 2] = -hz[is_step] * 0.4
  size[is_step, 2] = torch.stack((hx * 0.55, hy * 0.55, hz * 0.4), dim=-1)[is_step]
  pos[is_step, 2, 2] = hz[is_step] * 0.6

  # 3 L: two blocks meeting at the body origin, so the centre of mass sits off
  # the geometric centre and the principal axes rotate out of the body frame.
  size[is_ell, 0] = torch.stack((hx * 0.5, hy, hz), dim=-1)[is_ell]
  pos[is_ell, 0, 0] = -hx[is_ell] * 0.5
  size[is_ell, 2] = torch.stack((hx * 0.5, hy, hz * 0.45), dim=-1)[is_ell]
  pos[is_ell, 2, 0] = hx[is_ell] * 0.5
  pos[is_ell, 2, 2] = -hz[is_ell] * 0.55

  # 4 capped: a box with a cylinder standing on it.
  size[is_cap, 0] = torch.stack((hx, hy, hz * 0.55), dim=-1)[is_cap]
  pos[is_cap, 0, 2] = -hz[is_cap] * 0.45
  size[is_cap, 1, 0] = (torch.minimum(hx, hy) * 0.75)[is_cap]
  size[is_cap, 1, 1] = (hz * 0.45)[is_cap]
  size[is_cap, 1, 2] = 0.0
  pos[is_cap, 1, 2] = hz[is_cap] * 0.55

  # Park every collapsed part inside a part that is real, where nothing outside
  # the body can reach it.  The core is that part except for a pure cylinder,
  # whose core is itself collapsed onto the barrel's axis at the origin.
  tiny = size.max(dim=-1).values <= collapsed * 1.5   # (n, 3)
  host = torch.where(is_cyl.unsqueeze(-1), pos[:, 1], pos[:, 0])   # (n, 3)
  pos = torch.where(tiny.unsqueeze(-1), host.unsqueeze(1), pos)

  half = torch.stack((hx, hy, hz), dim=-1)
  half[is_cyl, 0] = mean_half[is_cyl]
  half[is_cyl, 1] = mean_half[is_cyl]
  return size, pos, half, cls


def _mass_properties(
  size: torch.Tensor, pos: torch.Tensor, mass: torch.Tensor, is_cylinder: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Body COM, principal inertia and inertial-frame quaternion.

  ``size``/``pos`` are (n, 3, 3); part 1 is the cylinder, the rest are boxes.
  Density is uniform across the parts, which is what a solid object made of one
  material actually is.
  """
  n = size.shape[0]
  device = size.device
  # Extents of each part along the body axes: a cylinder's third size entry is
  # unused, and its half-height lives in the second.
  cyl = torch.zeros(n, 3, dtype=torch.bool, device=device)
  cyl[:, 1] = True
  ext = torch.where(
    cyl.unsqueeze(-1),
    torch.stack((size[..., 0], size[..., 0], size[..., 1]), dim=-1),
    size,
  )
  vol = torch.where(
    cyl,
    math.pi * size[..., 0] ** 2 * 2.0 * size[..., 1],
    8.0 * size[..., 0] * size[..., 1] * size[..., 2],
  )
  # A part that is switched off still has a millimetre of volume; zeroing it
  # keeps the density from being diluted by parts that are not there.
  vol = torch.where(ext.max(dim=-1).values <= objects.COLLAPSED_HALF * 1.5,
                    torch.zeros_like(vol), vol)
  total = vol.sum(dim=1, keepdim=True).clamp(min=1e-12)
  part_mass = mass.unsqueeze(-1) * vol / total

  com = (part_mass.unsqueeze(-1) * pos).sum(dim=1) / mass.unsqueeze(-1).clamp(min=1e-12)

  a, b, c = ext[..., 0], ext[..., 1], ext[..., 2]
  box_i = torch.stack((b * b + c * c, a * a + c * c, a * a + b * b), dim=-1) / 3.0
  r, h = size[..., 0], size[..., 1]
  cyl_i = torch.stack(
    (
      (3.0 * r * r + 4.0 * h * h) / 12.0,
      (3.0 * r * r + 4.0 * h * h) / 12.0,
      r * r / 2.0,
    ),
    dim=-1,
  )
  local = torch.where(cyl.unsqueeze(-1), cyl_i, box_i) * part_mass.unsqueeze(-1)

  inertia = torch.diag_embed(local).sum(dim=1)
  d = pos - com.unsqueeze(1)
  d2 = (d * d).sum(dim=-1)
  eye = torch.eye(3, device=device).expand(n, 3, 3)
  shift = part_mass.unsqueeze(-1).unsqueeze(-1) * (
    d2.unsqueeze(-1).unsqueeze(-1) * eye.unsqueeze(1)
    - d.unsqueeze(-1) * d.unsqueeze(-2)
  )
  inertia = inertia + shift.sum(dim=1)

  # An L is not diagonal in the body frame, so the principal axes have to be
  # solved for rather than assumed.
  evals, evecs = torch.linalg.eigh(inertia.double())
  flip = torch.linalg.det(evecs) < 0
  evecs[flip, :, 0] = -evecs[flip, :, 0]
  quat = _quat_from_matrix(evecs.float())
  return com, evals.float().clamp(min=1e-9), quat


# Without this the writes below land only in world 0: these model fields are
# broadcast across environments until an event declares it needs them
# expanded, and reading one back afterwards returns world 0's value for
# every environment.  It fails silently -- every object is a copy of the
# first one, and nothing raises.
@requires_model_fields(
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "geom_pos",
  "geom_friction",
  "body_mass",
  "body_inertia",
  "body_ipos",
  "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def randomize_object_shape(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  mass_range: tuple[float, float] = objects.OBJECT_MASS_RANGE,
  friction_range: tuple[float, float] = objects.OBJECT_FRICTION_RANGE,
  variety: float = 1.0,
) -> None:
  """Draw a fresh object: shape class, size, mass, inertia and friction."""
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  local, _ = asset.find_geoms(list(objects.OBJECT_GEOMS), preserve_order=True)
  gids = torch.as_tensor(
    [int(asset.indexing.geom_ids[i]) for i in local], device=env.device
  )
  body_local, _ = asset.find_bodies(["object"], preserve_order=True)
  bid = int(asset.indexing.body_ids[body_local[0]])

  size, pos, half, cls = _compose(n, env.device, None, variety)
  lo, hi = mass_range
  mass = lo + (hi - lo) * torch.rand(n, device=env.device)
  com, inertia, iquat = _mass_properties(size, pos, mass, cls == 1)

  grid_env, grid_geom = torch.meshgrid(env_ids, gids, indexing="ij")
  env.sim.model.geom_size[grid_env, grid_geom] = size
  env.sim.model.geom_pos[grid_env, grid_geom] = pos

  flo, fhi = friction_range
  fric = flo + (fhi - flo) * torch.rand(n, 1, device=env.device)
  env.sim.model.geom_friction[grid_env, grid_geom, 0] = fric.expand(n, len(gids))

  env.sim.model.body_mass[env_ids, bid] = mass
  env.sim.model.body_ipos[env_ids, bid] = com
  env.sim.model.body_inertia[env_ids, bid] = inertia
  env.sim.model.body_iquat[env_ids, bid] = iquat

  _recompute_geom_bounds(env, env_ids.to(torch.int), asset_cfg)

  # The command term reads these for spawn height, the grasp test and the
  # observation, and recomputing them from the model would mean redoing the
  # union of three offset parts on every step.
  state = _state(env, asset_cfg.name)
  state["half"][env_ids] = half
  state["cls"][env_ids] = cls


def _state(env, asset_name: str = "object") -> dict:
  """The per-environment shape record for one asset, seeded from the model.

  Keyed by asset name.  A single record shared across assets is fine while
  there is one object and silently wrong the moment there are several: the
  second object's draw overwrites the first's half-extents, and those decide
  the spawn height, the lift threshold and the shape class every acceptance
  number is split by.

  The observation manager probes every term's shape while it is being built,
  which is before any reset event has run, so the record cannot depend on the
  composer having gone first.  Seeding it from the compiled geometry gives the
  same answer the composer would for the placeholder shape.
  """
  cache = getattr(env, "_object_shape_state", None)
  if cache is None:
    cache = env._object_shape_state = {}
  state = cache.get(asset_name)
  if state is not None:
    return state

  asset = env.scene[asset_name]
  local, _ = asset.find_geoms(list(objects.OBJECT_GEOMS), preserve_order=True)
  gids = torch.as_tensor(
    [int(asset.indexing.geom_ids[i]) for i in local], device=env.device
  )
  size = env.sim.model.geom_size[:, gids]
  pos = env.sim.model.geom_pos[:, gids]
  gtype = torch.as_tensor(env.sim.model.geom_type, device=env.device)[gids]
  is_cyl = (gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER)).view(1, -1, 1)
  ext = torch.where(
    is_cyl,
    torch.stack((size[..., 0], size[..., 0], size[..., 1]), dim=-1),
    size,
  )
  state = {
    "half": (pos.abs() + ext).max(dim=1).values,
    "cls": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
  }
  cache[asset_name] = state
  return state


def object_half_size(env, asset_name: str = "object") -> torch.Tensor:
  """Bounding half-extents of the composed object, per environment."""
  return _state(env, asset_name)["half"]


def object_shape_class(env, asset_name: str = "object") -> torch.Tensor:
  """Which shape class each environment drew, as an index into SHAPE_CLASSES."""
  return _state(env, asset_name)["cls"]
