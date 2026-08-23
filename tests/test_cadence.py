"""Cadence is a hold-time, not a distribution.

The whole point of splitting ``redraw`` out of ``randomize_object_shape`` is
that a cadence experiment must vary *only* how long a drawn value is held.  If
selecting a subset also changed the range that subset is drawn from, every
number the experiment produces would confound two effects and none of them
could be attributed.

These tests run against a stub rather than a built environment: the quantity
under test is which values get written and which are left alone, and that is
decidable without a simulator.  A GPU test here would be slower and would
check less.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest
import torch

from piper_push import objects, shapes


class _Model:
  def __init__(self, n: int, ngeom: int, nbody: int):
    self.geom_size = torch.zeros(n, ngeom, 3)
    self.geom_pos = torch.zeros(n, ngeom, 3)
    self.geom_friction = torch.zeros(n, ngeom, 3)
    self.geom_type = torch.zeros(ngeom, dtype=torch.long)
    self.body_mass = torch.zeros(n, nbody)
    self.body_ipos = torch.zeros(n, nbody, 3)
    self.body_inertia = torch.zeros(n, nbody, 3)
    self.body_iquat = torch.zeros(n, nbody, 4)


class _Indexing:
  def __init__(self, ngeom: int):
    self.geom_ids = list(range(ngeom))
    self.body_ids = [0]


class _Asset:
  def __init__(self, ngeom: int):
    self.indexing = _Indexing(ngeom)
    self._ngeom = ngeom

  def find_geoms(self, names, preserve_order=False):
    return list(range(self._ngeom)), list(names)

  def find_bodies(self, names, preserve_order=False):
    return [0], list(names)


class _Sim:
  def __init__(self, model):
    self.model = model

  def recompute_constants(self, *a, **k):
    pass


class _Env:
  """Just enough of ManagerBasedRlEnv for the randomiser to write into."""

  def __init__(self, n: int = 64):
    ngeom = len(objects.OBJECT_GEOMS)
    self.num_envs = n
    self.device = torch.device("cpu")
    self.sim = _Sim(_Model(n, ngeom, 1))
    self._asset = _Asset(ngeom)
    self.scene = {"object": self._asset}


class _Cfg:
  name = "object"


@pytest.fixture
def env(monkeypatch):
  # The bounds recompute reaches into mjlab's warp kernels; the fields it
  # derives are not what these tests are about.
  monkeypatch.setattr(shapes, "_recompute_geom_bounds", lambda *a, **k: None)
  e = _Env()
  # Seed the record the way a first draw would, so "held constant" has
  # something to hold.
  shapes.randomize_object_shape(
    e, None, _Cfg(), redraw=shapes.ALL_QUANTITIES)
  return e


def _snapshot(env):
  s = shapes._state(env, "object")
  return {k: s[k].clone() for k in ("size", "pos", "half", "cls", "mass", "fric")}


def test_all_quantities_move_when_all_are_redrawn(env):
  before = _snapshot(env)
  shapes.randomize_object_shape(env, None, _Cfg(), redraw=shapes.ALL_QUANTITIES)
  after = _snapshot(env)
  for k in ("size", "mass", "fric"):
    assert not torch.equal(before[k], after[k]), f"{k} did not change"


def test_empty_redraw_is_a_no_op(env):
  before = _snapshot(env)
  shapes.randomize_object_shape(env, None, _Cfg(), redraw=())
  after = _snapshot(env)
  for k in before:
    assert torch.equal(before[k], after[k]), f"{k} moved on an empty redraw"


@pytest.mark.parametrize(
  "held,moved",
  [
    ("shape", ("mass", "friction")),
    ("mass", ("shape", "friction")),
    ("friction", ("shape", "mass")),
  ],
)
def test_holding_one_quantity_leaves_it_alone(env, held, moved):
  """The named quantity is untouched and the others turn over."""
  before = _snapshot(env)
  shapes.randomize_object_shape(env, None, _Cfg(), redraw=moved)
  after = _snapshot(env)

  fields = {"shape": ("size", "pos", "half", "cls"), "mass": ("mass",),
            "friction": ("fric",)}
  for f in fields[held]:
    assert torch.equal(before[f], after[f]), f"{f} moved while {held} was held"
  for q in moved:
    assert any(not torch.equal(before[f], after[f]) for f in fields[q]), (
      f"{q} was asked for and did not change")


def test_marginals_are_identical_whatever_else_is_redrawn():
  """Mass drawn alone and mass drawn alongside shape come from one range.

  This is the assertion the cadence experiment rests on.  Drawing a subset
  must not narrow, widen or shift the distribution of what is drawn, or the
  comparison between two cadences is not a controlled one.
  """
  lo, hi = objects.OBJECT_MASS_RANGE
  samples = {}
  for redraw in (("mass",), shapes.ALL_QUANTITIES):
    e = _Env(n=4096)
    shapes._recompute_geom_bounds = lambda *a, **k: None
    torch.manual_seed(0)
    shapes.randomize_object_shape(e, None, _Cfg(), redraw=shapes.ALL_QUANTITIES)
    torch.manual_seed(1234)
    shapes.randomize_object_shape(e, None, _Cfg(), redraw=redraw)
    samples[redraw] = shapes._state(e, "object")["mass"].clone()

  a, b = samples[("mass",)], samples[shapes.ALL_QUANTITIES]
  assert a.min() >= lo - 1e-6 and a.max() <= hi + 1e-6
  assert b.min() >= lo - 1e-6 and b.max() <= hi + 1e-6
  # Same range, same mean to within the sampling error of 4096 uniforms.
  tol = 4.0 * (hi - lo) / (12 ** 0.5) / (4096 ** 0.5)
  assert abs(float(a.mean() - b.mean())) < tol


def test_unknown_quantity_is_rejected(env):
  with pytest.raises(ValueError, match="colour"):
    shapes.randomize_object_shape(env, None, _Cfg(), redraw=("colour",))
