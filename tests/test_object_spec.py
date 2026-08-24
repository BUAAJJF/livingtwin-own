"""The physical object set, checked against the distribution it stands in for.

``hardware/objects/README.md`` fixes eight sizes to go shopping with, and
``scripts/fit_object_distribution.py`` turns the weighed result into a change to
the mass parameters.  Both rest on one thing being right: that the volume this
script computes for a shape class is the volume the simulator composes for it.
If it is not, every density in the report is wrong by a constant that depends on
the class, and the retrain gets a parameter fitted to that error.  Nothing else
would show it -- the report prints, the numbers look like densities, and the
policy is trained for objects that weigh the wrong amount.

The rest of the file checks that the eight rows in the README are inside the
limits the README itself states, so that a later edit to the table cannot
quietly ship a size the gripper cannot hold.

Run with:  micromamba run -n mjlab python -m pytest tests/test_object_spec.py -q
"""

from __future__ import annotations

import pathlib
import re

import mjlab.tasks  # noqa: F401  -- before piper_push; see deploy/__init__.py
import pytest
import torch

from piper_push import objects, shapes

import importlib.util

_SPEC = importlib.util.spec_from_file_location(
  "fit_object_distribution",
  pathlib.Path(__file__).resolve().parents[1] / "scripts"
  / "fit_object_distribution.py",
)
fit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fit)

README = pathlib.Path(__file__).resolve().parents[1] / "hardware" / "objects" \
  / "README.md"


@pytest.mark.parametrize("cls", objects.SHAPE_CLASSES)
def test_nominal_volume_matches_the_sampler(cls: str) -> None:
  """The closed form per class equals what ``_compose`` actually builds.

  Drawn from the sampler rather than constructed here, so the test is against
  the shapes training uses and not against a second reading of the same
  formulas.
  """
  g = torch.Generator().manual_seed(7)
  size, _, half, idx = shapes._compose(20_000, torch.device("cpu"), g, 1.0)
  sel = idx == objects.SHAPE_CLASSES.index(cls)
  assert int(sel.sum()) > 100, f"too few {cls} draws to test"

  composed = fit.solid_volume(size[sel])
  wide = 2 * torch.maximum(half[sel, 0], half[sel, 1])
  narrow = 2 * torch.minimum(half[sel, 0], half[sel, 1])
  height = 2 * half[sel, 2]
  closed = torch.tensor([
    fit.nominal_solid_volume(cls, float(w), float(n), float(h))
    for w, n, h in zip(wide, narrow, height)
  ])
  # Relative, because the volumes span two orders of magnitude across sizes.
  rel = ((closed - composed).abs() / composed.clamp(min=1e-12)).max()
  assert float(rel) < 1e-5, f"{cls}: closed form off by {float(rel):.2%}"


def test_solid_volume_ignores_the_collapsed_parts() -> None:
  """A part that is switched off is a millimetre cube and must count as zero.

  A pure box carries a collapsed cylinder and a collapsed second box at all
  times -- the model has one topology -- and counting them would inflate the
  volume of the smallest objects most, which is exactly where the mass floor
  is decided.
  """
  g = torch.Generator().manual_seed(3)
  size, _, half, idx = shapes._compose(4_000, torch.device("cpu"), g, 1.0)
  box = idx == 0
  bbox = 8 * half[box, 0] * half[box, 1] * half[box, 2]
  assert torch.allclose(fit.solid_volume(size[box]), bbox, rtol=1e-6)


def _readme_rows() -> list[tuple[int, str, float, float, float]]:
  rows = []
  for line in README.read_text().splitlines():
    m = re.match(r"\|\s*(\d)\s*\|\s*([\w ]+?)\s*\|\s*(.+?)\s*\|", line)
    if not m:
      continue
    dims = m.group(3).replace("Ø", "").split("×")
    nums = [float(d.strip()) for d in dims]
    if len(nums) == 2:            # a diameter and a height
      nums = [nums[0], nums[0], nums[1]]
    rows.append((int(m.group(1)), m.group(2), *nums))
  return rows


def test_readme_table_is_inside_its_own_limits() -> None:
  rows = _readme_rows()
  assert len(rows) == 8, f"expected 8 objects in the README, parsed {len(rows)}"
  for i, cls, long_mm, short_mm, height in rows:
    bad = fit.check_limits({"long_mm": long_mm, "short_mm": short_mm,
                            "height_mm": height, "mass_g": 200})
    assert not bad, f"object {i} ({cls}): {bad}"


def test_readme_table_covers_the_distribution() -> None:
  """Every marginal reaches both tails, with no 40-point hole in between.

  Eight objects cannot be a sample of the distribution.  What they can be is a
  spanning set, and a gap larger than this would mean a whole region of sizes
  the policy was trained on and the table never shows it.
  """
  ref = fit.reference(50_000, "cpu", 0)
  rows = _readme_rows()
  for key, scale, values in (
    ("wide", 1000.0, [r[2] for r in rows]),
    ("height", 1000.0, [r[4] for r in rows]),
    ("aspect", 1.0, [r[4] / (0.5 * (r[2] + r[3])) for r in rows]),
  ):
    p = sorted(fit.pct(ref[key] * scale, v) for v in values)
    assert p[0] < 15, f"{key}: nothing below the {p[0]:.0f}th percentile"
    assert p[-1] > 75, f"{key}: nothing above the {p[-1]:.0f}th percentile"
    gap = max([p[0]] + [b - a for a, b in zip(p, p[1:])] + [100 - p[-1]])
    assert gap < 40, f"{key}: a {gap:.0f}-point gap in coverage"


def test_every_shape_class_is_represented() -> None:
  named = {r[1] for r in _readme_rows()}
  # The README writes the classes the way a person says them.
  alias = {"box": "box", "cylinder": "cylinder", "stepped": "stepped",
           "L": "l_shape", "capped": "capped"}
  have = {alias[n] for n in named if n in alias}
  assert have == set(objects.SHAPE_CLASSES), (
    f"missing from the table: {set(objects.SHAPE_CLASSES) - have}")
