"""The risk head must be deployable, and its metrics must suit a rare event.

Two things this file is here to stop.

A head that reads something a robot does not have. `RiskHead.CHANNELS` goes
through `wm_data.assert_deployable` at construction, so a head that names the
trip label, the damping, or the object's pose cannot be built at all.

And a head that is scored by a metric which flatters it. At the base rates
here -- the safety shell fires about twice an arm-hour at nominal damping --
ROC-AUC is decided by how the head orders the overwhelming majority of
negatives, and a head that never finds a positive can still look good. Average
precision is reported first for that reason, and these tests pin the
difference.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest
import torch

from piper_push import risk


# ---------------------------------------------------------------------------
# Deployability
# ---------------------------------------------------------------------------


def test_the_head_only_reads_deployable_channels():
  from piper_push import wm_data

  for c in risk.RiskHead.CHANNELS:
    assert c in wm_data.DEPLOYABLE
  assert "trip" not in risk.RiskHead.CHANNELS


def test_a_head_that_named_a_simulator_label_would_not_construct():
  class Leaky(risk.RiskHead):
    CHANNELS = ("enc", "trip")

  with pytest.raises(ValueError, match="simulator label"):
    Leaky({"enc": 4, "trip": 1})


def test_forward_gives_one_number_per_window():
  m = risk.RiskHead({"enc": 6, "proprio": 4, "action": 3, "servo": 1}, hidden=8)
  b = {"enc": torch.randn(12, 5, 6), "proprio": torch.randn(12, 5, 4),
       "action": torch.randn(12, 5, 3), "servo": torch.randn(12, 5, 1)}
  assert m(b).shape == (5,)
  p = m.probability(b)
  assert ((p >= 0) & (p <= 1)).all()


def test_round_trip(tmp_path):
  dims = {"enc": 6, "proprio": 4, "action": 3, "servo": 1}
  m = risk.RiskHead(dims, hidden=8)
  torch.save(m.state(), tmp_path / "r.pt")
  back = risk.RiskHead.load(torch.load(tmp_path / "r.pt", weights_only=False))
  b = {"enc": torch.randn(9, 2, 6), "proprio": torch.randn(9, 2, 4),
       "action": torch.randn(9, 2, 3), "servo": torch.randn(9, 2, 1)}
  assert torch.allclose(m(b), back(b), atol=1e-6)


# ---------------------------------------------------------------------------
# The label
# ---------------------------------------------------------------------------


def test_the_label_is_strictly_in_the_future():
  """A window that *contains* a trip is not thereby a positive.

  Labelling on the window itself would let the head read the trip out of the
  joint velocities it is shown, score beautifully, and predict nothing.
  """
  trip = torch.zeros(100, dtype=torch.bool)
  trip[10] = True                      # inside the window [0, 20)
  y = risk.labels_within_horizon(trip, [0], length=20, horizon=25)
  assert float(y[0]) == 0.0

  trip = torch.zeros(100, dtype=torch.bool)
  trip[25] = True                      # in (19, 44]
  assert float(risk.labels_within_horizon(trip, [0], 20, 25)[0]) == 1.0


def test_a_trip_beyond_the_horizon_is_not_a_positive():
  trip = torch.zeros(200, dtype=torch.bool)
  trip[80] = True
  assert float(risk.labels_within_horizon(trip, [0], 20, 25)[0]) == 0.0
  assert float(risk.labels_within_horizon(trip, [40], 20, 25)[0]) == 1.0


def test_a_window_whose_future_is_off_the_end_cannot_be_positive():
  trip = torch.zeros(30, dtype=torch.bool)
  y = risk.labels_within_horizon(trip, [10], length=20, horizon=25)
  assert float(y[0]) == 0.0


def test_labels_are_per_window():
  trip = torch.zeros(300, dtype=torch.bool)
  trip[130] = True
  y = risk.labels_within_horizon(trip, [0, 100, 200], 20, 25)
  assert y.tolist() == [0.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_average_precision_is_one_for_a_perfect_ranking():
  prob = torch.tensor([0.9, 0.8, 0.2, 0.1])
  label = torch.tensor([1.0, 1.0, 0.0, 0.0])
  assert risk.average_precision(prob, label) == pytest.approx(1.0)


def test_average_precision_punishes_what_roc_auc_forgives():
  """The reason average precision is the headline.

  One positive in a thousand, ranked 50th: ROC-AUC is 0.95, which reads as a
  good classifier, and average precision is 0.02, which reads as what it is.
  """
  n = 1000
  prob = torch.linspace(1.0, 0.0, n)
  label = torch.zeros(n)
  label[49] = 1.0
  assert risk.roc_auc(prob, label) > 0.94
  assert risk.average_precision(prob, label) < 0.05


def test_metrics_return_nan_rather_than_a_number_with_no_positives():
  prob = torch.rand(50)
  label = torch.zeros(50)
  assert risk.average_precision(prob, label) != risk.average_precision(prob, label)
  assert risk.roc_auc(prob, label) != risk.roc_auc(prob, label)


def test_brier_rewards_calibration_not_only_ranking():
  label = torch.tensor([1.0, 0.0, 1.0, 0.0])
  sharp_right = torch.tensor([0.9, 0.1, 0.9, 0.1])
  sharp_wrong = torch.tensor([0.6, 0.4, 0.6, 0.4])
  assert risk.brier(sharp_right, label) < risk.brier(sharp_wrong, label)


def test_calibration_bins_report_predicted_against_observed():
  prob = torch.tensor([0.05, 0.05, 0.95, 0.95])
  label = torch.tensor([0.0, 0.0, 1.0, 1.0])
  bins = risk.calibration(prob, label, bins=10)
  assert len(bins) == 2
  for b in bins:
    assert b["predicted"] == pytest.approx(b["observed"], abs=0.06)
    assert b["n"] == 2


def test_the_horizon_was_chosen_by_measurement():
  """0.2 s, and the module docstring carries the sweep that picked it.

  It started at 0.5 s because that seemed a reasonable lead time.  Measured
  within the target domain -- which is the only place the question "which
  window trips" is asked -- the best simple velocity feature is at AUC 0.57
  there and 0.66 at 0.2 s, so the number moved to where the evidence was.
  This test exists so that moving it again requires editing the recorded
  sweep, not just the constant.
  """
  import inspect

  assert risk.HORIZON == 10
  src = inspect.getsource(risk)
  head = src.split("class RiskHead")[0]
  for token in ("horizon  2 steps", "horizon 25 steps", "AUC"):
    assert token in head, f"the sweep that chose HORIZON is not recorded: {token}"


def test_the_label_horizon_is_a_parameter_not_a_constant():
  """The sweep that chose it has to be re-runnable at other horizons."""
  trip = torch.zeros(100, dtype=torch.bool)
  trip[40] = True                       # in (19, 44] but not in (19, 34]
  assert float(risk.labels_within_horizon(trip, [0], 20, horizon=25)[0]) == 1.0
  assert float(risk.labels_within_horizon(trip, [0], 20, horizon=15)[0]) == 0.0
