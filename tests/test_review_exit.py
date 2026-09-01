"""The exit-time review must never end the run it is reporting on."""

import pathlib

import pytest

from hardware.deploy import review, run


def test_review_run_raises_a_catchable_error_on_an_empty_session(tmp_path):
  """Not SystemExit.

  ``review.run`` is a library function; SystemExit is the CLI's business and it
  derives from BaseException, so it walks through every ``except Exception``
  its callers guard with.
  """
  (tmp_path / "meta.json").write_text("[]")
  with pytest.raises(ValueError, match="no frames"):
    review.run(tmp_path, stride=1, quality=80)
  # The property that matters, stated directly: the one an ``except Exception``
  # catches, and the one it does not.
  assert issubclass(ValueError, Exception)
  assert not issubclass(SystemExit, Exception)


def test_write_review_survives_a_session_with_no_frames(tmp_path, capsys):
  """A --dry-run has no camera frames by design.

  Before this, the review's SystemExit propagated out of ``_write_review``'s
  ``except Exception``, run.py exited nonzero, and scripts/bringup_d455.sh
  reported NO-GO for a policy whose export had just passed every check.
  """
  (tmp_path / "meta.json").write_text("[]")
  run._write_review(tmp_path)                     # must not raise, must not exit
  assert "could not be written" in capsys.readouterr().out


def test_write_review_swallows_a_bare_system_exit(tmp_path, monkeypatch, capsys):
  """Guard the promise itself, not just today's way of breaking it."""
  def boom(*a, **k):
    raise SystemExit("anything at all")
  monkeypatch.setattr(review, "run", boom)
  run._write_review(tmp_path)
  assert "could not be written" in capsys.readouterr().out


def test_event_frame_fields_tolerate_no_frame():
  """The loop logs a frame index for events that can happen before any frame.

  A --dry-run never gets one, and neither does any run for its first steps.
  Reaching through the None crashed the loop after the arm was already held --
  safe, but a nonzero exit that failed the bring-up gate.
  """
  assert run._frame_index(None) is None
  assert run._frame_stamp(None) is None

  class F:
    index, stamp = 7, 1.5
  assert run._frame_index(F()) == 7
  assert run._frame_stamp(F()) == pytest.approx(1.5)


def test_no_event_payload_dereferences_a_frame_directly():
  """Guard against the next one being written the old way."""
  import inspect
  src = inspect.getsource(run)
  assert "int(frame.index)," not in src
  assert "float(frame.stamp)," not in src
