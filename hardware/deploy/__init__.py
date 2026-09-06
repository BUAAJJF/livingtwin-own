"""Run the trained vision policy on the real PiPER-X and the real D455 (D405 before 2026-08-25).

The pipeline, in the order a frame travels through it:

    d455 -> rectify -> mask -> obs ---.
                                       >-- policy -> command -> arm
    arm  -> proprio ------------------'

Every stage has a counterpart in ``src/piper_push`` and the job of this package
is to agree with it.  Where it cannot -- the target mask, the pad contacts --
the disagreement is written down in that module's docstring, and where possible
the simulator has been changed to meet it rather than the other way round.

``selftest.py`` runs the whole thing against a scene rendered by mjlab and
compares the reconstructed observation to the one the simulator produced.  It
needs no camera and no robot, so it is the thing to run first and after every
change.
"""

# mjlab before anything of ours.  Importing mjlab runs its entry-point scan,
# which imports ``piper_push.tasks``, which builds the task configs, which
# reads ``piper_push.camera`` and ``piper_push.robot``.  A module here that
# reaches for either of those first starts that chain from inside itself; the
# task package then fails to register, with a warning rather than an error, so
# the first symptom is a task id that does not exist.
import mjlab.tasks  # noqa: F401,E402
