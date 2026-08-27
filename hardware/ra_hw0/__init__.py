"""Phase RA-HW-0: read-only hardware audit and motion design for the PiPER-X.

Nothing in this package enables a motor, sends a motion command, releases an
emergency stop, homes, clears a fault or writes a controller parameter.  The
entry points default to ``--dry-run`` and require an explicit ``--hardware``
flag to open CAN at all; even then the audit path uses only the SDK's ``Get*``
accessors and the three enquiry frames ``ConnectPort`` issues.

The gate that separates design from motion is a human sentence, recorded in
docs/ra_hw0_experiment_plan.md, and no code in this package can satisfy it.
"""
