#!/usr/bin/env python3
"""Start policygui with the deployment process's measured thread limits.

This tiny entry point exists so the limits are installed before policygui
imports numpy, OpenCV or Torch. Setting them afterwards is too late: those
libraries size their pools at import time. The browser and perception logic
remain in policygui.py.
"""

from __future__ import annotations

import os
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_var, "1")

from .policygui import main  # noqa: E402


if __name__ == "__main__":
  sys.exit(main())
