#!/usr/bin/env python3
"""Re-exec policygui with numerical thread limits in the process environment.

Some numerical runtimes inspect their environment before Python module code
runs. A normal wrapper that assigns os.environ and then imports the GUI is not
early enough on this workstation. Re-exec makes the limited environment the
initial environment of the process, exactly like prefixing the shell command.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
  if os.environ.get("POLICYGUI_LIMITED") != "1":
    env = dict(os.environ)
    env["POLICYGUI_LIMITED"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
      env[name] = "1"
    os.execvpe(sys.executable, [
      sys.executable, "-m", "hardware.deploy.policygui", *sys.argv[1:]
    ], env)
  from .policygui import main as gui_main
  return gui_main()


if __name__ == "__main__":
  sys.exit(main())
