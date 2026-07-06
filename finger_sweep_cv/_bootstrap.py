"""Zero-dependency venv relaunch guard.

Import and call ensure_venv() as the very first statement -- before any
third-party import (cv2, numpy, matplotlib, ...) -- in every entry-point
script in this folder. If the current interpreter isn't the dedicated
finger_sweep_cv\\.venv, this transparently re-execs the same script (same
argv) under that venv's python, so it doesn't matter whether the script was
launched via an IDE's Run button, a raw double-click, or a bare
`python foo.py` in an unactivated shell.

This module must stay stdlib-only: it runs before we know we're in the
right interpreter, so it can't rely on anything from requirements.txt.
"""

import os
import subprocess
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_PYTHON = os.path.join(_SCRIPTS_DIR, ".venv", "Scripts", "python.exe")


def ensure_venv():
    if not os.path.isfile(_VENV_PYTHON):
        return  # venv missing -- let the caller hit the real ImportError downstream
    current = os.path.normcase(os.path.abspath(sys.executable))
    target = os.path.normcase(os.path.abspath(_VENV_PYTHON))
    if current == target:
        return  # already running under the venv

    # Relaunch under the venv's python as a child process rather than
    # os.execv: execv-style process-image replacement is unreliable on
    # Windows (observed to mis-handle paths containing spaces in this
    # environment), so spawn-and-wait is the more portable choice here.
    result = subprocess.run([_VENV_PYTHON, os.path.abspath(sys.argv[0]), *sys.argv[1:]])
    sys.exit(result.returncode)
