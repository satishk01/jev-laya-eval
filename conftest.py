"""Shared test setup.

DeepEval's end-of-run results table prints Unicode marks (⚠, ✓) which crash
with UnicodeEncodeError on a cp1252 Windows console. Reconfiguring stdout and
stderr for UTF-8 here -- at conftest import, long before the table prints --
removes the need for `PYTHONUTF8=1`. Streams that can't be reconfigured
(capture wrappers, pipes) are left alone.
"""

import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
