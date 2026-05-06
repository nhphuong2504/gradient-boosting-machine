"""Pytest bootstrap that makes the local ``src/`` layout importable.

Adding ``src/`` to ``sys.path`` here means tests and experiment scripts can
``import treeboost`` without installing the package, which keeps the
project simple to run from a fresh clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
