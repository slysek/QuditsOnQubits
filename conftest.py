"""Pytest configuration shared across the test suite.

Without this file, pytest's default rootdir-based import logic does not place
the workspace root on ``sys.path``, so test modules cannot import top-level
modules such as ``find_best_diagonal_decomposition`` (the standalone script
at the workspace root) or top-level packages like ``QuditsOnQubits`` and
``encoding_search_v2``. We insert the directory containing this conftest at
position 0 of ``sys.path`` so all tests can rely on a consistent import
environment regardless of where pytest is invoked from.
"""

from __future__ import annotations

import os
import sys

_WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)
