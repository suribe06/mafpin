"""MAFPIN pipeline package.

Public API
----------
main        Entry point — equivalent to running ``python pipeline.py``.
STEPS       Ordered dict mapping step name → (description, callable).
ALL_STEPS   Ordered list of all step names.
"""

from __future__ import annotations

from pipeline._runner import ALL_STEPS, STEPS, main

__all__ = ["main", "STEPS", "ALL_STEPS"]
