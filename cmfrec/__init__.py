"""Compatibility shim for the local cmfrec source tree.

The real cmfrec package lives under ``cmfrec-master/cmfrec`` so its C/Cython
sources can be patched and rebuilt in place. This shim keeps normal imports
working from the repository root:

    import cmfrec
    from cmfrec import CMF
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

VENDOR_PACKAGE = Path(__file__).resolve().parents[1] / "cmfrec-master" / "cmfrec"
VENDOR_INIT = VENDOR_PACKAGE / "__init__.py"

if not VENDOR_INIT.exists():
    raise ModuleNotFoundError(f"Local cmfrec package not found at {VENDOR_PACKAGE}")

__file__ = str(VENDOR_INIT)
__path__ = [str(VENDOR_PACKAGE)]

vendor_spec = importlib.util.spec_from_file_location(
    "_mafpin_local_cmfrec",
    VENDOR_INIT,
    submodule_search_locations=[str(VENDOR_PACKAGE)],
)
if vendor_spec is None or vendor_spec.loader is None:
    raise ImportError(f"Unable to load local cmfrec package from {VENDOR_INIT}")

vendor_module = importlib.util.module_from_spec(vendor_spec)
sys.modules[vendor_spec.name] = vendor_module
vendor_spec.loader.exec_module(vendor_module)

for name, value in vars(vendor_module).items():
    if name not in {"__name__", "__package__", "__loader__", "__spec__"}:
        globals()[name] = value
