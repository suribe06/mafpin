"""Local cmfrec import bridge."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CMFREC_ROOT = PROJECT_ROOT / "cmfrec-master"

if not (LOCAL_CMFREC_ROOT / "cmfrec" / "__init__.py").exists():
    raise ModuleNotFoundError(
        f"Local cmfrec source tree not found at {LOCAL_CMFREC_ROOT}"
    )

local_path = str(LOCAL_CMFREC_ROOT)
if local_path in sys.path:
    sys.path.remove(local_path)
sys.path.insert(0, local_path)

if TYPE_CHECKING:

    class CMF:
        """Typing shim for the dynamically imported local cmfrec.CMF class."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...

        def __getattr__(self, name: str) -> Any: ...

        def fit(self, *_args: Any, **_kwargs: Any) -> CMF: ...

        def predict(self, *_args: Any, **_kwargs: Any) -> Any: ...

else:
    try:
        from cmfrec import CMF  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Local cmfrec is present but its Cython extensions are not built. "
            "Run `cd cmfrec-master && conda run -n mafpin python setup.py build_ext --inplace`."
        ) from exc

__all__ = ["CMF"]
