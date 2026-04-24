"""
Parallel worker functions for ProcessPoolExecutor-based network evaluation.

These functions MUST live at module level to be picklable by multiprocessing.

IMPORTANT: _eval_network_worker uses a LAZY import of evaluate_single_network
to avoid the circular import that would arise if network_eval.py (which imports
_eval_network_worker at module level) were imported here at module level.
"""

from __future__ import annotations

import pandas as pd

# Module-level cache populated once per worker process via the pool initializer.
# This avoids pickling the full ratings DataFrame once per task (~100× per run).
_WORKER_DATA: "pd.DataFrame | None" = None
_WORKER_SHARED: "dict | None" = None


def _worker_init(data: "pd.DataFrame", shared_kwargs: dict) -> None:
    """Pool initializer: cache the shared data/kwargs in the worker process.

    Also pins BLAS libraries (OpenBLAS, MKL, OMP) to a single thread per
    worker to prevent OpenMP thread oversubscription.
    """
    import os
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:
        pass

    global _WORKER_DATA, _WORKER_SHARED  # noqa: PLW0603
    _WORKER_DATA = data
    _WORKER_SHARED = shared_kwargs


def _eval_network_worker(packed: "tuple[int, dict]") -> "tuple[int, list[dict]]":
    """Top-level worker for :class:`concurrent.futures.ProcessPoolExecutor`.

    Receives a ``(network_index, extra_kwargs)`` tuple; the heavy ``data``
    DataFrame and other shared kwargs are read from the per-worker module
    cache populated by :func:`_worker_init`.

    Uses a lazy import of evaluate_single_network to avoid a circular import
    between workers.py and network_eval.py.
    """
    # Lazy import to break the circular dependency:
    # network_eval.py imports _eval_network_worker from workers.py at module
    # level, so we must NOT import from network_eval.py at module level here.
    from recommender.enhanced.network_eval import (
        evaluate_single_network,
    )  # noqa: PLC0415

    net_idx, extra = packed
    if _WORKER_DATA is not None and _WORKER_SHARED is not None:
        kwargs = {**_WORKER_SHARED, **extra, "data": _WORKER_DATA}
    else:
        kwargs = extra
    return net_idx, evaluate_single_network(**kwargs)
