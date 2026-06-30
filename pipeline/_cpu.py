"""CPU / thread-limit utilities."""

from __future__ import annotations

import argparse
import os


def _cpu_thread_limit(cpu_fraction: float) -> int:
    cpu_count = os.cpu_count() or 1
    safe_fraction = min(max(cpu_fraction, 0.05), 1.0)
    return max(1, int(cpu_count * safe_fraction))


def _resolve_cmf_nthreads(args: argparse.Namespace) -> int:
    explicit = getattr(args, "cmf_nthreads", 0)
    if explicit and explicit > 0:
        nthreads = int(explicit)
    else:
        nthreads = _cpu_thread_limit(float(args.cpu_fraction))
    # ponytail: cmfrec L-BFGS segfaults with nthreads>1 on typical OpenBLAS builds;
    # parallelize with --n-jobs (network workers), not BLAS inside each fit.
    if getattr(args, "cmf_method", "lbfgs") == "lbfgs" and nthreads != 1:
        if explicit and explicit > 1:
            print(
                f"WARNING: cmfrec L-BFGS is not thread-safe above 1; "
                f"capping CMF nthreads from {nthreads} to 1."
            )
        nthreads = 1
    return nthreads


def _configure_cpu_limits(args: argparse.Namespace) -> int:
    nthreads = _resolve_cmf_nthreads(args)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(nthreads)
    if getattr(args, "cmf_method", "lbfgs") == "lbfgs":
        print(
            "CPU limit: CMF/BLAS threads capped at 1 "
            "(cmfrec L-BFGS; use --n-jobs for parallel network evaluation)."
        )
    else:
        print(
            f"CPU limit: CMF/BLAS threads capped at {nthreads} "
            f"(~{float(args.cpu_fraction):.0%} of detected cores)."
        )
    return nthreads
