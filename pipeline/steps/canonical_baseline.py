"""Establish a single canonical M1 baseline reused by final_eval."""

from __future__ import annotations

import argparse

from pipeline._cpu import _resolve_cmf_nthreads
from recommender.experiment import final_eval as final_eval_mod


def run_canonical_baseline(args: argparse.Namespace) -> None:
    final_eval_mod.run_canonical_baseline(
        args.dataset,
        force=bool(getattr(args, "force", False)),
        cmf_method=args.cmf_method,
        cmf_maxiter=args.cmf_maxiter,
        cmf_nthreads=_resolve_cmf_nthreads(args),
        random_state=args.seed,
    )
