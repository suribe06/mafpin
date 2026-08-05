"""Global held-out test evaluation for core experiment variants."""

from __future__ import annotations

import argparse

from pipeline._cpu import _resolve_cmf_nthreads
from recommender.experiment import final_eval as final_eval_mod


def run_final_eval(args: argparse.Namespace) -> None:
    variant_ids = [args.model_variant] if args.model_variant else None
    final_eval_mod.run_final_eval(
        args.dataset,
        variant_ids=variant_ids,
        all_variants=bool(args.all_variants),
        cmf_method=args.cmf_method,
        cmf_maxiter=args.cmf_maxiter,
        cmf_nthreads=_resolve_cmf_nthreads(args),
        random_state=args.seed,
    )
