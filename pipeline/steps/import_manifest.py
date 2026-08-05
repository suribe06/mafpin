"""Import hyperparameters and CV network winners from existing recommend logs."""

from __future__ import annotations

import argparse

from recommender.experiment.manifest import import_manifest_from_logs


def run_import_manifest(args: argparse.Namespace) -> None:
    variant_ids = [args.model_variant] if args.model_variant else None
    import_manifest_from_logs(
        args.dataset,
        variant_ids=variant_ids,
        all_variants=bool(args.all_variants),
    )
