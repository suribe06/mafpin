"""Import hyperparameters and CV network winners from existing recommend logs."""

from __future__ import annotations

import argparse

from config import DatasetPaths
from recommender.experiment.manifest import build_manifest_from_logs, save_manifest
from recommender.experiment.variants import CORE_VARIANT_IDS, VARIANT_SPECS


def run_import_manifest(args: argparse.Namespace) -> None:
    dp = DatasetPaths(args.dataset)
    log_dir = dp.LOGS
    if not log_dir.exists():
        print(f"WARNING: log directory missing: {log_dir}")

    variant_ids = None
    if args.model_variant:
        variant_ids = [args.model_variant]
    elif args.all_variants:
        variant_ids = [v for v in VARIANT_SPECS if v != "M1"]
    else:
        variant_ids = [v for v in CORE_VARIANT_IDS if v != "M1"]

    manifest = build_manifest_from_logs(
        args.dataset,
        log_dir=log_dir,
        variant_ids=variant_ids,
    )
    n = len(manifest.get("variants", {}))
    print(f"Imported {n} variant(s) from {log_dir}")
    if n == 0:
        print(
            "No logs found. Expected files like data/<dataset>/logs/m3_recommend.log"
        )
    save_manifest(manifest, dp.EXPERIMENT_MANIFEST)
