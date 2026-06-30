"""Core experiment helpers: manifest, network selection, final test evaluation."""

from recommender.experiment.final_eval import evaluate_variant_global_test
from recommender.experiment.manifest import (
    archive_recommend_run,
    build_manifest_from_logs,
    load_manifest,
    save_manifest,
)
from recommender.experiment.network_selection import run_network_selection
from recommender.experiment.variants import ALL_VARIANT_IDS, VARIANT_SPECS, variant_cli_flags

__all__ = [
    "ALL_VARIANT_IDS",
    "VARIANT_SPECS",
    "archive_recommend_run",
    "build_manifest_from_logs",
    "evaluate_variant_global_test",
    "load_manifest",
    "run_network_selection",
    "save_manifest",
    "variant_cli_flags",
]
