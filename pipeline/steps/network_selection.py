"""Freeze diffusion model + alpha per variant from CV metrics."""

from __future__ import annotations

import argparse

from recommender.experiment.manifest import load_manifest
from recommender.experiment.network_selection import run_network_selection
from recommender.experiment.variants import CORE_VARIANT_IDS


def run_network_selection_step(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.dataset)
    variant_ids = None
    if args.model_variant:
        variant_ids = [args.model_variant]
    elif not args.all_variants:
        variant_ids = [v for v in CORE_VARIANT_IDS if v != "M1"]

    print("Selecting best network per variant …")
    run_network_selection(
        args.dataset,
        variant_ids=variant_ids,
        manifest=manifest,
    )
