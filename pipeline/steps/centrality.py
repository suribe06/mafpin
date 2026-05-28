"""Centrality computation step."""

from __future__ import annotations

import argparse

from pipeline._artifacts import _check_artifact_manifest


def run_centrality(args: argparse.Namespace) -> None:
    from config import Models, DatasetPaths, SideUserFeatures
    from networks.centrality import calculate_centrality_for_all_models
    from visualization.network_plots import plot_all_centrality_distributions

    _check_artifact_manifest(args.dataset, context="centrality computation")

    # Warn if pagerank_lph is enabled but community files are missing (Issue 19).
    if SideUserFeatures.FEATURES.get("pagerank_lph", False):
        _dp = DatasetPaths(args.dataset)
        _communities_exist = any(
            (_dp.COMMUNITIES / _mn).exists()
            and any((_dp.COMMUNITIES / _mn).glob(f"communities_{_mn}_*.csv"))
            for _mn in Models.ALL
        )
        if not _communities_exist:
            print(
                "  WARNING: pagerank_lph is enabled in SideUserFeatures.FEATURES "
                "but no community CSV files were found under "
                f"{_dp.COMMUNITIES}. The pagerank_lph column will be set to NaN "
                "for all networks. Run --steps communities first to compute LPH."
            )

    calculate_centrality_for_all_models(
        dataset=args.dataset if hasattr(args, "dataset") else None
    )

    for _mn in Models.ALL:
        plot_all_centrality_distributions(_mn, "000", save=True, dataset=args.dataset)
