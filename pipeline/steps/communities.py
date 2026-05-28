"""Community detection step."""

from __future__ import annotations

import argparse

from pipeline._artifacts import _check_artifact_manifest


def run_communities(args: argparse.Namespace) -> None:
    from networks.communities import calculate_communities_for_all_models
    from visualization.community_plots import (
        plot_alpha_vs_lph,
        plot_alpha_vs_num_communities,
        plot_community_correlation_heatmap,
        plot_lph_distribution,
        plot_lph_vs_centrality,
        plot_num_communities_dist,
    )

    _check_artifact_manifest(args.dataset, context="community detection")

    calculate_communities_for_all_models(
        dataset=args.dataset if hasattr(args, "dataset") else None
    )

    plot_lph_distribution(dataset=args.dataset)
    plot_num_communities_dist(dataset=args.dataset)
    plot_alpha_vs_lph(dataset=args.dataset)
    plot_alpha_vs_num_communities(dataset=args.dataset)
    plot_lph_vs_centrality(dataset=args.dataset)
    plot_community_correlation_heatmap(dataset=args.dataset)
