"""Network inference step (NetInf)."""

from __future__ import annotations

import argparse

from config import Defaults


def run_inference(args: argparse.Namespace) -> None:
    from config import DatasetPaths, Datasets
    from networks.delta import (
        compute_median_delta,
        alpha_centers_from_delta,
        log_alpha_grid,
    )
    from networks.inference import infer_networks_all_models, infer_networks

    dp = DatasetPaths(
        args.dataset if hasattr(args, "dataset") and args.dataset else Datasets.DEFAULT
    )
    model = args.model
    model_index_map = {"exponential": 0, "powerlaw": 1, "rayleigh": 2}
    if model:
        _ = compute_median_delta(dp.CASCADES)  # kept for reference
        _ = alpha_centers_from_delta(_)  # kept for reference
        _ = log_alpha_grid  # kept for reference
        infer_networks(
            cascades_file=dp.CASCADES,
            n=args.n_alphas,
            model=model_index_map[model],
            max_iter=args.max_iter,
            k_avg_degree=args.k_avg_degree if args.k_avg_degree > 0 else None,
            name_output=str(dp.NETWORKS / model),
            r=Defaults.RANGE_R,
            networks_dir=dp.NETWORKS,
        )
    else:
        infer_networks_all_models(
            n=args.n_alphas,
            max_iter=args.max_iter,
            k_avg_degree=args.k_avg_degree if args.k_avg_degree > 0 else None,
            networks_dir=dp.NETWORKS,
            cascades_file=dp.CASCADES,
        )

    from visualization.model_plots import plot_alpha_edges

    plot_alpha_edges(save_plot=True, dataset=args.dataset)
