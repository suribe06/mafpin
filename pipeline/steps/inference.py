"""Network inference step (NetInf)."""

from __future__ import annotations

import argparse

from config import Defaults
from pipeline._artifacts import _check_artifact_manifest
from pipeline._cpu import _resolve_n_jobs


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
    _check_artifact_manifest(args.dataset, context="network inference")
    model = args.model
    n_jobs = _resolve_n_jobs(args)
    print(f"NetInf parallel workers: {n_jobs}")
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
            n_jobs=n_jobs,
        )
    else:
        infer_networks_all_models(
            n=args.n_alphas,
            max_iter=args.max_iter,
            k_avg_degree=args.k_avg_degree if args.k_avg_degree > 0 else None,
            networks_dir=dp.NETWORKS,
            cascades_file=dp.CASCADES,
            n_jobs=n_jobs,
        )

    from visualization.model_plots import plot_alpha_edges

    plot_alpha_edges(save_plot=True, dataset=args.dataset)
