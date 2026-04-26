"""CLI entry-point: python -m visualization.model_plots"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from config import Models, DatasetPaths, Datasets
from visualization.model_plots.hypersearch import (
    plot_hyperparameter_search_results,
    plot_parameter_heatmap,
    plot_convergence_analysis,
)
from visualization.model_plots.metrics import plot_metrics_comparison
from visualization.model_plots.alpha import (
    plot_alpha_rmse_analysis,
    plot_alpha_delta_rmse,
    plot_alpha_edges,
)
from visualization.model_plots.ranking import (
    plot_ranking_metrics_per_alpha,
    plot_ranking_metrics_comparison,
)

_PLOT_CHOICES = [
    "alpha-rmse",
    "delta-rmse",
    "alpha-edges",
    "hyperparam",
    "heatmap",
    "metrics",
    "convergence",
    "ranking",
    "ranking-comparison",
    "all",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CMF evaluation plots from saved result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        choices=_PLOT_CHOICES,
        default=["all"],
        metavar="PLOT",
        help="Which plots to generate. Default: all.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=Models.ALL + ["all"],
        default=["all"],
        help="Models for alpha-rmse / delta-rmse plots.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Show plots interactively without saving.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (default: uses Datasets.DEFAULT).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    plots = set(args.plot)
    do_all = "all" in plots
    model_list = Models.ALL if "all" in args.models else args.models
    save = not args.no_save
    dataset = args.dataset

    search_json = DatasetPaths(dataset or Datasets.DEFAULT).BASELINE_RESULTS
    search_result: dict | None = None
    if do_all or plots & {"hyperparam", "metrics", "convergence", "heatmap"}:
        if search_json.exists():
            with open(search_json, encoding="utf-8") as fh:
                search_result = json.load(fh)
            print(f"Loaded search results from {search_json}")
        else:
            print(
                f"Warning: {search_json} not found. "
                "Run 'python -m recommender.baseline' first."
            )

    if search_result and (do_all or "hyperparam" in plots):
        print("\n--- Hyperparameter search overview ---")
        plot_hyperparameter_search_results(
            search_result,
            save_path="hyperparam_search.png" if save else None,
            dataset=dataset,
        )

    if search_result and (do_all or "heatmap" in plots):
        print("\n--- Parameter space heatmap ---")
        plot_parameter_heatmap(
            search_result,
            metric="rmse",
            save_path="heatmap_rmse.png" if save else None,
            dataset=dataset,
        )

    if search_result and (do_all or "metrics" in plots):
        print("\n--- Metrics distribution ---")
        plot_metrics_comparison(
            search_result,
            save_path="metrics_comparison.png" if save else None,
            dataset=dataset,
        )

    if search_result and (do_all or "convergence" in plots):
        print("\n--- Convergence analysis ---")
        plot_convergence_analysis(
            search_result,
            save_path="convergence.png" if save else None,
            dataset=dataset,
        )

    # Global baseline RMSE for alpha plots
    global_baseline_rmse: float | None = None
    if search_json.exists():
        with open(search_json, encoding="utf-8") as bf:
            bs = json.load(bf)
        if "global_test_rmse" in bs:
            global_baseline_rmse = float(bs["global_test_rmse"])
            print(f"Global baseline RMSE: {global_baseline_rmse:.4f}")

    if do_all or plots & {"alpha-rmse", "delta-rmse"}:
        for model_name in model_list:
            short = Models.SHORT[model_name]
            csv = (
                DatasetPaths(dataset or Datasets.DEFAULT).NETWORKS
                / model_name
                / f"inferred_edges_{short}.csv"
            )

            if not csv.exists():
                print(f"Skipping {model_name}: {csv} not found.")
                continue

            df = pd.read_csv(csv, sep="|")

            if "rmse_mean" not in df.columns or bool(df["rmse_mean"].isna().all()):
                print(
                    f"Skipping {model_name}: no rmse_mean data. "
                    "Run enhanced --all first."
                )
                continue

            rmse = df["rmse_mean"].tolist()
            std = (
                df["rmse_std"].tolist()
                if "rmse_std" in df.columns and not bool(df["rmse_std"].isna().all())
                else None
            )

            baseline: float
            if "baseline_rmse_mean" in df.columns and not bool(
                df["baseline_rmse_mean"].isna().all()
            ):
                baseline = float(df["baseline_rmse_mean"].dropna().mean())
            elif "improvement_pct" in df.columns and not bool(
                df["improvement_pct"].isna().all()
            ):
                valid = df.dropna(subset=["rmse_mean", "improvement_pct"])
                baseline = float(
                    (
                        valid["rmse_mean"] / (1.0 - valid["improvement_pct"] / 100.0)
                    ).mean()
                )
            elif global_baseline_rmse is not None:
                baseline = global_baseline_rmse
            else:
                baseline = float(df["rmse_mean"].mean())

            delta_pct: list[float] | None = None
            if "improvement_pct" in df.columns and not bool(
                df["improvement_pct"].isna().all()
            ):
                delta_pct = df["improvement_pct"].fillna(0.0).tolist()

            print(f"\n--- {model_name.upper()} (paired baseline≈{baseline:.4f}) ---")

            if do_all or "alpha-rmse" in plots:
                plot_alpha_rmse_analysis(
                    model_name,
                    rmse,
                    baseline_rmse=baseline,
                    rmse_std_values=std,
                    save_plot=save,
                    global_baseline_rmse=global_baseline_rmse,
                    dataset=dataset,
                )
            if do_all or "delta-rmse" in plots:
                plot_alpha_delta_rmse(
                    model_name,
                    rmse,
                    baseline_rmse=baseline,
                    save_plot=save,
                    delta_pct_values=delta_pct,
                    dataset=dataset,
                )

    if do_all or "alpha-edges" in plots:
        print("\n--- Alpha vs Inferred Edges (all models) ---")
        plot_alpha_edges(save_plot=save, dataset=dataset)

    if do_all or "ranking" in plots:
        print("\n--- Ranking metrics vs Alpha (per model) ---")
        for model_name in model_list:
            plot_ranking_metrics_per_alpha(model_name, save_plot=save, dataset=dataset)

    if do_all or "ranking-comparison" in plots:
        print("\n--- Ranking metrics comparison (all models) ---")
        plot_ranking_metrics_comparison(save_plot=save, dataset=dataset)

    sys.exit(0)


if __name__ == "__main__":
    main()
