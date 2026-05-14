"""Plots for Phase 6 social regularization smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt  # type: ignore[attr-defined]
from matplotlib.ticker import FuncFormatter
import pandas as pd

from config import DatasetPaths, Datasets

MODE_ORDER = [
    "uniform",
    "community_jaccard",
    "boundary_downweight",
    "bridge_preserve",
]

MODE_LABELS = {
    "uniform": "Uniform",
    "community_jaccard": "Community Jaccard",
    "boundary_downweight": "Boundary Downweight",
    "bridge_preserve": "Bridge Preserve",
}

MODE_COLORS = {
    "uniform": "#4C78A8",
    "community_jaccard": "#59A14F",
    "boundary_downweight": "#F28E2B",
    "bridge_preserve": "#B07AA1",
}

MODEL_ORDER = ["exponential", "powerlaw", "rayleigh"]

MODEL_LABELS = {
    "exponential": "Exponential",
    "powerlaw": "Powerlaw",
    "rayleigh": "Rayleigh",
}

MODEL_COLORS = {
    "exponential": "#4C78A8",
    "powerlaw": "#F28E2B",
    "rayleigh": "#59A14F",
}


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one DataFrame column with a precise type for Pyright."""
    return cast(pd.Series, frame[column])


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric column with a precise type for Pyright."""
    return cast(pd.Series, pd.to_numeric(_series(frame, column), errors="coerce"))


def _sorted(frame: pd.DataFrame, columns: str | list[str]) -> pd.DataFrame:
    """Sort a DataFrame while keeping pandas stubs out of call sites."""
    return cast(pd.DataFrame, frame.sort_values(by=columns))


def _metric_formatter(values: pd.Series) -> FuncFormatter:
    """Return a y-axis formatter that avoids offset/scientific notation."""
    value_range = float(values.max() - values.min()) if not values.empty else 0.0
    if value_range < 1e-5:
        decimals = 9
    elif value_range < 1e-3:
        decimals = 6
    else:
        decimals = 4
    return FuncFormatter(lambda value, _position: f"{value:.{decimals}f}")


def load_lambda_sweep_results(results_dir: Path) -> pd.DataFrame:
    """Load lambda-sweep JSON files into a compact plotting DataFrame."""
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        social_metrics = result["lambda_social_on"]
        baseline_metrics = result["lambda_social_0"]
        diagnostics = result["diagnostics"]
        social_edges = result["social_edges"]
        rows.append(
            {
                "mode": result["social_mode"],
                "lambda_social": float(result["lambda_social"]),
                "rmse": float(social_metrics["rmse"]),
                "mae": float(social_metrics["mae"]),
                "r2": float(social_metrics["r2"]),
                "baseline_rmse": float(baseline_metrics["rmse"]),
                "baseline_mae": float(baseline_metrics["mae"]),
                "baseline_r2": float(baseline_metrics["r2"]),
                "rmse_delta": float(result["rmse_delta"]),
                "edges": int(social_edges["n_edges"]),
                "min_weight": float(social_edges["min_weight"]),
                "max_weight": float(social_edges["max_weight"]),
                "baseline_reasonable": bool(
                    diagnostics["lambda_social_0_reasonable_scale"]
                ),
                "social_reasonable": bool(
                    diagnostics["lambda_social_on_reasonable_scale"]
                ),
                "path": str(path),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No lambda-sweep JSON files found in {results_dir}")

    return _sorted(pd.DataFrame(rows), ["mode", "lambda_social"])


def plot_lambda_sweep_metric(
    results: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Plot one metric against lambda_social for every social mode."""
    labels = {"rmse": "RMSE", "mae": "MAE", "r2": "R2"}
    subtitles = {
        "rmse": "Lower is better",
        "mae": "Lower is better",
        "r2": "Higher is better",
    }

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for mode in MODE_ORDER:
        mode_mask = _series(results, "mode") == mode
        subset = _sorted(
            cast(pd.DataFrame, results.loc[mode_mask].copy()), "lambda_social"
        )
        if subset.empty:
            continue
        ax.plot(
            _numeric_series(subset, "lambda_social"),
            _numeric_series(subset, metric),
            marker="o",
            linewidth=2,
            markersize=6,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )

    ax.set_xscale("log")
    ax.set_xlabel("lambda_social")
    ax.set_ylabel(labels[metric])
    ax.set_title(f"Step 2 Lambda Sweep: {labels[metric]}", fontweight="bold")
    ax.yaxis.set_major_formatter(_metric_formatter(_numeric_series(results, metric)))
    ax.text(0.01, 0.98, subtitles[metric], transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_sweep(
    results_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Generate lambda-sweep summary CSV and metric plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_lambda_sweep_results(results_dir)

    summary_path = results_dir / "lambda_sweep_summary.csv"
    results.to_csv(summary_path, index=False)

    plot_lambda_sweep_metric(results, "rmse", output_dir / "social_lambda_rmse.png")
    plot_lambda_sweep_metric(results, "mae", output_dir / "social_lambda_mae.png")
    plot_lambda_sweep_metric(results, "r2", output_dir / "social_lambda_r2.png")
    return results


def load_network_sweep_results(results_dir: Path) -> pd.DataFrame:
    """Load Step 3 network-sweep summary rows for plotting."""
    summary_path = results_dir / "network_sweep_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Network-sweep summary not found: {summary_path}")

    results = pd.read_csv(summary_path)
    if "status" in results.columns:
        status_mask = _series(results, "status") == "ok"
        results = cast(pd.DataFrame, results.loc[status_mask].copy())
    if results.empty:
        raise ValueError(f"No successful network-sweep rows found in {summary_path}")

    numeric_cols = [
        "network_index",
        "lambda_social",
        "rmse",
        "mae",
        "r2",
        "baseline_rmse",
        "baseline_mae",
        "baseline_r2",
        "rmse_delta",
        "edges",
        "min_weight",
        "max_weight",
    ]
    for col in numeric_cols:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")
    for col in ["baseline_reasonable", "social_regularized_reasonable"]:
        if col not in results.columns:
            results[col] = True
        elif results[col].dtype != bool:
            results[col] = (
                _series(results, col).astype(str).str.lower().isin({"true", "1"})
            )
    return _sorted(results, ["model_name", "network_index"])


def plot_network_sweep_metric(
    results: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Plot one Step 3 metric across sampled network indices."""
    labels = {"rmse": "RMSE", "mae": "MAE", "r2": "R2"}
    subtitles = {
        "rmse": "Lower is better",
        "mae": "Lower is better",
        "r2": "Higher is better",
    }

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for model_name in MODEL_ORDER:
        model_mask = _series(results, "model_name") == model_name
        subset = _sorted(
            cast(pd.DataFrame, results.loc[model_mask].copy()),
            "network_index",
        )
        if subset.empty:
            continue
        ax.plot(
            _numeric_series(subset, "network_index"),
            _numeric_series(subset, metric),
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )

    ax.set_xlabel("Sampled network index")
    ax.set_ylabel(labels[metric])
    ax.set_title(f"Step 3 Network Sweep: {labels[metric]}", fontweight="bold")
    ax.yaxis.set_major_formatter(_metric_formatter(_numeric_series(results, metric)))
    ax.text(0.01, 0.98, subtitles[metric], transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_network_sweep(
    results_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Generate Step 3 network-sweep aggregate CSV and metric plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_network_sweep_results(results_dir)
    social_mask = _series(results, "social_regularized_reasonable").astype(bool)
    metric_results = cast(pd.DataFrame, results.loc[social_mask].copy())
    baseline_mask = _series(metric_results, "baseline_reasonable").astype(bool)
    delta_results = cast(pd.DataFrame, metric_results.loc[baseline_mask].copy())
    if metric_results.empty:
        raise ValueError(
            "No rating-scale sane social-regularized network-sweep rows found."
        )

    metric_aggregate = (
        metric_results.groupby("model_name")[["rmse", "mae", "r2", "edges"]]
        .agg(["mean", "std", "min", "max"])
        .round(9)
    )
    metric_aggregate.columns = [
        f"{metric}_{stat}" for metric, stat in metric_aggregate.columns
    ]

    delta_aggregate = (
        delta_results.groupby("model_name")[["rmse_delta"]]
        .agg(["mean", "std", "min", "max"])
        .round(9)
    )
    delta_aggregate.columns = [
        f"{metric}_{stat}" for metric, stat in delta_aggregate.columns
    ]

    counts = (
        pd.DataFrame(
            {
                "sampled_runs": results.groupby("model_name").size(),
                "metric_runs": metric_results.groupby("model_name").size(),
                "delta_runs": delta_results.groupby("model_name").size(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    aggregate = (
        counts.join(metric_aggregate, how="left")
        .join(delta_aggregate, how="left")
        .reset_index()
    )
    aggregate.to_csv(results_dir / "network_sweep_by_model.csv", index=False)

    plot_network_sweep_metric(
        metric_results, "rmse", output_dir / "social_network_rmse.png"
    )
    plot_network_sweep_metric(
        metric_results, "mae", output_dir / "social_network_mae.png"
    )
    plot_network_sweep_metric(
        metric_results, "r2", output_dir / "social_network_r2.png"
    )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Phase 6 social regularization plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=Datasets.DEFAULT, choices=Datasets.ALL)
    parser.add_argument(
        "--plot-kind",
        default="lambda",
        choices=["lambda", "network", "all"],
        help="Which Phase 6 plots to generate.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing results for the selected plot kind.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where plots should be written.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    dataset_paths = DatasetPaths(args.dataset)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else dataset_paths.PLOTS / "models" / "social_regularization"
    )

    if args.plot_kind in {"lambda", "all"}:
        lambda_results_dir = (
            Path(args.results_dir)
            if args.results_dir is not None and args.plot_kind == "lambda"
            else dataset_paths.BASE / "social_smoke_results" / "lambda_sweep"
        )
        lambda_results = plot_lambda_sweep(lambda_results_dir, output_dir)
        print(
            f"Loaded {len(lambda_results)} lambda-sweep runs from {lambda_results_dir}"
        )
        print(f"Saved summary to {lambda_results_dir / 'lambda_sweep_summary.csv'}")

    if args.plot_kind in {"network", "all"}:
        network_results_dir = (
            Path(args.results_dir)
            if args.results_dir is not None and args.plot_kind == "network"
            else dataset_paths.BASE / "social_smoke_results" / "network_sweep"
        )
        network_results = plot_network_sweep(network_results_dir, output_dir)
        print(
            f"Loaded {len(network_results)} network-sweep runs from {network_results_dir}"
        )
        print(f"Saved summary to {network_results_dir / 'network_sweep_by_model.csv'}")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
