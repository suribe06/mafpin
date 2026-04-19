"""
Network-level visualizations: centrality metric distributions and cascade timelines.

Example usage
-------------
>>> from visualization.network_plots import plot_cascades_timeline, plot_centrality_distribution
>>> plot_cascades_timeline(n=20, save=True)
>>> plot_centrality_distribution("degree", "exponential", "001", save=True)
"""

from __future__ import annotations

import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from config import Models, DatasetPaths, Datasets


# ---------------------------------------------------------------------------
# Centrality metric distributions
# ---------------------------------------------------------------------------

METRIC_COLUMNS = [
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
    "clustering",
    "eccentricity",
]


def plot_centrality_distribution(
    metric: str,
    model_name: str,
    network_id: str,
    save: bool = True,
    dataset: str | None = None,
) -> None:
    """
    Histogram + KDE for a single centrality metric of one network.

    Args:
        metric:      Metric column name (see :data:`METRIC_COLUMNS`).
        model_name:  Diffusion model name, e.g. ``"exponential"``.
        network_id:  Zero-padded network index, e.g. ``"001"``.
        save:        Write PNG to ``data/centrality_metrics/<model>/plots/``.
    """
    csv_path = (
        DatasetPaths(dataset or Datasets.DEFAULT).CENTRALITY
        / model_name
        / f"centrality_metrics_{Models.SHORT.get(model_name, model_name)}_{network_id}.csv"
    )
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        print(f"Metric '{metric}' not found. Available: {list(df.columns)}")
        return

    data = np.asarray(df[metric].dropna(), dtype=float)
    positive = data[data > 0]
    if len(positive) == 0:
        print(f"No positive values for metric '{metric}'.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(positive, kde=True, color="darkblue", alpha=0.7)  # type: ignore[arg-type]
    plt.yscale("log")
    plt.xlabel(metric.capitalize())
    plt.title(
        f"{metric.capitalize()} Distribution — "
        f"{model_name.capitalize()} (Network {network_id})"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        out_dir = (
            DatasetPaths(dataset or Datasets.DEFAULT).PLOTS / "centrality" / model_name
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{metric}_{model_name}_{network_id}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close()


def plot_all_centrality_distributions(
    model_name: str,
    network_id: str,
    save: bool = True,
    dataset: str | None = None,
) -> None:
    """
    Seven-panel grid showing all centrality distributions for one network.

    Args:
        model_name:  Diffusion model name.
        network_id:  Zero-padded network index, e.g. ``"001"``.
        save:        Write PNG to ``plots/centrality/<model>/``.
    """
    csv_path = (
        DatasetPaths(dataset or Datasets.DEFAULT).CENTRALITY
        / model_name
        / f"centrality_metrics_{Models.SHORT.get(model_name, model_name)}_{network_id}.csv"
    )
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    available = [m for m in METRIC_COLUMNS if m in df.columns]
    if not available:
        print("No recognisable metric columns found.")
        return

    ncols = 4
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
    fig.suptitle(
        f"Centrality Distributions — {model_name.capitalize()} (Network {network_id})",
        fontsize=14,
        fontweight="bold",
    )

    for ax, metric in zip(axes_flat, available):
        data = np.asarray(df[metric].dropna(), dtype=float)
        positive = data[data > 0]
        if len(positive) == 0:
            ax.set_title(metric)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        sns.histplot(positive, kde=True, ax=ax, color="steelblue", alpha=0.7)  # type: ignore[arg-type]
        ax.set_yscale("log")
        ax.set_title(metric.capitalize())
        ax.grid(alpha=0.3)

    # Hide unused axes
    for ax in axes_flat[len(available) :]:
        ax.set_visible(False)

    plt.tight_layout()
    if save:
        out_dir = (
            DatasetPaths(dataset or Datasets.DEFAULT).PLOTS / "centrality" / model_name
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"all_centrality_{model_name}_{network_id}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Cascade timeline
# ---------------------------------------------------------------------------


def plot_cascades_timeline(
    cascade_file: str | None = None,
    n: int = 20,
    save: bool = True,
    dataset: str | None = None,
) -> None:
    """
    Plot the first *n* cascades as event timelines.

    Each cascade is drawn as a horizontal row of points at the event timestamps,
    ordered by time.

    Args:
        cascade_file:  Absolute path to the cascades file.  Defaults to
                       :data:`config.Paths.CASCADES`.
        n:             Number of cascades to show.
        save:          Write PNG to ``plots/cascades_timeline.png``.
    """
    from pathlib import Path as _Path

    path = (
        _Path(cascade_file) if cascade_file else DatasetPaths(Datasets.DEFAULT).CASCADES
    )
    if not path.exists():
        print(f"Cascade file not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    # Locate the first line that has cascade data (>2 comma-separated values)
    start_idx = 0
    for idx, line in enumerate(lines):
        if len(line.strip().split(",")) > 2:
            start_idx = idx
            break

    cascade_lines = lines[start_idx : start_idx + n]

    plt.figure(figsize=(14, max(6, n // 3)))
    for c_idx, line in enumerate(cascade_lines, 1):
        parts = line.strip().split(",")
        try:
            pairs = [
                (int(parts[i]), float(parts[i + 1]))
                for i in range(0, len(parts) - 1, 2)
            ]
        except (ValueError, IndexError):
            continue
        pairs.sort(key=lambda x: x[1])
        _, times = zip(*pairs)
        dates = np.array(
            [datetime.datetime.fromtimestamp(t) for t in times], dtype="datetime64[s]"
        )
        plt.plot(dates, [c_idx] * len(dates), "o-", linewidth=0.8, markersize=4)

    plt.xlabel("Date")
    plt.ylabel("Cascade index")
    plt.title(f"First {n} cascade timelines")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        out_dir = DatasetPaths(dataset or Datasets.DEFAULT).PLOTS
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cascades_timeline.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    _parser = argparse.ArgumentParser(
        description="Generate network and cascade visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _sub = _parser.add_subparsers(dest="command", required=True)

    # -- cascades-timeline ---------------------------------------------------
    _p_cas = _sub.add_parser("cascades", help="Plot cascade timelines.")
    _p_cas.add_argument("--n", type=int, default=30, help="Number of cascades to show.")
    _p_cas.add_argument("--no-save", action="store_true", help="Show without saving.")

    # -- centrality ----------------------------------------------------------
    _p_cen = _sub.add_parser("centrality", help="Plot centrality distributions.")
    _p_cen.add_argument(
        "--model",
        choices=Models.ALL,
        required=True,
        help="Diffusion model name.",
    )
    _p_cen.add_argument(
        "--network", default="000", help="Network index string (e.g. '000')."
    )
    _p_cen.add_argument(
        "--metric",
        choices=METRIC_COLUMNS + ["all"],
        default="all",
        help="Centrality metric to plot, or 'all' for the full grid.",
    )
    _p_cen.add_argument("--no-save", action="store_true", help="Show without saving.")

    _args = _parser.parse_args()

    if _args.command == "cascades":
        plot_cascades_timeline(n=_args.n, save=not _args.no_save)

    elif _args.command == "centrality":
        if _args.metric == "all":
            plot_all_centrality_distributions(
                _args.model, _args.network, save=not _args.no_save
            )
        else:
            plot_centrality_distribution(
                _args.metric, _args.model, _args.network, save=not _args.no_save
            )

    sys.exit(0)
