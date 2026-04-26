"""
Ranking metric visualizations (NDCG@K, Precision@K, Recall@K, MRR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Models, DatasetPaths, Datasets
from visualization.model_plots._common import _plots_dir

_RANKING_METRICS = [
    ("ndcg_at_k", "NDCG@K", "#4C72B0"),
    ("precision_at_k", "Precision@K", "#DD8452"),
    ("recall_at_k", "Recall@K", "#55A868"),
    ("mrr", "MRR", "#C44E52"),
]


def plot_ranking_metrics_per_alpha(
    model_name: str,
    save_plot: bool = True,
    figsize: tuple = (14, 5),
    dataset: str | None = None,
) -> None:
    """
    Four-panel line plot of ranking metrics vs alpha for one diffusion model.

    Args:
        model_name: Diffusion model name.
        save_plot:  Write PNG to ``plots/models/``.
        figsize:    Figure size.
        dataset:    Dataset name.
    """
    short = Models.SHORT[model_name]
    csv = (
        DatasetPaths(dataset or Datasets.DEFAULT).NETWORKS
        / model_name
        / f"inferred_edges_{short}.csv"
    )
    if not csv.exists():
        print(f"plot_ranking_metrics_per_alpha: {csv} not found.")
        return

    df = pd.read_csv(csv, sep="|")
    if "alpha" not in df.columns:
        print(f"plot_ranking_metrics_per_alpha: 'alpha' column missing in {csv}.")
        return

    available = [
        (col, label, colour)
        for col, label, colour in _RANKING_METRICS
        if col in df.columns and not bool(df[col].isna().all())
    ]
    if not available:
        print(
            f"plot_ranking_metrics_per_alpha: no ranking metrics found in {csv}. "
            "Run 'python -m recommender.enhanced --all --ranking' first."
        )
        return

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(figsize[0], figsize[1]))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f"Ranking Metrics vs Alpha — {model_name.capitalize()} Model",
        fontsize=14,
        fontweight="bold",
    )

    plot_df = df.dropna(subset=[col for col, _, _ in available]).sort_values("alpha")
    for ax, (col, label, colour) in zip(axes, available):
        ax.scatter(plot_df["alpha"], plot_df[col], color=colour, s=30, alpha=0.75)
        ax.plot(plot_df["alpha"], plot_df[col], color=colour, linewidth=1.2, alpha=0.5)
        ax.set_xlabel("Alpha", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout()
    if save_plot:
        full_path = f"{_plots_dir(dataset)}/ranking_metrics_alpha_{model_name}.png"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.close()


def plot_ranking_metrics_comparison(
    save_plot: bool = True,
    figsize: tuple = (14, 5),
    dataset: str | None = None,
) -> None:
    """
    Grouped bar chart comparing mean NDCG@K, Precision@K, Recall@K, and MRR
    across the three diffusion models.

    Args:
        save_plot: Write PNG to ``plots/models/``.
        figsize:   Figure size.
        dataset:   Dataset name.
    """
    _MODEL_CFG = [
        ("exponential", "expo", "#E07B54"),
        ("powerlaw", "power", "#5B8DB8"),
        ("rayleigh", "ray", "#6DBF82"),
    ]

    model_means: dict[str, dict[str, float]] = {}
    for model_name, short, _ in _MODEL_CFG:
        csv = (
            DatasetPaths(dataset or Datasets.DEFAULT).NETWORKS
            / model_name
            / f"inferred_edges_{short}.csv"
        )
        if not csv.exists():
            continue
        df = pd.read_csv(csv, sep="|")
        model_means[model_name] = {}
        for col, _, _ in _RANKING_METRICS:
            if col in df.columns and not bool(df[col].isna().all()):
                model_means[model_name][col] = float(df[col].mean(skipna=True))

    if not model_means:
        print(
            "plot_ranking_metrics_comparison: no ranking data found. "
            "Run 'python -m recommender.enhanced --all --ranking' first."
        )
        return

    present_metrics = [
        (col, label, colour)
        for col, label, colour in _RANKING_METRICS
        if any(col in m for m in model_means.values())
    ]
    if not present_metrics:
        print("plot_ranking_metrics_comparison: no ranking metrics available.")
        return

    n_metrics = len(present_metrics)
    n_models = len(model_means)
    bar_width = 0.25
    x = np.arange(n_metrics)

    _, ax = plt.subplots(figsize=figsize)
    for i, (model_name, _, colour) in enumerate(_MODEL_CFG):
        if model_name not in model_means:
            continue
        values = [
            model_means[model_name].get(col, float("nan"))
            for col, _, _ in present_metrics
        ]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=model_name.capitalize(),
            color=colour,
            alpha=0.85,
            edgecolor="white",
        )
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in present_metrics], fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Ranking Metrics Comparison — Enhanced CMF by Diffusion Model",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_plot:
        full_path = f"{_plots_dir(dataset)}/ranking_metrics_comparison.png"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.close()
