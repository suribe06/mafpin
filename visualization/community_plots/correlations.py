"""
Alpha–LPH and centrality correlation plots.
"""

from __future__ import annotations

import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from visualization.community_plots.loaders import (
    MODELS,
    PALETTE,
    _plots_dir,
    _load_community_csv,
    _load_centrality_csv,
    _load_alpha_csv,
    _aggregate_community_stats,
)


def plot_alpha_vs_lph(
    n_networks: int = 100, save: bool = True, dataset: str | None = None
) -> None:
    """
    3-row × 2-column grid showing how the alpha parameter affects LPH.

    Row per model; left = mean ± std, right = median.

    Args:
        n_networks: Number of networks aggregated per model.
        save:       If ``True``, write PNG to ``plots/communities/``.
        dataset:    Dataset name.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
    fig.suptitle(
        "Effect of Alpha on Local Pluralistic Homophily",
        fontsize=15,
        fontweight="bold",
    )

    for row, model in enumerate(MODELS):
        alpha_df = _load_alpha_csv(model, dataset=dataset)
        stats_df = _aggregate_community_stats(model, n_networks, dataset=dataset)
        col = PALETTE[model]
        ax_mean = axes[row, 0]
        ax_med = axes[row, 1]

        if alpha_df is not None and not stats_df.empty:
            merged = (
                alpha_df.merge(stats_df, on="network_index", how="inner")
                .query("alpha > 0")
                .sort_values("alpha")
            )
            ax_mean.plot(merged["alpha"], merged["mean_lph"], color=col, linewidth=1.8)
            ax_mean.fill_between(
                merged["alpha"],
                merged["mean_lph"] - merged["std_lph"],
                merged["mean_lph"] + merged["std_lph"],
                color=col,
                alpha=0.20,
            )
            ax_med.plot(
                merged["alpha"],
                merged["median_lph"],
                color=col,
                linewidth=1.8,
                linestyle="--",
            )

        for ax, ylabel in [(ax_mean, "Mean LPH (± std)"), (ax_med, "Median LPH")]:
            ax.set_xscale("log")
            ax.set_xlabel("Alpha", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(alpha=0.3)

        ax_mean.set_title(f"{model.capitalize()} — Mean LPH ± std", fontsize=11)
        ax_med.set_title(f"{model.capitalize()} — Median LPH", fontsize=11)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if save:
        path = f"{_plots_dir(dataset)}/alpha_vs_lph.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_alpha_vs_num_communities(
    n_networks: int = 100, save: bool = True, dataset: str | None = None
) -> None:
    """
    Three-panel column: alpha (log x-axis) vs mean community count per model.

    Args:
        n_networks: Number of networks aggregated per model.
        save:       If ``True``, write PNG to ``plots/communities/``.
        dataset:    Dataset name.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=False)
    fig.suptitle(
        "Alpha vs Mean Community Memberships per Node",
        fontsize=14,
        fontweight="bold",
    )

    for ax, model in zip(axes, MODELS):
        alpha_df = _load_alpha_csv(model, dataset=dataset)
        stats_df = _aggregate_community_stats(model, n_networks, dataset=dataset)
        col = PALETTE[model]

        if alpha_df is not None and not stats_df.empty:
            merged = (
                alpha_df.merge(stats_df, on="network_index", how="inner")
                .query("alpha > 0")
                .sort_values("alpha")
            )
            ax.plot(merged["alpha"], merged["mean_num_coms"], color=col, linewidth=1.8)
            std_val = float(merged["mean_num_coms"].std())
            ax.fill_between(
                merged["alpha"],
                merged["mean_num_coms"] - std_val,
                merged["mean_num_coms"] + std_val,
                color=col,
                alpha=0.18,
            )

        ax.set_title(f"{model.capitalize()}", fontsize=12)
        ax.set_xlabel("Alpha (inference parameter)", fontsize=10)
        ax.set_ylabel("Mean communities per node", fontsize=10)
        ax.set_xscale("log")
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if save:
        path = f"{_plots_dir(dataset)}/alpha_vs_num_communities.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_lph_vs_centrality(
    model_name: str | None = None,
    network_index: int | None = None,
    save: bool = True,
    dataset: str | None = None,
) -> None:
    """
    Scatter plot of LPH vs degree and PageRank, coloured by ``num_communities``.

    Args:
        model_name:    Diffusion model (random if ``None``).
        network_index: Zero-based network index (random if ``None``).
        save:          If ``True``, write PNG to ``plots/communities/``.
        dataset:       Dataset name.
    """
    if model_name is None:
        model_name = random.choice(MODELS)
    if network_index is None:
        network_index = random.randint(0, 99)

    comm_df = _load_community_csv(model_name, network_index, dataset=dataset)
    cent_df = _load_centrality_csv(model_name, network_index, dataset=dataset)
    if comm_df is None or cent_df is None:
        print(f"Data not found for {model_name} network {network_index}.")
        return

    merged = comm_df.merge(cent_df, on="UserId", how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    nc = merged["num_communities"].clip(upper=merged["num_communities"].quantile(0.99))

    for ax, xcol, xlabel in zip(
        axes,
        ["degree", "pagerank"],
        ["Degree centrality", "PageRank"],
    ):
        sc = ax.scatter(
            merged[xcol],
            merged["local_pluralistic_hom"],
            c=nc,
            cmap="viridis",
            alpha=0.5,
            s=15,
            linewidths=0,
        )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Local Pluralistic Homophily (LPH)", fontsize=11)
        ax.set_title(f"LPH vs {xlabel}", fontsize=12)
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label="num_communities")

    fig.suptitle(
        f"LPH vs Centrality Metrics — {model_name} network {network_index:03d}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save:
        path = (
            f"{_plots_dir(dataset)}/"
            f"lph_vs_centrality_{model_name}_{network_index:03d}.png"
        )
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_community_correlation_heatmap(
    model_name: str | None = None,
    network_index: int | None = None,
    save: bool = True,
    dataset: str | None = None,
) -> None:
    """
    Pearson correlation heatmap between LPH, ``num_communities``, and centrality.

    Args:
        model_name:    Diffusion model (random if ``None``).
        network_index: Zero-based network index (random if ``None``).
        save:          If ``True``, write PNG to ``plots/communities/``.
        dataset:       Dataset name.
    """
    if model_name is None:
        model_name = random.choice(MODELS)
    if network_index is None:
        network_index = random.randint(0, 99)

    comm_df = _load_community_csv(model_name, network_index, dataset=dataset)
    cent_df = _load_centrality_csv(model_name, network_index, dataset=dataset)
    if comm_df is None or cent_df is None:
        print(f"Data not found for {model_name} network {network_index}.")
        return

    comm_cols = ["UserId", "local_pluralistic_hom", "num_communities"]
    if "lph_score" in comm_df.columns:
        comm_cols.append("lph_score")
    merged = (
        comm_df[comm_cols]
        .merge(cent_df, on="UserId", how="inner")
        .drop(columns=["UserId"])
    )
    corr = merged.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        f"Pearson Correlation — {model_name} network {network_index:03d}",
        fontsize=13,
    )
    plt.tight_layout()

    if save:
        path = (
            f"{_plots_dir(dataset)}/"
            f"correlation_heatmap_{model_name}_{network_index:03d}.png"
        )
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()
