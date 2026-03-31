"""
Visualization of overlapping community detection results and Local Pluralistic
Homophily (LPH) analysis.

All six plots load their data via :mod:`config.Paths` so they work regardless
of the working directory from which the script is executed.

Available plots
---------------
1. :func:`plot_lph_distribution`         — LPH violin + box plot by model
2. :func:`plot_num_communities_dist`     — Distribution of per-node community count
3. :func:`plot_alpha_vs_lph`             — 3×2 grid: alpha vs mean/median LPH
4. :func:`plot_alpha_vs_num_communities` — Alpha vs mean community memberships
5. :func:`plot_lph_vs_centrality`        — LPH vs degree / PageRank scatter
6. :func:`plot_community_correlation_heatmap` — Pearson correlation heatmap

Usage (standalone)::

    python -m visualization.community_plots
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import Paths, Models


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PALETTE = {
    "exponential": "#2196F3",
    "powerlaw": "#FF5722",
    "rayleigh": "#4CAF50",
}
MODELS = Models.ALL


# ---------------------------------------------------------------------------
# Internal data-loading helpers
# ---------------------------------------------------------------------------


def _plots_dir() -> str:
    out = Paths.PLOTS / "communities"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _load_community_csv(model_name: str, network_index: int) -> pd.DataFrame | None:
    fp = (
        Paths.COMMUNITIES
        / model_name
        / f"communities_{model_name}_{network_index:03d}.csv"
    )
    return pd.read_csv(fp) if fp.exists() else None


def _load_centrality_csv(model_name: str, network_index: int) -> pd.DataFrame | None:
    fp = (
        Paths.CENTRALITY
        / model_name
        / f"centrality_metrics_{model_name}_{network_index:03d}.csv"
    )
    return pd.read_csv(fp) if fp.exists() else None


def _load_alpha_csv(model_name: str) -> pd.DataFrame | None:
    short = Models.SHORT[model_name]
    fp = Paths.NETWORKS / model_name / f"inferred_edges_{short}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, sep="|").reset_index(drop=True)
    df["network_index"] = df.index
    return df


def _aggregate_community_stats(model_name: str, n_networks: int = 100) -> pd.DataFrame:
    """
    Return per-network aggregated community statistics.

    Columns: ``network_index``, ``mean_lph``, ``median_lph``, ``std_lph``,
    ``mean_num_coms``, ``median_num_coms``.
    """
    rows = []
    for i in range(n_networks):
        raw = _load_community_csv(model_name, i)
        if raw is None:
            continue
        rows.append(
            {
                "network_index": i,
                "mean_lph": raw["local_pluralistic_hom"].mean(),
                "median_lph": raw["local_pluralistic_hom"].median(),
                "std_lph": raw["local_pluralistic_hom"].std(),
                "mean_num_coms": raw["num_communities"].mean(),
                "median_num_coms": raw["num_communities"].median(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 1 — LPH distribution
# ---------------------------------------------------------------------------


def plot_lph_distribution(
    n_networks: int = 100,
    sample_nodes: int = 500,
    save: bool = True,
) -> None:
    """
    Violin + strip plot comparing the LPH distribution across the three models.

    Each violin pools LPH values from up to *sample_nodes* randomly selected
    nodes per network, giving a distribution that reveals the full shape
    without excessive memory use.

    Args:
        n_networks:   Number of networks to sample per model.
        sample_nodes: Maximum nodes to sample from each network.
        save:         If ``True``, write PNG to ``plots/communities/``.
    """
    rng = np.random.default_rng(42)
    records = []
    for model in MODELS:
        for i in range(n_networks):
            raw = _load_community_csv(model, i)
            if raw is None:
                continue
            sample = raw["local_pluralistic_hom"].dropna()
            if len(sample) > sample_nodes:
                sample = sample.iloc[
                    rng.choice(len(sample), sample_nodes, replace=False)
                ]
            for v in sample:
                records.append({"model": model, "LPH": v})

    if not records:
        print("No community data found.")
        return

    data = pd.DataFrame(records)
    _, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(
        data=data,
        x="model",
        y="LPH",
        palette=PALETTE,
        inner=None,
        cut=0,
        ax=ax,
        alpha=0.7,
    )
    sns.boxplot(
        data=data,
        x="model",
        y="LPH",
        width=0.15,
        showcaps=True,
        boxprops={"zorder": 3},
        whiskerprops={"zorder": 3},
        medianprops={"color": "white", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
        palette=PALETTE,
        ax=ax,
    )
    ax.set_title("Local Pluralistic Homophily (LPH) Distribution by Model", fontsize=14)
    ax.set_xlabel("Inference Model", fontsize=12)
    ax.set_ylabel("LPH", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = f"{_plots_dir()}/lph_distribution.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2 — Number of communities distribution
# ---------------------------------------------------------------------------


def plot_num_communities_dist(n_networks: int = 100, save: bool = True) -> None:
    """
    Density-normalised histogram of per-node community count, one curve per model.

    Args:
        n_networks: Number of networks to aggregate per model.
        save:       If ``True``, write PNG to ``plots/communities/``.
    """
    _, ax = plt.subplots(figsize=(9, 5))
    for model in MODELS:
        all_counts: list[int] = []
        for i in range(n_networks):
            raw = _load_community_csv(model, i)
            if raw is None:
                continue
            all_counts.extend(raw["num_communities"].tolist())
        if not all_counts:
            continue
        counts = np.array(all_counts)
        ax.hist(
            counts,
            bins=range(0, int(counts.max()) + 2),
            density=True,
            alpha=0.5,
            color=PALETTE[model],
            label=model,
            edgecolor="none",
        )

    ax.set_title("Distribution of Community Memberships per Node", fontsize=14)
    ax.set_xlabel("Number of communities a node belongs to", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(title="Model", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = f"{_plots_dir()}/num_communities_dist.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3 — Alpha vs LPH (3 × 2 grid)
# ---------------------------------------------------------------------------


def plot_alpha_vs_lph(n_networks: int = 100, save: bool = True) -> None:
    """
    3-row × 2-column grid showing how the alpha parameter affects LPH.

    Row per model; left column = mean ± std, right column = median.

    Args:
        n_networks: Number of networks aggregated per model.
        save:       If ``True``, write PNG to ``plots/communities/``.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
    fig.suptitle(
        "Effect of Alpha on Local Pluralistic Homophily", fontsize=15, fontweight="bold"
    )

    for row, model in enumerate(MODELS):
        alpha_df = _load_alpha_csv(model)
        stats_df = _aggregate_community_stats(model, n_networks)
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
        path = f"{_plots_dir()}/alpha_vs_lph.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot 4 — Alpha vs number of community memberships
# ---------------------------------------------------------------------------


def plot_alpha_vs_num_communities(n_networks: int = 100, save: bool = True) -> None:
    """
    Three-panel column showing alpha (log x-axis) vs mean community count.

    Args:
        n_networks: Number of networks aggregated per model.
        save:       If ``True``, write PNG to ``plots/communities/``.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=False)
    fig.suptitle(
        "Alpha vs Mean Community Memberships per Node", fontsize=14, fontweight="bold"
    )

    for ax, model in zip(axes, MODELS):
        alpha_df = _load_alpha_csv(model)
        stats_df = _aggregate_community_stats(model, n_networks)
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
        path = f"{_plots_dir()}/alpha_vs_num_communities.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot 5 — LPH vs centrality metrics scatter
# ---------------------------------------------------------------------------


def plot_lph_vs_centrality(
    model_name: str | None = None,
    network_index: int | None = None,
    save: bool = True,
) -> None:
    """
    Scatter plot of LPH vs degree and PageRank, coloured by ``num_communities``.

    If *model_name* or *network_index* are ``None``, they are chosen randomly.

    Args:
        model_name:    Diffusion model (exponential / powerlaw / rayleigh).
        network_index: Zero-based network index.
        save:          If ``True``, write PNG to ``plots/communities/``.
    """
    if model_name is None:
        model_name = random.choice(MODELS)
    if network_index is None:
        network_index = random.randint(0, 99)

    comm_df = _load_community_csv(model_name, network_index)
    cent_df = _load_centrality_csv(model_name, network_index)
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
            f"{_plots_dir()}/" f"lph_vs_centrality_{model_name}_{network_index:03d}.png"
        )
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot 6 — Correlation heatmap
# ---------------------------------------------------------------------------


def plot_community_correlation_heatmap(
    model_name: str | None = None,
    network_index: int | None = None,
    save: bool = True,
) -> None:
    """
    Pearson correlation heatmap between LPH, ``num_communities``, and centrality
    metrics for a single network.

    Args:
        model_name:    Diffusion model.
        network_index: Zero-based network index.
        save:          If ``True``, write PNG to ``plots/communities/``.
    """
    if model_name is None:
        model_name = random.choice(MODELS)
    if network_index is None:
        network_index = random.randint(0, 99)

    comm_df = _load_community_csv(model_name, network_index)
    cent_df = _load_centrality_csv(model_name, network_index)
    if comm_df is None or cent_df is None:
        print(f"Data not found for {model_name} network {network_index}.")
        return

    merged = (
        comm_df[["UserId", "local_pluralistic_hom", "num_communities"]]
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
            f"{_plots_dir()}/"
            f"correlation_heatmap_{model_name}_{network_index:03d}.png"
        )
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Quick demo when run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_lph_distribution(n_networks=10)
    plot_num_communities_dist(n_networks=10)
