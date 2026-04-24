"""
LPH and community-count distribution plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from visualization.community_plots.loaders import (
    MODELS,
    PALETTE,
    _plots_dir,
    _load_community_csv,
)


def plot_lph_distribution(
    n_networks: int = 100,
    sample_nodes: int = 500,
    save: bool = True,
    dataset: str | None = None,
) -> None:
    """
    Violin + box plot comparing LPH distributions across the three models.

    Two side-by-side subplots:
    - Left: Jaccard-based ``local_pluralistic_hom`` (range [0, 1]).
    - Right: Normalized boundary score ``lph_score`` (when present).

    Args:
        n_networks:   Number of networks to sample per model.
        sample_nodes: Maximum nodes to sample from each network.
        save:         If ``True``, write PNG to ``plots/communities/``.
        dataset:      Dataset name.
    """
    rng = np.random.default_rng(42)
    records = []
    has_lph_score = False
    for model in MODELS:
        for i in range(n_networks):
            raw = _load_community_csv(model, i, dataset=dataset)
            if raw is None:
                continue
            if "lph_score" in raw.columns:
                has_lph_score = True
            sample_idx = np.arange(len(raw))
            if len(sample_idx) > sample_nodes:
                sample_idx = rng.choice(len(sample_idx), sample_nodes, replace=False)
            subset = raw.iloc[sample_idx]
            for _, row in subset.iterrows():
                rec: dict = {
                    "model": model,
                    "LPH (Jaccard)": row["local_pluralistic_hom"],
                }
                if "lph_score" in raw.columns:
                    rec["h̃v (paper)"] = row["lph_score"]
                records.append(rec)

    if not records:
        print("No community data found.")
        return

    data = pd.DataFrame(records)
    ncols = 2 if has_lph_score else 1
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    def _violin_box(ax: Axes, col: str, ylabel: str, ylim: tuple | None) -> None:
        sns.violinplot(
            data=data,
            x="model",
            y=col,
            palette=PALETTE,
            inner=None,
            cut=0,
            ax=ax,
            alpha=0.7,
        )
        sns.boxplot(
            data=data,
            x="model",
            y=col,
            width=0.15,
            showcaps=True,
            boxprops={"zorder": 3},
            whiskerprops={"zorder": 3},
            medianprops={"color": "white", "linewidth": 2},
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
            palette=PALETTE,
            ax=ax,
        )
        ax.set_xlabel("Inference Model", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)

    _violin_box(axes[0], "LPH (Jaccard)", "LPH (Jaccard)", (-0.05, 1.05))
    axes[0].set_title("Jaccard-based LPH", fontsize=13)

    if has_lph_score:
        _violin_box(axes[1], "h̃v (paper)", "h̃v (Barraza et al. 2025)", None)
        axes[1].set_title("Normalized h̃v — Boundary Score", fontsize=13)
        axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")

    fig.suptitle(
        "Local Pluralistic Homophily Distribution by Model",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save:
        path = f"{_plots_dir(dataset)}/lph_distribution.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_num_communities_dist(
    n_networks: int = 100, save: bool = True, dataset: str | None = None
) -> None:
    """
    Density-normalised histogram of per-node community count, one curve per model.

    Args:
        n_networks: Number of networks to aggregate per model.
        save:       If ``True``, write PNG to ``plots/communities/``.
        dataset:    Dataset name.
    """
    _, ax = plt.subplots(figsize=(9, 5))
    for model in MODELS:
        all_counts: list[int] = []
        for i in range(n_networks):
            raw = _load_community_csv(model, i, dataset=dataset)
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
        path = f"{_plots_dir(dataset)}/num_communities_dist.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()
