"""
Visualization module for overlapping community detection results and
Local Pluralistic Homophily (LPH) analysis.

Provides the following plots:

    1. plot_lph_distribution        — LPH distribution per model (violin + box)
    2. plot_num_communities_dist    — Distribution of node community count per model
    3. plot_alpha_vs_lph            — Alpha parameter vs mean LPH across 100 networks
    4. plot_alpha_vs_num_communities— Alpha parameter vs mean num_communities across 100 networks
    5. plot_lph_vs_centrality       — LPH vs degree/pagerank scatter (single network)
    6. plot_community_correlation   — Heatmap: Pearson correlations LPH + centrality metrics

All save-paths default to ``../plots/communities/<function_name>.png``.

Alpha files expected at:
    ../data/inferred_networks/<model>/inferred_edges_<short>.csv
    (pipe-separated, columns: alpha | inferred_edges_<short> | rmse)

Community files expected at:
    ../data/communities/<model>/communities_<model>_<NNN>.csv

Centrality files expected at:
    ../data/centrality_metrics/<model>/centrality_metrics_<model>_<NNN>.csv

Dependencies:
    - os, glob
    - numpy, pandas
    - matplotlib, seaborn
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── colour palette shared by all plots ─────────────────────────────────────
PALETTE = {
    "exponential": "#2196F3",  # blue
    "powerlaw": "#FF5722",  # deep orange
    "rayleigh": "#4CAF50",  # green
}
MODEL_SHORTS = {"exponential": "expo", "powerlaw": "power", "rayleigh": "ray"}
MODELS = list(PALETTE.keys())


# ── helpers ─────────────────────────────────────────────────────────────────


def _output_dir(subdir="communities"):
    path = os.path.join("..", "plots", subdir)
    os.makedirs(path, exist_ok=True)
    return path


def _load_community_csv(model_name, network_index):
    """Return community DataFrame or None."""
    fp = os.path.join(
        "..",
        "data",
        "communities",
        model_name,
        f"communities_{model_name}_{network_index:03d}.csv",
    )
    return pd.read_csv(fp) if os.path.exists(fp) else None


def _load_centrality_csv(model_name, network_index):
    """Return centrality DataFrame or None."""
    fp = os.path.join(
        "..",
        "data",
        "centrality_metrics",
        model_name,
        f"centrality_metrics_{model_name}_{network_index:03d}.csv",
    )
    return pd.read_csv(fp) if os.path.exists(fp) else None


def _load_alpha_csv(model_name):
    """
    Return DataFrame (alpha, network_index) from inferred_edges CSV.
    Rows are ordered 0..N-1 and a zero-based 'network_index' column is added.
    """
    short = MODEL_SHORTS[model_name]
    fp = os.path.join(
        "..", "data", "inferred_networks", model_name, f"inferred_edges_{short}.csv"
    )
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, sep="|")
    df = df.reset_index(drop=True)
    df["network_index"] = df.index
    return df


def _aggregate_community_stats(model_name, n_networks=100):
    """
    Return a DataFrame with per-network aggregated community stats:
        network_index, mean_lph, std_lph, mean_num_coms, median_lph
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


# ── Plot 1 ───────────────────────────────────────────────────────────────────


def plot_lph_distribution(n_networks=100, sample_nodes=500, save=True):
    """
    Violin + strip plot comparing the LPH distribution across the three models.

    Each violin pools LPH values of ``sample_nodes`` randomly selected nodes
    from each of the ``n_networks`` networks, giving a distribution of ~50 000
    values per model — enough to reveal the full shape without memory issues.

    WHY USEFUL: Shows whether a particular inference model (exponential /
    powerlaw / rayleigh) produces networks whose nodes tend to live in more
    homogeneous or more mixed community contexts.
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
        path = os.path.join(_output_dir(), "lph_distribution.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ── Plot 2 ───────────────────────────────────────────────────────────────────


def plot_num_communities_dist(n_networks=100, save=True):
    """
    Histogram (overlaid, density-normalised) of the number of communities
    each node belongs to, one curve per model.

    WHY USEFUL: Indicates whether nodes are multi-community (high overlap) or
    mostly singleton, which directly affects how meaningful LPH is.
    """
    _, ax = plt.subplots(figsize=(9, 5))
    for model in MODELS:
        all_counts = []
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
        path = os.path.join(_output_dir(), "num_communities_dist.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ── Plot 3 ───────────────────────────────────────────────────────────────────


def plot_alpha_vs_lph(n_networks=100, save=True):
    """
    3-row × 2-col grid: one row per model.
    Col 1 — alpha vs mean LPH (± std band).
    Col 2 — alpha vs median LPH.
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
            merged_raw = alpha_df.merge(stats_df, on="network_index", how="inner")
            merged = merged_raw.query("alpha > 0").sort_values("alpha")

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

        for ax, ylabel in [
            (ax_mean, "Mean LPH (± std)"),
            (ax_med, "Median LPH"),
        ]:
            ax.set_xscale("log")
            ax.set_xlabel("Alpha", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(alpha=0.3)

        ax_mean.set_title(f"{model.capitalize()} — Mean LPH ± std", fontsize=11)
        ax_med.set_title(f"{model.capitalize()} — Median LPH", fontsize=11)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if save:
        path = os.path.join(_output_dir(), "alpha_vs_lph.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ── Plot 4 ───────────────────────────────────────────────────────────────────


def plot_alpha_vs_num_communities(n_networks=100, save=True):
    """
    3-subplot column: one subplot per model, alpha (x, log) vs mean number of
    community memberships per node (y, with ± std band).
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
            ax.plot(
                merged["alpha"],
                merged["mean_num_coms"],
                color=col,
                linewidth=1.8,
            )
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
        path = os.path.join(_output_dir(), "alpha_vs_num_communities.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ── Plot 5 ───────────────────────────────────────────────────────────────────


def plot_lph_vs_centrality(model_name=None, network_index=None, save=True):
    """
    Scatter: LPH (y) vs degree & pagerank (x, two sub-plots), coloured by
    num_communities.

    If ``model_name`` or ``network_index`` are None, they are chosen at random.

    WHY USEFUL: Reveals whether structurally central nodes (hubs) sit inside
    many communities and whether that makes them more or less homophilic. If
    high-degree nodes have low LPH → hubs bridge disparate communities.
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
    cmap = "viridis"
    nc = merged["num_communities"].clip(upper=merged["num_communities"].quantile(0.99))

    for ax, xcol, xlabel in zip(
        axes, ["degree", "pagerank"], ["Degree centrality", "PageRank"]
    ):
        sc = ax.scatter(
            merged[xcol],
            merged["local_pluralistic_hom"],
            c=nc,
            cmap=cmap,
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
        path = os.path.join(
            _output_dir(), f"lph_vs_centrality_{model_name}_{network_index:03d}.png"
        )
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ── Plot 6 ───────────────────────────────────────────────────────────────────


def plot_community_correlation_heatmap(model_name=None, network_index=None, save=True):
    """
    Heatmap of Pearson correlations between LPH, num_communities and all
    centrality metrics for a single network.

    If ``model_name`` or ``network_index`` are None, they are chosen at random.

    WHY USEFUL: One-shot view of which centrality metric is most predictive of
    community overlap/homophily — directly informs feature selection for CMF.
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

    assert comm_df is not None and cent_df is not None
    comm: pd.DataFrame = comm_df
    cent: pd.DataFrame = cent_df
    merged = (
        comm[["UserId", "local_pluralistic_hom", "num_communities"]]
        .merge(cent, on="UserId", how="inner")
        .drop(columns=["UserId"])
    )

    corr = merged.corr()

    _, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
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
        f"Pearson Correlation — {model_name} network {network_index:03d}", fontsize=13
    )
    plt.tight_layout()

    if save:
        path = os.path.join(
            _output_dir(), f"correlation_heatmap_{model_name}_{network_index:03d}.png"
        )
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ── CLI entry point ──────────────────────────────────────────────────────────


def generate_all_plots(n_networks=100):
    """Generate and save all community plots."""
    print("Generating community plots...\n")

    print("[1/6] LPH distribution (violin)")
    plot_lph_distribution(n_networks=n_networks)

    print("[2/6] Number of communities distribution")
    plot_num_communities_dist(n_networks=n_networks)

    print("[3/6] Alpha vs mean LPH")
    plot_alpha_vs_lph(n_networks=n_networks)

    print("[4/6] Alpha vs mean num communities")
    plot_alpha_vs_num_communities(n_networks=n_networks)

    rand_model = random.choice(MODELS)
    rand_idx = random.randint(0, 99)
    print(f"[5/6] LPH vs centrality scatter ({rand_model}, net {rand_idx:03d})")
    plot_lph_vs_centrality(model_name=rand_model, network_index=rand_idx)

    print(f"[6/6] Correlation heatmap ({rand_model}, net {rand_idx:03d})")
    plot_community_correlation_heatmap(model_name=rand_model, network_index=rand_idx)

    print("\nAll plots saved to ../plots/communities/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate community analysis plots")
    parser.add_argument(
        "--n-networks",
        type=int,
        default=100,
        help="Number of networks per model to aggregate (default: 100)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="all",
        choices=[
            "all",
            "lph_dist",
            "num_coms",
            "alpha_lph",
            "alpha_coms",
            "lph_vs_centrality",
            "heatmap",
        ],
        help="Which plot to generate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["exponential", "powerlaw", "rayleigh"],
        help="Model for per-network plots (default: random)",
    )
    parser.add_argument(
        "--network",
        type=int,
        default=None,
        help="Network index for per-network plots (default: random 0-99)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Generate all plots for all models (alias for --plot all)",
    )
    args = parser.parse_args()
    if args.all_models:
        args.plot = "all"

    dispatch = {
        "lph_dist": lambda: plot_lph_distribution(args.n_networks),
        "num_coms": lambda: plot_num_communities_dist(args.n_networks),
        "alpha_lph": lambda: plot_alpha_vs_lph(args.n_networks),
        "alpha_coms": lambda: plot_alpha_vs_num_communities(args.n_networks),
        "lph_vs_centrality": lambda: plot_lph_vs_centrality(args.model, args.network),
        "heatmap": lambda: plot_community_correlation_heatmap(args.model, args.network),
        "all": lambda: generate_all_plots(args.n_networks),
    }
    dispatch[args.plot]()
