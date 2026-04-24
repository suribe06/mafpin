"""
Hyperparameter search visualizations: overview, heatmap, convergence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from visualization.model_plots._common import _plots_dir


def plot_hyperparameter_search_results(
    search_results: dict,
    save_path: str | None = None,
    figsize: tuple = (15, 10),
    dataset: str | None = None,
) -> None:
    """
    Six-panel overview of a hyperparameter search run.

    Args:
        search_results: Dict from :func:`recommender.baseline.search_best_params`.
                        Must contain keys ``all_results`` and ``best_params``.
        save_path:      Filename (relative to ``plots/models/``).  If ``None``,
                        the plot is only shown.
        figsize:        Matplotlib figure size.
    """
    if not search_results or "all_results" not in search_results:
        print("Error: no search results to plot.")
        return

    results = search_results["all_results"]
    if not results:
        print("Error: all_results is empty.")
        return

    k_values = [r["k"] for r in results]
    lambda_values = [r["lambda_reg"] for r in results]
    rmse_values = [r["rmse"] for r in results]
    mae_values = [r.get("mae") for r in results]
    r2_values = [r.get("r2") for r in results]
    has_mae = any(v is not None for v in mae_values)
    has_r2 = any(v is not None for v in r2_values)
    best_rmse = search_results["best_params"].get("rmse", min(rmse_values))

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Hyperparameter Search Results", fontsize=16, fontweight="bold")

    axes[0, 0].scatter(k_values, rmse_values, alpha=0.6, c="steelblue")
    axes[0, 0].set(xlabel="Latent factors (k)", ylabel="RMSE", title="RMSE vs k")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].scatter(lambda_values, rmse_values, alpha=0.6, c="tomato")
    axes[0, 1].set(xlabel="Regularisation (λ)", ylabel="RMSE", title="RMSE vs λ")
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].hist(
        rmse_values, bins=20, alpha=0.7, color="seagreen", edgecolor="black"
    )
    axes[0, 2].axvline(
        best_rmse, color="red", linestyle="--", label=f"Best: {best_rmse:.4f}"
    )
    axes[0, 2].set(xlabel="RMSE", ylabel="Frequency", title="RMSE Distribution")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    if has_mae:
        axes[1, 0].scatter(rmse_values, mae_values, alpha=0.6, c="mediumpurple")
        axes[1, 0].set(xlabel="RMSE", ylabel="MAE", title="MAE vs RMSE")
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "MAE not available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("MAE vs RMSE")
    axes[1, 0].grid(alpha=0.3)

    if has_r2:
        axes[1, 1].scatter(rmse_values, r2_values, alpha=0.6, c="darkorange")
        axes[1, 1].set(xlabel="RMSE", ylabel="R²", title="R² vs RMSE")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "R² not available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("R² vs RMSE")
    axes[1, 1].grid(alpha=0.3)

    sorted_r = sorted(results, key=lambda x: x["rmse"])[:20]
    x_pos = list(range(len(sorted_r)))
    rmse_m = [r["rmse"] for r in sorted_r]
    rmse_e = [r.get("rmse_std", 0.0) for r in sorted_r]
    axes[1, 2].errorbar(x_pos, rmse_m, yerr=rmse_e, fmt="o", capsize=3, alpha=0.7)
    axes[1, 2].set(
        xlabel="Rank (best first)", ylabel="RMSE", title="Top 20 Combinations"
    )
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        full_path = f"{_plots_dir(dataset)}/{save_path}"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.close()


def plot_parameter_heatmap(
    search_results: dict,
    metric: str = "rmse",
    save_path: str | None = None,
    figsize: tuple = (12, 8),
    dataset: str | None = None,
) -> None:
    """
    Heatmap of *metric* across the (k, λ) parameter space.

    Args:
        search_results: Dict from :func:`recommender.baseline.search_best_params`.
        metric:         One of ``"rmse"``, ``"mae"``, or ``"r2"``.
        save_path:      Filename for ``plots/models/``.
        figsize:        Figure size.
    """
    if not search_results or "all_results" not in search_results:
        print("Error: no search results.")
        return

    df = pd.DataFrame(search_results["all_results"])
    if metric not in df.columns:
        print(f"Metric '{metric}' not found. Available: {list(df.columns)}")
        return

    k_bins = np.linspace(df["k"].min(), df["k"].max(), 10)
    lambda_bins = np.linspace(df["lambda_reg"].min(), df["lambda_reg"].max(), 10)
    df["k_bin"] = pd.cut(df["k"], k_bins, include_lowest=True)
    df["lambda_bin"] = pd.cut(df["lambda_reg"], lambda_bins, include_lowest=True)

    pivot = df.pivot_table(
        values=metric, index="k_bin", columns="lambda_bin", aggfunc="mean"
    )

    cmap = "viridis_r" if metric in ("rmse", "mae") else "viridis"
    plt.figure(figsize=figsize)
    im = plt.imshow(pivot.values, cmap=cmap, aspect="auto")
    plt.colorbar(im, label=metric.upper())
    plt.title(
        f"{metric.upper()} — Parameter Space Heatmap", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Regularisation (λ) bins")
    plt.ylabel("Latent factors (k) bins")
    plt.xticks(
        range(len(pivot.columns)),
        [f"{iv.left:.2f}–{iv.right:.2f}" for iv in pivot.columns],  # type: ignore[union-attr]
        rotation=45,
    )
    plt.yticks(
        range(len(pivot.index)),
        [f"{iv.left:.0f}–{iv.right:.0f}" for iv in pivot.index],  # type: ignore[union-attr]
    )
    plt.tight_layout()

    if save_path:
        full_path = f"{_plots_dir(dataset)}/{save_path}"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.close()


def plot_convergence_analysis(
    search_results: dict,
    save_path: str | None = None,
    figsize: tuple = (12, 6),
    dataset: str | None = None,
) -> None:
    """
    Convergence plot: all trial RMSE values and running-best curve.

    Args:
        search_results: Dict from :func:`recommender.baseline.search_best_params`.
        save_path:      Filename for ``plots/models/``.
        figsize:        Figure size.
    """
    if not search_results or "all_results" not in search_results:
        print("Error: no search results.")
        return

    results = sorted(search_results["all_results"], key=lambda x: x.get("iteration", 0))
    iterations = list(range(1, len(results) + 1))
    rmse_values = [r["rmse"] for r in results]

    running_best: list[float] = []
    current_best = float("inf")
    for v in rmse_values:
        if v < current_best:
            current_best = v
        running_best.append(current_best)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.scatter(iterations, rmse_values, alpha=0.5, label="All trials")
    ax1.plot(iterations, running_best, "r-", linewidth=2, label="Best so far")
    ax1.set(
        xlabel="Iteration", ylabel="RMSE", title="Hyperparameter Search Convergence"
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    improvements = [
        (i + 1, running_best[i])
        for i in range(1, len(running_best))
        if running_best[i] < running_best[i - 1]
    ]
    if improvements:
        imp_iter, imp_rmse = zip(*improvements)
        ax2.plot(imp_iter, imp_rmse, "go-", linewidth=2, markersize=8)
        for it, rv in improvements:
            ax2.annotate(
                f"{rv:.4f}",
                xy=(it, rv),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "No improvements found",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )
    ax2.set(xlabel="Iteration", ylabel="Best RMSE", title="Improvement Steps")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        full_path = f"{_plots_dir(dataset)}/{save_path}"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.close()
