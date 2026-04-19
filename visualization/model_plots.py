"""
Visualizations for CMF model evaluation and hyperparameter analysis.

All save paths are derived from :mod:`config.Paths` so the plots are written
to ``plots/`` regardless of the working directory.

Available plots
---------------
:func:`plot_hyperparameter_search_results`
    Six-panel overview of a hyperparameter search run (scatter, histogram,
    error bars).
:func:`plot_parameter_heatmap`
    Heatmap of a selected metric across the (k, λ) parameter space.
:func:`plot_convergence_analysis`
    Convergence curve showing best RMSE improvement over iterations.
:func:`plot_metrics_comparison`
    Side-by-side distribution comparison of RMSE, MAE, and R².
:func:`plot_alpha_rmse_analysis`
    Alpha vs RMSE line plot with best-alpha marker and baseline reference.
:func:`plot_alpha_delta_rmse`
    Signed delta RMSE (enhanced − baseline) scatter plot per alpha.
:func:`plot_alpha_edges`
    Three-subplot figure showing alpha vs inferred edge count for each model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Paths, Models, DatasetPaths, Datasets


# ---------------------------------------------------------------------------
# Base output directory
# ---------------------------------------------------------------------------


def _plots_dir(dataset: str | None = None) -> str:
    out = DatasetPaths(dataset or Datasets.DEFAULT).PLOTS / "models"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


# ---------------------------------------------------------------------------
# Hyperparameter search visualizations
# ---------------------------------------------------------------------------


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
    mae_values = [r["mae"] for r in results]
    r2_values = [r["r2"] for r in results]
    best_rmse = search_results["best_params"].get("rmse", min(rmse_values))

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Hyperparameter Search Results", fontsize=16, fontweight="bold")

    # 1) RMSE vs k
    axes[0, 0].scatter(k_values, rmse_values, alpha=0.6, c="steelblue")
    axes[0, 0].set(xlabel="Latent factors (k)", ylabel="RMSE", title="RMSE vs k")
    axes[0, 0].grid(alpha=0.3)

    # 2) RMSE vs λ
    axes[0, 1].scatter(lambda_values, rmse_values, alpha=0.6, c="tomato")
    axes[0, 1].set(xlabel="Regularisation (λ)", ylabel="RMSE", title="RMSE vs λ")
    axes[0, 1].grid(alpha=0.3)

    # 3) RMSE histogram
    axes[0, 2].hist(
        rmse_values, bins=20, alpha=0.7, color="seagreen", edgecolor="black"
    )
    axes[0, 2].axvline(
        best_rmse, color="red", linestyle="--", label=f"Best: {best_rmse:.4f}"
    )
    axes[0, 2].set(xlabel="RMSE", ylabel="Frequency", title="RMSE Distribution")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # 4) MAE vs RMSE
    axes[1, 0].scatter(rmse_values, mae_values, alpha=0.6, c="mediumpurple")
    axes[1, 0].set(xlabel="RMSE", ylabel="MAE", title="MAE vs RMSE")
    axes[1, 0].grid(alpha=0.3)

    # 5) R² vs RMSE
    axes[1, 1].scatter(rmse_values, r2_values, alpha=0.6, c="darkorange")
    axes[1, 1].set(xlabel="RMSE", ylabel="R²", title="R² vs RMSE")
    axes[1, 1].grid(alpha=0.3)

    # 6) Top-20 results with error bars (requires per-split std — graceful fallback)
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
    plt.show()
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
        [
            f"{iv.left:.2f}–{iv.right:.2f}"  # type: ignore[union-attr]
            for iv in pivot.columns
        ],
        rotation=45,
    )
    plt.yticks(
        range(len(pivot.index)),
        [
            f"{iv.left:.0f}–{iv.right:.0f}"  # type: ignore[union-attr]
            for iv in pivot.index
        ],
    )
    plt.tight_layout()

    if save_path:
        full_path = f"{_plots_dir(dataset)}/{save_path}"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.show()
    plt.close()


def plot_convergence_analysis(
    search_results: dict,
    save_path: str | None = None,
    figsize: tuple = (12, 6),
    dataset: str | None = None,
) -> None:
    """
    Convergence plot: all trial RMSE values and running-best curve.

    Requires each result record to have an ``iteration`` key.

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
    plt.show()
    plt.close()


def plot_metrics_comparison(
    search_results: dict,
    save_path: str | None = None,
    figsize: tuple = (15, 5),
    dataset: str | None = None,
) -> None:
    """
    Side-by-side histograms of RMSE, MAE, and R².

    Args:
        search_results: Dict from :func:`recommender.baseline.search_best_params`.
        save_path:      Filename for ``plots/models/``.
        figsize:        Figure size.
    """
    if not search_results or "all_results" not in search_results:
        print("Error: no search results.")
        return

    results = search_results["all_results"]
    rmse_values = [r["rmse"] for r in results]
    mae_values = [r["mae"] for r in results]
    r2_values = [r["r2"] for r in results]

    best_idx = int(np.argmin(rmse_values))
    best_rmse = rmse_values[best_idx]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Metrics Distribution Comparison", fontsize=16, fontweight="bold")

    for ax, vals, label, colour in zip(
        axes,
        [rmse_values, mae_values, r2_values],
        ["RMSE", "MAE", "R²"],
        ["steelblue", "seagreen", "darkorange"],
    ):
        ax.hist(vals, bins=20, alpha=0.7, color=colour, edgecolor="black")
        ref_val = vals[best_idx]
        ax.axvline(
            ref_val, color="red", linestyle="--", label=f"Best RMSE idx: {ref_val:.4f}"
        )
        ax.set(xlabel=label, ylabel="Frequency", title=f"{label} Distribution")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        full_path = f"{_plots_dir(dataset)}/{save_path}"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.show()
    plt.close()

    print(f"\nRMSE: mean={np.mean(rmse_values):.4f}, best={best_rmse:.4f}")
    print(f"MAE : mean={np.mean(mae_values):.4f}")
    print(f"R²  : mean={np.mean(r2_values):.4f}")


# ---------------------------------------------------------------------------
# Alpha–RMSE analysis helpers
# ---------------------------------------------------------------------------


def _extract_alphas(model_name: str, dataset: str | None = None) -> np.ndarray:
    """
    Load the alpha column from ``data/inferred_networks/<model>/inferred_edges_<short>.csv``.

    Returns an empty array if the file is missing or malformed.
    """
    short = Models.SHORT.get(model_name, "")
    fp = DatasetPaths(dataset or Datasets.DEFAULT).NETWORKS / model_name / f"inferred_edges_{short}.csv"
    if not fp.exists():
        print(f"Alpha file not found: {fp}")
        return np.array([])
    df = pd.read_csv(fp, sep="|")
    if "alpha" not in df.columns:
        print("'alpha' column not found in inferred_edges file.")
        return np.array([])
    return np.asarray(df["alpha"].values, dtype=float)


def plot_alpha_rmse_analysis(
    model_name: str,
    rmse_values: list[float],
    baseline_rmse: float,
    rmse_std_values: list[float] | None = None,
    save_plot: bool = True,
    figsize: tuple = (12, 8),
    global_baseline_rmse: float | None = None,
    dataset: str | None = None,
) -> None:
    """
    Alpha vs RMSE line + scatter with best-alpha marker and baseline reference.

    Args:
        model_name:           Diffusion model name.
        rmse_values:          Per-alpha mean RMSE values.
        baseline_rmse:        Paired baseline RMSE (same user subset as the
                              enhanced model — fair comparison reference).
        rmse_std_values:      Optional per-alpha std values; if provided, a ±1σ
                              shaded band is drawn around the line.
        save_plot:            Write PNG to ``plots/models/``.
        figsize:              Figure size.
        global_baseline_rmse: Optional global plain-CMF baseline (all users).
                              When provided it is drawn as a second dashed line
                              for absolute reference.
    """
    alpha_values = _extract_alphas(model_name, dataset=dataset)
    if len(alpha_values) == 0 or len(alpha_values) != len(rmse_values):
        print("Mismatch between alpha values and RMSE values — cannot plot.")
        return

    best_idx = int(np.argmin(rmse_values))
    best_alpha = float(alpha_values[best_idx])
    best_rmse = float(rmse_values[best_idx])

    plt.figure(figsize=figsize)
    plt.plot(
        alpha_values, rmse_values, "b-", linewidth=2, alpha=0.7, label="RMSE vs Alpha"
    )
    if rmse_std_values is not None and len(rmse_std_values) == len(rmse_values):
        std_arr = np.asarray(rmse_std_values, dtype=float)
        mean_arr = np.asarray(rmse_values, dtype=float)
        plt.fill_between(
            alpha_values,
            mean_arr - std_arr,
            mean_arr + std_arr,
            alpha=0.2,
            color="steelblue",
            label="±1σ",
        )
    plt.scatter(alpha_values, rmse_values, c="steelblue", alpha=0.6, s=30)
    plt.scatter(
        best_alpha,
        best_rmse,
        c="red",
        s=120,
        marker="*",
        zorder=5,
        label=f"Best α={best_alpha:.2e}  RMSE={best_rmse:.4f}",
    )
    plt.axhline(
        baseline_rmse,
        color="seagreen",
        linestyle="--",
        linewidth=2,
        label=f"Paired baseline RMSE={baseline_rmse:.4f}",
    )
    if (
        global_baseline_rmse is not None
        and abs(global_baseline_rmse - baseline_rmse) > 1e-6
    ):
        plt.axhline(
            global_baseline_rmse,
            color="darkorange",
            linestyle=":",
            linewidth=1.5,
            label=f"Global baseline RMSE={global_baseline_rmse:.4f}",
        )

    alpha_range = (
        float(alpha_values.max()) / float(alpha_values.min())
        if float(alpha_values.min()) > 0
        else 1.0
    )
    if alpha_range > 1000:
        plt.xscale("log")
        plt.xlabel("Alpha (log scale)", fontsize=12)
    else:
        plt.xlabel("Alpha", fontsize=12)

    plt.ylabel("RMSE", fontsize=12)
    plt.title(
        f"Alpha vs RMSE — {model_name.capitalize()} Model",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_plot:
        full_path = f"{_plots_dir(dataset)}/alpha_rmse_{model_name}.png"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.show()
    plt.close()

    improvement = (baseline_rmse - best_rmse) / baseline_rmse * 100
    print(
        f"\n{model_name.capitalize()} — Best α={best_alpha:.4e}  "
        f"RMSE={best_rmse:.6f}  improvement={improvement:+.3f}%"
    )


def plot_alpha_delta_rmse(
    model_name: str,
    rmse_values: list[float],
    baseline_rmse: float,
    save_plot: bool = True,
    figsize: tuple = (12, 8),
    delta_pct_values: list[float] | None = None,
    dataset: str | None = None,
) -> None:
    """
    Signed delta RMSE scatter: positive = better than baseline (blue), negative =
    worse (red).

    Args:
        model_name:      Diffusion model name.
        rmse_values:     Per-alpha enhanced RMSE values.
        baseline_rmse:   Fallback scalar baseline RMSE (used only when
                         *delta_pct_values* is ``None``).
        save_plot:       Write PNG to ``plots/models/``.
        figsize:         Figure size.
        delta_pct_values: Pre-computed per-network improvement percentages
                          (paired comparison, same user subset).  When provided
                          these are used directly so the plot reflects a fair
                          comparison between models evaluated on the same users.
                          When ``None`` the delta is derived from *baseline_rmse*
                          and *rmse_values* (both must be on the same population).
    """
    alpha_values = _extract_alphas(model_name, dataset=dataset)
    if len(alpha_values) == 0 or len(alpha_values) != len(rmse_values):
        print("Mismatch between alpha values and RMSE values — cannot plot.")
        return

    if delta_pct_values is not None:
        deltas_pct = np.array(delta_pct_values)
    else:
        deltas_pct = (baseline_rmse - np.array(rmse_values)) / baseline_rmse * 100
    colours = ["steelblue" if d > 0 else "tomato" for d in deltas_pct]

    plt.figure(figsize=figsize)
    plt.scatter(alpha_values, deltas_pct, c=colours, s=40, alpha=0.7)
    plt.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Baseline Reference (Δ = 0%)",
    )
    plt.xlabel("Alpha", fontsize=12)
    plt.ylabel("Δ RMSE % (baseline − enhanced) / baseline × 100", fontsize=12)
    plt.title(
        f"Delta RMSE (%) per Alpha — {model_name.capitalize()} Model",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_plot:
        full_path = f"{_plots_dir(dataset)}/alpha_delta_rmse_{model_name}.png"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.show()
    plt.close()


def plot_alpha_edges(
    save_plot: bool = True,
    figsize: tuple = (18, 5),
    dataset: str | None = None,
) -> None:
    """
    Three-subplot figure: one subplot per diffusion model showing alpha (x-axis)
    vs the number of inferred edges (y-axis).

    Args:
        save_plot: Write PNG to ``plots/models/``.
        figsize:   Figure size.
    """
    _MODEL_CFG = [
        ("exponential", "expo", "#E07B54"),
        ("powerlaw", "power", "#5B8DB8"),
        ("rayleigh", "ray", "#6DBF82"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Alpha vs Inferred Edge Count", fontsize=15, fontweight="bold", y=1.02)

    for ax, (model_name, short, colour) in zip(axes, _MODEL_CFG):
        csv = DatasetPaths(dataset or Datasets.DEFAULT).NETWORKS / model_name / f"inferred_edges_{short}.csv"
        if not csv.exists():
            ax.set_title(f"{model_name.capitalize()}\n(data not found)")
            ax.axis("off")
            continue

        df = pd.read_csv(csv, sep="|")
        edge_col = f"inferred_edges_{short}"
        if edge_col not in df.columns or "alpha" not in df.columns:
            ax.set_title(f"{model_name.capitalize()}\n(missing columns)")
            ax.axis("off")
            continue

        ax.scatter(df["alpha"], df[edge_col], color=colour, s=30, alpha=0.75)
        ax.set_xlabel("Alpha", fontsize=11)
        ax.set_ylabel("Inferred Edges", fontsize=11)
        ax.set_title(f"{model_name.capitalize()} Model", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout()

    if save_plot:
        full_path = f"{_plots_dir(dataset)}/alpha_edges.png"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import json
    import sys

    _PLOT_CHOICES = [
        "alpha-rmse",
        "delta-rmse",
        "alpha-edges",
        "hyperparam",
        "heatmap",
        "metrics",
        "convergence",
        "all",
    ]

    _parser = argparse.ArgumentParser(
        description="Generate CMF evaluation plots from saved result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _parser.add_argument(
        "--plot",
        nargs="+",
        choices=_PLOT_CHOICES,
        default=["all"],
        metavar="PLOT",
        help=(
            "Which plots to generate. Choices: "
            + ", ".join(_PLOT_CHOICES)
            + ". Default: all."
        ),
    )
    _parser.add_argument(
        "--models",
        nargs="+",
        choices=Models.ALL + ["all"],
        default=["all"],
        help="Models for alpha-rmse / delta-rmse plots.",
    )
    _parser.add_argument(
        "--no-save",
        action="store_true",
        help="Show plots interactively without saving.",
    )
    _parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (default: uses Datasets.DEFAULT).",
    )
    _args = _parser.parse_args()

    _plots = set(_args.plot)
    _do_all = "all" in _plots
    _model_list = Models.ALL if "all" in _args.models else _args.models
    _save = not _args.no_save
    _dataset = _args.dataset

    # ------------------------------------------------------------------
    # Baseline search results (hyperparam / metrics / convergence plots)
    # ------------------------------------------------------------------
    _search_json = DatasetPaths(_dataset or Datasets.DEFAULT).BASELINE_RESULTS
    _search_result: dict | None = None
    if _do_all or _plots & {"hyperparam", "metrics", "convergence", "heatmap"}:
        if _search_json.exists():
            with open(_search_json, encoding="utf-8") as _fh:
                _search_result = json.load(_fh)
            print(f"Loaded search results from {_search_json}")
        else:
            print(
                f"Warning: {_search_json} not found. "
                "Run 'python -m recommender.baseline' or "
                "'python -m recommender.enhanced --all' first."
            )

    if _search_result and (_do_all or "hyperparam" in _plots):
        print("\n--- Hyperparameter search overview ---")
        plot_hyperparameter_search_results(
            _search_result,
            save_path="hyperparam_search.png" if _save else None,
        )

    if _search_result and (_do_all or "heatmap" in _plots):
        print("\n--- Parameter space heatmap ---")
        plot_parameter_heatmap(
            _search_result,
            metric="rmse",
            save_path="heatmap_rmse.png" if _save else None,
        )

    if _search_result and (_do_all or "metrics" in _plots):
        print("\n--- Metrics distribution ---")
        plot_metrics_comparison(
            _search_result,
            save_path="metrics_comparison.png" if _save else None,
        )

    if _search_result and (_do_all or "convergence" in _plots):
        print("\n--- Convergence analysis ---")
        plot_convergence_analysis(
            _search_result,
            save_path="convergence.png" if _save else None,
        )

    # ------------------------------------------------------------------
    # Per-model alpha plots (reads inferred_edges CSVs)
    # ------------------------------------------------------------------
    # Load the globally-tuned baseline RMSE saved by the recommend step.
    # Falls back to None — in which case the per-CSV baseline is used.
    _global_baseline_rmse: float | None = None
    if _search_json.exists():
        with open(_search_json, encoding="utf-8") as _bf:
            _bs = json.load(_bf)
        if "global_test_rmse" in _bs:
            _global_baseline_rmse = float(_bs["global_test_rmse"])
            print(
                f"Global baseline RMSE (from recommend step): {_global_baseline_rmse:.4f}"
            )

    if _do_all or _plots & {"alpha-rmse", "delta-rmse"}:
        for _model_name in _model_list:
            _short = Models.SHORT[_model_name]
            _csv = DatasetPaths(_dataset or Datasets.DEFAULT).NETWORKS / _model_name / f"inferred_edges_{_short}.csv"

            if not _csv.exists():
                print(f"Skipping {_model_name}: {_csv} not found.")
                continue

            _df = pd.read_csv(_csv, sep="|")

            if "rmse_mean" not in _df.columns or bool(_df["rmse_mean"].isna().all()):
                print(
                    f"Skipping {_model_name}: no rmse_mean data. Run enhanced --all first."
                )
                continue

            _rmse = _df["rmse_mean"].tolist()
            _std = (
                _df["rmse_std"].tolist()
                if "rmse_std" in _df.columns and not bool(_df["rmse_std"].isna().all())
                else None
            )

            # --- Paired baseline for alpha-rmse plot ------------------------
            # Use the per-network paired baseline (same user subset) so the
            # reference is on the same population as the enhanced RMSE values.
            # The global baseline is passed separately as a secondary line.
            _baseline: float
            if "baseline_rmse_mean" in _df.columns and not bool(
                _df["baseline_rmse_mean"].isna().all()
            ):
                _baseline = float(_df["baseline_rmse_mean"].dropna().mean())
            elif "improvement_pct" in _df.columns and not bool(
                _df["improvement_pct"].isna().all()
            ):
                _valid = _df.dropna(subset=["rmse_mean", "improvement_pct"])
                _baseline = float(
                    (
                        _valid["rmse_mean"] / (1.0 - _valid["improvement_pct"] / 100.0)
                    ).mean()
                )
            elif _global_baseline_rmse is not None:
                _baseline = _global_baseline_rmse
            else:
                _baseline = float(_df["rmse_mean"].mean())

            # --- Pre-computed paired deltas for delta-rmse plot --------------
            # improvement_pct was computed in _save_rmses as:
            #   (baseline_rmse_mean - rmse_mean) / baseline_rmse_mean * 100
            # Both values come from the same user subset (network users only),
            # so this is a fair paired comparison.  Using the global baseline
            # here would mix populations (network users vs all users).
            _delta_pct: list[float] | None = None
            if "improvement_pct" in _df.columns and not bool(
                _df["improvement_pct"].isna().all()
            ):
                _delta_pct = _df["improvement_pct"].fillna(0.0).tolist()

            print(f"\n--- {_model_name.upper()} (paired baseline≈{_baseline:.4f}) ---")

            if _do_all or "alpha-rmse" in _plots:
                plot_alpha_rmse_analysis(
                    _model_name,
                    _rmse,
                    baseline_rmse=_baseline,
                    rmse_std_values=_std,
                    save_plot=_save,
                    global_baseline_rmse=_global_baseline_rmse,
                    dataset=_dataset,
                )
            if _do_all or "delta-rmse" in _plots:
                plot_alpha_delta_rmse(
                    _model_name,
                    _rmse,
                    baseline_rmse=_baseline,
                    save_plot=_save,
                    delta_pct_values=_delta_pct,
                    dataset=_dataset,
                )

    if _do_all or "alpha-edges" in _plots:
        print("\n--- Alpha vs Inferred Edges (all models) ---")
        plot_alpha_edges(save_plot=_save, dataset=_dataset)

    sys.exit(0)
