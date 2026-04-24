"""
Metrics comparison visualizations.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from visualization.model_plots._common import _plots_dir


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
    mae_values_raw = [r.get("mae") for r in results]
    r2_values_raw = [r.get("r2") for r in results]
    mae_values = [v for v in mae_values_raw if v is not None]
    r2_values = [v for v in r2_values_raw if v is not None]

    best_idx = int(np.argmin(rmse_values))
    best_rmse = rmse_values[best_idx]

    metrics_data = [("RMSE", rmse_values, "steelblue")]
    if mae_values:
        metrics_data.append(("MAE", mae_values, "seagreen"))
    if r2_values:
        metrics_data.append(("R²", r2_values, "darkorange"))

    n_plots = len(metrics_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0] * n_plots / 3, figsize[1]))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Metrics Distribution Comparison", fontsize=16, fontweight="bold")

    for ax, (label, vals, colour) in zip(axes, metrics_data):
        ax.hist(vals, bins=20, alpha=0.7, color=colour, edgecolor="black")
        ref_val = vals[best_idx] if label == "RMSE" else vals[0]
        ax.axvline(ref_val, color="red", linestyle="--", label=f"Best: {ref_val:.4f}")
        ax.set(xlabel=label, ylabel="Frequency", title=f"{label} Distribution")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        full_path = f"{_plots_dir(dataset)}/{save_path}"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {full_path}")
    plt.close()

    print(f"\nRMSE: mean={np.mean(rmse_values):.4f}, best={best_rmse:.4f}")
    print(f"MAE : mean={np.mean(mae_values):.4f}")
    print(f"R²  : mean={np.mean(r2_values):.4f}")
