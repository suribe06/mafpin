"""
Alpha–RMSE analysis plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Models, DatasetPaths, Datasets
from visualization.model_plots._common import _plots_dir


def _extract_alphas(model_name: str, dataset: str | None = None) -> np.ndarray:
    """
    Load the alpha column from ``data/inferred_networks/<model>/inferred_edges_<short>.csv``.

    Returns an empty array if the file is missing or malformed.
    """
    short = Models.SHORT.get(model_name, "")
    fp = (
        DatasetPaths(dataset or Datasets.DEFAULT).NETWORKS
        / model_name
        / f"inferred_edges_{short}.csv"
    )
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
        baseline_rmse:        Paired baseline RMSE.
        rmse_std_values:      Optional per-alpha std values (±1σ band).
        save_plot:            Write PNG to ``plots/models/``.
        figsize:              Figure size.
        global_baseline_rmse: Optional global plain-CMF baseline.
        dataset:              Dataset name.
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
        model_name:       Diffusion model name.
        rmse_values:      Per-alpha enhanced RMSE values.
        baseline_rmse:    Fallback scalar baseline RMSE.
        save_plot:        Write PNG to ``plots/models/``.
        figsize:          Figure size.
        delta_pct_values: Pre-computed per-network improvement percentages.
        dataset:          Dataset name.
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
        dataset:   Dataset name.
    """
    _MODEL_CFG = [
        ("exponential", "expo", "#E07B54"),
        ("powerlaw", "power", "#5B8DB8"),
        ("rayleigh", "ray", "#6DBF82"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Alpha vs Inferred Edge Count", fontsize=15, fontweight="bold", y=1.02)

    for ax, (model_name, short, colour) in zip(axes, _MODEL_CFG):
        csv = (
            DatasetPaths(dataset or Datasets.DEFAULT).NETWORKS
            / model_name
            / f"inferred_edges_{short}.csv"
        )
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
    plt.close()
