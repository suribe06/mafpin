"""
SHAP feature importance visualizations for the enhanced CMF recommender.

All save paths are derived from :mod:`config.DatasetPaths` so the plots are written
to ``plots/<dataset>/shap/`` regardless of the working directory.

Available plots
---------------
:func:`plot_shap_importance_comparison`
    Grouped horizontal bar chart comparing mean |SHAP| across all three
    cascade models.  Features sorted by global mean importance.
:func:`plot_shap_beeswarm`
    Per-model strip/beeswarm plot of raw SHAP values per feature, built from
    the saved ``.npy`` SHAP matrices.
:func:`plot_all_shap`
    Convenience wrapper that generates both plots above for every model.

Usage (standalone)::

    python -m visualization.shap_plots
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import DatasetPaths, Datasets


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PALETTE = {
    "exponential": "#2196F3",
    "powerlaw": "#FF5722",
    "rayleigh": "#4CAF50",
}

_SHAP_RESULTS_PATH = DatasetPaths(Datasets.DEFAULT).SHAP_RESULTS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plots_dir(dataset: str | None = None) -> str:
    out = DatasetPaths(dataset or Datasets.DEFAULT).PLOTS / "shap"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _load_results(results_path: str | Path | None, dataset: str | None = None) -> dict:
    path = (
        Path(results_path)
        if results_path
        else DatasetPaths(dataset or Datasets.DEFAULT).SHAP_RESULTS
    )
    if not path.exists():
        raise FileNotFoundError(
            f"SHAP results not found at {path}. "
            "Run `python pipeline.py --steps shap` first."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1 – Grouped bar chart (all models side-by-side)
# ---------------------------------------------------------------------------


def plot_shap_importance_comparison(
    results_path: str | Path | None = None,
    save: bool = True,
    dataset: str | None = None,
) -> None:
    """
    Grouped horizontal bar chart comparing mean |SHAP| for every feature
    across all three cascade models.

    Features are sorted by global mean importance (average over models).

    Args:
        results_path: Path to ``shap_results.json``.  ``None`` uses the
                      default location under ``data/``.
        save:         When ``True``, write the figure to ``plots/shap/``.
    """
    results = _load_results(results_path, dataset=dataset)

    models = list(results.keys())
    if not models:
        print(
            "[shap_plots] No SHAP results available — skipping importance comparison chart."
        )
        return
    feature_names = results[models[0]]["feature_names"]

    # (n_models, n_features) matrix of mean |SHAP|
    abs_matrix = np.array([results[m]["mean_shap_abs"] for m in models])

    # sort features ascending by global mean so most important is on top
    global_mean = abs_matrix.mean(axis=0)
    order = np.argsort(global_mean)
    sorted_features = [feature_names[i] for i in order]
    abs_sorted = abs_matrix[:, order]

    n_features = len(sorted_features)
    n_models = len(models)
    bar_height = 0.25
    y_base = np.arange(n_features)

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * bar_height
        ax.barh(
            y_base + offset,
            abs_sorted[i],
            height=bar_height * 0.9,
            color=PALETTE.get(model, f"C{i}"),
            label=model.capitalize(),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        "Feature Importance (SHAP) — Enhanced CMF by Cascade Model",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(title="Model", fontsize=10, title_fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    sns.despine(ax=ax, left=False, bottom=False)

    fig.tight_layout()

    if save:
        path = Path(_plots_dir(dataset)) / "shap_importance_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[shap_plots] Saved → {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 – Per-model beeswarm from raw SHAP matrices
# ---------------------------------------------------------------------------


def plot_shap_beeswarm(
    model_name: str,
    results_path: str | Path | None = None,
    save: bool = True,
    max_display_pts: int = 2000,
    dataset: str | None = None,
) -> None:
    """
    Beeswarm-style strip plot of SHAP values per feature for *model_name*.

    Loads all saved ``.npy`` SHAP matrices for the requested model,
    concatenates them into a single ``(total_users, n_features)`` array, and
    draws every user as a semi-transparent dot.  Features are sorted by mean
    |SHAP| (same order as the comparison chart).

    Args:
        model_name:       One of ``exponential``, ``powerlaw``, ``rayleigh``.
        results_path:     Path to ``shap_results.json``.  ``None`` uses default.
        save:             When ``True``, write to ``plots/shap/``.
        max_display_pts:  Maximum rows to plot (down-sampled when exceeded).
    """
    results = _load_results(results_path, dataset=dataset)

    if model_name not in results:
        raise ValueError(
            f"Model '{model_name}' not in SHAP results. " f"Available: {list(results)}"
        )

    data = results[model_name]
    feature_names = data["feature_names"]
    matrix_paths = data["matrix_paths"]

    # --- load and stack matrices --------------------------------------------
    matrices = []
    for p in matrix_paths:
        fp = Path(p)
        if fp.exists():
            matrices.append(np.load(fp))
        else:
            print(f"[shap_plots] Warning: matrix not found – {fp}")

    if not matrices:
        print(f"[shap_plots] No matrices found for '{model_name}'. Skipping beeswarm.")
        return

    all_shap = np.vstack(matrices)  # (total_users, n_features)

    # --- optional down-sample -----------------------------------------------
    if all_shap.shape[0] > max_display_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(all_shap.shape[0], size=max_display_pts, replace=False)
        all_shap = all_shap[idx]

    # --- sort features by mean |SHAP| ascending (bottom → top) -------------
    mean_abs = np.abs(all_shap).mean(axis=0)
    order = np.argsort(mean_abs)
    sorted_features = [feature_names[i] for i in order]
    all_shap_sorted = all_shap[:, order]

    # --- build long-form DataFrame for seaborn ------------------------------
    rows = []
    for feat_idx, feat_name in enumerate(sorted_features):
        for val in all_shap_sorted[:, feat_idx]:
            rows.append({"feature": feat_name, "shap_value": float(val)})
    df = pd.DataFrame(rows)

    # --- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    color = PALETTE.get(model_name, "steelblue")
    sns.stripplot(
        data=df,
        x="shap_value",
        y="feature",
        ax=ax,
        color=color,
        alpha=0.25,
        size=2.5,
        jitter=True,
        order=sorted_features,
    )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(
        f"SHAP Value Distribution — {model_name.capitalize()} Model",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", alpha=0.3)
    sns.despine(ax=ax, left=False, bottom=False)

    fig.tight_layout()

    if save:
        path = Path(_plots_dir(dataset)) / f"shap_beeswarm_{model_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[shap_plots] Saved → {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def plot_all_shap(
    results_path: str | Path | None = None,
    save: bool = True,
    dataset: str | None = None,
) -> None:
    """
    Generate all SHAP plots:

    1. ``shap_importance_comparison.png`` — grouped bar chart (all models).
    2. ``shap_beeswarm_<model>.png``       — one beeswarm per cascade model.

    Args:
        results_path: Path to ``shap_results.json``.  ``None`` uses default.
        save:         When ``True``, write figures to ``plots/shap/``.
    """
    print("[shap_plots] Generating SHAP importance comparison chart…")
    plot_shap_importance_comparison(
        results_path=results_path, save=save, dataset=dataset
    )

    results = _load_results(results_path, dataset=dataset)
    if not results:
        print("[shap_plots] No SHAP results available — skipping beeswarm plots.")
        return
    for model_name in results:
        print(f"[shap_plots] Generating beeswarm for '{model_name}'…")
        plot_shap_beeswarm(
            model_name, results_path=results_path, save=save, dataset=dataset
        )


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_all_shap()
