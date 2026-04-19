"""
SHAP feature importance analysis for the MAFPIN enhanced CMF recommender.

Strategy
--------
For each sampled (diffusion model, network) pair:

1. Train the enhanced CMF with the best hyperparameters found during the
   ``recommend`` step, loaded from ``data/enhanced_search_results.json``.
2. Predict ratings on the **test set** using the fitted CMF.  The per-user
   mean predicted rating becomes the target variable — it captures which users
   the model anticipates will rate items highly, independently of their actual
   ratings.
3. Fit a ``GradientBoostingRegressor`` surrogate on
   ``(scaled network features) → mean predicted rating``.  The surrogate is
   a thin wrapper whose sole purpose is enabling efficient SHAP computation.
4. Apply ``shap.TreeExplainer`` (exact, fast) to the surrogate.  Because the
   surrogate is trained on CMF outputs, the resulting SHAP values explain
   *the CMF's behaviour*, not the surrogate's.
5. Average |SHAP| values across the ``k`` sampled networks per diffusion model
   to obtain a robust, model-level feature importance ranking.

Why a surrogate instead of KernelExplainer directly on CMF?
-----------------------------------------------------------
In ALS-based CMF, user embeddings are fixed after ``fit()``.  Calling
``model.predict()`` with a perturbed ``U`` matrix does not change the output
for already-seen users — their factors are baked in.  Retraining the model for
every KernelExplainer perturbation (~2^n_features evaluations) would be
computationally infeasible.  The surrogate approach gives *exact* SHAP values
for a model trained directly on CMF outputs, faithfully approximating CMF's
feature-to-prediction sensitivity.

Usage
-----
From the command line (after running ``--steps recommend``)::

    python pipeline.py --steps shap

Programmatic::

    from analysis.shap_analysis import run_shap_analysis
    results = run_shap_analysis(k_networks=5, include_communities=True)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from cmfrec import CMF  # type: ignore[import-untyped]
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from config import DatasetPaths, Datasets, Models
from recommender.data import load_and_split_dataset
from recommender.enhanced import load_network_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENHANCED_PARAMS_PATH = DatasetPaths(Datasets.DEFAULT).ENHANCED_RESULTS
_SHAP_RESULTS_PATH = DatasetPaths(Datasets.DEFAULT).SHAP_RESULTS
_SHAP_MATRICES_DIR = DatasetPaths(Datasets.DEFAULT).SHAP_MATRICES

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "normalizer": Normalizer,
}


# ---------------------------------------------------------------------------
# Hyperparameter loading
# ---------------------------------------------------------------------------


def load_enhanced_params(path: Path | None = None) -> dict:
    """
    Load the best enhanced CMF hyperparameters saved by the ``recommend`` step.

    Args:
        path: Override JSON path.  Defaults to
              ``data/enhanced_search_results.json``.

    Returns:
        Dict with keys ``k``, ``lambda_reg``, ``w_main``, ``w_user``.

    Raises:
        FileNotFoundError: If the JSON has not been created yet.
    """
    p = path or _ENHANCED_PARAMS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Enhanced hyperparameters not found at {p}.\n"
            "Run 'python pipeline.py --steps recommend' first to generate it."
        )
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["best_params"]


# ---------------------------------------------------------------------------
# Network index helpers
# ---------------------------------------------------------------------------


def _available_indices(model_name: str, dataset: str | None = None) -> list[int]:
    """Return the sorted list of network indices available for *model_name*."""
    centrality_dir = DatasetPaths(dataset or Datasets.DEFAULT).CENTRALITY / model_name
    if not centrality_dir.exists():
        return []
    return sorted(
        int(p.stem.rsplit("_", 1)[-1])
        for p in centrality_dir.glob(f"centrality_metrics_{model_name}_*.csv")
    )


def _sample_indices(
    model_name: str, k: int, rng: random.Random, dataset: str | None = None
) -> list[int]:
    """Sample up to *k* network indices without replacement."""
    available = _available_indices(model_name, dataset=dataset)
    if not available:
        return []
    return sorted(rng.sample(available, min(k, len(available))))


# ---------------------------------------------------------------------------
# CMF training helper
# ---------------------------------------------------------------------------


def _train_enhanced_cmf(
    train_df: pd.DataFrame,
    features: pd.DataFrame,
    params: dict,
    transform: str,
) -> tuple[CMF, pd.DataFrame]:
    """
    Train the enhanced CMF for a single network and return the fitted model
    together with the scaled feature DataFrame (indexed by ``UserId``).

    Scaling is fitted on training users only to prevent leakage.

    Args:
        train_df:  Training ratings DataFrame.
        features:  Raw feature DataFrame indexed by ``UserId`` (0-based).
        params:    Best-params dict (``k``, ``lambda_reg``, ``w_main``,
                   ``w_user``).
        transform: Scaler key — ``"standard"``, ``"minmax"``, or
                   ``"normalizer"``.

    Returns:
        ``(fitted_model, scaled_features_df)``
    """
    feat_users = set(features.index)
    train_users = sorted(u for u in train_df["UserId"].unique() if u in feat_users)

    scaler = _SCALERS[transform]()
    scaler.fit(features.loc[train_users].values)

    scaled = pd.DataFrame(
        scaler.transform(features.values),
        index=features.index,
        columns=features.columns,
    )
    u_matrix = scaled.reset_index()  # cmfrec requires UserId as a column

    model = CMF(
        method="als",
        k=params["k"],
        lambda_=params["lambda_reg"],
        w_main=params["w_main"],
        w_user=params["w_user"],
        verbose=False,
    )
    model.fit(X=train_df, U=u_matrix)

    return model, scaled


# ---------------------------------------------------------------------------
# Per-network SHAP computation
# ---------------------------------------------------------------------------


def compute_shap_for_network(
    model_name: str,
    network_index: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    include_communities: bool = True,
    transform: str = "standard",
    surrogate_n_estimators: int = 100,
    surrogate_random_state: int = 42,
    min_users: int = 10,
) -> tuple[np.ndarray, list[str]] | None:
    """
    Train enhanced CMF on one (model, network) pair and compute SHAP values.

    The surrogate GBT is trained to predict per-user mean CMF-predicted
    ratings on the test set.  ``shap.TreeExplainer`` then provides exact SHAP
    values for the surrogate, which serve as an efficient proxy for CMF's
    feature sensitivities.

    Args:
        model_name:             Diffusion model name.
        network_index:          Zero-based network index.
        train_df:               Training ratings DataFrame.
        test_df:                Test ratings DataFrame.
        params:                 Best enhanced CMF hyperparameters.
        include_communities:    Include LPH and ``num_communities`` features.
        transform:              Feature scaling method.
        surrogate_n_estimators: Trees in the GBT surrogate.
        surrogate_random_state: Seed for the surrogate.
        min_users:              Minimum users required; returns ``None`` if
                                fewer are available.

    Returns:
        ``(shap_values, feature_names)`` where ``shap_values`` has shape
        ``(n_users, n_features)``, or ``None`` if data is insufficient.
    """
    features = load_network_features(model_name, network_index, include_communities)
    if features is None:
        return None

    model, scaled_features = _train_enhanced_cmf(train_df, features, params, transform)

    # --- Per-user mean predicted rating on test interactions -----------------
    feat_users = set(scaled_features.index)
    test_filtered = test_df[test_df["UserId"].isin(list(feat_users))].copy()
    if test_filtered.empty:
        return None

    preds = model.predict(
        user=test_filtered["UserId"].values,  # type: ignore[union-attr]
        item=test_filtered["ItemId"].values,  # type: ignore[union-attr]
    )
    test_filtered["_pred"] = preds
    per_user_pred = test_filtered.groupby("UserId")["_pred"].mean()

    # --- Align features and predictions --------------------------------------
    common_users = sorted(per_user_pred.index.intersection(scaled_features.index))
    if len(common_users) < min_users:
        return None

    X = scaled_features.loc[common_users].values
    y = per_user_pred.loc[common_users].values
    feature_names = list(scaled_features.columns)

    # --- GBT surrogate trained on CMF predictions ----------------------------
    surrogate = GradientBoostingRegressor(
        n_estimators=surrogate_n_estimators,
        random_state=surrogate_random_state,
    )
    surrogate.fit(X, y)

    # TreeSHAP: exact, O(n_trees * n_features), fast
    explainer = shap.TreeExplainer(surrogate)
    shap_values: np.ndarray = explainer.shap_values(X)  # (n_users, n_features)

    return shap_values, feature_names


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------


def run_shap_analysis(
    k_networks: int | None = 20,
    include_communities: bool = True,
    seed: int = 42,
    model_names: list[str] | None = None,
    params_path: Path | None = None,
    transform: str = "standard",
    dataset: str | None = None,
) -> dict[str, dict]:
    """
    Run SHAP feature importance analysis over ``k_networks`` random networks
    per diffusion model.

    For each model the mean absolute SHAP value per feature is computed by
    averaging |SHAP| across all successfully processed networks.  The signed
    mean is also recorded to indicate the *direction* of each feature's effect
    (positive = higher feature value → higher predicted rating).

    Args:
        k_networks:          Number of networks to sample per diffusion model.
                             Pass ``None`` to use **all** available networks.
        include_communities: Include LPH and ``num_communities`` features.
        seed:                Random seed for reproducible network sampling.
        model_names:         Subset of diffusion models to analyse.  Defaults
                             to all three (exponential, powerlaw, rayleigh).
        params_path:         Override path for the enhanced search results JSON.
        transform:           Feature scaling method (``"standard"`` recommended).

    Returns:
        Dict mapping model name → result dict::

            {
                "mean_shap_abs":   list[float],  # mean |SHAP| per feature
                "mean_shap":       list[float],  # mean SHAP per feature (signed)
                "feature_names":   list[str],
                "n_networks":      int,
                "network_indices": list[int],
            }
    """
    params = load_enhanced_params(params_path)
    _, train_df, test_df = load_and_split_dataset(dataset=dataset)
    dp = DatasetPaths(dataset or Datasets.DEFAULT)

    if model_names is None:
        model_names = Models.ALL

    rng = random.Random(seed)
    results: dict[str, dict] = {}

    for model_name in model_names:
        print(f"\n{'='*55}\nModel: {model_name.upper()}\n{'='*55}")

        if k_networks is None:
            indices = _available_indices(model_name, dataset=dataset)
        else:
            indices = _sample_indices(model_name, k_networks, rng, dataset=dataset)
        if not indices:
            print("  No networks found, skipping.")
            continue

        all_shap: list[np.ndarray] = []
        feature_names: list[str] = []
        valid_indices: list[int] = []

        for idx in indices:
            print(f"  [{model_name}] network {idx:03d} ...", end=" ", flush=True)
            result = compute_shap_for_network(
                model_name,
                idx,
                train_df,
                test_df,
                params,
                include_communities=include_communities,
                transform=transform,
            )
            if result is None:
                print("skipped (insufficient data).")
                continue

            sv, fn = result
            all_shap.append(sv)
            feature_names = fn
            valid_indices.append(idx)

            # Persist full matrix so plots can be regenerated without re-running.
            model_matrices_dir = dp.SHAP_MATRICES / model_name
            model_matrices_dir.mkdir(parents=True, exist_ok=True)
            matrix_path = model_matrices_dir / f"{model_name}_{idx:03d}.npy"
            np.save(matrix_path, sv)

            print(f"OK  ({sv.shape[0]} users, {sv.shape[1]} features)")

        if not all_shap:
            print(f"  No valid networks processed for {model_name}.")
            continue

        # Average per-network statistics to produce model-level importances.
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in all_shap], axis=0)
        mean_signed = np.mean([sv.mean(axis=0) for sv in all_shap], axis=0)

        matrix_paths = [
            str(dp.SHAP_MATRICES / model_name / f"{model_name}_{i:03d}.npy")
            for i in valid_indices
        ]
        results[model_name] = {
            "mean_shap_abs": mean_abs.tolist(),
            "mean_shap": mean_signed.tolist(),
            "feature_names": feature_names,
            "n_networks": len(all_shap),
            "network_indices": valid_indices,
            "matrix_paths": matrix_paths,
        }

        # Pretty-print ranked feature importances.
        order = np.argsort(mean_abs)[::-1]
        print(f"\n  Feature importance ({model_name}, {len(all_shap)} networks):")
        for rank, i in enumerate(order, 1):
            direction = "+" if mean_signed[i] >= 0 else "-"
            print(
                f"    {rank:2d}. {feature_names[i]:<30s}"
                f"|SHAP|={mean_abs[i]:.5f}  dir={direction}"
            )

    return results


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def save_shap_results(results: dict, path: Path | None = None) -> None:
    """
    Save *results* from :func:`run_shap_analysis` to a JSON file.

    Args:
        results: Output of :func:`run_shap_analysis`.
        path:    Override destination.  Defaults to
                 ``data/shap_results.json``.
    """
    dest = path or _SHAP_RESULTS_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"SHAP results saved → {dest}")
