"""
SHAP feature importance analysis for the MAFPIN enhanced CMF recommender.

IMPORTANT — Post-hoc interpretability
--------------------------------------
The SHAP values produced here are *post-hoc* explanations of an already-fitted
model.  They describe *how* the model uses its inputs, not causal relationships
between network features and real-world preferences.  Results should be
interpreted as "the model behaves as if feature X matters" and not as evidence
that X causes higher ratings.  Comparisons across datasets or hyperparameter
configurations should be made with care.

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
from typing import Any, cast

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from config import DatasetPaths, Datasets, Defaults, Models
from recommender.data import load_and_split_dataset
from recommender._cmfrec import CMF
from recommender.enhanced import load_network_features
from recommender.enhanced.social_regularization import (
    SocialNormalization,
    SocialMode,
    build_social_edges,
    fit_social_cmf_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WARNING: "normalizer" uses sklearn.preprocessing.Normalizer, which normalises
# each *sample* (row) to unit norm, NOT each *feature* (column).  This is
# semantically incorrect for feature scaling and should not be the first choice.
# Prefer "standard" (StandardScaler) or "minmax" (MinMaxScaler).
_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "normalizer": Normalizer,  # row-normalisation — see warning above
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# Hyperparameter loading
# ---------------------------------------------------------------------------


def load_enhanced_params(path: Path | None = None, dataset: str | None = None) -> dict:
    """
    Load the best enhanced CMF hyperparameters saved by the ``recommend`` step.

    Args:
        path:    Override JSON path.  Defaults to the dataset-specific
                 ``enhanced_search_results.json`` resolved via
                 :class:`~config.DatasetPaths`.
        dataset: Dataset name used to resolve the default path when *path* is
                 not provided.  Defaults to ``Datasets.DEFAULT``.

    Returns:
        Dict with keys ``k``, ``lambda_reg``, ``w_main``, ``w_user``.

    Raises:
        FileNotFoundError: If the JSON has not been created yet.
    """
    p = path or DatasetPaths(dataset or Datasets.DEFAULT).ENHANCED_RESULTS
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
    test_df: pd.DataFrame,
    features: pd.DataFrame,
    params: dict,
    transform: str,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    random_state: int = Defaults.CMF_RANDOM_STATE,
    cmf_nthreads: int = -1,
    social_regularization: bool = False,
    dataset: str | None = None,
    model_name: str | None = None,
    network_index: int | None = None,
    social_mode: SocialMode = "boundary_downweight",
    lambda_social: float = 0.001,
    social_beta: float = 0.5,
    social_gamma: float = 1.0,
    social_normalization: SocialNormalization = "mean_weight",
) -> tuple[CMF, pd.DataFrame]:
    """
    Train the enhanced CMF for a single network and return the fitted model
    together with the scaled feature DataFrame (indexed by ``UserId``).

    Scaling is fitted on training users only to prevent leakage.

    Args:
        train_df:  Training ratings DataFrame.
        test_df:   Test ratings DataFrame used to size L-BFGS factors for later
                predictions.
        features:  Raw feature DataFrame indexed by ``UserId`` (0-based).
        params:    Best-params dict.
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
    if social_regularization:
        if model_name is None or network_index is None:
            raise ValueError("Social SHAP requires model_name and network_index.")
        selected_social_mode = cast(SocialMode, params.get("social_mode", social_mode))
        selected_lambda_social = float(params.get("lambda_social", lambda_social))
        selected_beta = float(params.get("beta", social_beta))
        selected_gamma = float(params.get("gamma", social_gamma))
        social_edges = build_social_edges(
            dataset=dataset or Datasets.DEFAULT,
            model_name=model_name,
            network_index=network_index,
            user_index=features.index,
            mode=selected_social_mode,
            beta=selected_beta,
            gamma=selected_gamma,
            normalization=cast(
                SocialNormalization,
                params.get("social_normalization", social_normalization),
            ),
            dtype=np.float32,
        )
        n_users = int(
            max(
                train_df["UserId"].max(),
                test_df["UserId"].max(),
                int(np.max(features.index.to_numpy(dtype=np.int64))),
            )
            + 1
        )
        n_items = int(max(train_df["ItemId"].max(), test_df["ItemId"].max()) + 1)
        model = fit_social_cmf_model(
            train_df,
            features,
            social_edges,
            k=int(params["k"]),
            lambda_reg=float(params["lambda_reg"]),
            w_main=float(params["w_main"]),
            w_user=float(params["w_user"]),
            lambda_social=selected_lambda_social,
            transform=transform,
            maxiter=maxiter,
            nthreads=cmf_nthreads,
            random_state=random_state,
            include_user_attributes=True,
            n_users=n_users,
            n_items=n_items,
        )
    else:
        u_matrix = scaled.rename_axis("UserId").reset_index()
        kwargs = {
            "method": method,
            "k": params["k"],
            "lambda_": params["lambda_reg"],
            "w_main": params["w_main"],
            "w_user": params["w_user"],
            "nthreads": cmf_nthreads,
            "verbose": False,
        }
        if method == "lbfgs":
            kwargs.update({"maxiter": maxiter, "random_state": random_state})
        model = CMF(**kwargs)
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
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    cmf_nthreads: int = -1,
    social_regularization: bool = False,
    social_mode: SocialMode = "boundary_downweight",
    lambda_social: float = 0.001,
    social_beta: float = 0.5,
    social_gamma: float = 1.0,
    social_normalization: SocialNormalization = "mean_weight",
    surrogate_n_estimators: int = 100,
    surrogate_random_state: int = 42,
    min_users: int = 30,
    surrogate_r2_threshold: float = 0.05,
    dataset: str | None = None,
) -> dict[str, Any]:
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
        min_users:              Minimum users required (≥30 recommended);
                                returns ``None`` if fewer are available.
        surrogate_r2_threshold: Minimum held-out R² for the surrogate to be
                                considered reliable.  Networks with a lower
                                surrogate R² are skipped to avoid reporting
                                SHAP values based on an overfit model.
        dataset:                Dataset name.  Defaults to ``Datasets.DEFAULT``.

    Returns:
        Dict with ``status="ok"`` and SHAP payload, or ``status="skipped"``
        with audit metadata explaining why the network was skipped.
    """
    features = load_network_features(
        model_name, network_index, include_communities, dataset=dataset
    )
    if features is None:
        return {
            "status": "skipped",
            "reason": "features_not_found",
            "network_index": network_index,
            "n_users": 0,
        }

    try:
        model, scaled_features = _train_enhanced_cmf(
            train_df,
            test_df,
            features,
            params,
            transform,
            method=method,
            maxiter=maxiter,
            cmf_nthreads=cmf_nthreads,
            social_regularization=social_regularization,
            dataset=dataset,
            model_name=model_name,
            network_index=network_index,
            social_mode=social_mode,
            lambda_social=lambda_social,
            social_beta=social_beta,
            social_gamma=social_gamma,
            social_normalization=social_normalization,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        return {
            "status": "skipped",
            "reason": "training_failed",
            "error": str(exc),
            "network_index": network_index,
            "n_users": int(len(features)),
        }

    # --- Per-user mean predicted rating on test interactions -----------------
    feat_users = set(scaled_features.index)
    test_filtered = test_df[test_df["UserId"].isin(list(feat_users))].copy()
    if test_filtered.empty:
        return {
            "status": "skipped",
            "reason": "no_test_users_with_features",
            "network_index": network_index,
            "n_users": 0,
        }

    preds = model.predict(
        user=test_filtered["UserId"].values,  # type: ignore[union-attr]
        item=test_filtered["ItemId"].values,  # type: ignore[union-attr]
    )
    test_filtered["_pred"] = preds
    per_user_pred = test_filtered.groupby("UserId")["_pred"].mean()

    # --- Align features and predictions --------------------------------------
    common_users = sorted(per_user_pred.index.intersection(scaled_features.index))
    if len(common_users) < min_users:
        return {
            "status": "skipped",
            "reason": "insufficient_users",
            "network_index": network_index,
            "n_users": int(len(common_users)),
            "min_users": int(min_users),
        }

    X = scaled_features.loc[common_users].values
    y = per_user_pred.loc[common_users].values
    feature_names = list(scaled_features.columns)

    # --- GBT surrogate with held-out quality check ---------------------------
    # Fit on 80% of users, evaluate on the remaining 20% to guard against
    # trivial overfitting when n_users is small.  Skip this network if the
    # surrogate cannot explain even a small fraction of variance on the
    # held-out set (R² < surrogate_r2_threshold).
    surrogate = GradientBoostingRegressor(
        n_estimators=surrogate_n_estimators,
        random_state=surrogate_random_state,
    )
    val_split = max(1, int(0.8 * len(common_users)))
    surrogate_r2: float | None = None
    if val_split < len(common_users):  # enough data for a held-out set
        surrogate.fit(X[:val_split], y[:val_split])
        surrogate_r2 = float(r2_score(y[val_split:], surrogate.predict(X[val_split:])))
        if not np.isfinite(surrogate_r2) or surrogate_r2 < surrogate_r2_threshold:
            print(
                f"  surrogate R²={surrogate_r2:.3f} below threshold "
                f"{surrogate_r2_threshold} — skipping network {network_index:03d}."
            )
            return {
                "status": "skipped",
                "reason": "low_surrogate_r2",
                "network_index": network_index,
                "n_users": int(len(common_users)),
                "surrogate_r2": surrogate_r2,
                "surrogate_r2_threshold": surrogate_r2_threshold,
            }
    # Refit on full data before computing SHAP values
    surrogate.fit(X, y)

    # TreeSHAP: exact, O(n_trees * n_features), fast
    explainer = shap.TreeExplainer(surrogate)
    shap_values: np.ndarray = explainer.shap_values(X)  # (n_users, n_features)

    return {
        "status": "ok",
        "network_index": network_index,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "n_users": int(len(common_users)),
        "surrogate_r2": surrogate_r2,
    }


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
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    cmf_nthreads: int = -1,
    social_regularization: bool = False,
    social_mode: SocialMode = "boundary_downweight",
    lambda_social: float = 0.001,
    social_beta: float = 0.5,
    social_gamma: float = 1.0,
    social_normalization: SocialNormalization = "mean_weight",
) -> dict[str, dict]:
    """
    Run SHAP feature importance analysis over ``k_networks`` random networks
    per diffusion model.

    **Post-hoc note**: The SHAP values are post-hoc explanations of the fitted
    CMF.  They reflect model behaviour, not causal relationships between
    network features and user preferences.  Do not over-interpret directional
    effects — the GBT surrogate adds an additional approximation layer.

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

        A ``shap_skipped_networks.json`` audit file is also written alongside
        the main results when any networks are skipped.
    """
    params = load_enhanced_params(params_path, dataset=dataset)
    _, train_df, test_df = load_and_split_dataset(dataset=dataset)
    dp = DatasetPaths(dataset or Datasets.DEFAULT)

    if model_names is None:
        model_names = Models.ALL

    rng = random.Random(seed)
    results: dict[str, dict] = {}
    skipped_networks: list[dict] = []

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
        model_skipped: list[dict[str, Any]] = []
        surrogate_r2_values: list[float] = []

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
                method=method,
                maxiter=maxiter,
                cmf_nthreads=cmf_nthreads,
                social_regularization=social_regularization,
                social_mode=social_mode,
                lambda_social=lambda_social,
                social_beta=social_beta,
                social_gamma=social_gamma,
                social_normalization=social_normalization,
                dataset=dataset,
            )
            if result.get("status") != "ok":
                reason = str(result.get("reason", "unknown"))
                print(f"skipped ({reason}).")
                audit = {
                    "model": model_name,
                    "index": idx,
                    **{
                        key: _json_ready(value)
                        for key, value in result.items()
                        if key not in {"shap_values", "feature_names"}
                    },
                }
                skipped_networks.append(audit)
                model_skipped.append(audit)
                continue

            sv = cast(np.ndarray, result["shap_values"])
            fn = cast(list[str], result["feature_names"])
            all_shap.append(sv)
            feature_names = fn
            valid_indices.append(idx)
            surrogate_r2 = result.get("surrogate_r2")
            if isinstance(surrogate_r2, (int, float, np.integer, np.floating)):
                surrogate_r2_values.append(float(surrogate_r2))

            # Persist full matrix so plots can be regenerated without re-running.
            # Use atomic tmp+replace so interrupted runs leave no partial .npy files.
            model_matrices_dir = dp.SHAP_MATRICES / model_name
            model_matrices_dir.mkdir(parents=True, exist_ok=True)
            matrix_path = model_matrices_dir / f"{model_name}_{idx:03d}.npy"
            matrix_tmp = matrix_path.with_suffix(".tmp.npy")
            np.save(matrix_tmp, sv)
            matrix_tmp.replace(matrix_path)

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

        # Write a completion manifest so downstream code can verify the matrix
        # set is complete and was not left in a partial state.
        manifest_path = dp.SHAP_MATRICES / model_name / "manifest.json"
        manifest_tmp = manifest_path.with_suffix(".tmp")
        manifest_tmp.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "n_networks": len(all_shap),
                    "n_attempted": len(indices),
                    "n_skipped": len(model_skipped),
                    "network_indices": valid_indices,
                    "matrix_paths": matrix_paths,
                    "feature_names": feature_names,
                    "model_variant": "social" if social_regularization else "enhanced",
                    "social_normalization": (
                        social_normalization if social_regularization else None
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_tmp.replace(manifest_path)

        results[model_name] = {
            "mean_shap_abs": mean_abs.tolist(),
            "mean_shap": mean_signed.tolist(),
            "feature_names": feature_names,
            "n_networks": len(all_shap),
            "n_attempted": len(indices),
            "n_valid": len(all_shap),
            "n_skipped": len(model_skipped),
            "network_indices": valid_indices,
            "matrix_paths": matrix_paths,
            "skipped_networks": model_skipped,
            "surrogate_r2_summary": {
                "count": len(surrogate_r2_values),
                "mean": (
                    float(np.mean(surrogate_r2_values)) if surrogate_r2_values else None
                ),
                "min": (
                    float(np.min(surrogate_r2_values)) if surrogate_r2_values else None
                ),
                "max": (
                    float(np.max(surrogate_r2_values)) if surrogate_r2_values else None
                ),
            },
            "model_variant": "social" if social_regularization else "enhanced",
            "resolved_params": _json_ready(params),
            "post_hoc_target": "per-user mean CMF prediction on held-out test interactions",
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

    # Persist skipped-networks audit log for reproducibility.
    if skipped_networks:
        audit_path = dp.BASE / "shap_skipped_networks.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_tmp = audit_path.with_suffix(".tmp")
        audit_tmp.write_text(json.dumps(skipped_networks, indent=2), encoding="utf-8")
        audit_tmp.replace(audit_path)
        print(f"  Skipped-networks audit saved \u2192 {audit_path}")

    return results


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def save_shap_results(
    results: dict,
    path: Path | None = None,
    dataset: str | None = None,
) -> None:
    """
    Save *results* from :func:`run_shap_analysis` to a JSON file.

    Args:
        results: Output of :func:`run_shap_analysis`.
        path:    Override destination.  Defaults to the dataset-specific
                 ``shap_results.json`` resolved via :class:`~config.DatasetPaths`.
        dataset: Dataset name used to resolve the default path when *path* is
                 not provided.  Defaults to ``Datasets.DEFAULT``.
    """
    dest = path or DatasetPaths(dataset or Datasets.DEFAULT).SHAP_RESULTS
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(dest)
    print(f"SHAP results saved \u2192 {dest}")
