"""Train cold-start variants and aggregate metrics by stratum."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import DatasetPaths, Defaults
from recommender.baseline import train_final_model
from recommender.data import (
    evaluate_ranking,
    predict_ratings,
    rating_reasonableness_limit,
)
from recommender.enhanced.social_regularization import (
    build_social_edges,
    fit_social_cmf_model,
)
from recommender.experiment.cold_start.deltas import per_user_rmse
from recommender.experiment.cold_start.features import load_variant_features
from recommender.experiment.cold_start.strata import STRATA_ORDER, strata_user_map
from recommender.experiment.final_eval import train_enhanced_final
from recommender.experiment.variants import VARIANT_SPECS


def _sane_pred_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ratings_for_scale: np.ndarray,
) -> np.ndarray:
    """Finite preds whose absolute error is within the shared RMSE sanity scale."""
    limit = rating_reasonableness_limit(ratings_for_scale)
    return (
        np.isfinite(y_pred)
        & np.isfinite(y_true)
        & (np.abs(y_pred - y_true) <= limit)
        & (np.abs(y_pred) <= limit)
    )


def _hyperparams_for_variant(
    variant_id: str,
    manifest: dict[str, Any],
    baseline_params: dict[str, Any],
) -> dict[str, Any]:
    entry = (manifest.get("variants") or {}).get(variant_id) or {}
    hp = dict(entry.get("hyperparameters") or {})
    if variant_id == "M1":
        return {
            "k": baseline_params["k"],
            "lambda_reg": baseline_params["lambda_reg"],
        }
    if variant_id == "M3_soft" and not hp:
        entry = (manifest.get("variants") or {}).get("M3") or {}
        hp = dict(entry.get("hyperparameters") or {})
    if not hp:
        # Fall back to M3 / baseline when a variant was never run.
        m3 = ((manifest.get("variants") or {}).get("M3") or {}).get(
            "hyperparameters"
        ) or {}
        return {
            "k": int(m3.get("k", baseline_params["k"])),
            "lambda_reg": float(m3.get("lambda_reg", baseline_params["lambda_reg"])),
            "w_main": float(m3.get("w_main", Defaults.W_MAIN)),
            "w_user": float(m3.get("w_user", Defaults.W_USER)),
            "lambda_social": float(m3.get("lambda_social", 0.1)),
            "beta": float(m3.get("beta", 0.5)),
            "gamma": float(m3.get("gamma", 1.0)),
        }
    return hp


def train_variant_model(
    variant_id: str,
    train_df: pd.DataFrame,
    *,
    dataset: str,
    hyperparameters: dict[str, Any],
    baseline_params: dict[str, Any],
    selected_network: dict[str, Any] | None,
    user_attributes: pd.DataFrame | None = None,
    all_user_ids: list[int] | None = None,
    paths: DatasetPaths | None = None,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    nthreads: int = 1,
    random_state: int = Defaults.CMF_RANDOM_STATE,
):
    """Train one variant; return (model, meta_row_fields)."""
    spec = VARIANT_SPECS[variant_id]
    meta: dict[str, Any] = {
        "k": None,
        "lambda_reg": None,
        "w_main": None,
        "w_user": None,
        "lambda_social": None,
        "social_mode": spec.get("social_mode"),
        "social_normalization": spec.get("social_normalization"),
        "beta": None,
        "gamma": None,
        "diffusion_model": None,
        "alpha_index": None,
        "alpha_value": None,
    }

    def _align_attrs(attrs: pd.DataFrame) -> pd.DataFrame:
        users = set(map(int, train_df["UserId"].unique()))
        if all_user_ids is not None:
            users |= set(map(int, all_user_ids))
        return attrs.reindex(sorted(users | set(map(int, attrs.index)))).fillna(0.0)

    if variant_id == "M1":
        model = train_final_model(
            train_df,
            k=int(baseline_params["k"]),
            lambda_reg=float(baseline_params["lambda_reg"]),
            method=method,
            maxiter=maxiter,
            nthreads=nthreads,
            random_state=random_state,
        )
        meta["k"] = int(baseline_params["k"])
        meta["lambda_reg"] = float(baseline_params["lambda_reg"])
        return model, meta

    if spec.get("trust_features"):
        if user_attributes is None or user_attributes.empty:
            raise ValueError(f"{variant_id} requires trust user_attributes")
        user_attributes = _align_attrs(user_attributes)
        model = train_enhanced_final(
            train_df,
            user_attributes,
            k=int(hyperparameters.get("k", baseline_params["k"])),
            lambda_reg=float(
                hyperparameters.get("lambda_reg", baseline_params["lambda_reg"])
            ),
            w_main=float(hyperparameters.get("w_main", Defaults.W_MAIN)),
            w_user=float(hyperparameters.get("w_user", Defaults.W_USER)),
            method=method,
            maxiter=maxiter,
            nthreads=nthreads,
            random_state=random_state,
        )
        meta.update(
            {
                "k": int(hyperparameters.get("k", baseline_params["k"])),
                "lambda_reg": float(
                    hyperparameters.get("lambda_reg", baseline_params["lambda_reg"])
                ),
                "w_main": float(hyperparameters.get("w_main", Defaults.W_MAIN)),
                "w_user": float(hyperparameters.get("w_user", Defaults.W_USER)),
            }
        )
        return model, meta

    if not selected_network:
        raise ValueError(f"{variant_id} requires selected_network")
    model_name = selected_network["diffusion_model"]
    net_idx = int(selected_network["alpha_index"])

    if user_attributes is None:
        user_attributes = load_variant_features(
            dataset=dataset,
            model_name=model_name,
            network_index=net_idx,
            include_communities=bool(spec["include_communities"]),
            paths=paths,
        )
    if user_attributes is None:
        raise FileNotFoundError(
            f"Missing features for {model_name} network {net_idx}"
        )
    user_attributes = _align_attrs(user_attributes)

    if spec.get("soft_communities"):
        from networks.artifacts import NetworkArtifacts
        from recommender.experiment.route_b.soft_assignment import (
            merge_soft_into_user_attributes,
            soft_community_feature_frame,
        )

        arts = NetworkArtifacts(dataset, paths=paths)
        com_csv = arts.communities_csv(model_name, net_idx)
        if not com_csv.exists():
            raise FileNotFoundError(f"Missing communities for soft assignment: {com_csv}")
        com = pd.read_csv(com_csv).set_index("UserId")
        soft = soft_community_feature_frame(
            train_df,
            com.reset_index(),
            user_ids=list(map(int, user_attributes.index)),
        )
        user_attributes = merge_soft_into_user_attributes(user_attributes, soft)

    if spec["social_regularization"]:
        social_edges = build_social_edges(
            dataset=dataset,
            model_name=model_name,
            network_index=net_idx,
            user_index=list(map(int, train_df["UserId"].unique())),
            mode=spec["social_mode"],  # type: ignore[arg-type]
            beta=float(hyperparameters.get("beta", 0.5)),
            gamma=float(hyperparameters.get("gamma", 1.0)),
            normalization=spec["social_normalization"],  # type: ignore[arg-type]
            paths=paths,
        )
        model = fit_social_cmf_model(
            train_df,
            user_attributes,
            social_edges,
            k=int(hyperparameters["k"]),
            lambda_reg=float(hyperparameters["lambda_reg"]),
            w_main=float(hyperparameters.get("w_main", Defaults.W_MAIN)),
            w_user=float(hyperparameters.get("w_user", Defaults.W_USER)),
            lambda_social=float(hyperparameters["lambda_social"]),
            maxiter=maxiter,
            nthreads=nthreads,
            random_state=random_state,
        )
        meta.update(
            {
                "k": int(hyperparameters["k"]),
                "lambda_reg": float(hyperparameters["lambda_reg"]),
                "w_main": float(hyperparameters.get("w_main", Defaults.W_MAIN)),
                "w_user": float(hyperparameters.get("w_user", Defaults.W_USER)),
                "lambda_social": float(hyperparameters["lambda_social"]),
                "beta": float(hyperparameters.get("beta", 0.5)),
                "gamma": float(hyperparameters.get("gamma", 1.0)),
                "diffusion_model": model_name,
                "alpha_index": net_idx,
                "alpha_value": selected_network.get("alpha_value"),
            }
        )
        return model, meta

    model = train_enhanced_final(
        train_df,
        user_attributes,
        k=int(hyperparameters["k"]),
        lambda_reg=float(hyperparameters["lambda_reg"]),
        w_main=float(hyperparameters.get("w_main", Defaults.W_MAIN)),
        w_user=float(hyperparameters.get("w_user", Defaults.W_USER)),
        method=method,
        maxiter=maxiter,
        nthreads=nthreads,
        random_state=random_state,
    )
    meta.update(
        {
            "k": int(hyperparameters["k"]),
            "lambda_reg": float(hyperparameters["lambda_reg"]),
            "w_main": float(hyperparameters.get("w_main", Defaults.W_MAIN)),
            "w_user": float(hyperparameters.get("w_user", Defaults.W_USER)),
            "diffusion_model": model_name,
            "alpha_index": net_idx,
            "alpha_value": selected_network.get("alpha_value"),
        }
    )
    return model, meta


def metrics_by_stratum(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    user_strata: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Aggregate RMSE/MAE/coverage for each stratum."""
    stratum_users = strata_user_map(user_strata)
    scale = test_df["Rating"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for stratum in STRATA_ORDER:
        users = stratum_users.get(stratum, set())
        mask = test_df["UserId"].isin(list(users)).to_numpy()
        n_ratings = int(mask.sum())
        n_users = int(test_df.loc[mask, "UserId"].nunique()) if n_ratings else 0
        if n_ratings == 0:
            rows.append(
                {
                    "stratum": stratum,
                    "n_users": 0,
                    "n_ratings": 0,
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "coverage": 0.0,
                }
            )
            continue
        sane = mask & _sane_pred_mask(y_true, y_pred, scale)
        coverage = float(sane.sum() / n_ratings) if n_ratings else 0.0
        if not sane.any():
            rmse = mae = float("nan")
        else:
            err = y_true[sane] - y_pred[sane]
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
        rows.append(
            {
                "stratum": stratum,
                "n_users": n_users,
                "n_ratings": n_ratings,
                "rmse": rmse,
                "mae": mae,
                "coverage": coverage,
            }
        )
    return rows


def evaluate_variants_by_stratum(
    *,
    dataset: str,
    mode: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_strata: pd.DataFrame,
    variant_ids: list[str],
    manifest: dict[str, Any],
    baseline_params: dict[str, Any],
    selected_network: dict[str, Any] | None,
    paths: DatasetPaths | None = None,
    trust_attributes: dict[str, pd.DataFrame] | None = None,
    include_ranking: bool = False,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    nthreads: int = 1,
    random_state: int = Defaults.CMF_RANDOM_STATE,
) -> tuple[list[dict[str, Any]], dict[str, pd.Series]]:
    """Train each variant once; return stratum result rows + per-user RMSE map."""
    from recommender.experiment.cold_start.features import resolve_selected_network

    y_true = test_df["Rating"].to_numpy(dtype=float)
    result_rows: list[dict[str, Any]] = []
    rmse_by_variant: dict[str, pd.Series] = {}
    trust_attributes = trust_attributes or {}

    for variant_id in variant_ids:
        print(f"[cold_start] training {variant_id} ({mode})")
        hp = _hyperparams_for_variant(variant_id, manifest, baseline_params)
        user_attrs = trust_attributes.get(variant_id)
        net = resolve_selected_network(manifest, variant_id) or selected_network
        model, meta = train_variant_model(
            variant_id,
            train_df,
            dataset=dataset,
            hyperparameters=hp,
            baseline_params=baseline_params,
            selected_network=net,
            user_attributes=user_attrs,
            all_user_ids=list(map(int, test_df["UserId"].unique())),
            paths=paths,
            method=method,
            maxiter=maxiter,
            nthreads=nthreads,
            random_state=random_state,
        )
        y_pred = predict_ratings(model, test_df)
        rmse_by_variant[variant_id] = per_user_rmse(test_df, y_true, y_pred)
        ranking: dict[str, float] = {}
        if include_ranking:
            ranking = evaluate_ranking(model, train_df, test_df, k=10)

        for stratum_row in metrics_by_stratum(
            test_df, y_true, y_pred, user_strata
        ):
            row = {
                "dataset": dataset,
                "mode": mode,
                "model_variant": variant_id,
                **stratum_row,
                **meta,
            }
            if include_ranking:
                row.update(
                    {
                        "ndcg_at_10": ranking.get("ndcg_at_k"),
                        "precision_at_10": ranking.get("precision_at_k"),
                        "recall_at_10": ranking.get("recall_at_k"),
                        "mrr": ranking.get("mrr"),
                    }
                )
            result_rows.append(row)
    return result_rows, rmse_by_variant
