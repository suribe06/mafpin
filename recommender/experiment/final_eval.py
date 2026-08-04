"""Global held-out test evaluation for core experiment variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DatasetPaths, Defaults
from recommender._cmfrec import CMF
from recommender.baseline import train_final_model
from recommender.data import (
    evaluate_ranking,
    evaluate_single_split,
    metrics_are_reasonable,
)
from recommender.enhanced.features import _SCALERS, load_network_features
from recommender.enhanced.social_regularization import (
    build_social_edges,
    fit_social_cmf_model,
)
from recommender.experiment.variants import VARIANT_SPECS


def train_enhanced_final(
    train_df: pd.DataFrame,
    user_attributes: pd.DataFrame,
    *,
    k: int,
    lambda_reg: float,
    w_main: float,
    w_user: float,
    method: str,
    maxiter: int,
    nthreads: int,
    random_state: int,
    transform: str = "standard",
) -> CMF:
    train_users = sorted(train_df["UserId"].unique())
    scaler = _SCALERS[transform]()
    scaler.fit(user_attributes.loc[train_users].values)
    scaled_all = pd.DataFrame(
        scaler.transform(user_attributes.values),
        index=user_attributes.index,
        columns=user_attributes.columns,
    )
    u_matrix = scaled_all.rename_axis("UserId").reset_index()
    kwargs: dict[str, Any] = {
        "method": method,
        "k": k,
        "lambda_": lambda_reg,
        "w_main": w_main,
        "w_user": w_user,
        "nthreads": nthreads,
        "verbose": False,
    }
    if method == "lbfgs":
        kwargs.update({"maxiter": maxiter, "random_state": random_state})
    model = CMF(**kwargs)
    model.fit(X=train_df, U=u_matrix)
    return model


def load_canonical_baseline(dataset: str) -> dict[str, Any]:
    path = DatasetPaths(dataset).CANONICAL_BASELINE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing canonical baseline at {path}. "
            "Run --steps canonical_baseline first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_variant_global_test(
    variant_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    dataset: str,
    hyperparameters: dict[str, Any],
    baseline_params: dict[str, Any],
    selected_network: dict[str, Any] | None,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    nthreads: int = 1,
    random_state: int = Defaults.CMF_RANDOM_STATE,
    ranking_k: int = 10,
) -> dict[str, Any]:
    """Train on full train split and evaluate once on the global test split."""
    spec = VARIANT_SPECS[variant_id]
    row: dict[str, Any] = {
        "dataset": dataset,
        "model_variant": variant_id,
        "k": None,
        "lambda_reg": None,
        "w_main": None,
        "w_user": None,
        "lambda_social": None,
        "social_mode": spec["social_mode"],
        "social_normalization": spec["social_normalization"],
        "beta": None,
        "gamma": None,
        "diffusion_model": None,
        "alpha_index": None,
        "alpha_value": None,
        "baseline_k": baseline_params["k"],
        "baseline_lambda": baseline_params["lambda_reg"],
    }

    if spec.get("trust_features"):
        raise NotImplementedError(
            f"{variant_id} is evaluated via recommender.experiment.cold_start"
        )
    if variant_id == "M1" or not spec["needs_network"]:
        model = train_final_model(
            train_df,
            k=baseline_params["k"],
            lambda_reg=baseline_params["lambda_reg"],
            method=method,
            maxiter=maxiter,
            nthreads=nthreads,
            random_state=random_state,
        )
        row["k"] = baseline_params["k"]
        row["lambda_reg"] = baseline_params["lambda_reg"]
    elif spec["social_regularization"]:
        if not selected_network:
            raise ValueError(f"{variant_id} requires selected_network")
        model_name = selected_network["diffusion_model"]
        net_idx = int(selected_network["alpha_index"])
        user_attrs = load_network_features(
            model_name,
            net_idx,
            include_communities=spec["include_communities"],
            dataset=dataset,
        )
        if user_attrs is None:
            raise FileNotFoundError(
                f"Missing features for {model_name} network {net_idx}"
            )
        social_edges = build_social_edges(
            dataset=dataset,
            model_name=model_name,
            network_index=net_idx,
            user_index=list(map(int, train_df["UserId"].unique())),
            mode=spec["social_mode"],  # type: ignore[arg-type]
            beta=float(hyperparameters.get("beta", 0.5)),
            gamma=float(hyperparameters.get("gamma", 1.0)),
            normalization=spec["social_normalization"],  # type: ignore[arg-type]
        )
        model = fit_social_cmf_model(
            train_df,
            user_attrs,
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
        row.update(
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
    else:
        if not selected_network:
            raise ValueError(f"{variant_id} requires selected_network")
        model_name = selected_network["diffusion_model"]
        net_idx = int(selected_network["alpha_index"])
        user_attrs = load_network_features(
            model_name,
            net_idx,
            include_communities=spec["include_communities"],
            dataset=dataset,
        )
        if user_attrs is None:
            raise FileNotFoundError(
                f"Missing features for {model_name} network {net_idx}"
            )
        model = train_enhanced_final(
            train_df,
            user_attrs,
            k=int(hyperparameters["k"]),
            lambda_reg=float(hyperparameters["lambda_reg"]),
            w_main=float(hyperparameters.get("w_main", Defaults.W_MAIN)),
            w_user=float(hyperparameters.get("w_user", Defaults.W_USER)),
            method=method,
            maxiter=maxiter,
            nthreads=nthreads,
            random_state=random_state,
        )
        row.update(
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

    metrics = evaluate_single_split(model, test_df)
    ranking = evaluate_ranking(model, train_df, test_df, k=ranking_k)
    row.update(
        {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "ndcg_at_10": ranking["ndcg_at_k"],
            "precision_at_10": ranking["precision_at_k"],
            "recall_at_10": ranking["recall_at_k"],
            "mrr": ranking["mrr"],
            "valid_metric_row": bool(np.isfinite(metrics["rmse"])),
        }
    )
    return row


def apply_final_eval_deltas(
    rows: list[dict[str, Any]],
    *,
    canonical_baseline_rmse: float | None = None,
    ratings: pd.Series | np.ndarray | None = None,
) -> None:
    """Set rmse_delta_vs_baseline / rmse_delta_vs_m3 from the same-session M1 row."""
    m1_rmse = next(
        (float(r["rmse"]) for r in rows if r.get("model_variant") == "M1"),
        None,
    )
    if m1_rmse is None and canonical_baseline_rmse is not None:
        candidate = {"rmse": float(canonical_baseline_rmse)}
        if ratings is None or metrics_are_reasonable(candidate, ratings):
            m1_rmse = float(canonical_baseline_rmse)

    m3_rmse = next(
        (float(r["rmse"]) for r in rows if r.get("model_variant") == "M3"),
        None,
    )
    for row in rows:
        rmse = float(row["rmse"])
        if m1_rmse is not None and np.isfinite(m1_rmse):
            row["rmse_delta_vs_baseline"] = m1_rmse - rmse
        else:
            row["rmse_delta_vs_baseline"] = float("nan")
        if row.get("model_variant") == "M3":
            row["rmse_delta_vs_m3"] = 0.0
        elif m3_rmse is not None:
            row["rmse_delta_vs_m3"] = m3_rmse - rmse
        else:
            row["rmse_delta_vs_m3"] = float("nan")


def append_core_results(rows: list[dict[str, Any]], path: Path) -> None:
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_old = pd.read_csv(path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new = df_new.drop_duplicates(subset=["dataset", "model_variant"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df_new.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f"Core experiment results → {path}")
