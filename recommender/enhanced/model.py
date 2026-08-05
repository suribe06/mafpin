"""
CMF model training and evaluation with user-side attributes.

Warm-split / scale / fit live here so social eval and final_eval share one core.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd

from config import Defaults
from recommender._cmfrec import CMF
from recommender.data import evaluate_ranking, evaluate_single_split, split_data_single
from recommender.enhanced.features import _SCALERS


def filter_to_feature_users(
    data: pd.DataFrame,
    user_attributes: pd.DataFrame,
) -> pd.DataFrame | None:
    """Ratings restricted to users present in the feature matrix, or None."""
    filtered = data.loc[data["UserId"].isin(list(user_attributes.index))].copy()
    if filtered.empty:
        print("  Warning: no overlap between rating users and network users.")
        return None
    return filtered


def iter_warm_splits(
    filtered: pd.DataFrame,
    *,
    n_splits: int = 5,
    test_size: float = 0.2,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(split_idx, train_df, warm_test)``; skip empty warm tests (M-4)."""
    for split_idx in range(n_splits):
        train_df, test_df = split_data_single(
            filtered, test_size=test_size, random_state=split_idx
        )
        seen_users = list(train_df["UserId"].unique())
        seen_items = list(train_df["ItemId"].unique())
        warm_test = test_df.loc[
            test_df["UserId"].isin(seen_users) & test_df["ItemId"].isin(seen_items)
        ].copy()
        if warm_test.empty:
            continue
        yield split_idx, train_df, warm_test


def scaled_u_matrix(
    user_attributes: pd.DataFrame,
    train_users: list[Any],
    transform: str = "standard",
) -> pd.DataFrame:
    """Fit scaler on *train_users* only (M-2), return U frame with UserId column."""
    if transform not in _SCALERS:
        raise ValueError(
            f"Unknown transform: {transform!r}. Use one of {list(_SCALERS)}."
        )
    scaler = _SCALERS[transform]()
    scaler.fit(user_attributes.loc[train_users].values)
    scaled_all = pd.DataFrame(
        scaler.transform(user_attributes.values),
        index=user_attributes.index,
        columns=user_attributes.columns,
    )
    return scaled_all.rename_axis("UserId").reset_index()


def fit_enhanced_cmf(
    train_df: pd.DataFrame,
    user_attributes: pd.DataFrame,
    *,
    k: int,
    lambda_reg: float,
    w_main: float = Defaults.W_MAIN,
    w_user: float = Defaults.W_USER,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    nthreads: int = -1,
    random_state: int = Defaults.CMF_RANDOM_STATE,
    transform: str = "standard",
) -> CMF:
    """Scale user attributes on train users and fit enhanced CMF."""
    train_users = sorted(train_df["UserId"].unique())
    u_matrix = scaled_u_matrix(user_attributes, train_users, transform=transform)
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


def evaluate_cmf_with_user_attributes(
    data: pd.DataFrame,
    user_attributes: pd.DataFrame,
    k: int = 20,
    lambda_reg: float = 1.0,
    w_main: float = Defaults.W_MAIN,
    w_user: float = Defaults.W_USER,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    random_state: int = Defaults.CMF_RANDOM_STATE,
    n_splits: int = 5,
    test_size: float = 0.2,
    transform: str = "standard",
    baseline_k: int | None = None,
    baseline_lambda: float | None = None,
    compute_ranking: bool = False,
    ranking_k: int = 10,
    cmf_nthreads: int = -1,
) -> list[dict]:
    """
    Evaluate enhanced CMF via repeated warm train/test splits.

    Feature scaling is fitted on training users only (M-2).  A paired baseline
    CMF (no side information) shares the same split (M-3).
    """
    if transform not in _SCALERS:
        raise ValueError(
            f"Unknown transform: {transform!r}. Use one of {list(_SCALERS)}."
        )

    filtered = filter_to_feature_users(data, user_attributes)
    if filtered is None:
        return []

    results: list[dict] = []
    for split_idx, train_df, warm_test in iter_warm_splits(
        filtered, n_splits=n_splits, test_size=test_size
    ):
        enhanced_model = fit_enhanced_cmf(
            train_df,
            user_attributes,
            k=k,
            lambda_reg=lambda_reg,
            w_main=w_main,
            w_user=w_user,
            method=method,
            maxiter=maxiter,
            nthreads=cmf_nthreads,
            random_state=random_state + split_idx,
            transform=transform,
        )
        enhanced_rmse = evaluate_single_split(enhanced_model, warm_test)["rmse"]

        if baseline_k is not None and baseline_lambda is not None:
            baseline_kwargs: dict[str, Any] = {
                "method": method,
                "k": baseline_k,
                "lambda_": baseline_lambda,
                "nthreads": cmf_nthreads,
                "verbose": False,
            }
            if method == "lbfgs":
                baseline_kwargs.update(
                    {"maxiter": maxiter, "random_state": random_state + split_idx}
                )
            baseline_model = CMF(**baseline_kwargs)
            baseline_model.fit(X=train_df)
            baseline_rmse = evaluate_single_split(baseline_model, warm_test)["rmse"]
        else:
            baseline_rmse = float("nan")

        result: dict = {
            "rmse_enhanced": enhanced_rmse,
            "rmse_baseline": baseline_rmse,
            "improvement": baseline_rmse - enhanced_rmse,
        }
        if compute_ranking:
            ranking = evaluate_ranking(enhanced_model, train_df, warm_test, k=ranking_k)
            result.update(ranking)
        results.append(result)

    return results
