"""Smoke-test runner for Phase 6 social regularization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets, Defaults, Models
from recommender.data import load_dataset, split_data_single
from recommender.enhanced.features import load_network_features
from recommender.enhanced.social_regularization import (
    SocialMode,
    build_social_edges,
    fit_social_cmf_split,
)


def run_social_smoke_test(
    dataset: str = Datasets.DEFAULT,
    model_name: str = "exponential",
    network_index: int = 0,
    social_mode: SocialMode = "boundary_downweight",
    lambda_social: float = 0.01,
    beta: float = 0.5,
    gamma: float = 1.0,
    max_ratings: int = 5000,
    test_size: float = 0.2,
    k: int = 8,
    lambda_reg: float = 1.0,
    w_main: float = Defaults.W_MAIN,
    w_user: float = Defaults.W_USER,
    transform: str = "standard",
    maxiter: int = 5,
    random_state: int = 42,
    nthreads: int = 1,
    include_user_attributes: bool = False,
    output_path: str | Path | None = None,
) -> dict:
    """Run a small before/after smoke test for Phase 6."""
    data = load_dataset(dataset=dataset)
    user_attributes = load_network_features(
        model_name,
        network_index,
        include_communities=True,
        dataset=dataset,
    )
    if user_attributes is None:
        raise FileNotFoundError(
            f"Feature file not found for {dataset}/{model_name}/{network_index:03d}."
        )

    valid_users = sorted(map(int, user_attributes.index))
    data = cast(pd.DataFrame, data.loc[data["UserId"].isin(valid_users)].copy())
    if max_ratings and len(data) > max_ratings:
        data = data.sample(n=max_ratings, random_state=random_state).copy()

    train_df, test_df = split_data_single(
        data,
        test_size=test_size,
        random_state=random_state,
    )
    seen_users = sorted(map(int, train_df["UserId"].unique()))
    seen_items = sorted(map(int, train_df["ItemId"].unique()))
    test_df = cast(
        pd.DataFrame,
        test_df.loc[
            test_df["UserId"].isin(seen_users) & test_df["ItemId"].isin(seen_items)
        ].copy(),
    )
    if test_df.empty:
        raise ValueError("Smoke-test split has no warm test rows after filtering.")
    social_edges = build_social_edges(
        dataset=dataset,
        model_name=model_name,
        network_index=network_index,
        user_index=user_attributes.index,
        mode=social_mode,
        beta=beta,
        gamma=gamma,
        dtype=np.float32,
    )
    if social_edges.n_edges == 0:
        raise ValueError("No usable social edges were built for the smoke test.")

    _, no_social_metrics = fit_social_cmf_split(
        train_df,
        test_df,
        user_attributes,
        social_edges,
        k=k,
        lambda_reg=lambda_reg,
        w_main=w_main,
        w_user=w_user,
        lambda_social=0.0,
        transform=transform,
        maxiter=maxiter,
        nthreads=nthreads,
        random_state=random_state,
        include_user_attributes=include_user_attributes,
    )
    _, social_metrics = fit_social_cmf_split(
        train_df,
        test_df,
        user_attributes,
        social_edges,
        k=k,
        lambda_reg=lambda_reg,
        w_main=w_main,
        w_user=w_user,
        lambda_social=lambda_social,
        transform=transform,
        maxiter=maxiter,
        nthreads=nthreads,
        random_state=random_state,
        include_user_attributes=include_user_attributes,
    )

    rating_span = float(data["Rating"].max() - data["Rating"].min())
    reasonableness_limit = max(10.0, 10.0 * rating_span)

    result = {
        "dataset": dataset,
        "model_name": model_name,
        "network_index": network_index,
        "social_mode": social_mode,
        "lambda_social": lambda_social,
        "beta": beta,
        "gamma": gamma,
        "max_ratings": max_ratings,
        "k": k,
        "lambda_reg": lambda_reg,
        "w_main": w_main,
        "w_user": w_user,
        "include_user_attributes": include_user_attributes,
        "transform": transform,
        "maxiter": maxiter,
        "random_state": random_state,
        "train_ratings": int(len(train_df)),
        "test_ratings": int(len(test_df)),
        "warm_test_only": True,
        "social_edges": {
            key: value
            for key, value in asdict(social_edges).items()
            if key not in {"row", "col", "val"}
        },
        "lambda_social_0": no_social_metrics,
        "lambda_social_on": social_metrics,
        "rmse_delta": social_metrics["rmse"] - no_social_metrics["rmse"],
        "diagnostics": {
            "lambda_social_0_finite": bool(
                all(np.isfinite(value) for value in no_social_metrics.values())
            ),
            "lambda_social_on_finite": bool(
                all(np.isfinite(value) for value in social_metrics.values())
            ),
            "lambda_social_0_reasonable_scale": bool(
                no_social_metrics["rmse"] <= reasonableness_limit
            ),
            "lambda_social_on_reasonable_scale": bool(
                social_metrics["rmse"] <= reasonableness_limit
            ),
        },
    }

    out_path = (
        Path(output_path)
        if output_path is not None
        else DatasetPaths(dataset).BASE / "social_smoke_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small Phase 6 social-regularized CMF smoke test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=Datasets.DEFAULT, choices=Datasets.ALL)
    parser.add_argument(
        "--model", dest="model_name", default="exponential", choices=Models.ALL
    )
    parser.add_argument("--network-index", type=int, default=0)
    parser.add_argument(
        "--social-mode",
        default="boundary_downweight",
        choices=[
            "uniform",
            "community_jaccard",
            "boundary_downweight",
            "bridge_preserve",
        ],
    )
    parser.add_argument("--lambda-social", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max-ratings", type=int, default=5000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--w-user", type=float, default=Defaults.W_USER)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional JSON output path. Defaults to data/<dataset>/social_smoke_results.json.",
    )
    parser.add_argument(
        "--include-user-attributes",
        action="store_true",
        help="Also pass enhanced side-user features as U; off by default for smoke stability.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_social_smoke_test(**vars(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
