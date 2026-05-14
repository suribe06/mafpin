"""Smoke-test runner for Phase 6 social regularization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, cast

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


def _float_slug(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _grid_result_row(
    path: Path,
    result: dict,
    status: str,
    error: str | None = None,
) -> dict:
    social_metrics = result.get("lambda_social_on", {})
    baseline_metrics = result.get("lambda_social_0", {})
    diagnostics = result.get("diagnostics", {})
    social_edges = result.get("social_edges", {})
    return {
        "status": status,
        "dataset": result.get("dataset"),
        "model_name": result.get("model_name"),
        "network_index": result.get("network_index"),
        "social_mode": result.get("social_mode"),
        "lambda_social": result.get("lambda_social"),
        "lambda_reg": result.get("lambda_reg"),
        "w_user": result.get("w_user"),
        "rmse": social_metrics.get("rmse"),
        "mae": social_metrics.get("mae"),
        "r2": social_metrics.get("r2"),
        "baseline_rmse": baseline_metrics.get("rmse"),
        "baseline_mae": baseline_metrics.get("mae"),
        "baseline_r2": baseline_metrics.get("r2"),
        "rmse_delta": result.get("rmse_delta"),
        "edges": social_edges.get("n_edges"),
        "baseline_reasonable": diagnostics.get("lambda_social_0_reasonable_scale"),
        "social_regularized_reasonable": diagnostics.get(
            "lambda_social_on_reasonable_scale"
        ),
        "path": str(path),
        "error": error,
    }


def run_user_attribute_grid(
    dataset: str = Datasets.DEFAULT,
    model_name: str = "exponential",
    network_index: int = 0,
    social_mode: SocialMode = "boundary_downweight",
    lambda_social: float = 0.001,
    beta: float = 0.5,
    gamma: float = 1.0,
    max_ratings: int = 5000,
    test_size: float = 0.2,
    k: int = 8,
    lambda_reg_grid: Iterable[float] = (1.0, 3.0, 10.0),
    w_user_grid: Iterable[float] = (0.01, 0.05, 0.1),
    w_main: float = Defaults.W_MAIN,
    transform: str = "standard",
    maxiter: int = 20,
    random_state: int = 42,
    nthreads: int = 1,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run Step 4: side-user attributes over a constrained parameter grid."""
    base_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else DatasetPaths(dataset).BASE / "social_smoke_results" / "user_attribute_grid"
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for lambda_reg_value in lambda_reg_grid:
        for w_user_value in w_user_grid:
            output_path = base_output_dir / (
                f"{model_name}_{network_index:03d}_{social_mode}"
                f"_lambda_social_{_float_slug(lambda_social)}"
                f"_lambda_reg_{_float_slug(lambda_reg_value)}"
                f"_w_user_{_float_slug(w_user_value)}.json"
            )
            try:
                if output_path.exists() and not overwrite:
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                else:
                    result = run_social_smoke_test(
                        dataset=dataset,
                        model_name=model_name,
                        network_index=network_index,
                        social_mode=social_mode,
                        lambda_social=lambda_social,
                        beta=beta,
                        gamma=gamma,
                        max_ratings=max_ratings,
                        test_size=test_size,
                        k=k,
                        lambda_reg=float(lambda_reg_value),
                        w_main=w_main,
                        w_user=float(w_user_value),
                        transform=transform,
                        maxiter=maxiter,
                        random_state=random_state,
                        nthreads=nthreads,
                        include_user_attributes=True,
                        output_path=output_path,
                    )
                rows.append(_grid_result_row(output_path, result, status="ok"))
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                rows.append(
                    _grid_result_row(
                        output_path,
                        {
                            "dataset": dataset,
                            "model_name": model_name,
                            "network_index": network_index,
                            "social_mode": social_mode,
                            "lambda_social": lambda_social,
                            "lambda_reg": float(lambda_reg_value),
                            "w_user": float(w_user_value),
                        },
                        status="error",
                        error=str(exc),
                    )
                )

    summary = pd.DataFrame(rows).sort_values(["lambda_reg", "w_user"])
    summary.to_csv(base_output_dir / "user_attribute_grid_summary.csv", index=False)
    return summary


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
    parser.add_argument(
        "--lambda-reg-grid",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 10.0],
        help="Step 4 grid values for lambda_reg when --user-attribute-grid is set.",
    )
    parser.add_argument("--w-user", type=float, default=Defaults.W_USER)
    parser.add_argument(
        "--w-user-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1],
        help="Step 4 grid values for w_user when --user-attribute-grid is set.",
    )
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional JSON output path. Defaults to data/<dataset>/social_smoke_results.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for --user-attribute-grid summaries and JSON files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-user-attributes",
        action="store_true",
        help="Also pass enhanced side-user features as U; off by default for smoke stability.",
    )
    parser.add_argument(
        "--user-attribute-grid",
        action="store_true",
        help="Run Step 4 over lambda_reg_grid x w_user_grid with side-user attributes enabled.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.user_attribute_grid:
        summary = run_user_attribute_grid(
            dataset=args.dataset,
            model_name=args.model_name,
            network_index=args.network_index,
            social_mode=args.social_mode,
            lambda_social=args.lambda_social,
            beta=args.beta,
            gamma=args.gamma,
            max_ratings=args.max_ratings,
            k=args.k,
            lambda_reg_grid=args.lambda_reg_grid,
            w_user_grid=args.w_user_grid,
            maxiter=args.maxiter,
            nthreads=args.nthreads,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        output_dir = (
            Path(args.output_dir)
            if args.output_dir is not None
            else DatasetPaths(args.dataset).BASE
            / "social_smoke_results"
            / "user_attribute_grid"
        )
        ok_count = int((summary["status"] == "ok").sum())
        print(f"Completed {ok_count}/{len(summary)} Step 4 user-attribute-grid runs")
        print(f"Saved summary to {output_dir / 'user_attribute_grid_summary.csv'}")
        return

    result = run_social_smoke_test(
        dataset=args.dataset,
        model_name=args.model_name,
        network_index=args.network_index,
        social_mode=args.social_mode,
        lambda_social=args.lambda_social,
        beta=args.beta,
        gamma=args.gamma,
        max_ratings=args.max_ratings,
        k=args.k,
        lambda_reg=args.lambda_reg,
        w_user=args.w_user,
        maxiter=args.maxiter,
        nthreads=args.nthreads,
        include_user_attributes=args.include_user_attributes,
        output_path=args.output_path,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
