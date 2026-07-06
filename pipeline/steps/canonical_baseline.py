"""Establish a single canonical M1 baseline reused by final_eval."""

from __future__ import annotations

import argparse
import json

import mlflow

from config import DatasetPaths, MLflow as MlflowCfg
from pipeline._cpu import _resolve_cmf_nthreads
from recommender.baseline import save_search_results, search_baseline_params
from recommender.data import (
    evaluate_single_split,
    load_and_split_dataset,
    metrics_are_reasonable,
)

_CANONICAL_TEST_ATTEMPTS = 3


def run_canonical_baseline(args: argparse.Namespace) -> None:
    dp = DatasetPaths(args.dataset)
    dest = dp.CANONICAL_BASELINE

    if dest.exists() and not getattr(args, "force", False):
        existing = json.loads(dest.read_text(encoding="utf-8"))
        params = existing.get("best_params", {})
        print(
            f"Canonical baseline already exists → {dest}\n"
            f"  k={params.get('k')} lambda_reg={params.get('lambda_reg')}\n"
            "  Pass --force to re-run Optuna search."
        )
        return

    mlflow.set_tracking_uri(MlflowCfg.TRACKING_URI)
    mlflow.set_experiment(MlflowCfg.EXPERIMENT_NAME)

    _, train_df, test_df = load_and_split_dataset(dataset=args.dataset)
    cmf_nthreads = _resolve_cmf_nthreads(args)

    with mlflow.start_run(run_name="canonical_baseline"):
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "n_trials": 50,
                "n_cv_splits": 3,
                "cmf_method": args.cmf_method,
            }
        )
        print("Searching canonical baseline hyperparameters (50 Optuna trials) …")
        search = search_baseline_params(
            train_df,
            n_trials=50,
            n_splits=3,
            method=args.cmf_method,
            maxiter=args.cmf_maxiter,
            nthreads=cmf_nthreads,
            random_state=args.seed,
        )
        best = search["best_params"]
        print(f"Best baseline CV params: {best}")

        from recommender.baseline import train_final_model

        test_metrics: dict[str, float] | None = None
        candidate: dict[str, float] = {"rmse": float("nan")}
        for attempt in range(_CANONICAL_TEST_ATTEMPTS):
            fit_seed = args.seed + attempt
            if attempt:
                print(
                    f"Retrying canonical baseline global test "
                    f"(attempt {attempt + 1}/{_CANONICAL_TEST_ATTEMPTS}, seed={fit_seed}) …"
                )
            model = train_final_model(
                train_df,
                k=best["k"],
                lambda_reg=best["lambda_reg"],
                method=args.cmf_method,
                maxiter=args.cmf_maxiter,
                nthreads=cmf_nthreads,
                random_state=fit_seed,
            )
            candidate = evaluate_single_split(model, test_df)
            if metrics_are_reasonable(candidate, test_df["Rating"]):
                test_metrics = candidate
                break

        if test_metrics is None:
            last_rmse = candidate["rmse"] if candidate else float("nan")
            raise RuntimeError(
                f"Canonical baseline global test metrics degenerate after "
                f"{_CANONICAL_TEST_ATTEMPTS} attempt(s) (last RMSE={last_rmse}). "
                "Re-run with --force or a different --seed."
            )

        search["global_test_rmse"] = test_metrics["rmse"]
        search["global_test_mae"] = test_metrics["mae"]
        search["global_test_r2"] = test_metrics["r2"]
        print(
            f"Canonical baseline (global test) — RMSE: {test_metrics['rmse']:.4f}  "
            f"MAE: {test_metrics['mae']:.4f}  R²: {test_metrics['r2']:.4f}"
        )

        mlflow.log_metrics(
            {
                "baseline_rmse": test_metrics["rmse"],
                "baseline_mae": test_metrics["mae"],
                "baseline_r2": test_metrics["r2"],
            }
        )

    save_search_results(search, path=dest)
    print(f"Canonical baseline saved → {dest}")
