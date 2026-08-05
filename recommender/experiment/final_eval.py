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
from recommender.enhanced.features import load_network_features
from recommender.enhanced.model import fit_enhanced_cmf
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
    return fit_enhanced_cmf(
        train_df,
        user_attributes,
        k=k,
        lambda_reg=lambda_reg,
        w_main=w_main,
        w_user=w_user,
        method=method,
        maxiter=maxiter,
        nthreads=nthreads,
        random_state=random_state,
        transform=transform,
    )


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
    save_predictions_path: Path | None = None,
    beyond_accuracy: bool = False,
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
    reasonable = metrics_are_reasonable(metrics, pd.Series(test_df["Rating"]))
    if not reasonable:
        print(
            f"  WARNING: {variant_id} RMSE {metrics['rmse']:.4g} is off the rating "
            "scale — the fit diverged; row flagged invalid_metric_row"
        )
    row.update(
        {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "ndcg_at_10": ranking["ndcg_at_k"],
            "precision_at_10": ranking["precision_at_k"],
            "recall_at_10": ranking["recall_at_k"],
            "mrr": ranking["mrr"],
            "valid_metric_row": reasonable,
        }
    )

    if save_predictions_path is not None:
        from recommender.data import predict_ratings

        pred = test_df.loc[:, ["UserId", "ItemId", "Rating"]].copy()
        assert isinstance(pred, pd.DataFrame)
        pred["Prediction"] = predict_ratings(model, pred)
        pred["variant"] = variant_id
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pred.to_parquet(save_predictions_path, index=False)
        except Exception:
            # ponytail: parquet engine optional; CSV always works
            pred.to_csv(save_predictions_path.with_suffix(".csv"), index=False)
        print(f"  Predictions → {save_predictions_path}")

    if beyond_accuracy:
        from recommender.experiment.route_b.beyond_accuracy import (
            compute_beyond_accuracy_metrics,
        )

        agg, per_user = compute_beyond_accuracy_metrics(
            model, train_df, test_df, dataset=dataset, k=ranking_k
        )
        row.update({f"ba_{k}": v for k, v in agg.items()})
        row["_beyond_accuracy_per_user"] = per_user

    return row


def apply_final_eval_deltas(
    rows: list[dict[str, Any]],
    *,
    canonical_baseline_rmse: float | None = None,
    ratings: pd.Series | np.ndarray | None = None,
) -> None:
    """Set rmse_delta_vs_baseline / rmse_delta_vs_m3 from the same-session M1 row."""
    m1_rmse = next(
        (
            float(r["rmse"])
            for r in rows
            if r.get("model_variant") == "M1" and r.get("valid_metric_row", True)
        ),
        None,
    )
    if m1_rmse is None and canonical_baseline_rmse is not None:
        candidate = {"rmse": float(canonical_baseline_rmse)}
        if ratings is None or metrics_are_reasonable(candidate, ratings):
            m1_rmse = float(canonical_baseline_rmse)

    # A diverged reference would turn every delta into garbage, so ignore it.
    m3_rmse = next(
        (
            float(r["rmse"])
            for r in rows
            if r.get("model_variant") == "M3" and r.get("valid_metric_row", True)
        ),
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


def run_final_eval(
    dataset: str,
    *,
    variant_ids: list[str] | None = None,
    all_variants: bool = False,
    cmf_method: str = Defaults.CMF_METHOD,
    cmf_maxiter: int = Defaults.CMF_MAXITER,
    cmf_nthreads: int = 1,
    random_state: int = Defaults.CMF_RANDOM_STATE,
    beyond_accuracy: bool = False,
    save_predictions: bool = False,
) -> list[dict[str, Any]]:
    """Run global-test final eval for core experiment variants (MLflow + skip rules)."""
    import mlflow

    from config import MLflow as MlflowCfg
    from recommender.data import load_and_split_dataset
    from recommender.experiment.manifest import load_manifest
    from recommender.experiment.route_b.paths import RouteBPaths
    from recommender.experiment.variants import CORE_VARIANT_IDS

    dp = DatasetPaths(dataset)
    route_paths = RouteBPaths(dataset)
    if beyond_accuracy or save_predictions:
        route_paths.ensure_dirs()
    manifest = load_manifest(dataset)
    baseline_search = load_canonical_baseline(dataset)
    baseline_params = baseline_search["best_params"]

    selection_path = dp.NETWORK_SELECTION
    network_selection: dict = {}
    if selection_path.exists():
        network_selection = json.loads(selection_path.read_text(encoding="utf-8"))

    _, train_df, test_df = load_and_split_dataset(dataset=dataset)

    if all_variants or not variant_ids:
        resolved = list(CORE_VARIANT_IDS)
    else:
        resolved = list(variant_ids)

    mlflow.set_tracking_uri(MlflowCfg.TRACKING_URI)
    mlflow.set_experiment(MlflowCfg.EXPERIMENT_NAME)

    rows: list[dict[str, Any]] = []
    ba_per_user_frames: list[pd.DataFrame] = []
    with mlflow.start_run(run_name="final_eval"):
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("variants", ",".join(resolved))
        mlflow.log_param("route_b", str(beyond_accuracy or save_predictions))
        mlflow.set_tags({"route_b": str(beyond_accuracy or save_predictions).lower()})

        for variant_id in resolved:
            print(f"\n--- Final eval: {variant_id} ---", flush=True)
            spec = VARIANT_SPECS[variant_id]
            entry = manifest.get("variants", {}).get(variant_id, {})
            hyperparams = entry.get("hyperparameters") or {}

            if variant_id == "M1":
                hyperparams = dict(baseline_params)
            elif not hyperparams:
                print(f"  SKIP {variant_id}: no hyperparameters in manifest")
                continue

            selected = None
            if spec["needs_network"]:
                selected = entry.get("selected_network") or network_selection.get(
                    "variants", {}
                ).get(variant_id)
                if not selected:
                    print(
                        f"  SKIP {variant_id}: no selected network (run network_selection)"
                    )
                    continue

            pred_path = (
                route_paths.prediction_path(variant_id) if save_predictions else None
            )
            with mlflow.start_run(run_name=f"final_{variant_id}", nested=True):
                row = evaluate_variant_global_test(
                    variant_id,
                    train_df,
                    test_df,
                    dataset=dataset,
                    hyperparameters=hyperparams,
                    baseline_params=baseline_params,
                    selected_network=selected,
                    method=cmf_method,
                    maxiter=cmf_maxiter,
                    nthreads=cmf_nthreads,
                    random_state=random_state,
                    save_predictions_path=pred_path,
                    beyond_accuracy=beyond_accuracy,
                )
                per_user = row.pop("_beyond_accuracy_per_user", None)
                if per_user is not None and isinstance(per_user, pd.DataFrame):
                    tmp = per_user.copy()
                    tmp.insert(0, "model_variant", variant_id)
                    tmp.insert(0, "dataset", dataset)
                    ba_per_user_frames.append(tmp)
                rows.append(row)
                print(
                    f"  Global test — RMSE: {row['rmse']:.4f}  "
                    f"MAE: {row['mae']:.4f}  R²: {row['r2']:.4f}  "
                    f"NDCG@10: {row['ndcg_at_10']:.4f}",
                    flush=True,
                )
                metrics_log = {
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                    "r2": row["r2"],
                    "ndcg_at_10": row["ndcg_at_10"],
                }
                if beyond_accuracy and "ba_cce_at_k_mean" in row:
                    metrics_log["cce_at_10"] = float(row["ba_cce_at_k_mean"])
                mlflow.log_metrics(metrics_log)

    apply_final_eval_deltas(
        rows,
        canonical_baseline_rmse=baseline_search.get("global_test_rmse"),
        ratings=pd.Series(test_df["Rating"]),
    )
    if beyond_accuracy and rows:
        ba_rows = []
        for row in rows:
            ba_rows.append(
                {
                    "dataset": row["dataset"],
                    "model_variant": row["model_variant"],
                    **{k[3:]: v for k, v in row.items() if k.startswith("ba_")},
                    "rmse": row.get("rmse"),
                    "ndcg_at_10": row.get("ndcg_at_10"),
                }
            )
        pd.DataFrame(ba_rows).to_csv(route_paths.BEYOND_ACCURACY, index=False)
        print(f"Beyond-accuracy → {route_paths.BEYOND_ACCURACY}")
        if ba_per_user_frames:
            per_user_all = pd.concat(ba_per_user_frames, ignore_index=True)
            try:
                per_user_all.to_parquet(
                    route_paths.BEYOND_ACCURACY_PER_USER, index=False
                )
            except Exception:
                per_user_all.to_csv(
                    route_paths.BEYOND_ACCURACY_PER_USER.with_suffix(".csv"),
                    index=False,
                )
            print(f"Beyond-accuracy per-user → {route_paths.BEYOND_ACCURACY_PER_USER}")

    # Keep core CSV free of Route B-only columns.
    core_rows = [
        {k: v for k, v in row.items() if not k.startswith("ba_")} for row in rows
    ]
    if core_rows:
        append_core_results(core_rows, dp.CORE_EXPERIMENT_RESULTS)
    else:
        print("No final_eval rows produced.")

    return rows


_CANONICAL_TEST_ATTEMPTS = 3


def run_canonical_baseline(
    dataset: str,
    *,
    force: bool = False,
    cmf_method: str = Defaults.CMF_METHOD,
    cmf_maxiter: int = Defaults.CMF_MAXITER,
    cmf_nthreads: int = 1,
    random_state: int = Defaults.CMF_RANDOM_STATE,
    n_trials: int = 50,
) -> dict[str, Any] | None:
    """Optuna + global-test canonical M1 baseline; skip if file exists unless *force*."""
    import mlflow

    from config import MLflow as MlflowCfg
    from recommender.baseline import save_search_results, search_baseline_params, train_final_model
    from recommender.data import load_and_split_dataset

    dp = DatasetPaths(dataset)
    dest = dp.CANONICAL_BASELINE
    if dest.exists() and not force:
        existing = json.loads(dest.read_text(encoding="utf-8"))
        params = existing.get("best_params", {})
        print(
            f"Canonical baseline already exists → {dest}\n"
            f"  k={params.get('k')} lambda_reg={params.get('lambda_reg')}\n"
            "  Pass --force to re-run Optuna search."
        )
        return existing

    mlflow.set_tracking_uri(MlflowCfg.TRACKING_URI)
    mlflow.set_experiment(MlflowCfg.EXPERIMENT_NAME)

    _, train_df, test_df = load_and_split_dataset(dataset=dataset)
    with mlflow.start_run(run_name="canonical_baseline"):
        mlflow.log_params(
            {
                "dataset": dataset,
                "n_trials": n_trials,
                "n_cv_splits": 3,
                "cmf_method": cmf_method,
            }
        )
        print(f"Searching canonical baseline hyperparameters ({n_trials} Optuna trials) …")
        search = search_baseline_params(
            train_df,
            n_trials=n_trials,
            n_splits=3,
            method=cmf_method,
            maxiter=cmf_maxiter,
            nthreads=cmf_nthreads,
            random_state=random_state,
        )
        best = search["best_params"]
        print(f"Best baseline CV params: {best}")

        test_metrics: dict[str, float] | None = None
        candidate: dict[str, float] = {"rmse": float("nan")}
        for attempt in range(_CANONICAL_TEST_ATTEMPTS):
            fit_seed = random_state + attempt
            if attempt:
                print(
                    f"Retrying canonical baseline global test "
                    f"(attempt {attempt + 1}/{_CANONICAL_TEST_ATTEMPTS}, seed={fit_seed}) …"
                )
            model = train_final_model(
                train_df,
                k=best["k"],
                lambda_reg=best["lambda_reg"],
                method=cmf_method,
                maxiter=cmf_maxiter,
                nthreads=cmf_nthreads,
                random_state=fit_seed,
            )
            candidate = evaluate_single_split(model, test_df)
            if metrics_are_reasonable(candidate, pd.Series(test_df["Rating"])):
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
    return search
