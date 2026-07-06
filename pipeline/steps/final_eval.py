"""Global held-out test evaluation for core experiment variants."""

from __future__ import annotations

import argparse
import json

import mlflow

from config import DatasetPaths, MLflow as MlflowCfg
from pipeline._cpu import _resolve_cmf_nthreads
from recommender.data import load_and_split_dataset
from recommender.experiment.final_eval import (
    append_core_results,
    apply_final_eval_deltas,
    evaluate_variant_global_test,
    load_canonical_baseline,
)
from recommender.experiment.manifest import load_manifest
from recommender.experiment.variants import CORE_VARIANT_IDS, VARIANT_SPECS


def run_final_eval(args: argparse.Namespace) -> None:
    dp = DatasetPaths(args.dataset)
    manifest = load_manifest(args.dataset)
    baseline_search = load_canonical_baseline(args.dataset)
    baseline_params = baseline_search["best_params"]

    selection_path = dp.NETWORK_SELECTION
    network_selection: dict = {}
    if selection_path.exists():
        network_selection = json.loads(selection_path.read_text(encoding="utf-8"))

    _, train_df, test_df = load_and_split_dataset(dataset=args.dataset)
    cmf_nthreads = _resolve_cmf_nthreads(args)

    if args.all_variants:
        variant_ids = list(CORE_VARIANT_IDS)
    elif args.model_variant:
        variant_ids = [args.model_variant]
    else:
        variant_ids = list(CORE_VARIANT_IDS)

    mlflow.set_tracking_uri(MlflowCfg.TRACKING_URI)
    mlflow.set_experiment(MlflowCfg.EXPERIMENT_NAME)

    rows: list[dict] = []

    with mlflow.start_run(run_name="final_eval"):
        mlflow.log_param("dataset", args.dataset)
        mlflow.log_param("variants", ",".join(variant_ids))

        for variant_id in variant_ids:
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
                selected = (
                    entry.get("selected_network")
                    or network_selection.get("variants", {}).get(variant_id)
                )
                if not selected:
                    print(f"  SKIP {variant_id}: no selected network (run network_selection)")
                    continue

            with mlflow.start_run(run_name=f"final_{variant_id}", nested=True):
                row = evaluate_variant_global_test(
                    variant_id,
                    train_df,
                    test_df,
                    dataset=args.dataset,
                    hyperparameters=hyperparams,
                    baseline_params=baseline_params,
                    selected_network=selected,
                    method=args.cmf_method,
                    maxiter=args.cmf_maxiter,
                    nthreads=cmf_nthreads,
                    random_state=args.seed,
                )
                rows.append(row)
                print(
                    f"  Global test — RMSE: {row['rmse']:.4f}  "
                    f"MAE: {row['mae']:.4f}  R²: {row['r2']:.4f}  "
                    f"NDCG@10: {row['ndcg_at_10']:.4f}",
                    flush=True,
                )
                mlflow.log_metrics(
                    {
                        "rmse": row["rmse"],
                        "mae": row["mae"],
                        "r2": row["r2"],
                        "ndcg_at_10": row["ndcg_at_10"],
                    }
                )

    apply_final_eval_deltas(
        rows,
        canonical_baseline_rmse=baseline_search.get("global_test_rmse"),
        ratings=test_df["Rating"],
    )

    if rows:
        append_core_results(rows, dp.CORE_EXPERIMENT_RESULTS)
    else:
        print("No final_eval rows produced.")
