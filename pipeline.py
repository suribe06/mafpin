"""
MAFPIN unified pipeline.

Run individual steps or the full pipeline from the command line.

Examples
--------
Full pipeline (all steps for all models)::

    python pipeline.py --all

Cascade generation only::

    python pipeline.py --steps cascade

Network inference + centrality for the exponential model only::

    python pipeline.py --steps inference centrality --model exponential

Recommendation with community side information::

    python pipeline.py --steps recommend --include-communities

Available steps
---------------
cascade
    Convert ratings CSV → cascades.txt for NetInf input.
delta
    Compute median inter-event delta and alpha grid parameters.
inference
    Run NetInf to infer diffusion networks (generates per-alpha CSV files).
centrality
    Compute seven SNAP centrality metrics for every inferred network.
    Requires the communities step to have run first so that the
    ``pagerank_lph`` column (LPH-weighted custom PageRank) can be included.
communities
    Detect overlapping communities (Demon / ASLPAw) and compute LPH.
    Run baseline CMF + enhanced CMF with network side information.
hypertune
    Optuna TPE search for enhanced CMF hyperparameters only (k, lambda_reg,
    w_main, w_user).  Saves best params to data/enhanced_search_results.json
    without running the full network evaluation.  Run this before ``shap``
    if you want to avoid re-running the recommendation evaluation.
shap
    SHAP feature importance analysis for the enhanced CMF.  Loads the best
    enhanced hyperparameters from data/enhanced_search_results.json, samples
    k networks per diffusion model, trains a GBT surrogate on CMF outputs,
    and applies TreeSHAP.  Saves per-model importance rankings to
    data/shap_results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from config import DatasetPaths as ConfigDatasetPaths, Defaults


class _TeeStream:
    def __init__(self, primary: TextIO, log_file: TextIO) -> None:
        self.primary = primary
        self.log_file = log_file
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.log_file.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self.primary, name)


def _cpu_thread_limit(cpu_fraction: float) -> int:
    cpu_count = os.cpu_count() or 1
    safe_fraction = min(max(cpu_fraction, 0.05), 1.0)
    return max(1, int(cpu_count * safe_fraction))


def _resolve_cmf_nthreads(args: argparse.Namespace) -> int:
    explicit = getattr(args, "cmf_nthreads", 0)
    if explicit and explicit > 0:
        return int(explicit)
    return _cpu_thread_limit(float(args.cpu_fraction))


def _configure_cpu_limits(args: argparse.Namespace) -> int:
    nthreads = _resolve_cmf_nthreads(args)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(nthreads)
    print(
        f"CPU limit: CMF/BLAS threads capped at {nthreads} "
        f"(~{float(args.cpu_fraction):.0%} of detected cores)."
    )
    return nthreads


def _default_log_path(dataset: str) -> Path:
    return ConfigDatasetPaths(dataset).BASE / "pipeline.log"


def _open_pipeline_log(args: argparse.Namespace) -> TextIO | None:
    if args.no_log:
        return None
    log_path = Path(args.log_file) if args.log_file else _default_log_path(args.dataset)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    print(f"Pipeline log: {log_path}", flush=True)
    return log_file


def _best_rmse_from_results(search_result: dict[str, Any]) -> float | None:
    values: list[float] = []
    for row in search_result.get("all_results", []):
        rmse = row.get("rmse") if isinstance(row, dict) else None
        if isinstance(rmse, (int, float)) and math.isfinite(float(rmse)):
            values.append(float(rmse))
    return min(values) if values else None


def _print_best_hyperparams(label: str, search_result: dict[str, Any]) -> None:
    best_params = search_result.get("best_params") or {}
    print(f"\n{label} best hyperparameters:", flush=True)
    if best_params:
        print(json.dumps(best_params, indent=2, sort_keys=True), flush=True)
    else:
        print("{}", flush=True)

    best_value = search_result.get("best_value")
    if best_value is None:
        best_value = _best_rmse_from_results(search_result)
    if isinstance(best_value, (int, float)) and math.isfinite(float(best_value)):
        print(f"{label} best RMSE: {float(best_value):.6f}", flush=True)

    best_metrics = search_result.get("best_metrics") or {}
    if best_metrics:
        print(f"{label} best metrics:", flush=True)
        print(json.dumps(best_metrics, indent=2, sort_keys=True), flush=True)


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def _run_cascade(args: argparse.Namespace) -> None:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from networks.cascades import generate_cascades_from_df
    from config import DatasetPaths, Datasets, Split

    ds_name = args.dataset
    cfg = Datasets.CONFIG[ds_name]
    csv_path = Datasets.ROOT / ds_name / cfg["file"]
    cols = [cfg["col_user"], cfg["col_item"], cfg["col_rating"], cfg["col_time"]]
    df = pd.read_csv(  # type: ignore[call-overload]
        csv_path,
        sep=cfg["sep"],
        header=cfg["header"],
        usecols=cols,  # type: ignore[call-overload]
        engine="python",
    )
    df.columns = pd.Index(["UserId", "ItemId", "Rating", "timestamp"])

    # Apply the global split so NetInf learns from training interactions only.
    # Pass all_user_ids=df["UserId"] so the cascade header declares the full
    # user-ID space — keeping network compact IDs aligned with LabelEncoder.
    train_df, _ = train_test_split(
        df, test_size=Split.TEST_SIZE, random_state=Split.RANDOM_STATE
    )
    generate_cascades_from_df(
        pd.DataFrame(train_df),
        all_user_ids=df["UserId"],
        output_file=DatasetPaths(ds_name).CASCADES,
    )

    from networks.cascades import compute_cascade_user_stats

    compute_cascade_user_stats(dataset=ds_name)

    from visualization.network_plots import plot_cascades_timeline

    plot_cascades_timeline(
        cascade_file=str(DatasetPaths(ds_name).CASCADES),
        save=True,
        dataset=ds_name,
    )


def _run_delta(_args: argparse.Namespace) -> None:
    from networks.delta import compute_median_delta, alpha_centers_from_delta
    from config import DatasetPaths

    delta = compute_median_delta(DatasetPaths(_args.dataset).CASCADES)
    print(f"Median delta: {delta:.4f} days")
    centers = alpha_centers_from_delta(delta)
    for model, info in centers.items():
        print(f"  {model}: alpha0 = {info['alpha0']:.4e} days⁻¹")


def _run_inference(args: argparse.Namespace) -> None:
    from networks.inference import infer_networks_all_models, infer_networks
    from networks.delta import (
        compute_median_delta,
        alpha_centers_from_delta,
        log_alpha_grid,
    )
    from config import DatasetPaths, Datasets

    dp = DatasetPaths(
        args.dataset if hasattr(args, "dataset") and args.dataset else Datasets.DEFAULT
    )
    model = args.model
    model_index_map = {"exponential": 0, "powerlaw": 1, "rayleigh": 2}
    if model:
        _ = compute_median_delta(dp.CASCADES)  # kept for reference
        _ = alpha_centers_from_delta(_)  # kept for reference
        _ = log_alpha_grid  # kept for reference
        infer_networks(
            cascades_file=dp.CASCADES,
            n=args.n_alphas,
            model=model_index_map[model],
            max_iter=args.max_iter,
            k_avg_degree=args.k_avg_degree if args.k_avg_degree > 0 else None,
            name_output=str(dp.NETWORKS / model),
            r=Defaults.RANGE_R,
            networks_dir=dp.NETWORKS,
        )
    else:
        infer_networks_all_models(
            n=args.n_alphas,
            max_iter=args.max_iter,
            k_avg_degree=args.k_avg_degree if args.k_avg_degree > 0 else None,
            networks_dir=dp.NETWORKS,
            cascades_file=dp.CASCADES,
        )

    from visualization.model_plots import plot_alpha_edges

    plot_alpha_edges(save_plot=True, dataset=args.dataset)


def _run_centrality(args: argparse.Namespace) -> None:
    from networks.centrality import calculate_centrality_for_all_models
    from visualization.network_plots import plot_all_centrality_distributions
    from config import Models

    calculate_centrality_for_all_models(
        dataset=args.dataset if hasattr(args, "dataset") else None
    )

    for _mn in Models.ALL:
        plot_all_centrality_distributions(_mn, "000", save=True, dataset=args.dataset)


def _run_communities(args: argparse.Namespace) -> None:
    from networks.communities import calculate_communities_for_all_models
    from visualization.community_plots import (
        plot_lph_distribution,
        plot_num_communities_dist,
        plot_alpha_vs_lph,
        plot_alpha_vs_num_communities,
        plot_lph_vs_centrality,
        plot_community_correlation_heatmap,
    )

    calculate_communities_for_all_models(
        dataset=args.dataset if hasattr(args, "dataset") else None
    )

    plot_lph_distribution(dataset=args.dataset)
    plot_num_communities_dist(dataset=args.dataset)
    plot_alpha_vs_lph(dataset=args.dataset)
    plot_alpha_vs_num_communities(dataset=args.dataset)
    plot_lph_vs_centrality(dataset=args.dataset)
    plot_community_correlation_heatmap(dataset=args.dataset)


def _run_recommend(args: argparse.Namespace) -> None:
    import mlflow
    from recommender.data import load_and_split_dataset, evaluate_single_split
    from recommender.baseline import train_final_model, search_baseline_params
    from recommender.enhanced import (
        run_network_evaluation,
        search_enhanced_params,
        save_enhanced_search_results,
        load_network_features,
    )
    from config import Models, DatasetPaths, MLflow as MlflowCfg

    dp = DatasetPaths(args.dataset)

    mlflow.set_tracking_uri(MlflowCfg.TRACKING_URI)
    mlflow.set_experiment(MlflowCfg.EXPERIMENT_NAME)

    with mlflow.start_run(run_name="recommend"):
        cmf_nthreads = _resolve_cmf_nthreads(args)
        mlflow.log_params(
            {
                "include_communities": args.include_communities,
                "sample_networks": args.sample_networks,
                "all_networks": args.all_networks,
                "model": args.model or "all",
                "n_baseline_optuna_trials": 50,
                "n_enhanced_optuna_trials": 50,
                "n_social_optuna_trials": (
                    args.social_n_trials if args.social_regularization else 0
                ),
                "n_cv_splits": 3,
                "cmf_method": args.cmf_method,
                "cmf_maxiter": args.cmf_maxiter,
                "cmf_nthreads": cmf_nthreads,
                "cpu_fraction": args.cpu_fraction,
                "social_regularization": args.social_regularization,
            }
        )

        _, train_df, test_df = load_and_split_dataset(dataset=args.dataset)
        selected_models = [args.model] if args.model else Models.ALL
        selected_social_mode = args.social_mode
        selected_lambda_social = args.lambda_social
        selected_social_beta = args.social_beta
        selected_social_gamma = args.social_gamma

        # Find first available feature file to represent the feature space.
        sample_features = None
        sample_model_name = None
        for _mn in selected_models:
            sample_features = load_network_features(
                _mn,
                0,
                include_communities=args.include_communities,
                dataset=args.dataset,
            )
            if sample_features is not None:
                sample_model_name = _mn
                break

        enhanced_search = None
        if sample_features is not None:
            # Independent Optuna search for the baseline (k, lambda_reg).
            print(
                "Searching best baseline hyperparameters "
                "(Optuna TPE — k, lambda_reg) …"
            )
            with mlflow.start_run(run_name="baseline_search", nested=True):
                baseline_search = search_baseline_params(
                    train_df,
                    n_trials=50,
                    n_splits=3,
                    method=args.cmf_method,
                    maxiter=args.cmf_maxiter,
                    nthreads=cmf_nthreads,
                )
            best_k_b = baseline_search["best_params"]["k"]
            best_lambda_b = baseline_search["best_params"]["lambda_reg"]
            _print_best_hyperparams("Baseline CMF", baseline_search)

            if args.social_regularization:
                from recommender.enhanced.social_search import (
                    search_social_regularized_params,
                )

                print(
                    f"Searching best social CMF hyperparameters (Optuna TPE) "
                    f"using {sample_model_name} network #000 "
                    f"({args.social_n_trials} trials) …"
                )
                with mlflow.start_run(run_name="social_search", nested=True):
                    enhanced_search = search_social_regularized_params(
                        dataset=args.dataset,
                        model_name=sample_model_name or selected_models[0],
                        network_index=0,
                        n_trials=args.social_n_trials,
                        max_ratings=args.social_search_max_ratings,
                        maxiter=args.cmf_maxiter,
                        random_state=args.seed,
                        nthreads=cmf_nthreads,
                        include_user_attributes=True,
                        output_path=dp.SOCIAL_RESULTS,
                    )
                if not enhanced_search["best_params"]:
                    print("Social hyperparameter search produced no usable trials.")
                    sys.exit(1)
            else:
                # Independent Optuna search for the enhanced model
                print(
                    f"Searching best enhanced hyperparameters (Optuna TPE — k, "
                    f"lambda_reg, w_main, w_user) using first "
                    f"{sample_model_name} network …"
                )
                with mlflow.start_run(run_name="enhanced_search", nested=True):
                    enhanced_search = search_enhanced_params(
                        train_df,
                        sample_features,
                        n_trials=50,
                        n_splits=3,
                        method=args.cmf_method,
                        maxiter=args.cmf_maxiter,
                        cmf_nthreads=cmf_nthreads,
                    )
                save_enhanced_search_results(enhanced_search, path=dp.ENHANCED_RESULTS)
            best_k_e = enhanced_search["best_params"]["k"]
            best_lambda_e = enhanced_search["best_params"]["lambda_reg"]
            best_w_main = enhanced_search["best_params"]["w_main"]
            best_w_user = enhanced_search["best_params"]["w_user"]
            if args.social_regularization:
                selected_social_mode = enhanced_search["best_params"]["social_mode"]
                selected_lambda_social = enhanced_search["best_params"]["lambda_social"]
                selected_social_beta = enhanced_search["best_params"]["beta"]
                selected_social_gamma = enhanced_search["best_params"]["gamma"]
            _print_best_hyperparams(
                "Social CMF" if args.social_regularization else "Enhanced CMF",
                enhanced_search,
            )
        else:
            print("No feature files found — using default params.")
            best_k_b = Defaults.K
            best_lambda_b = Defaults.LAMBDA_REG
            best_k_e = Defaults.K
            best_lambda_e = Defaults.LAMBDA_REG
            best_w_main = Defaults.W_MAIN
            best_w_user = Defaults.W_USER
            baseline_search = {
                "best_params": {"k": best_k_b, "lambda_reg": best_lambda_b},
                "all_results": [],
            }
            enhanced_search = {
                "best_params": (
                    {
                        "k": best_k_e,
                        "lambda_reg": best_lambda_e,
                        "w_main": best_w_main,
                        "w_user": best_w_user,
                        "lambda_social": selected_lambda_social,
                        "social_mode": selected_social_mode,
                        "beta": selected_social_beta,
                        "gamma": selected_social_gamma,
                    }
                    if args.social_regularization
                    else {
                        "k": best_k_e,
                        "lambda_reg": best_lambda_e,
                        "w_main": best_w_main,
                        "w_user": best_w_user,
                    }
                ),
                "all_results": [],
            }
            _print_best_hyperparams("Baseline CMF", baseline_search)
            _print_best_hyperparams(
                "Social CMF" if args.social_regularization else "Enhanced CMF",
                enhanced_search,
            )

        mlflow.log_params(
            {
                "k_baseline": best_k_b,
                "lambda_baseline": best_lambda_b,
                "k_enhanced": best_k_e,
                "lambda_enhanced": best_lambda_e,
                "w_main": best_w_main,
                "w_user": best_w_user,
                "social_mode": selected_social_mode,
                "lambda_social": selected_lambda_social,
                "social_beta": selected_social_beta,
                "social_gamma": selected_social_gamma,
            }
        )

        # Train final baseline model with its own independently tuned k/lambda.
        print(
            f"Training final baseline: method={args.cmf_method}, "
            f"k={best_k_b}, lambda_reg={best_lambda_b:.4f}"
        )
        baseline_model = train_final_model(
            train_df,
            k=best_k_b,
            lambda_reg=best_lambda_b,
            method=args.cmf_method,
            maxiter=args.cmf_maxiter,
            nthreads=cmf_nthreads,
        )
        baseline_metrics = evaluate_single_split(baseline_model, test_df)
        print(
            f"Baseline (global test) — RMSE: {baseline_metrics['rmse']:.4f}  "
            f"MAE: {baseline_metrics['mae']:.4f}  R²: {baseline_metrics['r2']:.4f}"
        )

        mlflow.log_metrics(
            {
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_r2": baseline_metrics["r2"],
            }
        )

        # Persist the global test-set baseline RMSE.
        from recommender.baseline import save_search_results as _save_baseline

        baseline_search["global_test_rmse"] = baseline_metrics["rmse"]
        _save_baseline(baseline_search, path=dp.BASELINE_RESULTS)

        # Enhanced evaluation — pass pre-tuned enhanced params.
        all_results = run_network_evaluation(
            data=train_df,
            model_names=selected_models,
            include_communities=args.include_communities,
            sample_networks=999_999 if args.all_networks else args.sample_networks,
            k=best_k_e,
            lambda_reg=best_lambda_e,
            w_main=best_w_main,
            w_user=best_w_user,
            baseline_k=best_k_b,
            baseline_lambda=best_lambda_b,
            compute_ranking=True,
            dataset=args.dataset,
            n_jobs=cmf_nthreads if args.n_jobs == -1 else args.n_jobs,
            method=args.cmf_method,
            maxiter=args.cmf_maxiter,
            cmf_nthreads=cmf_nthreads,
            use_social_regularization=args.social_regularization,
            social_mode=selected_social_mode,
            lambda_social=selected_lambda_social,
            social_beta=selected_social_beta,
            social_gamma=selected_social_gamma,
        )

        for _artifact in [
            dp.BASELINE_RESULTS,
            dp.SOCIAL_RESULTS if args.social_regularization else dp.ENHANCED_RESULTS,
        ]:
            if _artifact.exists():
                mlflow.log_artifact(str(_artifact))

    # --- plots ---------------------------------------------------------------
    from visualization.model_plots import (
        plot_hyperparameter_search_results,
        plot_parameter_heatmap,
        plot_convergence_analysis,
        plot_metrics_comparison,
        plot_alpha_rmse_analysis,
        plot_alpha_delta_rmse,
        plot_alpha_edges,
        plot_ranking_metrics_per_alpha,
        plot_ranking_metrics_comparison,
    )

    for _search, _prefix in [
        (baseline_search, "baseline"),
        (
            enhanced_search if sample_features is not None else None,
            "social" if args.social_regularization else "enhanced",
        ),
    ]:
        if _search and _search.get("all_results"):
            plot_hyperparameter_search_results(
                _search, save_path=f"{_prefix}_hyper_search.png", dataset=args.dataset
            )
            plot_parameter_heatmap(
                _search, save_path=f"{_prefix}_param_heatmap.png", dataset=args.dataset
            )
            plot_convergence_analysis(
                _search, save_path=f"{_prefix}_convergence.png", dataset=args.dataset
            )
            plot_metrics_comparison(
                _search, save_path=f"{_prefix}_metrics.png", dataset=args.dataset
            )

    global_rmse = baseline_metrics["rmse"]
    for _mn, _rmse_list in all_results.items():
        if _rmse_list:
            plot_alpha_rmse_analysis(
                model_name=_mn,
                rmse_values=_rmse_list,
                baseline_rmse=global_rmse,
                save_plot=True,
                dataset=args.dataset,
            )
            plot_alpha_delta_rmse(
                model_name=_mn,
                rmse_values=_rmse_list,
                baseline_rmse=global_rmse,
                save_plot=True,
                dataset=args.dataset,
            )

    plot_alpha_edges(save_plot=True, dataset=args.dataset)

    for _mn in Models.ALL:
        plot_ranking_metrics_per_alpha(_mn, save_plot=True, dataset=args.dataset)
    plot_ranking_metrics_comparison(save_plot=True, dataset=args.dataset)


def _run_hypertune(args: argparse.Namespace) -> None:
    import mlflow
    from recommender.data import load_and_split_dataset
    from recommender.enhanced import (
        search_enhanced_params,
        save_enhanced_search_results,
        load_network_features,
    )
    from config import Models, MLflow as MlflowCfg, DatasetPaths

    dp = DatasetPaths(args.dataset)
    mlflow.set_tracking_uri(MlflowCfg.TRACKING_URI)
    mlflow.set_experiment(MlflowCfg.EXPERIMENT_NAME)

    with mlflow.start_run(run_name="hypertune"):
        cmf_nthreads = _resolve_cmf_nthreads(args)
        mlflow.log_params(
            {
                "include_communities": args.include_communities,
                "n_enhanced_optuna_trials": 50,
                "n_social_optuna_trials": (
                    args.social_n_trials if args.social_regularization else 0
                ),
                "n_cv_splits": 3,
                "cmf_method": args.cmf_method,
                "cmf_maxiter": args.cmf_maxiter,
                "cmf_nthreads": cmf_nthreads,
                "cpu_fraction": args.cpu_fraction,
                "social_regularization": args.social_regularization,
            }
        )

        _, train_df, _ = load_and_split_dataset(dataset=args.dataset)
        selected_models = [args.model] if args.model else Models.ALL

        sample_features = None
        sample_model_name = None
        for _mn in selected_models:
            sample_features = load_network_features(
                _mn,
                0,
                include_communities=args.include_communities,
                dataset=args.dataset,
            )
            if sample_features is not None:
                sample_model_name = _mn
                break

        if sample_features is None:
            print("No feature files found. Run --steps centrality first.")
            sys.exit(1)

        if args.social_regularization:
            from recommender.enhanced.social_search import (
                search_social_regularized_params,
            )

            print(
                f"Searching best social CMF hyperparameters (Optuna TPE) "
                f"using {sample_model_name} network #000 "
                f"({args.social_n_trials} trials) …"
            )
            enhanced_search = search_social_regularized_params(
                dataset=args.dataset,
                model_name=sample_model_name or selected_models[0],
                network_index=0,
                n_trials=args.social_n_trials,
                max_ratings=args.social_search_max_ratings,
                maxiter=args.cmf_maxiter,
                random_state=args.seed,
                nthreads=cmf_nthreads,
                include_user_attributes=True,
                output_path=dp.SOCIAL_RESULTS,
            )
            _artifact = dp.SOCIAL_RESULTS
        else:
            print(
                f"Searching best enhanced hyperparameters (Optuna TPE — k, "
                f"lambda_reg, w_main, w_user) using first {sample_model_name} network …"
            )
            enhanced_search = search_enhanced_params(
                train_df,
                sample_features,
                n_trials=50,
                n_splits=3,
                method=args.cmf_method,
                maxiter=args.cmf_maxiter,
                cmf_nthreads=cmf_nthreads,
            )
            save_enhanced_search_results(enhanced_search, path=dp.ENHANCED_RESULTS)
            _artifact = dp.ENHANCED_RESULTS

        _print_best_hyperparams(
            "Social CMF" if args.social_regularization else "Enhanced CMF",
            enhanced_search,
        )

        if _artifact.exists():
            mlflow.log_artifact(str(_artifact))

    # --- plots ---------------------------------------------------------------
    from visualization.model_plots import (
        plot_hyperparameter_search_results,
        plot_parameter_heatmap,
        plot_convergence_analysis,
        plot_metrics_comparison,
    )

    if enhanced_search.get("all_results"):
        plot_hyperparameter_search_results(
            enhanced_search,
            save_path=(
                "social_hyper_search.png"
                if args.social_regularization
                else "enhanced_hyper_search.png"
            ),
            dataset=args.dataset,
        )
        plot_parameter_heatmap(
            enhanced_search,
            save_path=(
                "social_param_heatmap.png"
                if args.social_regularization
                else "enhanced_param_heatmap.png"
            ),
            dataset=args.dataset,
        )
        plot_convergence_analysis(
            enhanced_search,
            save_path=(
                "social_convergence.png"
                if args.social_regularization
                else "enhanced_convergence.png"
            ),
            dataset=args.dataset,
        )
        plot_metrics_comparison(
            enhanced_search,
            save_path=(
                "social_metrics.png"
                if args.social_regularization
                else "enhanced_metrics.png"
            ),
            dataset=args.dataset,
        )


def _run_shap(args: argparse.Namespace) -> None:
    import mlflow
    from analysis.shap_analysis import run_shap_analysis, save_shap_results
    from visualization.shap_plots import plot_all_shap
    from config import MLflow as MlflowCfg, DatasetPaths

    dp = DatasetPaths(args.dataset)
    mlflow.set_tracking_uri(MlflowCfg.TRACKING_URI)
    mlflow.set_experiment(MlflowCfg.EXPERIMENT_NAME)

    with mlflow.start_run(run_name="shap"):
        cmf_nthreads = _resolve_cmf_nthreads(args)
        mlflow.log_params(
            {
                "k_networks": args.k_networks,
                "include_communities": args.include_communities,
                "seed": args.seed,
                "all_networks": args.all_networks,
                "model": args.model or "all",
                "cmf_method": args.cmf_method,
                "cmf_maxiter": args.cmf_maxiter,
                "cmf_nthreads": cmf_nthreads,
                "cpu_fraction": args.cpu_fraction,
                "social_regularization": args.social_regularization,
                "social_mode": args.social_mode,
                "lambda_social": args.lambda_social,
            }
        )

        results = run_shap_analysis(
            k_networks=None if args.all_networks else args.k_networks,
            include_communities=args.include_communities,
            seed=args.seed,
            model_names=[args.model] if args.model else None,
            params_path=(
                dp.SOCIAL_RESULTS if args.social_regularization else dp.ENHANCED_RESULTS
            ),
            dataset=args.dataset,
            method=args.cmf_method,
            maxiter=args.cmf_maxiter,
            cmf_nthreads=cmf_nthreads,
            social_regularization=args.social_regularization,
            social_mode=args.social_mode,
            lambda_social=args.lambda_social,
            social_beta=args.social_beta,
            social_gamma=args.social_gamma,
        )
        save_shap_results(results, path=dp.SHAP_RESULTS)
        plot_all_shap(dataset=args.dataset)

        for model_name, model_results in results.items():
            mlflow.log_metric(f"{model_name}_n_networks", model_results["n_networks"])
            for fname, fval in zip(
                model_results["feature_names"], model_results["mean_shap_abs"]
            ):
                safe_name = fname.replace(" ", "_").replace("/", "_")
                mlflow.log_metric(f"shap_{model_name}_{safe_name}", fval)

        _artifact = dp.SHAP_RESULTS
        if _artifact.exists():
            mlflow.log_artifact(str(_artifact))


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS: dict[str, tuple[str, object]] = {
    "cascade": ("Generate diffusion cascades from ratings", _run_cascade),
    "delta": ("Compute median inter-event delta", _run_delta),
    "inference": ("Infer diffusion networks (NetInf)", _run_inference),
    "communities": ("Detect overlapping communities + LPH", _run_communities),
    "centrality": ("Compute SNAP centrality metrics", _run_centrality),
    "recommend": ("Train and evaluate CMF recommender", _run_recommend),
    "hypertune": ("Optuna search for enhanced CMF hyperparameters", _run_hypertune),
    "shap": ("SHAP feature importance for enhanced CMF", _run_shap),
}

ALL_STEPS = list(STEPS.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="MAFPIN unified pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    step_group = parser.add_mutually_exclusive_group(required=True)
    step_group.add_argument(
        "--all",
        action="store_true",
        help="Run all pipeline steps in order.",
    )
    step_group.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEPS.keys()),
        metavar="STEP",
        help=(
            "One or more steps to execute in the given order.  "
            "Choices: " + ", ".join(STEPS.keys())
        ),
    )

    parser.add_argument(
        "--model",
        choices=["exponential", "powerlaw", "rayleigh"],
        default=None,
        help="Restrict inference and recommendation to a single diffusion model.",
    )
    parser.add_argument(
        "--dataset",
        choices=["movielens", "ciao", "epinions"],
        default="movielens",
        help="Dataset to use for the pipeline (reads from datasets/<name>/).",
    )
    parser.add_argument(
        "--n-alphas",
        type=int,
        default=100,
        dest="n_alphas",
        help="Number of alpha values for the NetInf grid search.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=Defaults.MAX_ITER,
        dest="max_iter",
        help="Fallback edge budget k when --k-fraction is disabled.",
    )
    parser.add_argument(
        "--k-avg-degree",
        type=float,
        default=Defaults.K_AVG_DEGREE,
        dest="k_avg_degree",
        help="k = avg_degree × N edges per network (0 to disable; paper default: 2).",
    )
    parser.add_argument(
        "--no-communities",
        action="store_false",
        dest="include_communities",
        help="Exclude community membership features from the enhanced CMF.",
    )
    parser.set_defaults(include_communities=True)
    parser.add_argument(
        "--cmf-method",
        choices=["lbfgs", "als"],
        default=Defaults.CMF_METHOD,
        help="CMF optimizer used by pipeline recommender fits.",
    )
    parser.add_argument(
        "--cmf-maxiter",
        type=int,
        default=Defaults.CMF_MAXITER,
        help="L-BFGS iteration budget for CMF fits.",
    )
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=Defaults.CPU_FRACTION,
        help=(
            "Approximate fraction of detected CPU cores to use for CMF/BLAS "
            "workloads when --cmf-nthreads is not set."
        ),
    )
    parser.add_argument(
        "--cmf-nthreads",
        type=int,
        default=0,
        help=("Explicit CMF/BLAS thread cap. 0 chooses a cap from --cpu-fraction."),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path for a tee log file. Defaults to data/<dataset>/pipeline.log.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable the pipeline tee log file.",
    )
    parser.add_argument(
        "--social-regularization",
        action="store_true",
        help="Use Phase 6 social-regularized CMF in recommend/hypertune/shap.",
    )
    parser.add_argument(
        "--social-mode",
        choices=[
            "uniform",
            "community_jaccard",
            "boundary_downweight",
            "bridge_preserve",
        ],
        default="boundary_downweight",
        help="Social edge weighting mode for social-regularized CMF.",
    )
    parser.add_argument(
        "--lambda-social",
        type=float,
        default=0.001,
        help="Fallback social regularization strength when no search params exist.",
    )
    parser.add_argument(
        "--social-beta",
        type=float,
        default=0.5,
        help="Boundary penalty parameter for social edge weighting.",
    )
    parser.add_argument(
        "--social-gamma",
        type=float,
        default=1.0,
        help="Shared-community gain parameter for social edge weighting.",
    )
    parser.add_argument(
        "--social-search-max-ratings",
        type=int,
        default=5000,
        help="Rating cap for social Optuna search; use 0 to disable the cap.",
    )
    parser.add_argument(
        "--social-n-trials",
        type=int,
        default=Defaults.SOCIAL_N_TRIALS,
        help="Optuna trial budget for the larger social CMF search space.",
    )
    parser.add_argument(
        "--sample-networks",
        type=int,
        default=5,
        dest="sample_networks",
        help="Number of networks to sample per model for the recommend step.",
    )
    parser.add_argument(
        "--k-networks",
        type=int,
        default=20,
        dest="k_networks",
        help="Networks to sample per diffusion model for SHAP analysis.",
    )
    parser.add_argument(
        "--all-networks",
        action="store_true",
        dest="all_networks",
        help="Use ALL available networks for SHAP analysis (overrides --k-networks).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for network sampling in SHAP analysis.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        dest="n_jobs",
        help=(
            "Number of parallel worker processes for the recommend step. "
            "1 = sequential (default). -1 = CPU cap from --cpu-fraction."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the MAFPIN pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = _open_pipeline_log(args)
    if log_file is not None:
        sys.stdout = _TeeStream(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _TeeStream(original_stderr, log_file)  # type: ignore[assignment]

    try:
        print(
            f"\n=== Pipeline run started {datetime.now().isoformat(timespec='seconds')} ===",
            flush=True,
        )
        print(f"Command: python pipeline.py {' '.join(sys.argv[1:])}", flush=True)
        _configure_cpu_limits(args)

        steps = ALL_STEPS if args.all else args.steps

        print(f"Running steps: {', '.join(steps)}", flush=True)
        print("-" * 50, flush=True)

        for step in steps:
            description, runner = STEPS[step]
            print(f"\n[{step.upper()}] {description}", flush=True)
            print("=" * 50, flush=True)
            runner(args)  # type: ignore[operator]
            print(f"[{step.upper()}] Done.", flush=True)

        print("\nPipeline finished.", flush=True)
    finally:
        print(
            f"=== Pipeline run ended {datetime.now().isoformat(timespec='seconds')} ===\n",
            flush=True,
        )
        if log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
