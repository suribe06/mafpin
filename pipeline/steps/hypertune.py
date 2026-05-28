"""Hyperparameter tuning step (Optuna search only, no full evaluation)."""

from __future__ import annotations

import argparse
import sys

from pipeline._cpu import _resolve_cmf_nthreads
from pipeline._artifacts import _check_artifact_manifest
from pipeline._results import _print_best_hyperparams


def run_hypertune(args: argparse.Namespace) -> None:
    import mlflow

    from config import DatasetPaths, MLflow as MlflowCfg, Models
    from recommender.data import load_and_split_dataset
    from recommender.enhanced import (
        load_network_features,
        save_enhanced_search_results,
        search_enhanced_params,
    )

    dp = DatasetPaths(args.dataset)
    _check_artifact_manifest(args.dataset, context="hyperparameter tuning")
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
                "social_normalization": args.social_normalization,
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
                social_normalization=args.social_normalization,
                output_path=dp.SOCIAL_RESULTS,
                train_df=train_df,
            )
            _artifact = dp.SOCIAL_RESULTS
        else:
            print(
                f"Searching best enhanced hyperparameters (Optuna TPE — k, "
                f"lambda_reg, w_main, w_user) using first "
                f"{sample_model_name} network …"
            )
            enhanced_search = search_enhanced_params(
                train_df,
                sample_features,
                n_trials=50,
                n_splits=3,
                method=args.cmf_method,
                maxiter=args.cmf_maxiter,
                cmf_nthreads=cmf_nthreads,
                random_state=args.seed,
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
        plot_convergence_analysis,
        plot_hyperparameter_search_results,
        plot_metrics_comparison,
        plot_parameter_heatmap,
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
