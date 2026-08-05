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
    from recommender.enhanced.tuning import run_hyperparam_campaign

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

        try:
            campaign = run_hyperparam_campaign(
                train_df,
                dataset=args.dataset,
                selected_models=selected_models,
                include_communities=args.include_communities,
                social_regularization=args.social_regularization,
                social_mode=args.social_mode,
                social_normalization=args.social_normalization,
                lambda_social=args.lambda_social,
                social_beta=args.social_beta,
                social_gamma=args.social_gamma,
                social_n_trials=args.social_n_trials,
                social_search_max_ratings=args.social_search_max_ratings,
                cmf_method=args.cmf_method,
                cmf_maxiter=args.cmf_maxiter,
                cmf_nthreads=cmf_nthreads,
                random_state=args.seed,
                search_baseline=False,
                require_features=True,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(exc)
            sys.exit(1)

        enhanced_search = campaign.enhanced_search
        _artifact = (
            dp.SOCIAL_RESULTS if args.social_regularization else dp.ENHANCED_RESULTS
        )
        _print_best_hyperparams(
            "Social CMF" if args.social_regularization else "Enhanced CMF",
            enhanced_search,
        )

        if _artifact.exists():
            mlflow.log_artifact(str(_artifact))

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
