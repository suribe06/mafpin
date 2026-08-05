"""Recommendation evaluation step (baseline + enhanced CMF)."""

from __future__ import annotations

import argparse
import sys

from pipeline._cpu import _resolve_cmf_nthreads, _cpu_thread_limit
from pipeline._artifacts import _check_artifact_manifest
from pipeline._results import _print_best_hyperparams


def run_recommend(args: argparse.Namespace) -> None:
    import mlflow

    from config import DatasetPaths, MLflow as MlflowCfg, Models
    from recommender.baseline import save_search_results as _save_baseline
    from recommender.baseline import train_final_model
    from recommender.data import evaluate_single_split, load_and_split_dataset
    from recommender.enhanced import run_network_evaluation
    from recommender.enhanced.tuning import run_hyperparam_campaign

    dp = DatasetPaths(args.dataset)
    _check_artifact_manifest(args.dataset, context="recommendation evaluation")

    mlflow.set_tracking_uri(MlflowCfg.TRACKING_URI)
    mlflow.set_experiment(MlflowCfg.EXPERIMENT_NAME)

    with mlflow.start_run(run_name="recommend"):
        cmf_nthreads = _resolve_cmf_nthreads(args)
        network_n_jobs = (
            _cpu_thread_limit(args.cpu_fraction)
            if args.n_jobs == -1
            else args.n_jobs
        )
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
                "network_eval_n_jobs": network_n_jobs,
                "social_regularization": args.social_regularization,
                "social_normalization": args.social_normalization,
            }
        )

        _, train_df, test_df = load_and_split_dataset(dataset=args.dataset)
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
                search_baseline=True,
                require_features=False,
            )
        except RuntimeError as exc:
            print(exc)
            sys.exit(1)

        if campaign.sample_features is None:
            print("No feature files found — using default params.")

        baseline_search = campaign.baseline_search
        enhanced_search = campaign.enhanced_search
        _print_best_hyperparams("Baseline CMF", baseline_search)
        _print_best_hyperparams(
            "Social CMF" if args.social_regularization else "Enhanced CMF",
            enhanced_search,
        )

        mlflow.log_params(
            {
                "k_baseline": campaign.best_k_b,
                "lambda_baseline": campaign.best_lambda_b,
                "k_enhanced": campaign.best_k_e,
                "lambda_enhanced": campaign.best_lambda_e,
                "w_main": campaign.best_w_main,
                "w_user": campaign.best_w_user,
                "social_mode": campaign.social_mode,
                "lambda_social": campaign.lambda_social,
                "social_beta": campaign.social_beta,
                "social_gamma": campaign.social_gamma,
                "social_normalization": args.social_normalization,
            }
        )

        print(
            f"Training final baseline: method={args.cmf_method}, "
            f"k={campaign.best_k_b}, lambda_reg={campaign.best_lambda_b:.4f}"
        )
        baseline_model = train_final_model(
            train_df,
            k=campaign.best_k_b,
            lambda_reg=campaign.best_lambda_b,
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

        baseline_search["global_test_rmse"] = baseline_metrics["rmse"]
        _save_baseline(baseline_search, path=dp.BASELINE_RESULTS)

        all_results = run_network_evaluation(
            data=train_df,
            model_names=selected_models,
            include_communities=args.include_communities,
            sample_networks=999_999 if args.all_networks else args.sample_networks,
            k=campaign.best_k_e,
            lambda_reg=campaign.best_lambda_e,
            w_main=campaign.best_w_main,
            w_user=campaign.best_w_user,
            baseline_k=campaign.best_k_b,
            baseline_lambda=campaign.best_lambda_b,
            compute_ranking=True,
            dataset=args.dataset,
            n_jobs=network_n_jobs,
            method=args.cmf_method,
            maxiter=args.cmf_maxiter,
            cmf_nthreads=cmf_nthreads,
            use_social_regularization=args.social_regularization,
            social_mode=campaign.social_mode,
            lambda_social=campaign.lambda_social,
            social_beta=campaign.social_beta,
            social_gamma=campaign.social_gamma,
            social_normalization=args.social_normalization,
        )

        for _artifact in [
            dp.BASELINE_RESULTS,
            dp.SOCIAL_RESULTS if args.social_regularization else dp.ENHANCED_RESULTS,
        ]:
            if _artifact.exists():
                mlflow.log_artifact(str(_artifact))

    if getattr(args, "run_id", None):
        from recommender.experiment.manifest import archive_recommend_run

        archive_recommend_run(
            args.dataset,
            args.run_id,
            baseline_path=dp.BASELINE_RESULTS,
            enhanced_path=dp.ENHANCED_RESULTS if not args.social_regularization else None,
            social_path=dp.SOCIAL_RESULTS if args.social_regularization else None,
        )

    from visualization.model_plots import (
        plot_alpha_delta_rmse,
        plot_alpha_edges,
        plot_alpha_rmse_analysis,
        plot_convergence_analysis,
        plot_hyperparameter_search_results,
        plot_metrics_comparison,
        plot_parameter_heatmap,
        plot_ranking_metrics_comparison,
        plot_ranking_metrics_per_alpha,
    )

    for _search, _prefix in [
        (baseline_search, "baseline"),
        (
            enhanced_search if campaign.sample_features is not None else None,
            "social" if args.social_regularization else "enhanced",
        ),
    ]:
        if _search and _search.get("all_results"):
            plot_hyperparameter_search_results(
                _search,
                save_path=f"{_prefix}_hyper_search.png",
                dataset=args.dataset,
            )
            plot_parameter_heatmap(
                _search,
                save_path=f"{_prefix}_param_heatmap.png",
                dataset=args.dataset,
            )
            plot_convergence_analysis(
                _search,
                save_path=f"{_prefix}_convergence.png",
                dataset=args.dataset,
            )
            plot_metrics_comparison(
                _search,
                save_path=f"{_prefix}_metrics.png",
                dataset=args.dataset,
            )

    global_rmse = baseline_metrics["rmse"]
    for _mn, _net_results in all_results.items():
        _enhanced_list = _net_results.get("enhanced", [])
        _baseline_list = _net_results.get("baseline", [])
        if _enhanced_list:
            _mean_paired_baseline = (
                float(sum(_baseline_list) / len(_baseline_list))
                if _baseline_list
                else global_rmse
            )
            plot_alpha_rmse_analysis(
                model_name=_mn,
                rmse_values=_enhanced_list,
                baseline_rmse=_mean_paired_baseline,
                global_baseline_rmse=global_rmse,
                save_plot=True,
                dataset=args.dataset,
            )
            plot_alpha_delta_rmse(
                model_name=_mn,
                rmse_values=_enhanced_list,
                baseline_rmse=_mean_paired_baseline,
                save_plot=True,
                dataset=args.dataset,
            )

    plot_alpha_edges(save_plot=True, dataset=args.dataset)

    for _mn in Models.ALL:
        plot_ranking_metrics_per_alpha(_mn, save_plot=True, dataset=args.dataset)
    plot_ranking_metrics_comparison(save_plot=True, dataset=args.dataset)
