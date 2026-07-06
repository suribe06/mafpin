"""Recommendation evaluation step (baseline + enhanced CMF)."""

from __future__ import annotations

import argparse
import sys

from config import Defaults
from pipeline._cpu import _resolve_cmf_nthreads, _cpu_thread_limit
from pipeline._artifacts import _check_artifact_manifest
from pipeline._results import _print_best_hyperparams


def run_recommend(args: argparse.Namespace) -> None:
    import mlflow

    from config import DatasetPaths, MLflow as MlflowCfg, Models
    from recommender.baseline import save_search_results as _save_baseline
    from recommender.baseline import search_baseline_params, train_final_model
    from recommender.data import evaluate_single_split, load_and_split_dataset
    from recommender.enhanced import (
        load_network_features,
        run_network_evaluation,
        save_enhanced_search_results,
        search_enhanced_params,
    )

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
                    random_state=args.seed,
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
                        social_modes=(args.social_mode,),
                        social_normalization=args.social_normalization,
                        output_path=dp.SOCIAL_RESULTS,
                        train_df=train_df,
                    )
                if not enhanced_search["best_params"]:
                    print("Social hyperparameter search produced no usable trials.")
                    sys.exit(1)
            else:
                # Independent Optuna search for the enhanced model.
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
                        random_state=args.seed,
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
                "social_normalization": args.social_normalization,
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
            n_jobs=network_n_jobs,
            method=args.cmf_method,
            maxiter=args.cmf_maxiter,
            cmf_nthreads=cmf_nthreads,
            use_social_regularization=args.social_regularization,
            social_mode=selected_social_mode,
            lambda_social=selected_lambda_social,
            social_beta=selected_social_beta,
            social_gamma=selected_social_gamma,
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

    # --- plots ---------------------------------------------------------------
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
            enhanced_search if sample_features is not None else None,
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
