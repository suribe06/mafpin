"""SHAP feature importance step."""

from __future__ import annotations

import argparse

from pipeline._cpu import _resolve_cmf_nthreads
from pipeline._artifacts import _check_artifact_manifest


def run_shap(args: argparse.Namespace) -> None:
    import mlflow

    from analysis.shap_analysis import run_shap_analysis, save_shap_results
    from config import DatasetPaths, MLflow as MlflowCfg
    from visualization.shap_plots import plot_all_shap

    dp = DatasetPaths(args.dataset)
    _check_artifact_manifest(args.dataset, context="SHAP analysis")
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
                "social_normalization": args.social_normalization,
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
            social_normalization=args.social_normalization,
        )
        save_shap_results(results, path=dp.SHAP_RESULTS)
        plot_all_shap(
            dataset=args.dataset,
            social_regularization=args.social_regularization,
        )

        # Log the actual hyperparameters loaded from disk (not CLI defaults).
        try:
            from analysis.shap_analysis import load_enhanced_params as _lep

            _actual_params = _lep(
                path=(
                    dp.SOCIAL_RESULTS
                    if args.social_regularization
                    else dp.ENHANCED_RESULTS
                ),
                dataset=args.dataset,
            )
            mlflow.log_params(
                {f"shap_best_{_k}": _v for _k, _v in _actual_params.items()}
            )
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as _exc:
            print(f"  Warning: could not log actual SHAP hyperparameters: {_exc}")

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
