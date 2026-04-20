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
import sys

from config import Defaults

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
    df = pd.read_csv(
        csv_path,
        sep=cfg["sep"],
        header=cfg["header"],
        usecols=cols,
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
        mlflow.log_params(
            {
                "include_communities": args.include_communities,
                "sample_networks": args.sample_networks,
                "all_networks": args.all_networks,
                "model": args.model or "all",
                "n_optuna_trials": 50,
                "n_cv_splits": 3,
            }
        )

        _, train_df, test_df = load_and_split_dataset(dataset=args.dataset)

        # Find first available feature file to represent the feature space.
        sample_features = None
        sample_model_name = None
        for _mn in Models.ALL:
            sample_features = load_network_features(
                _mn,
                0,
                include_communities=args.include_communities,
                dataset=args.dataset,
            )
            if sample_features is not None:
                sample_model_name = _mn
                break

        if sample_features is not None:
            # Independent Optuna search for the baseline (k, lambda_reg).
            print(
                "Searching best baseline hyperparameters "
                "(Optuna TPE — k, lambda_reg) …"
            )
            with mlflow.start_run(run_name="baseline_search", nested=True):
                baseline_search = search_baseline_params(
                    train_df, n_trials=50, n_splits=3
                )
            best_k_b = baseline_search["best_params"]["k"]
            best_lambda_b = baseline_search["best_params"]["lambda_reg"]

            # Independent Optuna search for the enhanced model
            print(
                f"Searching best enhanced hyperparameters (Optuna TPE — k, "
                f"lambda_reg, w_main, w_user) using first "
                f"{sample_model_name} network …"
            )
            with mlflow.start_run(run_name="enhanced_search", nested=True):
                enhanced_search = search_enhanced_params(
                    train_df, sample_features, n_trials=50, n_splits=3
                )
            save_enhanced_search_results(enhanced_search, path=dp.ENHANCED_RESULTS)
            best_k_e = enhanced_search["best_params"]["k"]
            best_lambda_e = enhanced_search["best_params"]["lambda_reg"]
            best_w_main = enhanced_search["best_params"]["w_main"]
            best_w_user = enhanced_search["best_params"]["w_user"]
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

        mlflow.log_params(
            {
                "k_baseline": best_k_b,
                "lambda_baseline": best_lambda_b,
                "k_enhanced": best_k_e,
                "lambda_enhanced": best_lambda_e,
                "w_main": best_w_main,
                "w_user": best_w_user,
            }
        )

        # Train final baseline model with its own independently tuned k/lambda.
        print(f"Training final baseline: k={best_k_b}, lambda_reg={best_lambda_b:.4f}")
        baseline_model = train_final_model(
            train_df, k=best_k_b, lambda_reg=best_lambda_b
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
        )

        for _artifact in [
            dp.BASELINE_RESULTS,
            dp.ENHANCED_RESULTS,
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
        (enhanced_search if sample_features is not None else None, "enhanced"),
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
        mlflow.log_params(
            {
                "include_communities": args.include_communities,
                "n_optuna_trials": 50,
                "n_cv_splits": 3,
            }
        )

        _, train_df, _ = load_and_split_dataset(dataset=args.dataset)

        sample_features = None
        sample_model_name = None
        for _mn in Models.ALL:
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

        print(
            f"Searching best enhanced hyperparameters (Optuna TPE — k, "
            f"lambda_reg, w_main, w_user) using first {sample_model_name} network …"
        )
        enhanced_search = search_enhanced_params(
            train_df, sample_features, n_trials=50, n_splits=3
        )
        save_enhanced_search_results(enhanced_search, path=dp.ENHANCED_RESULTS)

        _artifact = dp.ENHANCED_RESULTS
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
            save_path="enhanced_hyper_search.png",
            dataset=args.dataset,
        )
        plot_parameter_heatmap(
            enhanced_search,
            save_path="enhanced_param_heatmap.png",
            dataset=args.dataset,
        )
        plot_convergence_analysis(
            enhanced_search,
            save_path="enhanced_convergence.png",
            dataset=args.dataset,
        )
        plot_metrics_comparison(
            enhanced_search,
            save_path="enhanced_metrics.png",
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
        mlflow.log_params(
            {
                "k_networks": args.k_networks,
                "include_communities": args.include_communities,
                "seed": args.seed,
                "all_networks": args.all_networks,
                "model": args.model or "all",
            }
        )

        results = run_shap_analysis(
            k_networks=None if args.all_networks else args.k_networks,
            include_communities=args.include_communities,
            seed=args.seed,
            model_names=[args.model] if args.model else None,
            params_path=dp.ENHANCED_RESULTS,
            dataset=args.dataset,
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
        "--include-communities",
        action="store_true",
        dest="include_communities",
        help="Include community membership features in the enhanced CMF.",
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
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the MAFPIN pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    steps = ALL_STEPS if args.all else args.steps

    print(f"Running steps: {', '.join(steps)}")
    print("-" * 50)

    for step in steps:
        description, runner = STEPS[step]
        print(f"\n[{step.upper()}] {description}")
        print("=" * 50)
        runner(args)  # type: ignore[operator]
        print(f"[{step.upper()}] Done.")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
