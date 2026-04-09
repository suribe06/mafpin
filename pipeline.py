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
communities
    Detect overlapping communities (Demon / ASLPAw) and compute LPH.
recommend
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

    from networks.cascades import generate_cascades_from_df, list_available_datasets
    from config import Paths, Split

    dataset = args.dataset
    if dataset is None:
        datasets = list_available_datasets()
        if not datasets:
            print("No datasets found in data/. Provide --dataset explicitly.")
            sys.exit(1)
        dataset = datasets[0]
        print(f"Using dataset: {dataset}")

    csv_path = Paths.DATA / f"{dataset}.csv"
    df = pd.read_csv(csv_path, usecols=["UserId", "ItemId", "Rating", "timestamp"])

    # Apply the global split so NetInf learns from training interactions only.
    # Pass all_user_ids=df["UserId"] so the cascade header declares the full
    # user-ID space — keeping network compact IDs aligned with LabelEncoder.
    train_df, _ = train_test_split(
        df, test_size=Split.TEST_SIZE, random_state=Split.RANDOM_STATE
    )
    generate_cascades_from_df(train_df, all_user_ids=df["UserId"])


def _run_delta(_args: argparse.Namespace) -> None:
    from networks.delta import compute_median_delta, alpha_centers_from_delta

    delta = compute_median_delta()
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
    from config import Paths, Defaults

    model = args.model
    model_index_map = {"exponential": 0, "powerlaw": 1, "rayleigh": 2}
    if model:
        _ = compute_median_delta()  # kept for reference
        _ = alpha_centers_from_delta(_)  # kept for reference
        _ = log_alpha_grid  # kept for reference
        infer_networks(
            cascades_file=Paths.CASCADES,
            n=args.n_alphas,
            model=model_index_map[model],
            max_iter=args.max_iter,
            name_output=str(Paths.NETWORKS / model),
            r=Defaults.RANGE_R,
        )
    else:
        infer_networks_all_models(n=args.n_alphas, max_iter=args.max_iter)


def _run_centrality(_args: argparse.Namespace) -> None:
    from networks.centrality import calculate_centrality_for_all_models

    calculate_centrality_for_all_models()


def _run_communities(_args: argparse.Namespace) -> None:
    from networks.communities import calculate_communities_for_all_models

    calculate_communities_for_all_models()


def _run_recommend(args: argparse.Namespace) -> None:
    from recommender.data import load_and_split_dataset, evaluate_single_split
    from recommender.baseline import train_final_model, search_baseline_params
    from recommender.enhanced import (
        run_network_evaluation,
        search_enhanced_params,
        save_enhanced_search_results,
        load_network_features,
    )
    from config import Models, Defaults

    _, train_df, test_df = load_and_split_dataset()

    # Find first available feature file to represent the feature space.
    sample_features = None
    sample_model_name = None
    for _mn in Models.ALL:
        sample_features = load_network_features(
            _mn, 0, include_communities=args.include_communities
        )
        if sample_features is not None:
            sample_model_name = _mn
            break

    if sample_features is not None:
        # Independent Optuna search for the baseline (k, lambda_reg).
        # lambda_reg here is calibrated for a plain CMF loss without any
        # side-information term, so it is not biased toward the enhanced model.
        print(
            "Searching best baseline hyperparameters " "(Optuna TPE — k, lambda_reg) …"
        )
        baseline_search = search_baseline_params(train_df, n_trials=50, n_splits=3)
        best_k_b = baseline_search["best_params"]["k"]
        best_lambda_b = baseline_search["best_params"]["lambda_reg"]

        # Independent Optuna search for the enhanced model
        # (k, lambda_reg, w_main, w_user).
        print(
            f"Searching best enhanced hyperparameters (Optuna TPE — k, "
            f"lambda_reg, w_main, w_user) using first "
            f"{sample_model_name} network …"
        )
        enhanced_search = search_enhanced_params(
            train_df, sample_features, n_trials=50, n_splits=3
        )
        save_enhanced_search_results(enhanced_search)
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

    # Train final baseline model with its own independently tuned k/lambda.
    print(f"Training final baseline: k={best_k_b}, lambda_reg={best_lambda_b:.4f}")
    baseline_model = train_final_model(train_df, k=best_k_b, lambda_reg=best_lambda_b)
    baseline_metrics = evaluate_single_split(baseline_model, test_df)
    print(
        f"Baseline (global test) — RMSE: {baseline_metrics['rmse']:.4f}  "
        f"MAE: {baseline_metrics['mae']:.4f}  R²: {baseline_metrics['r2']:.4f}"
    )

    # Persist the global test-set baseline RMSE so the visualization can use
    # the correctly tuned reference instead of the paired per-fold baseline.
    from recommender.baseline import save_search_results as _save_baseline

    baseline_search["global_test_rmse"] = baseline_metrics["rmse"]
    _save_baseline(baseline_search)

    # Enhanced evaluation — pass pre-tuned enhanced params.
    run_network_evaluation(
        data=train_df,
        include_communities=args.include_communities,
        sample_networks=999_999 if args.all_networks else args.sample_networks,
        k=best_k_e,
        lambda_reg=best_lambda_e,
        w_main=best_w_main,
        w_user=best_w_user,
        baseline_k=best_k_b,
        baseline_lambda=best_lambda_b,
    )


def _run_hypertune(args: argparse.Namespace) -> None:
    from recommender.data import load_and_split_dataset
    from recommender.enhanced import (
        search_enhanced_params,
        save_enhanced_search_results,
        load_network_features,
    )
    from config import Models

    _, train_df, _ = load_and_split_dataset()

    sample_features = None
    sample_model_name = None
    for _mn in Models.ALL:
        sample_features = load_network_features(
            _mn, 0, include_communities=args.include_communities
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
    save_enhanced_search_results(enhanced_search)


def _run_shap(args: argparse.Namespace) -> None:
    from analysis.shap_analysis import run_shap_analysis, save_shap_results
    from visualization.shap_plots import plot_all_shap

    results = run_shap_analysis(
        k_networks=None if args.all_networks else args.k_networks,
        include_communities=args.include_communities,
        seed=args.seed,
        model_names=[args.model] if args.model else None,
    )
    save_shap_results(results)
    plot_all_shap()


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS: dict[str, tuple[str, object]] = {
    "cascade": ("Generate diffusion cascades from ratings", _run_cascade),
    "delta": ("Compute median inter-event delta", _run_delta),
    "inference": ("Infer diffusion networks (NetInf)", _run_inference),
    "centrality": ("Compute SNAP centrality metrics", _run_centrality),
    "communities": ("Detect overlapping communities + LPH", _run_communities),
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
        default=None,
        help="Ratings dataset filename inside data/ (auto-detected if omitted).",
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
        help="Maximum NetInf iterations per alpha.",
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
