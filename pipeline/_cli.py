"""CLI argument parser for the MAFPIN pipeline."""

from __future__ import annotations

import argparse

from config import Defaults


def _build_parser() -> argparse.ArgumentParser:
    # Import here so STEPS is available at call time (avoids circular import).
    from pipeline._runner import STEPS

    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="MAFPIN unified pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    step_group = parser.add_mutually_exclusive_group(required=False)
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
        help="Explicit CMF/BLAS thread cap. 0 chooses a cap from --cpu-fraction.",
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
        "--social-normalization",
        choices=[
            "none",
            "mean",
            "mean_weight",
            "edges",
            "n_edges",
            "sum_weight",
            "normalized_laplacian",
        ],
        default="mean_weight",
        help="Social edge normalization strategy for social-regularized CMF.",
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
    parser.add_argument(
        "--preregister-networks",
        action="store_true",
        dest="preregister_networks",
        help=(
            "Compute and save a pre-registered stratified sample of networks "
            "spanning sparse/medium/dense alpha quantiles × all diffusion models. "
            "Outputs data/<dataset>/preregistered_network_sample.json."
        ),
    )
    return parser
