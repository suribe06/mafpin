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
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def _run_cascade(args: argparse.Namespace) -> None:
    from networks.cascades import generate_cascades, list_available_datasets

    dataset = args.dataset
    if dataset is None:
        datasets = list_available_datasets()
        if not datasets:
            print("No datasets found in data/. Provide --dataset explicitly.")
            sys.exit(1)
        dataset = datasets[0]
        print(f"Using dataset: {dataset}")
    generate_cascades(dataset)


def _run_delta(_args: argparse.Namespace) -> None:
    from networks.delta import compute_median_delta, alpha_centers_from_delta

    delta = compute_median_delta()
    print(f"Median delta: {delta:.4f} seconds")
    centers = alpha_centers_from_delta(delta)
    for model, info in centers.items():
        print(f"  {model}: alpha0 = {info['alpha0_seconds']:.4e} s")


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
    from recommender.data import load_dataset
    from recommender.baseline import train_final_model
    from recommender.enhanced import run_network_evaluation

    data = load_dataset()

    # Baseline
    baseline_model = train_final_model(data)
    print(f"Baseline model trained — type: {type(baseline_model).__name__}")

    # Enhanced
    run_network_evaluation(
        include_communities=args.include_communities,
        sample_networks=args.sample_networks,
    )


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
        default=2000,
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
        help="Number of networks to sample per model. Use a very large number (e.g. 9999) to run all.",
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
