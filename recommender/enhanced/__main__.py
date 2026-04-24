"""CLI entry-point: python -m recommender.enhanced"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from recommender.enhanced.network_eval import run_network_evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate enhanced CMF with network features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every available network for all models.",
    )
    parser.add_argument(
        "--sample-networks",
        type=int,
        default=5,
        metavar="N",
        help="Number of networks to randomly sample per model.",
    )
    parser.add_argument(
        "--transform",
        default="standard",
        choices=["standard", "minmax", "normalizer"],
        help="Feature normalisation method.",
    )
    parser.add_argument(
        "--no-communities",
        action="store_true",
        help="Exclude community features (LPH, num_communities).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Cross-validation splits per network.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    sample = 999_999 if args.all else args.sample_networks

    from recommender.data import load_and_split_dataset

    _, train_df, _ = load_and_split_dataset()
    eval_results = run_network_evaluation(
        data=train_df,
        sample_networks=sample,
        transform=args.transform,
        include_communities=not args.no_communities,
        n_splits=args.n_splits,
    )
    print("\n" + "=" * 55)
    print("Summary")
    print("=" * 55)
    for model_name, rmse_list in eval_results.items():
        if rmse_list:
            print(
                f"{model_name}: n={len(rmse_list)}, "
                f"mean={np.mean(rmse_list):.4f}, "
                f"min={np.min(rmse_list):.4f}, "
                f"max={np.max(rmse_list):.4f}"
            )
        else:
            print(f"{model_name}: no results")
    sys.exit(0)


if __name__ == "__main__":
    main()
