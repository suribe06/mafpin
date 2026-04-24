"""CLI entry-point: python -m networks.centrality"""

from __future__ import annotations

import argparse
import sys

from config import DatasetPaths, Datasets, Models
from networks.centrality.batch import (
    calculate_centrality_for_all_models,
    calculate_centrality_for_network,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute centrality metrics for inferred networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=Models.ALL,
        help="Process all networks for a single diffusion model.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all models.",
    )
    group.add_argument(
        "--network",
        metavar="FILE",
        help="Process a single network file.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.network:
        success = calculate_centrality_for_network(args.network)
        sys.exit(0 if success else 1)

    elif args.all:
        summary = calculate_centrality_for_all_models()
        total = sum(summary.values())
        print(f"\nTotal networks processed: {total}")
        sys.exit(0)

    else:
        dp = DatasetPaths(Datasets.DEFAULT)
        model_dir = dp.NETWORKS / args.model
        if not model_dir.exists():
            print(f"Error: directory not found: {model_dir}")
            sys.exit(1)
        network_files = sorted(model_dir.glob("inferred-network-*.txt"))
        success_count = sum(
            1
            for nf in network_files
            if calculate_centrality_for_network(
                nf,
                communities_dir=dp.COMMUNITIES,
                centrality_dir=dp.CENTRALITY,
            )
        )
        print(f"\nProcessed: {success_count}/{len(network_files)}")
        sys.exit(0)


if __name__ == "__main__":
    main()
