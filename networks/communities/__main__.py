"""CLI entry-point: python -m networks.communities"""

from __future__ import annotations

import argparse
import sys

from config import DatasetPaths, Datasets, Models, Defaults
from networks.network_io import SYMMETRIZATION_METHODS
from networks.communities.batch import (
    calculate_communities_for_all_models,
    calculate_communities_for_network,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect communities and compute LPH for inferred networks.",
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
    parser.add_argument(
        "--algorithm",
        choices=["demon", "aslpaw"],
        default="demon",
        help="Community detection algorithm.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=Defaults.EPSILON,
        help="Merging threshold for Demon.",
    )
    parser.add_argument(
        "--min-community",
        type=int,
        default=Defaults.MIN_COM,
        help="Minimum community size for Demon.",
    )
    parser.add_argument(
        "--symmetrization",
        choices=sorted(SYMMETRIZATION_METHODS),
        default="union",
        help=(
            "Method to convert the directed inferred network to undirected "
            "before community detection and h\u0303v computation. "
            "'union': edge u\u2013v exists if u\u2192v OR v\u2192u (default, more edges). "
            "'intersection': edge u\u2013v exists only if BOTH directions present (fewer edges)."
        ),
    )
    parser.add_argument(
        "--boundary-percentile",
        type=float,
        default=20.0,
        metavar="P",
        help=(
            "Percentile threshold (0\u2013100) for the binary is_boundary flag. "
            "Nodes whose h\u0303v score falls at or below this percentile are "
            "marked as boundary-spanners (is_boundary=1). Default: 20."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    kwargs = {
        "algorithm": args.algorithm,
        "epsilon": args.epsilon,
        "min_community": args.min_community,
        "symmetrization": args.symmetrization,
        "boundary_percentile": args.boundary_percentile,
    }

    if args.network:
        success = calculate_communities_for_network(args.network, **kwargs)
        sys.exit(0 if success else 1)

    elif args.all:
        summary = calculate_communities_for_all_models(**kwargs)
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
            if calculate_communities_for_network(
                nf,
                **kwargs,
                output_dir=dp.COMMUNITIES / args.model,
            )
        )
        print(f"\nProcessed: {success_count}/{len(network_files)}")
        sys.exit(0)


if __name__ == "__main__":
    main()
