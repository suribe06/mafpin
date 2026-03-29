#!/usr/bin/env python3
"""
Community Detection Runner Script

Detects overlapping communities for all inferred networks and computes
Local Pluralistic Homophily (LPH) per node using the Demon algorithm (cdlib).

Results are saved to:
    ../data/communities/<model>/communities_<model>_<id>.csv

Each CSV has columns:
    UserId, num_communities, community_ids, local_pluralistic_hom

Usage examples:
    python run_communities.py --help
    python run_communities.py --all-models
    python run_communities.py --all-models --algorithm aslpaw
    python run_communities.py --model exponential --epsilon 0.3
    python run_communities.py --single-network ../data/inferred_networks/exponential/inferred-network-expo-000.txt
"""

import argparse
import os
import sys

from calculate_communities import (
    calculate_communities_for_all_models,
    calculate_communities_for_network,
)


def main():
    parser = argparse.ArgumentParser(
        description="Detect overlapping communities in inferred networks and compute LPH",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Network selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--single-network",
        type=str,
        metavar="FILE",
        help="Process a single network .txt file",
    )
    group.add_argument(
        "--all-models",
        action="store_true",
        help="Process all networks across all models (exponential, powerlaw, rayleigh)",
    )
    group.add_argument(
        "--model",
        type=str,
        choices=["exponential", "powerlaw", "rayleigh"],
        help="Process all networks from a single model",
    )

    # Algorithm options
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["demon", "aslpaw"],
        default="demon",
        help="Overlapping community detection algorithm",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.25,
        help="Demon merge threshold — lower values produce more communities (ignored for aslpaw)",
    )
    parser.add_argument(
        "--min-community",
        type=int,
        default=3,
        dest="min_community",
        help="Minimum community size to keep (Demon only)",
    )

    args = parser.parse_args()

    print("Community Detection Runner")
    print("=" * 50)
    print(f"Algorithm  : {args.algorithm}")
    if args.algorithm == "demon":
        print(f"Epsilon    : {args.epsilon}")
        print(f"Min size   : {args.min_community}")
    print()

    alg: str = str(args.algorithm)
    eps: float = float(args.epsilon)
    min_com: int = int(args.min_community)

    # ------------------------------------------------------------------ #
    if args.single_network:
        if not os.path.exists(args.single_network):
            print(f"Error: file not found: {args.single_network}")
            sys.exit(1)
        ok = calculate_communities_for_network(
            args.single_network, algorithm=alg, epsilon=eps, min_community=min_com
        )
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------ #
    elif args.all_models:
        base_path = os.path.join("..", "data", "inferred_networks")
        if not os.path.exists(base_path):
            print(f"Error: directory not found: {base_path}")
            sys.exit(1)
        results = calculate_communities_for_all_models(
            algorithm=alg, epsilon=eps, min_community=min_com
        )
        total_ok = sum(r["processed"] for r in results.values())
        sys.exit(0 if total_ok > 0 else 1)

    # ------------------------------------------------------------------ #
    elif args.model:
        import glob as _glob

        model_path = os.path.join("..", "data", "inferred_networks", args.model)

        if not os.path.exists(model_path):
            print(f"Error: directory not found: {model_path}")
            sys.exit(1)

        network_files = sorted(
            f
            for f in _glob.glob(os.path.join(model_path, "*.txt"))
            if not f.endswith(".csv")
        )

        if not network_files:
            print(f"No network files found in {model_path}")
            sys.exit(1)

        print(f"Processing {len(network_files)} networks for model '{args.model}'")
        processed, failed = 0, 0
        for nf in network_files:
            try:
                ok = calculate_communities_for_network(
                    nf, algorithm=alg, epsilon=eps, min_community=min_com
                )
                if ok:
                    processed += 1
                else:
                    failed += 1
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error: {nf}: {e}")
                failed += 1

        print(f"\nDone — {processed} processed, {failed} failed")
        sys.exit(0 if processed > 0 else 1)


if __name__ == "__main__":
    main()
