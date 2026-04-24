"""CLI entry-point: python -m networks.inference"""

from __future__ import annotations

import argparse
import sys

from config import DatasetPaths, Datasets, Models, Defaults
from networks.inference.core import infer_networks
from networks.inference.batch import infer_networks_all_models


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer diffusion networks from cascade data using NetInf.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=Models.ALL,
        help="Run inference for a single diffusion model.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run inference for all three models.",
    )
    parser.add_argument(
        "--dataset",
        choices=Datasets.ALL,
        default=Datasets.DEFAULT,
        help="Dataset whose cascades/networks to use (default: %(default)s).",
    )
    parser.add_argument(
        "--cascades",
        default=None,
        help="Override path to the cascades file.",
    )
    parser.add_argument(
        "--n-alphas",
        type=int,
        default=Defaults.N_ALPHAS,
        help="Number of alpha values in the grid.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=Defaults.MAX_ITER,
        help="Fallback edge budget k when --k-fraction is disabled.",
    )
    parser.add_argument(
        "--k-avg-degree",
        type=float,
        default=Defaults.K_AVG_DEGREE,
        help="k = avg_degree × N edges per network (0 to disable; paper default: 2).",
    )
    parser.add_argument(
        "--range-r",
        type=float,
        default=Defaults.RANGE_R,
        help="Multiplicative range factor for the log alpha grid.",
    )
    parser.add_argument(
        "--name-output",
        default="inferred-network",
        help="Base name for per-alpha output files.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dp = DatasetPaths(args.dataset)
    cascades_file = args.cascades or dp.CASCADES
    networks_dir = dp.NETWORKS
    k_avg_degree = args.k_avg_degree if args.k_avg_degree > 0 else None

    if args.all:
        results = infer_networks_all_models(
            cascades_file=cascades_file,
            n=args.n_alphas,
            max_iter=args.max_iter,
            k_avg_degree=k_avg_degree,
            name_output=args.name_output,
            r=args.range_r,
            networks_dir=networks_dir,
        )
        any_failed = any(not v for v in results.values())
        sys.exit(1 if any_failed else 0)
    else:
        model_idx = Models.ALL.index(args.model)
        success = infer_networks(
            cascades_file=cascades_file,
            n=args.n_alphas,
            model=model_idx,
            max_iter=args.max_iter,
            k_avg_degree=k_avg_degree,
            name_output=args.name_output,
            r=args.range_r,
            networks_dir=networks_dir,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
