"""CLI entry-point: python -m networks.cascades"""

from __future__ import annotations

import argparse
import sys

from config import DatasetPaths, Datasets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate cascade file from a rating dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=Datasets.ALL,
        default=Datasets.DEFAULT,
        help="Dataset to process (default: %(default)s).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from networks.cascades.generation import generate_cascades_from_df
    from recommender.data import load_and_split_dataset

    # Same train slice as the rest of the pipeline (no independent random split).
    full_df, train_df, _test_df = load_and_split_dataset(dataset=args.dataset)

    dp = DatasetPaths(args.dataset)
    success = generate_cascades_from_df(
        train_df,
        output_file=dp.CASCADES,
        all_user_ids=full_df["UserId"],
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
