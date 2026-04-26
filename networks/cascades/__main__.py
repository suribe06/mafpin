"""CLI entry-point: python -m networks.cascades"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

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

    from sklearn.model_selection import train_test_split
    from config import Split
    from networks.cascades.generation import generate_cascades_from_df

    cfg = Datasets.CONFIG[args.dataset]
    csv_path = Datasets.ROOT / args.dataset / cfg["file"]
    cols: list[int] = [
        cfg["col_user"],
        cfg["col_item"],
        cfg["col_rating"],
        cfg["col_time"],
    ]
    df = pd.read_csv(
        csv_path,
        sep=cfg["sep"],
        header=cfg["header"],
        usecols=cols,  # type: ignore[call-overload]
        engine="python",
    )
    df.columns = pd.Index(["UserId", "ItemId", "Rating", "timestamp"])

    train_df, _ = train_test_split(
        df, test_size=Split.TEST_SIZE, random_state=Split.RANDOM_STATE
    )

    dp = DatasetPaths(args.dataset)
    success = generate_cascades_from_df(
        pd.DataFrame(train_df),
        output_file=dp.CASCADES,
        all_user_ids=df["UserId"],
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
