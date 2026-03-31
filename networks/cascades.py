"""
Cascade generation from rating datasets.

A *cascade* represents the temporal sequence of user interactions with a
specific item (e.g. movie ratings).  The cascade format is required as input
for the NetInf algorithm used in the network inference step.

Input CSV columns (order matters):
    UserId    – unique user identifier
    ItemId    – unique item identifier
    Rating    – numeric rating
    timestamp – Unix epoch seconds

Output file (``data/cascades.txt``):
    - Header block: one ``user_id,user_id`` line per unique user
    - Empty separator line
    - One cascade line per item:
      ``user_node,timestamp,user_node,timestamp,...``

Usage (CLI)::

    python -m networks.cascades ratings_small
    python -m networks.cascades --help
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import sys
import numpy as np
import pandas as pd

from config import Paths


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def list_available_datasets() -> list[str]:
    """Return dataset names (without ``.csv`` extension) found in ``data/``."""
    if not Paths.DATA.exists():
        print(f"Error: data directory '{Paths.DATA}' not found.")
        return []
    return [f.stem for f in Paths.DATA.iterdir() if f.suffix == ".csv"]


def generate_cascades(dataset_name: str) -> bool:
    """
    Build cascade file from a rating dataset and write it to disk.

    Args:
        dataset_name: Name of the CSV file (without extension) inside ``data/``.

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    data_file = Paths.DATA / f"{dataset_name}.csv"
    output_file = Paths.CASCADES

    if not data_file.exists():
        print(f"Error: dataset '{data_file}' not found.")
        return False

    print(f"Processing dataset: {data_file}")

    # Load the first four columns: UserId, ItemId, Rating, timestamp
    interactions = pd.read_csv(data_file, usecols=range(4))  # type: ignore[call-overload]

    num_users = interactions["UserId"].nunique()
    num_items = interactions["ItemId"].nunique()
    print(f"Found {num_items} unique items and {num_users} unique users")

    # Map original IDs to consecutive integers
    item_mapper = dict(zip(np.unique(interactions["ItemId"]), range(int(num_items))))
    user_mapper = dict(
        zip(
            np.unique(interactions["UserId"]),
            range(int(num_items), int(num_items) + int(num_users)),
        )
    )

    # Build cascade dict: item_id → [(user_id, datetime), ...]
    cascades: dict[int, list] = {item_id: [] for item_id in item_mapper.values()}
    for _, row in interactions.iterrows():
        item_id = item_mapper[row["ItemId"]]
        user_id = user_mapper[row["UserId"]]
        timestamp = datetime.datetime.fromtimestamp(float(row["timestamp"]))
        cascades[item_id].append([user_id, timestamp])

    # Sort descending by time and flatten to [user, unix_ts, user, unix_ts, ...]
    for item_id, records in cascades.items():
        records.sort(key=lambda x: x[1], reverse=True)
        cascades[item_id] = list(
            itertools.chain(*[[u, dt.timestamp()] for u, dt in records])
        )

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fh:
        for u in user_mapper.values():
            fh.write(f"{u},{u}\n")
        fh.write("\n")
        for record in cascades.values():
            if record:
                fh.write(",".join(map(str, record)) + "\n")

    print(f"Cascades saved to: {output_file}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate cascade file from a rating dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (without .csv extension). Defaults to 'ratings_small'.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    datasets = list_available_datasets()
    if args.dataset is None:
        print("Available datasets:")
        for d in datasets:
            print(f"  - {d}")
        print()
        dataset_name = "ratings_small"
        print(f"No dataset specified — using default: {dataset_name}")
    else:
        dataset_name = args.dataset

    success = generate_cascades(dataset_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
