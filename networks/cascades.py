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
import sys
from pathlib import Path

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


def generate_cascades_from_df(
    interactions: pd.DataFrame,
    output_file: str | Path | None = None,
    all_user_ids=None,
) -> bool:
    """
    Build a cascade file from an already-loaded ratings DataFrame and write it
    to disk.

    This is the core implementation.  Use :func:`generate_cascades` for the
    convenience wrapper that reads a CSV from disk.

    The DataFrame must have at least the columns ``UserId``, ``ItemId``, and
    ``timestamp`` (any additional columns are ignored).

    Args:
        interactions: Ratings DataFrame (may be a train-only split).
        output_file:  Destination path.  Defaults to ``Paths.CASCADES``.
        all_user_ids: Optional array-like of *all* user IDs in the full dataset
            (including those absent from ``interactions``).  When provided the
            cascade header declares the complete user-ID space, which keeps
            NetInf node IDs aligned with the LabelEncoder mapping used by the
            CMF recommender.  Pass this whenever ``interactions`` is a
            train-only subset so that compact network IDs produced by NetInf
            still correspond to LabelEncoder user IDs (C-3 fix).

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    if output_file is None:
        output_file = Paths.CASCADES
    output_file = Path(output_file)

    num_users = interactions["UserId"].nunique()
    num_items = interactions["ItemId"].nunique()
    print(f"Found {num_items} unique items and {num_users} unique users")

    # Build the user mapping over the FULL user-ID space when provided.
    # This guarantees that cascade node IDs are consistent with the
    # LabelEncoder-assigned IDs used by the recommender (C-3 fix).
    users_for_mapping = (
        np.unique(all_user_ids)
        if all_user_ids is not None
        else np.unique(interactions["UserId"])
    )
    num_users_total = len(users_for_mapping)

    # Map original IDs to consecutive integers.
    # Users are offset past items so that item and user IDs are disjoint in
    # the NetInf node namespace.
    item_mapper = dict(zip(np.unique(interactions["ItemId"]), range(int(num_items))))
    user_mapper = dict(
        zip(
            users_for_mapping,
            range(int(num_items), int(num_items) + num_users_total),
        )
    )

    # Build cascade dict: item_id → [(user_id, timestamp_in_days), ...]
    # Timestamps are converted from Unix seconds to days so that the resulting
    # alpha values (α = ln(2)/Δ) are numerically in the range [1e-4, 1] rather
    # than [1e-9, 1e-5].  NetInf only uses relative time differences, so the
    # unit choice does not affect topology — only the numeric stability of the
    # log-likelihood surface and therefore the quality of NetInf's grid search.
    _SECONDS_PER_DAY = 86_400.0
    cascades: dict[int, list] = {item_id: [] for item_id in item_mapper.values()}
    for _, row in interactions.iterrows():
        item_id = item_mapper[row["ItemId"]]
        user_id = user_mapper[row["UserId"]]
        cascades[item_id].append([user_id, float(row["timestamp"]) / _SECONDS_PER_DAY])

    # Sort ascending by time (seed = earliest infected user comes first,
    # as required by NetInf's diffusion models) and flatten to
    # [user, unix_ts, user, unix_ts, ...].
    for item_id, records in cascades.items():
        records.sort(key=lambda x: x[1])  # ascending — C-1 fix
        flat: list = []
        for pair in records:
            flat.extend(pair)
        cascades[item_id] = flat

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fh:
        # Header declares ALL users (incl. test-only) for ID-space consistency.
        for u in user_mapper.values():
            fh.write(f"{u},{u}\n")
        fh.write("\n")
        for record in cascades.values():
            # Skip single-user cascades (< 2 user-timestamp pairs = < 4 elements).
            # They carry no diffusion signal and can confuse NetInf.
            if len(record) >= 4:  # m-2 fix
                fh.write(",".join(map(str, record)) + "\n")

    print(f"Cascades saved to: {output_file}")
    return True


def generate_cascades(dataset_name: str) -> bool:
    """
    Build cascade file from a rating dataset CSV and write it to disk.

    Loads the full dataset and delegates to
    :func:`generate_cascades_from_df`.  For leakage-free evaluation, call
    :func:`generate_cascades_from_df` directly with a train-only split.

    Args:
        dataset_name: Name of the CSV file (without extension) inside ``data/``.

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    data_file = Paths.DATA / f"{dataset_name}.csv"

    if not data_file.exists():
        print(f"Error: dataset '{data_file}' not found.")
        return False

    print(f"Processing dataset: {data_file}")

    # Load the first four columns: UserId, ItemId, Rating, timestamp
    interactions = pd.read_csv(data_file, usecols=range(4))  # type: ignore[call-overload]
    interactions.columns = pd.Index(["UserId", "ItemId", "Rating", "timestamp"])

    return generate_cascades_from_df(interactions)


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
