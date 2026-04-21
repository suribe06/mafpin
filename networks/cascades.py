"""
Cascade generation from rating datasets.

A *cascade* represents the temporal sequence of user interactions with a
specific item (e.g. movie ratings).  The cascade format is required as input
for the NetInf algorithm used in the network inference step.

Expected DataFrame columns (after dataset loading):
    UserId    – unique user identifier
    ItemId    – unique item identifier
    Rating    – numeric rating
    timestamp – Unix epoch seconds

Output file (``data/<dataset>/cascades.txt``):
    - Header block: one ``user_id,user_id`` line per unique user
    - Empty separator line
    - One cascade line per item:
      ``user_node,timestamp,user_node,timestamp,...``

Usage (CLI)::

    python -m networks.cascades --dataset movielens
    python -m networks.cascades --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def list_available_datasets() -> list[str]:
    """Return dataset names found as subdirectories of ``datasets/``."""
    if not Datasets.ROOT.exists():
        print(f"Error: datasets directory '{Datasets.ROOT}' not found.")
        return []
    return [d.name for d in sorted(Datasets.ROOT.iterdir()) if d.is_dir()]


def compute_cascade_user_stats(
    cascade_file: str | Path | None = None,
    min_cascades: int = 5,
    output_file: str | Path | None = None,
    dataset: str | None = None,
) -> pd.DataFrame:
    """
    Parse a NetInf cascade file and compute per-user temporal influence stats.

    For each user the following three scalars are derived from the cascade
    file produced by :func:`generate_cascades_from_df`:

    * ``mean_cascade_position`` — average 1-indexed rank across all cascades
      the user appears in.  Rank 1 = seed (earliest adopter).  Lower values
      indicate users who consistently act as influence *sources*.
    * ``min_cascade_position`` — minimum rank observed; ``1`` means the user
      was the seed in at least one cascade.
    * ``cascade_breadth`` — number of distinct items/cascades the user
      participates in.

    Users who appear in fewer than *min_cascades* cascades receive ``NaN`` for
    ``mean_cascade_position`` and ``min_cascade_position`` (their positional
    estimates are unreliable).  ``cascade_breadth`` is always recorded.
    When the output is merged into the feature matrix by
    :func:`recommender.enhanced.load_network_features`, ``NaN`` values are
    replaced with ``0.0`` by the downstream ``fillna(0.0)`` call.

    The compact user IDs used as keys are derived from the cascade header by
    sorting declared node IDs and assigning 0-based indices — exactly the same
    logic as ``_build_mapper`` in ``networks/network_io.py`` — so the result
    aligns directly with centrality-metric CSVs indexed by the same ``UserId``.

    Args:
        cascade_file:  Path to ``cascades.txt``.  Defaults to
            ``DatasetPaths(dataset).CASCADES``.
        min_cascades:  Minimum number of cascade appearances required for
            positional statistics to be reported (default 5).
        output_file:   Where to write the output CSV.  Defaults to
            ``DatasetPaths(dataset).CASCADE_USER_STATS``.
        dataset:       Dataset name.  Defaults to ``Datasets.DEFAULT``.

    Returns:
        DataFrame with columns ``UserId``, ``mean_cascade_position``,
        ``min_cascade_position``, ``cascade_breadth``.

    Raises:
        FileNotFoundError: If *cascade_file* does not exist.
        ValueError: If no node declarations are found in the file header.
    """
    dataset = dataset or Datasets.DEFAULT
    if cascade_file is None:
        cascade_file = DatasetPaths(dataset).CASCADES
    cascade_file = Path(cascade_file)

    if not cascade_file.exists():
        raise FileNotFoundError(f"Cascade file not found: {cascade_file}")

    # ------------------------------------------------------------------ #
    # 1. Parse file: header → node IDs; remainder → cascade lines         #
    # ------------------------------------------------------------------ #
    node_ids: list[int] = []
    cascade_lines: list[str] = []
    in_header = True

    with open(cascade_file, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if in_header:
                if not line:
                    in_header = False
                    continue
                parts = line.split(",")
                if len(parts) == 2 and parts[0] == parts[1]:
                    node_ids.append(int(parts[0]))
            else:
                if line:
                    cascade_lines.append(line)

    if not node_ids:
        raise ValueError("No node declarations found in the cascade file header.")

    # ------------------------------------------------------------------ #
    # 2. Build compact-ID mapper (mirrors _build_mapper in network_io.py) #
    # ------------------------------------------------------------------ #
    # Sorted ascending: compact_id = sorted-rank of the original node ID.
    compact_mapper: dict[int, int] = {nid: i for i, nid in enumerate(sorted(node_ids))}

    # ------------------------------------------------------------------ #
    # 3. Accumulate per-user cascade positions                            #
    # ------------------------------------------------------------------ #
    # Each cascade line is already sorted ascending by timestamp (seeds first).
    # Position 1 = first user in the cascade (seed / earliest adopter).
    position_lists: dict[int, list[int]] = {cid: [] for cid in compact_mapper.values()}

    for line in cascade_lines:
        parts = line.split(",")
        # Extract user nodes (every even-indexed token; odd-indexed = timestamp)
        users_in_cascade: list[int] = []
        for i in range(0, len(parts) - 1, 2):
            try:
                node_id = int(parts[i])
                compact_id = compact_mapper.get(node_id)
                if compact_id is not None:
                    users_in_cascade.append(compact_id)
            except ValueError:
                continue

        for pos, compact_id in enumerate(users_in_cascade, start=1):
            position_lists[compact_id].append(pos)

    # ------------------------------------------------------------------ #
    # 4. Build summary DataFrame                                          #
    # ------------------------------------------------------------------ #
    records = []
    for compact_id in sorted(compact_mapper.values()):
        positions = position_lists[compact_id]
        breadth = len(positions)
        if breadth >= min_cascades:
            mean_pos: float = float(np.mean(positions))
            min_pos: float = float(np.min(positions))
        else:
            mean_pos = float("nan")
            min_pos = float("nan")
        records.append(
            {
                "UserId": compact_id,
                "mean_cascade_position": mean_pos,
                "min_cascade_position": min_pos,
                "cascade_breadth": breadth,
            }
        )

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # 5. Save                                                             #
    # ------------------------------------------------------------------ #
    if output_file is None:
        output_file = DatasetPaths(dataset).CASCADE_USER_STATS
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Cascade user stats saved to: {output_file}")
    return df


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
        output_file = DatasetPaths(Datasets.DEFAULT).CASCADES
    output_file = Path(output_file)

    num_users = interactions["UserId"].nunique()
    num_items = interactions["ItemId"].nunique()
    print(f"Found {num_items} unique items and {num_users} unique users")

    # Fixed user-ID offset that keeps item and user namespaces disjoint in the
    # NetInf node namespace.  A fixed constant (rather than num_items) is
    # required so that the mapping is stable across datasets and across train /
    # test subsets that may have different numbers of unique items — which would
    # silently shift the user-ID range and break any subsequent join on UserId.
    USER_ID_OFFSET = 1_000_000

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
    item_mapper = dict(zip(np.unique(interactions["ItemId"]), range(int(num_items))))
    user_mapper = dict(
        zip(
            users_for_mapping,
            range(USER_ID_OFFSET, USER_ID_OFFSET + num_users_total),
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
