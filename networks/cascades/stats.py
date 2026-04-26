"""
Per-user cascade statistics derived from a NetInf cascade file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets


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
