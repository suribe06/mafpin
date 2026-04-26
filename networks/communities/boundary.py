"""
Boundary-spanner indicator and community result persistence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DatasetPaths, Datasets


def compute_boundary_indicator(
    lph_scores: dict[int, float],
    percentile: float = 20.0,
) -> dict[int, int]:
    """
    Compute a binary boundary indicator for each node based on low h̃v values.

    A node is flagged as a *boundary-spanner* (``1``) when its h̃v score falls
    at or below the *percentile*-th percentile across all nodes in the network.
    Nodes above that threshold are flagged ``0``.

    The lower the percentile, the more selective the criterion.  A 20th-
    percentile threshold (the default) captures roughly the most negative
    fifth of the distribution — the nodes whose community profile diverges
    most from their neighbourhood.

    Args:
        lph_scores: Dict mapping node_id → h̃v score (from :func:`~networks.communities.lph.compute_lph_paper`).
        percentile: Percentile threshold in the range (0, 100].  Nodes with
            h̃v ≤ this percentile are flagged as boundary-spanners.

    Returns:
        Dict mapping node_id → ``1`` (boundary) or ``0`` (non-boundary).
    """
    if not lph_scores:
        return {}

    scores = list(lph_scores.values())
    threshold = float(pd.Series(scores).quantile(percentile / 100.0))
    return {node: int(score <= threshold) for node, score in lph_scores.items()}


def save_community_results(
    user_ids: list[int],
    lph: dict[int, float],
    lph_paper: dict[int, float],
    membership: dict[int, set[int]],
    model_name: str,
    network_id: str,
    output_dir: str | Path | None = None,
    s_values: dict[int, int] | None = None,
    delta_values: dict[int, float] | None = None,
    boundary_indicator: dict[int, int] | None = None,
) -> Path:
    """
    Write community and LPH results to CSV.

    The output CSV includes all Phase-4 boundary metrics:

    * ``UserId``                — node identifier.
    * ``num_communities``       — m(v) = |C(v)|, membership count.
    * ``community_ids``         — semicolon-separated community indices.
    * ``local_pluralistic_hom`` — Jaccard-based LPH ∈ [0, 1].
    * ``s_v``                   — s(v) neighborhood alignment count (Eq. 3).
    * ``delta_v``               — δv local dissimilarity (Eq. 4).
    * ``lph_score``             — h̃v normalized boundary score (Eq. 6).
    * ``is_boundary``           — binary boundary indicator (1 = boundary-spanner).

    Args:
        user_ids:           Ordered list of node identifiers.
        lph:                Dict of Jaccard-based LPH scores per node.
        lph_paper:          Dict of normalized h̃v scores per node.
        membership:         Dict of community-index sets per node.
        model_name:         Diffusion model name.
        network_id:         Zero-padded index string (e.g. ``"007"``).
        output_dir:         Target directory.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).COMMUNITIES / model_name``.
        s_values:           Dict of s(v) neighborhood alignment counts.
        delta_values:       Dict of δv local dissimilarity values.
        boundary_indicator: Dict of binary boundary flags.

    Returns:
        Path of the written CSV file.
    """
    if output_dir is None:
        output_dir = DatasetPaths(Datasets.DEFAULT).COMMUNITIES / model_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for uid in user_ids:
        coms = sorted(membership.get(uid, set()))
        records.append(
            {
                "UserId": uid,
                "num_communities": len(coms),
                "community_ids": ";".join(map(str, coms)),
                "local_pluralistic_hom": lph.get(uid, 0.0),
                "s_v": s_values.get(uid) if s_values is not None else float("nan"),
                "delta_v": (
                    delta_values.get(uid) if delta_values is not None else float("nan")
                ),
                "lph_score": lph_paper.get(uid, 0.0),
                "is_boundary": (
                    boundary_indicator.get(uid)
                    if boundary_indicator is not None
                    else float("nan")
                ),
            }
        )

    df = pd.DataFrame(records)
    out_file = output_dir / f"communities_{model_name}_{network_id}.csv"
    df.to_csv(out_file, index=False)
    print(f"  Saved: {out_file}")
    return out_file
