"""
Feature loading for enhanced CMF: centrality, community, and cascade-stats CSVs.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from config import DatasetPaths, Datasets, SideUserFeatures

# Scaler type alias used inside the per-split loop.
# WARNING: "normalizer" uses sklearn.preprocessing.Normalizer, which normalises
# each *sample* (row) to unit norm, NOT each *feature* (column).  This is
# semantically incorrect for feature scaling and should not be the first choice.
# Prefer "standard" (StandardScaler) or "minmax" (MinMaxScaler).
_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "normalizer": Normalizer,  # row-normalisation — see warning above
}


def load_network_features(
    model_name: str,
    network_index: int,
    include_communities: bool = True,
    include_cascade_stats: bool = True,
    dataset: str | None = None,
    feature_config: "dict[str, bool] | None" = None,
) -> pd.DataFrame | None:
    """
    Load centrality (and optionally community and cascade-stats) features for
    one inferred network.

    Returns a raw (unscaled) DataFrame indexed by ``UserId``.  Scaling is
    intentionally deferred to :func:`~recommender.enhanced.model.evaluate_cmf_with_user_attributes`
    where it is fitted on training users only, preventing test-set leakage (M-2).

    Args:
        model_name:             Diffusion model name (exponential / powerlaw / rayleigh).
        network_index:          Zero-based network index (selects the CSV by filename).
        include_communities:    If ``True``, merge LPH and ``num_communities`` from the
                                corresponding community CSV.
        include_cascade_stats:  If ``True``, merge ``mean_cascade_position``,
                                ``min_cascade_position``, and ``cascade_breadth``
                                from ``cascade_user_stats.csv``.
        dataset:                Dataset name.  Defaults to ``Datasets.DEFAULT``.
        feature_config:         Dict mapping feature name → ``bool``.  Columns whose
                                key is ``False`` are dropped.
                                Defaults to ``SideUserFeatures.FEATURES``.

    Returns:
        Raw feature DataFrame indexed by ``UserId``, or ``None`` if the file is missing.
    """
    dp = DatasetPaths(dataset or Datasets.DEFAULT)
    index_str = f"{network_index:03d}"
    centrality_dir = dp.CENTRALITY / model_name
    centrality_csv = centrality_dir / f"centrality_metrics_{model_name}_{index_str}.csv"

    if not centrality_csv.exists():
        return None

    df = pd.read_csv(centrality_csv)

    if include_communities:
        community_dir = dp.COMMUNITIES / model_name
        community_csv = community_dir / f"communities_{model_name}_{index_str}.csv"
        if community_csv.exists():
            com_raw = pd.read_csv(community_csv)
            com_cols = ["UserId", "local_pluralistic_hom", "num_communities"]
            for _opt_col in ("lph_score", "s_v", "delta_v", "is_boundary"):
                if _opt_col in com_raw.columns:
                    com_cols.append(_opt_col)
            df = df.merge(com_raw[com_cols], on="UserId", how="left")

            if "community_ids" in com_raw.columns:
                TOP_K_COMMUNITIES = 20
                from collections import Counter

                parsed = (
                    com_raw["community_ids"]
                    .fillna("")
                    .apply(lambda s: set(map(int, filter(None, s.split(";")))))
                )
                community_counts: Counter = Counter()
                for cids in parsed:
                    community_counts.update(cids)
                top_communities = [
                    cid for cid, _ in community_counts.most_common(TOP_K_COMMUNITIES)
                ]
                for cid in top_communities:
                    col = f"community_{cid}"
                    com_raw[col] = parsed.apply(lambda cids, c=cid: int(c in cids))

                binary_cols = [f"community_{cid}" for cid in top_communities]
                df = df.merge(
                    com_raw[["UserId"] + binary_cols], on="UserId", how="left"
                )

    if include_cascade_stats:
        cascade_stats_csv = dp.CASCADE_USER_STATS
        if cascade_stats_csv.exists():
            cascade_stats = pd.read_csv(cascade_stats_csv)
            df = df.merge(
                cascade_stats[
                    [
                        "UserId",
                        "mean_cascade_position",
                        "min_cascade_position",
                        "cascade_breadth",
                    ]
                ],
                on="UserId",
                how="left",
            )

    df = df.set_index("UserId").fillna(0.0)

    cfg = feature_config if feature_config is not None else SideUserFeatures.FEATURES
    disabled = {k for k, v in cfg.items() if not v}
    cols_to_drop = [
        col
        for col in df.columns
        if col in disabled
        or (col.startswith("community_") and "community_binary" in disabled)
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df
