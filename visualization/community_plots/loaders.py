"""
Shared constants and data-loading helpers for community_plots subpackage.
"""

from __future__ import annotations

import pandas as pd

from config import Models, DatasetPaths, Datasets


PALETTE = {
    "exponential": "#2196F3",
    "powerlaw": "#FF5722",
    "rayleigh": "#4CAF50",
}
MODELS = Models.ALL


def _plots_dir(dataset: str | None = None) -> str:
    """Return (and create) the ``plots/communities/`` directory for *dataset*."""
    out = DatasetPaths(dataset or Datasets.DEFAULT).PLOTS / "communities"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _load_community_csv(
    model_name: str, network_index: int, dataset: str | None = None
) -> pd.DataFrame | None:
    fp = (
        DatasetPaths(dataset or Datasets.DEFAULT).COMMUNITIES
        / model_name
        / f"communities_{model_name}_{network_index:03d}.csv"
    )
    return pd.read_csv(fp) if fp.exists() else None


def _load_centrality_csv(
    model_name: str, network_index: int, dataset: str | None = None
) -> pd.DataFrame | None:
    fp = (
        DatasetPaths(dataset or Datasets.DEFAULT).CENTRALITY
        / model_name
        / f"centrality_metrics_{model_name}_{network_index:03d}.csv"
    )
    return pd.read_csv(fp) if fp.exists() else None


def _load_alpha_csv(model_name: str, dataset: str | None = None) -> pd.DataFrame | None:
    short = Models.SHORT[model_name]
    fp = (
        DatasetPaths(dataset or Datasets.DEFAULT).NETWORKS
        / model_name
        / f"inferred_edges_{short}.csv"
    )
    if not fp.exists():
        return None
    df = pd.read_csv(fp, sep="|").reset_index(drop=True)
    df["network_index"] = df.index
    return df


def _aggregate_community_stats(
    model_name: str, n_networks: int = 100, dataset: str | None = None
) -> pd.DataFrame:
    """
    Return per-network aggregated community statistics.

    Columns: ``network_index``, ``mean_lph``, ``median_lph``, ``std_lph``,
    ``mean_lph_score``, ``median_lph_score``, ``std_lph_score``,
    ``mean_num_coms``, ``median_num_coms``.
    """
    rows = []
    for i in range(n_networks):
        raw = _load_community_csv(model_name, i, dataset=dataset)
        if raw is None:
            continue
        row: dict = {
            "network_index": i,
            "mean_lph": raw["local_pluralistic_hom"].mean(),
            "median_lph": raw["local_pluralistic_hom"].median(),
            "std_lph": raw["local_pluralistic_hom"].std(),
            "mean_num_coms": raw["num_communities"].mean(),
            "median_num_coms": raw["num_communities"].median(),
        }
        if "lph_score" in raw.columns:
            row["mean_lph_score"] = raw["lph_score"].mean()
            row["median_lph_score"] = raw["lph_score"].median()
            row["std_lph_score"] = raw["lph_score"].std()
        rows.append(row)
    return pd.DataFrame(rows)
