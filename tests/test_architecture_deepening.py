"""Checks for architecture deepenings (artefact locator + warm-eval helpers)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from networks.artifacts import NetworkArtifacts
from recommender.enhanced.model import (
    filter_to_feature_users,
    iter_warm_splits,
    scaled_u_matrix,
)


def test_network_artifacts_paths(tmp_path: Path) -> None:
    from config import DatasetPaths

    class _FakePaths(DatasetPaths):
        def __init__(self, root: Path) -> None:
            super().__init__("movielens")
            self.BASE = root
            self.NETWORKS = root / "inferred_networks"
            self.CENTRALITY = root / "centrality_metrics"
            self.COMMUNITIES = root / "communities"

    root = tmp_path / "ds"
    paths = _FakePaths(root)
    (paths.NETWORKS / "exponential").mkdir(parents=True)
    (paths.CENTRALITY / "exponential").mkdir(parents=True)
    (paths.COMMUNITIES / "exponential").mkdir(parents=True)
    (paths.NETWORKS / "exponential" / "inferred-network-expo-007.txt").write_text("x")
    (paths.CENTRALITY / "exponential" / "centrality_metrics_exponential_007.csv").write_text(
        "UserId\n0\n"
    )
    (paths.COMMUNITIES / "exponential" / "communities_exponential_007.csv").write_text(
        "UserId\n0\n"
    )

    arts = NetworkArtifacts("movielens", paths=paths)
    assert arts.network_txt("exponential", 7).exists()
    assert arts.list_complete_indices("exponential") == [7]


def test_warm_eval_helpers_filter_and_scale() -> None:
    rng = list(range(20))
    data = pd.DataFrame(
        {
            "UserId": [u for u in rng for _ in range(4)],
            "ItemId": [i % 5 for i in range(80)],
            "Rating": [float((i % 5) + 1) for i in range(80)],
        }
    )
    attrs = pd.DataFrame(
        {"degree": [float(u) for u in rng[:10]]},
        index=pd.Index(rng[:10], name="UserId"),
    )
    filtered = filter_to_feature_users(data, attrs)
    assert filtered is not None
    assert set(filtered["UserId"]) <= set(rng[:10])

    splits = list(iter_warm_splits(filtered, n_splits=3, test_size=0.25))
    assert splits
    train_df, _warm = splits[0][1], splits[0][2]
    u = scaled_u_matrix(attrs, sorted(train_df["UserId"].unique()), transform="standard")
    assert "UserId" in u.columns
    assert len(u) == 10
