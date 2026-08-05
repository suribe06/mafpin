"""Rebuild cascade → NetInf → centrality → communities under ColdStartPaths."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import Defaults, Models
from networks.artifacts import NetworkArtifacts
from networks.cascades import compute_cascade_user_stats, generate_cascades_from_df
from networks.centrality.batch import calculate_centrality_for_all_models
from networks.communities.batch import calculate_communities_for_all_models
from networks.inference import infer_networks_all_models
from recommender.experiment.cold_start.paths import ColdStartPaths


def _count_network_files(paths: ColdStartPaths) -> dict[str, int]:
    arts = NetworkArtifacts(paths.BASE.parent.name, paths=paths)
    return {
        model_name: len(arts.list_network_indices(model_name))
        for model_name in Models.ALL
    }

def run_feature_pipeline(
    dataset: str,
    train_df: pd.DataFrame,
    *,
    all_user_ids: pd.Series | list[int],
    paths: ColdStartPaths,
    n_alphas: int = Defaults.N_ALPHAS,
    max_iter: int = Defaults.MAX_ITER,
    k_avg_degree: int = Defaults.K_AVG_DEGREE,
) -> dict[str, Any]:
    """Build cold-start network features from *train_df* only (anti-leakage)."""
    paths.ensure_dirs()
    print(f"[cold_start] cascades → {paths.CASCADES}")
    ok = generate_cascades_from_df(
        train_df,
        all_user_ids=all_user_ids,
        output_file=paths.CASCADES,
    )
    if not ok:
        raise RuntimeError("Cascade generation failed for cold-start rebuild")

    compute_cascade_user_stats(
        cascade_file=paths.CASCADES,
        output_file=paths.CASCADE_USER_STATS,
        dataset=dataset,
    )

    print(f"[cold_start] NetInf → {paths.NETWORKS} (n_alphas={n_alphas})")
    infer_networks_all_models(
        n=n_alphas,
        max_iter=max_iter,
        k_avg_degree=k_avg_degree if k_avg_degree > 0 else None,
        networks_dir=paths.NETWORKS,
        cascades_file=paths.CASCADES,
    )

    net_counts = _count_network_files(paths)
    total = sum(net_counts.values())
    if total == 0:
        raise RuntimeError(
            "Cold-start NetInf produced 0 network files under "
            f"{paths.NETWORKS}. Check that cascades are readable with an "
            "absolute path (NetInf cwd is networks/). "
            f"Per-model counts: {net_counts}"
        )

    print("[cold_start] centrality")
    cent_summary = calculate_centrality_for_all_models(dataset=dataset, paths=paths)
    print("[cold_start] communities")
    com_summary = calculate_communities_for_all_models(dataset=dataset, paths=paths)

    return {
        "cascades": str(paths.CASCADES),
        "networks": str(paths.NETWORKS),
        "network_file_counts": net_counts,
        "centrality_summary": cent_summary,
        "communities_summary": com_summary,
    }
