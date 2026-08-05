"""Cascade generation step."""

from __future__ import annotations

import argparse


def run_cascade(args: argparse.Namespace) -> None:
    from config import DatasetPaths, Datasets, Split
    from networks.cascades import generate_cascades_from_df, compute_cascade_user_stats
    from recommender.data import load_and_split_dataset
    from pipeline._artifacts import _write_artifact_manifest

    ds_name = args.dataset
    cfg = Datasets.CONFIG[ds_name]
    csv_path = Datasets.ROOT / ds_name / cfg["file"]

    # One encoding path: LabelEncoder UserId/ItemId match recommender + NetInf
    # compact IDs (C-3). Cascades are written from the encoded train split.
    full_df, train_df, test_df = load_and_split_dataset(dataset=ds_name)
    generate_cascades_from_df(
        train_df,
        all_user_ids=full_df["UserId"],
        output_file=DatasetPaths(ds_name).CASCADES,
    )

    temporal_cutoff = None
    if Split.STRATEGY == "temporal" and not train_df.empty:
        temporal_cutoff = train_df["timestamp"].max()
        if hasattr(temporal_cutoff, "item"):
            temporal_cutoff = temporal_cutoff.item()
    _write_artifact_manifest(
        ds_name,
        source_path=csv_path,
        train_rows=len(train_df),
        test_rows=len(test_df),
        total_rows=len(full_df),
        n_users=int(full_df["UserId"].nunique()),
        n_items=int(full_df["ItemId"].nunique()),
        temporal_cutoff=temporal_cutoff,
    )

    compute_cascade_user_stats(dataset=ds_name)

    from visualization.network_plots import plot_cascades_timeline

    plot_cascades_timeline(
        cascade_file=str(DatasetPaths(ds_name).CASCADES),
        save=True,
        dataset=ds_name,
    )
