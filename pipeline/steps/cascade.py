"""Cascade generation step."""

from __future__ import annotations

import argparse


def run_cascade(args: argparse.Namespace) -> None:
    import pandas as pd

    from config import DatasetPaths, Datasets, Split
    from networks.cascades import generate_cascades_from_df, compute_cascade_user_stats
    from recommender.data import split_data_temporal, split_data_single
    from pipeline._artifacts import _write_artifact_manifest

    ds_name = args.dataset
    cfg = Datasets.CONFIG[ds_name]
    csv_path = Datasets.ROOT / ds_name / cfg["file"]
    cols = [cfg["col_user"], cfg["col_item"], cfg["col_rating"], cfg["col_time"]]
    df = pd.read_csv(  # type: ignore[call-overload]
        csv_path,
        sep=cfg["sep"],
        header=cfg["header"],
        usecols=cols,  # type: ignore[call-overload]
        engine="python",
    )
    df.columns = pd.Index(["UserId", "ItemId", "Rating", "timestamp"])

    # Apply the global split respecting config.Split.STRATEGY so that NetInf
    # learns only from training interactions.  Pass all_user_ids=df["UserId"]
    # so the cascade header declares the full user-ID space, keeping compact
    # network IDs aligned with LabelEncoder (C-3 fix).
    if Split.STRATEGY == "temporal":
        train_df, _ = split_data_temporal(df, test_size=Split.TEST_SIZE)
    else:
        train_df, _ = split_data_single(
            df, test_size=Split.TEST_SIZE, random_state=Split.RANDOM_STATE
        )
    generate_cascades_from_df(
        train_df,
        all_user_ids=df["UserId"],
        output_file=DatasetPaths(ds_name).CASCADES,
    )

    # Persist split config so downstream steps can detect stale artifacts.
    _write_artifact_manifest(ds_name)

    compute_cascade_user_stats(dataset=ds_name)

    from visualization.network_plots import plot_cascades_timeline

    plot_cascades_timeline(
        cascade_file=str(DatasetPaths(ds_name).CASCADES),
        save=True,
        dataset=ds_name,
    )
