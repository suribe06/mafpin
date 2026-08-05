"""WP2 — boundary / LPH heterogeneous effects on frozen predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommender.experiment.cold_start.deltas import bootstrap_mean_ci
from recommender.experiment.route_b.communities_freeze import (
    item_dominant_communities,
    load_frozen_communities,
)
from recommender.experiment.route_b.paths import RouteBPaths


def assign_lph_strata(
    lph: pd.Series,
    *,
    min_n: int = 30,
) -> pd.Series:
    """Map users to B10 / B25 / MID / E75 by lph_score percentiles.

    Lower ``lph_score`` (h̃v) = stronger boundary in the Appl. Sci. convention
    used by this repo (see networks/communities/boundary.py).
    """
    valid = lph.dropna()
    if valid.empty:
        return pd.Series(dtype=object)
    p10, p25, p75 = np.nanpercentile(valid.to_numpy(dtype=float), [10, 25, 75])
    labels = pd.Series(index=lph.index, dtype=object)
    labels[lph <= p10] = "B10"
    labels[(lph > p10) & (lph <= p25)] = "B25"
    labels[(lph > p25) & (lph < p75)] = "MID"
    labels[lph >= p75] = "E75"
    # Merge tiny B10 into B25 if needed
    if int((labels == "B10").sum()) < min_n:
        labels = labels.replace({"B10": "B25"})
    return labels


def per_user_rmse_from_predictions(preds: pd.DataFrame) -> pd.Series:
    err2 = (preds["Rating"] - preds["Prediction"]) ** 2
    frame = preds[["UserId"]].copy()
    frame["err2"] = err2
    means = pd.Series(frame.groupby("UserId")["err2"].mean(), dtype=float)
    return pd.Series(np.sqrt(means.to_numpy()), index=means.index, name="rmse")


def load_variant_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def run_boundary_strata(
    dataset: str,
    *,
    variants: list[str] | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    min_n: int = 30,
) -> dict[str, Path]:
    """Compute RMSE by LPH stratum and paired deltas; write Route B artifacts."""
    paths = RouteBPaths(dataset)
    paths.ensure_dirs()
    variants = variants or ["M1", "M2", "M3", "M4c", "M4d"]
    com = load_frozen_communities(dataset)
    if "lph_score" not in com.columns:
        raise ValueError("Frozen communities CSV lacks lph_score")
    strata = assign_lph_strata(pd.Series(com["lph_score"]), min_n=min_n)

    rmse_by_variant: dict[str, pd.Series] = {}
    for vid in variants:
        pred_path = paths.prediction_path(vid)
        if not pred_path.exists():
            # allow csv fallback
            alt = pred_path.with_suffix(".csv")
            pred_path = alt if alt.exists() else pred_path
        if not pred_path.exists():
            print(f"[route_b] skip {vid}: missing predictions at {paths.prediction_path(vid)}")
            continue
        preds = load_variant_predictions(pred_path)
        rmse_by_variant[vid] = per_user_rmse_from_predictions(preds)

    if "M3" not in rmse_by_variant:
        raise FileNotFoundError("Need at least M3 predictions under route_b/predictions/")

    strata_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    for stratum in ("B10", "B25", "MID", "E75"):
        mask = strata == stratum
        users = {int(i) for i in list(pd.Series(strata).loc[mask].index)}
        for vid, series in rmse_by_variant.items():
            vals = series.reindex(sorted(users)).dropna()
            strata_rows.append(
                {
                    "dataset": dataset,
                    "stratum": stratum,
                    "model_variant": vid,
                    "n_users": int(vals.shape[0]),
                    "rmse_mean": float(vals.mean()) if len(vals) else float("nan"),
                }
            )
        # Paired deltas vs M3 / M2 / M1
        m3 = rmse_by_variant["M3"]
        for other, label in (
            ("M2", "M3_vs_M2"),
            ("M1", "M3_vs_M1"),
            ("M4c", "M4c_vs_M3"),
            ("M4d", "M4d_vs_M3"),
        ):
            if other not in rmse_by_variant:
                continue
            o = rmse_by_variant[other]
            # positive delta ⇒ second name better when defined as first-second?
            # Protocol: Δ(M3−M2) > 0 means M3 better (lower RMSE) ⇒ use M2-M3
            if label.startswith("M3_vs_"):
                paired = (o - m3).reindex(sorted(users)).dropna()
            else:
                paired = (m3 - o).reindex(sorted(users)).dropna()
            stats = bootstrap_mean_ci(
                paired.to_numpy(dtype=float),
                n_samples=bootstrap_samples,
                seed=seed,
            )
            boot_rows.append(
                {
                    "dataset": dataset,
                    "stratum": stratum,
                    "comparison": label,
                    "n_users": int(stats["n"]),
                    "mean_delta": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                }
            )

    strata_df = pd.DataFrame(strata_rows)
    boot_df = pd.DataFrame(boot_rows)
    strata_df.to_csv(paths.BOUNDARY_STRATA, index=False)
    boot_df.to_csv(paths.BOUNDARY_BOOTSTRAP, index=False)
    print(f"Boundary strata → {paths.BOUNDARY_STRATA}")
    print(f"Boundary bootstrap → {paths.BOUNDARY_BOOTSTRAP}")

    # Cross-community item subset using M3 predictions if available
    m3_path = paths.prediction_path("M3")
    if not m3_path.exists():
        m3_path = m3_path.with_suffix(".csv")
    if m3_path.exists():
        preds = load_variant_predictions(m3_path)
        # Need train for item dominant — load split from data module
        from recommender.data import load_and_split_dataset

        _full, train_df, _test = load_and_split_dataset(dataset=dataset)
        item_dom = item_dominant_communities(train_df, pd.Series(com["community_set"]))
        rows = []
        for vid, series in rmse_by_variant.items():
            pred_path = paths.prediction_path(vid)
            if not pred_path.exists():
                pred_path = pred_path.with_suffix(".csv")
            if not pred_path.exists():
                continue
            p = load_variant_predictions(pred_path)
            mask = []
            for _, row in p.iterrows():
                uid = int(row["UserId"])
                iid = int(row["ItemId"])
                ucom = com["community_set"].get(uid, set()) if uid in com.index else set()
                d = item_dom.get(iid, set())
                mask.append(bool(ucom) and bool(d) and set(ucom).isdisjoint(d))
            p = p.copy()
            p["_cross"] = mask
            sub = p[p["_cross"]]
            if sub.empty:
                rmse = float("nan")
                n = 0
            else:
                rmse = float(np.sqrt(((sub["Rating"] - sub["Prediction"]) ** 2).mean()))
                n = int(len(sub))
            rows.append(
                {
                    "dataset": dataset,
                    "model_variant": vid,
                    "n_ratings": n,
                    "rmse": rmse,
                }
            )
        pd.DataFrame(rows).to_csv(paths.CROSS_COMMUNITY_ITEMS, index=False)
        print(f"Cross-community items → {paths.CROSS_COMMUNITY_ITEMS}")

    return {
        "strata": paths.BOUNDARY_STRATA,
        "bootstrap": paths.BOUNDARY_BOOTSTRAP,
        "cross_items": paths.CROSS_COMMUNITY_ITEMS,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Route B WP2 boundary strata analysis")
    parser.add_argument("--dataset", required=True, choices=["movielens", "ciao", "epinions"])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["M1", "M2", "M3", "M4c", "M4d"],
    )
    args = parser.parse_args(argv)
    run_boundary_strata(
        args.dataset,
        variants=list(args.variants),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        min_n=args.min_n,
    )


if __name__ == "__main__":
    main()
