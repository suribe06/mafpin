#!/usr/bin/env python3
"""B2: M4c boundary_downweight mechanism check (edge stats + per-user correlation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DatasetPaths  # noqa: E402
from recommender.data import load_and_split_dataset  # noqa: E402
from recommender.enhanced.social_regularization import (  # noqa: E402
    compute_boundary_mechanism_stats,
)
from recommender.experiment.route_b.paths import RouteBPaths  # noqa: E402


def _m4c_manifest(dataset: str) -> dict:
    manifest = json.loads(DatasetPaths(dataset).EXPERIMENT_MANIFEST.read_text())
    entry = (manifest.get("variants") or {}).get("M4c")
    if not entry:
        raise FileNotFoundError(f"No M4c entry in experiment_manifest for {dataset}")
    return entry


def _per_user_rmse(pred_path: Path) -> pd.Series:
    df = pd.read_parquet(pred_path)
    err = (df["Rating"].astype(float) - df["Prediction"].astype(float)) ** 2
    return err.groupby(df["UserId"]).mean().pow(0.5).rename("rmse")


def _user_delta_table(
    *,
    dataset: str,
    ba_per_user: Path,
    predictions_dir: Path,
) -> pd.DataFrame:
    ba = pd.read_parquet(ba_per_user)
    # n_communities is a user property; take from any variant (prefer M1).
    if (ba["model_variant"] == "M1").any():
        com = ba.loc[ba["model_variant"] == "M1", ["UserId", "n_communities"]]
    else:
        com = ba.groupby("UserId", as_index=False)["n_communities"].first()
    com = cast(pd.DataFrame, com).drop_duplicates(subset=["UserId"])

    m1 = _per_user_rmse(predictions_dir / "M1.parquet")
    m4 = _per_user_rmse(predictions_dir / "M4c.parquet")
    # positive delta = M4c better (lower RMSE)
    delta = (m1 - m4).rename("rmse_delta_m4c_minus_m1")

    # beyond-accuracy deltas (same users ranked)
    def _metric(variant: str, col: str) -> pd.Series:
        sub = ba.loc[ba["model_variant"] == variant, ["UserId", col]].drop_duplicates(
            subset=["UserId"]
        )
        return sub.set_index("UserId")[col]

    out = com.set_index("UserId").join(delta, how="inner")
    for col in ("cce_at_k", "ild_latent_at_k", "novelty_at_k"):
        if col in ba.columns:
            out[f"delta_{col}"] = _metric("M4c", col) - _metric("M1", col)
    out["dataset"] = dataset
    return out.reset_index()


def _correlation_summary(deltas: pd.DataFrame) -> dict:
    n_com = deltas["n_communities"].astype(float)
    rmse_d = deltas["rmse_delta_m4c_minus_m1"].astype(float)
    spearman = cast(Any, spearmanr(n_com, rmse_d, nan_policy="omit"))
    zero = deltas[n_com <= 0]
    pos = deltas[n_com >= 1]
    multi = deltas[n_com >= 2]
    return {
        "n_users": int(len(deltas)),
        "frac_n_communities_0": float((n_com <= 0).mean()),
        "spearman_n_communities_vs_rmse_delta": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "mean_rmse_delta_ncom_0": float(zero["rmse_delta_m4c_minus_m1"].mean())
        if len(zero)
        else float("nan"),
        "mean_rmse_delta_ncom_ge1": float(pos["rmse_delta_m4c_minus_m1"].mean())
        if len(pos)
        else float("nan"),
        "mean_rmse_delta_ncom_ge2": float(multi["rmse_delta_m4c_minus_m1"].mean())
        if len(multi)
        else float("nan"),
        "n_ncom_0": int(len(zero)),
        "n_ncom_ge1": int(len(pos)),
        "n_ncom_ge2": int(len(multi)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument(
        "--ba-per-user",
        type=Path,
        default=None,
        help="beyond_accuracy_per_user.parquet (default: B1 seed_42 archive)",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Directory with M1.parquet / M4c.parquet predictions",
    )
    args = parser.parse_args()

    entry = _m4c_manifest(args.dataset)
    net = entry["selected_network"]
    hp = entry["hyperparameters"]
    model = str(net["diffusion_model"])
    alpha_index = int(net["alpha_index"])
    beta = float(hp["beta"])

    _, train_df, test_df = load_and_split_dataset(dataset=args.dataset)
    users = sorted(
        set(train_df["UserId"].astype(int)).union(set(test_df["UserId"].astype(int)))
    )

    stats = compute_boundary_mechanism_stats(
        args.dataset,
        model,
        alpha_index,
        users,
        beta=beta,
    )

    route = RouteBPaths(args.dataset)
    out_dir = route.BASE / "m4c_mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)

    ba_path = args.ba_per_user or (
        route.BASE / "multiseed" / "seed_42" / "beyond_accuracy_per_user.parquet"
    )
    pred_dir = args.predictions_dir or route.PREDICTIONS

    deltas = _user_delta_table(
        dataset=args.dataset, ba_per_user=ba_path, predictions_dir=pred_dir
    )
    corr = _correlation_summary(deltas)

    # Falsification heuristic from the plan: improvement concentrated on
    # users with no community info → not boundary-aware.
    mean0 = corr["mean_rmse_delta_ncom_0"]
    mean1 = corr["mean_rmse_delta_ncom_ge1"]
    mechanism_ok = (
        stats["frac_endpoint_has_community"] >= 0.25
        and stats["frac_downweighted_given_jaccard"] >= 0.05
        and (
            np.isnan(mean0)
            or np.isnan(mean1)
            or mean1 >= mean0 - 1e-6  # community users at least as helped
        )
    )

    payload = {
        "edge_stats": stats,
        "per_user_correlation": corr,
        "verdict": {
            "mechanism_plausible": bool(mechanism_ok),
            "note": (
                "plausible if enough edges touch communities, some weights are "
                "actually downweighted, and RMSE gains are not larger on "
                "n_communities=0 than on n_communities≥1"
            ),
        },
        "sources": {
            "ba_per_user": str(ba_path),
            "predictions_dir": str(pred_dir),
            "beta": beta,
            "network": {"diffusion_model": model, "alpha_index": alpha_index},
        },
    }

    (out_dir / "mechanism_summary.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    deltas.to_csv(out_dir / "per_user_deltas.csv", index=False)
    pd.DataFrame([stats]).to_csv(out_dir / "edge_stats.csv", index=False)

    print(json.dumps(payload, indent=2, default=float))
    print(f"\nWrote → {out_dir}")


if __name__ == "__main__":
    main()
