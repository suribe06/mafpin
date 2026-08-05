"""User strata, per-user deltas, and bootstrap CIs for cold-start evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STRATA_ORDER = ("0", "1-3", "4-10", ">10")


def assign_stratum(n_train_ratings: int) -> str:
    """Map train rating count to a cold-start stratum label."""
    if n_train_ratings <= 0:
        return "0"
    if n_train_ratings <= 3:
        return "1-3"
    if n_train_ratings <= 10:
        return "4-10"
    return ">10"


def build_user_strata(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-user stratum table from train/test rating counts.

    Optional *feature_coverage* must be indexed by ``UserId`` with boolean
    columns among ``appears_in_netinf_graph``, ``has_centrality_features``,
    ``has_lph_features``, ``has_trust_features``.
    """
    train_counts = train_df.groupby("UserId").size().rename("n_train_ratings")
    test_counts = test_df.groupby("UserId").size().rename("n_test_ratings")
    users = sorted(set(train_counts.index) | set(test_counts.index))
    frame = pd.DataFrame({"user_id": users}).set_index("user_id")
    frame["n_train_ratings"] = train_counts.reindex(users).fillna(0).astype(int)
    frame["n_test_ratings"] = test_counts.reindex(users).fillna(0).astype(int)
    frame["stratum"] = frame["n_train_ratings"].map(assign_stratum)

    for col in (
        "appears_in_netinf_graph",
        "has_centrality_features",
        "has_lph_features",
        "has_trust_features",
    ):
        if feature_coverage is not None and col in feature_coverage.columns:
            frame[col] = (
                feature_coverage.reindex(users)[col].fillna(False).astype(bool)
            )
        else:
            frame[col] = False

    # Only users with at least one test rating are evaluation targets.
    frame = frame[frame["n_test_ratings"] > 0].copy()
    frame = frame.reset_index().rename(columns={"user_id": "user_id"})
    return frame


def strata_user_map(user_strata: pd.DataFrame) -> dict[str, set[int]]:
    """Return stratum → set of user ids (with test ratings)."""
    out: dict[str, set[int]] = {s: set() for s in STRATA_ORDER}
    for row in user_strata.itertuples(index=False):
        out.setdefault(str(row.stratum), set()).add(int(row.user_id))
    return out


def coverage_from_feature_index(
    user_ids: list[int],
    *,
    centrality_users: set[int] | None = None,
    lph_users: set[int] | None = None,
    trust_users: set[int] | None = None,
) -> pd.DataFrame:
    """Build a feature-coverage frame indexed by UserId."""
    centrality_users = centrality_users or set()
    lph_users = lph_users or set()
    trust_users = trust_users or set()
    rows: list[dict[str, Any]] = []
    for uid in user_ids:
        rows.append(
            {
                "appears_in_netinf_graph": uid in centrality_users,
                "has_centrality_features": uid in centrality_users,
                "has_lph_features": uid in lph_users,
                "has_trust_features": uid in trust_users,
            }
        )
    return pd.DataFrame(rows, index=pd.Index(user_ids, name="UserId"))


def per_user_rmse(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.Series:
    """RMSE per UserId on test rows (drops unreasonable predictions)."""
    from recommender.data import rating_reasonableness_limit

    limit = rating_reasonableness_limit(test_df["Rating"].to_numpy(dtype=float))
    frame = test_df[["UserId"]].copy()
    err2 = (y_true - y_pred) ** 2
    sane = (
        np.isfinite(y_pred)
        & np.isfinite(y_true)
        & (np.abs(y_pred - y_true) <= limit)
        & (np.abs(y_pred) <= limit)
    )
    frame["err2"] = np.where(sane, err2, np.nan)
    grouped = frame.groupby("UserId")["err2"]
    return np.sqrt(grouped.mean()).rename("rmse")


def build_user_deltas(
    *,
    dataset: str,
    mode: str,
    user_strata: pd.DataFrame,
    rmse_by_variant: dict[str, pd.Series],
) -> pd.DataFrame:
    """Wide per-user RMSE table plus M3−M1 / M3−M2 (or trust) deltas."""
    base = user_strata[["user_id", "stratum", "n_train_ratings", "n_test_ratings"]].copy()
    base = base.rename(columns={"user_id": "UserId"})
    for variant_id, series in rmse_by_variant.items():
        base[f"rmse_{variant_id}"] = base["UserId"].map(series)

    m3_col = "rmse_M3" if "rmse_M3" in base.columns else (
        "rmse_M3_trust" if "rmse_M3_trust" in base.columns else None
    )
    m2_col = "rmse_M2" if "rmse_M2" in base.columns else (
        "rmse_M2_trust" if "rmse_M2_trust" in base.columns else None
    )
    if "rmse_M1" in base.columns and m3_col is not None:
        base["delta_m3_m1"] = base["rmse_M1"] - base[m3_col]
    else:
        base["delta_m3_m1"] = np.nan
    if m2_col is not None and m3_col is not None:
        base["delta_m3_m2"] = base[m2_col] - base[m3_col]
    else:
        base["delta_m3_m2"] = np.nan
    if "rmse_M1" in base.columns and m2_col is not None:
        base["delta_m2_m1"] = base["rmse_M1"] - base[m2_col]
    else:
        base["delta_m2_m1"] = np.nan

    base.insert(0, "dataset", dataset)
    base.insert(1, "mode", mode)
    return base


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap mean and percentile CI for a 1-D array."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    means = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = float(sample.mean())
    return {
        "n": float(values.size),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
    }


def bootstrap_delta_table(
    user_deltas: pd.DataFrame,
    *,
    dataset: str,
    mode: str,
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap CIs for delta_m3_m1 / delta_m3_m2 / delta_m2_m1 by stratum."""
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("delta_m3_m1", "M3_vs_M1"),
        ("delta_m3_m2", "M3_vs_M2"),
        ("delta_m2_m1", "M2_vs_M1"),
    )
    for stratum in STRATA_ORDER:
        subset = user_deltas[user_deltas["stratum"] == stratum]
        for col, comparison in comparisons:
            if col not in subset.columns:
                continue
            stats = bootstrap_mean_ci(
                subset[col].to_numpy(dtype=float),
                n_samples=n_samples,
                seed=seed,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "stratum": stratum,
                    "comparison": comparison,
                    "n_users": int(stats["n"]),
                    "mean_delta": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                }
            )
    return pd.DataFrame(rows)
