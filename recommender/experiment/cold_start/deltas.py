"""Per-user deltas and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from recommender.experiment.cold_start.strata import STRATA_ORDER


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
