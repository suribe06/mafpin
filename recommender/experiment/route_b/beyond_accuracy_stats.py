"""Bootstrap + Wilcoxon/Holm for Route B beyond-accuracy per-user deltas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommender.experiment.cold_start.deltas import bootstrap_mean_ci

# Preregistered paired comparisons (protocol §2.4 / §4.4).
_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("M3", "M2", "M3_vs_M2"),
    ("M4c", "M3", "M4c_vs_M3"),
    ("M4d", "M3", "M4d_vs_M3"),
    ("M3", "M1", "M3_vs_M1"),
)

_METRIC_COLS = ("cce_at_k", "ild_latent_at_k", "novelty_at_k")


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment; NaNs stay NaN."""
    n = len(p_values)
    out = [float("nan")] * n
    order = sorted(
        (i for i, p in enumerate(p_values) if np.isfinite(p)),
        key=lambda i: p_values[i],
    )
    running = 0.0
    for rank, idx in enumerate(order):
        m_remaining = n - rank
        adj = min(1.0, p_values[idx] * m_remaining)
        running = max(running, adj)
        out[idx] = running
    return out


def _wilcoxon_p(deltas: np.ndarray) -> float:
    vals = deltas[np.isfinite(deltas)]
    if vals.size < 5 or np.allclose(vals, 0.0):
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        # positive Δ ⇒ first variant better on metrics where higher is better
        res = wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return float("nan")


def paired_metric_deltas(
    per_user: pd.DataFrame,
    *,
    metric: str,
    variant_a: str,
    variant_b: str,
) -> np.ndarray:
    """Δ_u = metric(a) − metric(b); positive ⇒ a higher (better for CCE/ILD/novelty)."""
    a = per_user.loc[per_user["model_variant"] == variant_a, ["UserId", metric]]
    b = per_user.loc[per_user["model_variant"] == variant_b, ["UserId", metric]]
    if a.empty or b.empty:
        return np.asarray([], dtype=float)
    merged = a.merge(b, on="UserId", suffixes=("_a", "_b"))
    return (merged[f"{metric}_a"] - merged[f"{metric}_b"]).to_numpy(dtype=float)


def write_beyond_accuracy_bootstrap(
    per_user: pd.DataFrame,
    path: Path,
    *,
    n_samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Path:
    """Write CIs + Wilcoxon p + Holm-adjusted p for preregistered BA deltas."""
    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    row_idx_for_p: list[int] = []

    dataset = str(per_user["dataset"].iloc[0]) if "dataset" in per_user.columns else ""
    for metric in _METRIC_COLS:
        if metric not in per_user.columns:
            continue
        for va, vb, label in _COMPARISONS:
            deltas = paired_metric_deltas(
                per_user, metric=metric, variant_a=va, variant_b=vb
            )
            stats = bootstrap_mean_ci(
                deltas, n_samples=n_samples, seed=seed, alpha=alpha
            )
            p_raw = _wilcoxon_p(deltas)
            row = {
                "dataset": dataset,
                "comparison": label,
                "metric": metric,
                "variant_a": va,
                "variant_b": vb,
                "n_users": stats["n"],
                "mean_delta": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "wilcoxon_p": p_raw,
                "wilcoxon_p_holm": float("nan"),
            }
            row_idx_for_p.append(len(rows))
            raw_p.append(p_raw)
            rows.append(row)

    holm = _holm_adjust(raw_p)
    for i, adj in zip(row_idx_for_p, holm):
        rows[i]["wilcoxon_p_holm"] = adj

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f"Beyond-accuracy bootstrap → {path}")
    return path
