"""Shared synthetic data and path helpers for unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ratings_frame(
    rows: list[tuple[int, int, float]],
    *,
    timestamps: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal ratings DataFrame from (user, item, rating) tuples."""
    data: dict[str, Any] = {
        "UserId": [row[0] for row in rows],
        "ItemId": [row[1] for row in rows],
        "Rating": [row[2] for row in rows],
    }
    if timestamps is not None:
        data["timestamp"] = timestamps
    return pd.DataFrame(data)


def user_attributes_frame(user_ids: list[int], n_features: int = 2) -> pd.DataFrame:
    """Feature matrix indexed by UserId (matches load_network_features shape)."""
    values = [[float(i + j) for j in range(n_features)] for i in user_ids]
    columns = [f"f{j}" for j in range(n_features)]
    frame = pd.DataFrame(
        values, index=pd.Index(user_ids, name="UserId"), columns=pd.Index(columns)
    )
    return frame


def write_alpha_grid_csv(
    path: Path,
    alphas: list[float],
    *,
    rmse_col: str = "enhanced_rmse_mean",
    rmse_values: list[float] | None = None,
) -> None:
    """Write a pipe-separated inferred-edges summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rmse_values = rmse_values or [0.9 - 0.01 * i for i in range(len(alphas))]
    lines = ["alpha|" + rmse_col]
    for alpha, rmse in zip(alphas, rmse_values):
        lines.append(f"{alpha}|{rmse}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class FakeDatasetPaths:
    """Minimal DatasetPaths stand-in rooted at a temp directory."""

    def __init__(self, root: Path, dataset: str = "movielens") -> None:
        base = root / dataset
        self.BASE = base
        self.NETWORKS = base / "inferred_networks"
        self.RUNS = base / "runs"
        self.LOGS = base / "logs"
        self.EXPERIMENT_MANIFEST = base / "experiment_manifest.json"
        self.NETWORK_SELECTION = base / "network_selection_results.json"
        self.CANONICAL_BASELINE = base / "canonical_baseline.json"


def recommend_log_m3() -> str:
    return """
[RECOMMEND] Done.
Enhanced CMF best hyperparameters:
{
  "k": 9,
  "lambda_reg": 0.5,
  "w_main": 1.0,
  "w_user": 0.1
}
Exponential — Best α=1.5119e-01  RMSE=0.889019  improvement=+0.30%
"""


def recommend_log_m4c() -> str:
    return """
[RECOMMEND] Done.
Enhanced CMF best hyperparameters:
{
  "k": 9,
  "lambda_reg": 0.5,
  "w_main": 1.0,
  "w_user": 0.1
}
Social CMF best hyperparameters:
{
  "k": 7,
  "lambda_reg": 0.3,
  "lambda_social": 0.01,
  "beta": 0.4,
  "gamma": 1.2
}
Exponential — Best α=0.15  RMSE=0.88  improvement=+0.30%
"""


def sample_manifest_entry(
    *,
    variant_id: str = "M3",
    run_id: str = "m3_recommend",
    social: bool = False,
) -> dict[str, Any]:
    network_best = {
        "exponential": {"alpha": 0.15, "cv_rmse": 0.89},
        "powerlaw": {"alpha": 2.1, "cv_rmse": 0.91},
    }
    entry: dict[str, Any] = {
        "variant_id": variant_id,
        "run_id": run_id,
        "social_regularization": social,
        "network_best": network_best,
        "hyperparameters": {"k": 9, "lambda_reg": 0.5},
    }
    return entry
