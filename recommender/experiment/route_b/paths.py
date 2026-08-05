"""Artifact paths under ``data/<dataset>/route_b/``."""

from __future__ import annotations

from pathlib import Path

from config import DatasetPaths


class RouteBPaths:
    """Route B output layout (does not overwrite core experiment CSVs)."""

    def __init__(self, dataset: str, root: Path | str | None = None) -> None:
        base = Path(root).expanduser().resolve() if root else DatasetPaths(dataset).ROUTE_B
        self.BASE = base
        self.BEYOND_ACCURACY = base / "beyond_accuracy_results.csv"
        self.BEYOND_ACCURACY_PER_USER = base / "beyond_accuracy_per_user.parquet"
        self.BEYOND_ACCURACY_BOOTSTRAP = base / "beyond_accuracy_bootstrap.csv"
        self.PREDICTIONS = base / "predictions"
        self.BOUNDARY_STRATA = base / "boundary_strata_results.csv"
        self.BOUNDARY_BOOTSTRAP = base / "boundary_strata_bootstrap.csv"
        self.CROSS_COMMUNITY_ITEMS = base / "cross_community_items_results.csv"
        self.COMMUNITY_STABILITY = base / "community_stability.csv"
        self.SOFT_ASSIGNMENT = base / "soft_assignment"
        self.LOGS = DatasetPaths(dataset).LOGS / "route_b"

    def ensure_dirs(self) -> None:
        for path in (
            self.BASE,
            self.PREDICTIONS,
            self.SOFT_ASSIGNMENT,
            self.LOGS,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def prediction_path(self, variant_id: str) -> Path:
        return self.PREDICTIONS / f"{variant_id}.parquet"
