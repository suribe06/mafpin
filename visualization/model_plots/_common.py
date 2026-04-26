"""
Shared helper for the model_plots subpackage.
"""

from __future__ import annotations

from config import DatasetPaths, Datasets


def _plots_dir(dataset: str | None = None) -> str:
    """Return (and create) the ``plots/models/`` directory for *dataset*."""
    out = DatasetPaths(dataset or Datasets.DEFAULT).PLOTS / "models"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)
