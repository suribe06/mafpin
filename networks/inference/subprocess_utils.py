"""
Subprocess / filesystem utilities for NetInf invocations.
"""

from __future__ import annotations

import glob
import shutil
from pathlib import Path

from config import DatasetPaths, Datasets


def _create_output_dirs(
    model_name: str, networks_dir: Path | None = None
) -> tuple[Path, Path]:
    """
    Create ``<networks_dir>/<model>`` and its ``edge_info`` sub-directory.

    Args:
        model_name:   Diffusion model name.
        networks_dir: Root directory for inferred networks.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).NETWORKS``.

    Returns:
        Tuple of (model_dir, edge_info_dir) as Path objects.
    """
    root = networks_dir or DatasetPaths(Datasets.DEFAULT).NETWORKS
    model_dir = root / model_name
    edge_info_dir = model_dir / "edge_info"
    model_dir.mkdir(parents=True, exist_ok=True)
    edge_info_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, edge_info_dir


def _cleanup_leftover_edge_info(model_suffix: str, edge_info_dir: Path) -> None:
    """
    Move any remaining ``*-<suffix>-*-edge.info`` files from cwd into *edge_info_dir*.

    NetInf sometimes leaves files in the working directory when it exits early.
    """
    for leftover in glob.glob(f"*-{model_suffix}-*-edge.info"):
        try:
            shutil.move(leftover, edge_info_dir / Path(leftover).name)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  Warning: could not move {leftover}: {exc}")
