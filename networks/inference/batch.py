"""
Batch runner: infer networks for all three diffusion models.
"""

from __future__ import annotations

from pathlib import Path

from config import Defaults, Models
from networks.inference.core import infer_networks


def infer_networks_all_models(
    cascades_file: str | Path | None = None,
    n: int = Defaults.N_ALPHAS,
    max_iter: int = Defaults.MAX_ITER,
    k_avg_degree: float | None = Defaults.K_AVG_DEGREE,
    name_output: str = "inferred-network",
    r: float = Defaults.RANGE_R,
    networks_dir: Path | None = None,
) -> dict[str, bool]:
    """
    Run network inference for all three diffusion models.

    Args:
        cascades_file: Path to cascades file.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).CASCADES``.
        n:              Number of alpha grid points.
        max_iter:       Fallback edge budget k when *k_avg_degree* is ``None``.
        k_avg_degree:   Target average out-degree used to compute k.  Defaults
            to ``Defaults.K_AVG_DEGREE`` (2).
        name_output:    Base name for output files.
        r:              Range factor for the log alpha grid.
        networks_dir:   Root directory for output networks.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).NETWORKS``.

    Returns:
        Dict mapping model name → success flag.
    """
    results: dict[str, bool] = {}
    for idx, model_name in enumerate(Models.ALL):
        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()}")
        print("=" * 60)
        results[model_name] = infer_networks(
            cascades_file=cascades_file,
            n=n,
            model=idx,
            max_iter=max_iter,
            k_avg_degree=k_avg_degree,
            name_output=name_output,
            r=r,
            networks_dir=networks_dir,
        )
    return results
