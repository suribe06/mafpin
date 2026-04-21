"""
Temporal delta computation and alpha grid construction.

The *delta* parameter represents the median inter-event time observed across
all cascade pairs.  It is used as the centre of the log-spaced alpha grid that
controls the decay rate of NetInf's diffusion models (exponential, Rayleigh).

Functions
---------
compute_median_delta
    Reads ``cascades.txt`` and returns the median positive inter-event time.
alpha_centers_from_delta
    Derives canonical alpha centre values (one per model) from a delta in
    seconds.
log_alpha_grid
    Produces a symmetric log-spaced grid around a given anchor point.

Usage (CLI)::

    python -m networks.delta
    python -m networks.delta --cascades path/to/cascades.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from config import Defaults, DatasetPaths, Datasets


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def compute_median_delta(cascade_file: str | Path | None = None) -> float:
    """
    Compute the median of all positive consecutive inter-event time
    differences across cascades.

    Each cascade line encodes ``user,timestamp,user,timestamp,...``.  Within
    each cascade only adjacent (consecutive) timestamp pairs are compared;
    only positive differences are retained.

    Timestamps in the cascade file are expected in **days** (as written by
    ``generate_cascades_from_df``).  The returned delta is therefore in days,
    and ``alpha_centers_from_delta`` will produce alpha values in days⁻¹,
    which keeps NetInf's log-likelihood surface numerically well-conditioned.

    Args:
        cascade_file: Path to the cascades file.  Defaults to
            ``Paths.CASCADES``.

    Returns:
        Median delta in days (float).

    Raises:
        FileNotFoundError: If the cascade file does not exist.
        ValueError: If no positive deltas can be computed from the file.
    """
    if cascade_file is None:
        cascade_file = DatasetPaths(Datasets.DEFAULT).CASCADES
    cascade_file = Path(cascade_file)

    if not cascade_file.exists():
        raise FileNotFoundError(f"Cascade file not found: {cascade_file}")

    deltas: list[float] = []

    with open(cascade_file, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue  # header or single-event cascades — skip

            # Timestamps are at odd indices: 1, 3, 5, …
            timestamps = [float(parts[i]) for i in range(1, len(parts), 2)]
            # Use only consecutive (adjacent) differences.
            # Cascades are now written in ascending time order (C-1 fix), so
            # timestamps[i+1] >= timestamps[i] always holds; no abs() needed.
            timestamps.sort()  # defensive sort in case file was generated earlier
            for i in range(len(timestamps) - 1):
                diff = timestamps[i + 1] - timestamps[i]
                if diff > 0:
                    deltas.append(diff)

    if not deltas:
        raise ValueError("No positive time deltas found in cascade file.")

    median = float(np.median(deltas))
    print(f"Median delta: {median:.4f} days")
    return median


def alpha_centers_from_delta(delta_days: float) -> dict:
    """
    Derive alpha centre values for each diffusion model from a delta (days).

    Returns a nested dict with keys ``exponential`` and ``rayleigh``, each
    containing the sub-key ``alpha0`` (in days⁻¹ for exponential,
    days⁻² for Rayleigh).

    Note: The power-law model uses a different parametrisation (exponent),
    so no alpha centre is derived for it here.

    Args:
        delta_days: Median inter-event time in days (as returned by
            ``compute_median_delta`` when cascades use day-scale timestamps).

    Returns:
        Dict mapping model name → dict with key ``alpha0``.
    """
    # Exponential: median m = ln(2)/α  ⟹  α = ln(2)/m  [days⁻¹]
    alpha0_exp = np.log(2.0) / delta_days
    # Rayleigh: median m = sqrt(2·ln(2)/α)  ⟹  α = 2·ln(2)/m²  [days⁻²]
    alpha0_ray = 2.0 * np.log(2.0) / (delta_days**2)

    return {
        "exponential": {"alpha0": alpha0_exp},
        "rayleigh": {"alpha0": alpha0_ray},
    }


def count_cascade_nodes(cascade_file: str | Path | None = None) -> int:
    """
    Count the number of unique nodes declared in a NetInf cascade file.

    The file format has two blocks separated by a blank line.  The first block
    contains one ``id,name`` line per node.  Counting these lines gives N, from
    which the maximum number of directed edges in the inferred network can be
    derived as N*(N-1).

    Args:
        cascade_file: Path to the cascades file.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).CASCADES``.

    Returns:
        Integer number of nodes N.

    Raises:
        FileNotFoundError: If the cascade file does not exist.
        ValueError: If no node lines are found before the first blank line.
    """
    if cascade_file is None:
        cascade_file = DatasetPaths(Datasets.DEFAULT).CASCADES
    cascade_file = Path(cascade_file)

    if not cascade_file.exists():
        raise FileNotFoundError(f"Cascade file not found: {cascade_file}")

    n_nodes = 0
    with open(cascade_file, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            if not raw_line.strip():
                break  # blank line separates the two blocks
            n_nodes += 1

    if n_nodes == 0:
        raise ValueError("No node lines found in cascade file.")

    return n_nodes


def compute_k_from_nodes(n_nodes: int, avg_degree: float = 2.0) -> int:
    """
    Estimate the NetInf edge budget k as ``avg_degree × N``.

    The NetInf paper's real-data experiments (Gomez-Rodriguez et al. 2011,
    Section 5.2) infer networks whose edge count is roughly *linear* in the
    number of nodes N — equivalent to an average out-degree of 1–4.  Using
    ``k = avg_degree × N`` keeps the budget tractable for any dataset size
    and matches the sparsity regime studied in the paper.  A quadratic
    formula (fraction of N²) would give millions of edges for large N and
    is inconsistent with the paper's experimental setup.

    Args:
        n_nodes:    Number of unique nodes N.
        avg_degree: Target average out-degree (default 2, matching the
            paper's typical synthetic-network density).

    Returns:
        Integer edge budget k (at least 1).
    """
    return max(1, round(avg_degree * n_nodes))


def log_alpha_grid(
    alpha0: float,
    r: float = Defaults.RANGE_R,
    n: int = Defaults.N_ALPHAS,
) -> np.ndarray:
    """
    Build a symmetric log-spaced grid of alpha values centred on *alpha0*.

    The grid spans ``[alpha0 / r, alpha0 * r]`` with *n* evenly spaced points
    on a logarithmic scale.

    Args:
        alpha0: Centre of the grid (canonical alpha value).
        r:      Multiplicative range factor (default ``Defaults.RANGE_R``).
        n:      Number of grid points (default ``Defaults.N_ALPHAS``).

    Returns:
        1-D numpy array of length *n*.
    """
    return np.logspace(np.log10(alpha0 / r), np.log10(alpha0 * r), n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute median delta and alpha grids from cascade data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cascades",
        default=str(DatasetPaths(Datasets.DEFAULT).CASCADES),
        help="Path to the cascades file.",
    )
    parser.add_argument(
        "--range-r",
        type=float,
        default=Defaults.RANGE_R,
        help="Multiplicative range factor for the log alpha grid.",
    )
    parser.add_argument(
        "--n-alphas",
        type=int,
        default=Defaults.N_ALPHAS,
        help="Number of alpha values in the grid.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        delta = compute_median_delta(args.cascades)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    centers = alpha_centers_from_delta(delta)
    for model, vals in centers.items():
        print(f"\n{model.capitalize()} model alpha centre:")
        print(f"  alpha0 = {vals['alpha0']:.6e} days⁻¹")

    print("\nSample log alpha grids (first 5 values):")
    for model in ("exponential", "rayleigh"):
        alpha0 = centers[model]["alpha0"]
        grid = log_alpha_grid(alpha0, r=args.range_r, n=args.n_alphas)
        print(f"  {model}: {grid[:5]} ...")


if __name__ == "__main__":
    main()
