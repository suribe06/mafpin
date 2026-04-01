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

from config import Paths, Defaults


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def compute_median_delta(cascade_file: str | Path | None = None) -> float:
    """
    Compute the median of all positive inter-event time differences across
    cascades.

    Each cascade line encodes ``user,timestamp,user,timestamp,...``.  Within
    each cascade every pair of timestamps is compared; only positive
    differences are retained.

    Args:
        cascade_file: Path to the cascades file.  Defaults to
            ``Paths.CASCADES``.

    Returns:
        Median delta in seconds (float).

    Raises:
        FileNotFoundError: If the cascade file does not exist.
        ValueError: If no positive deltas can be computed from the file.
    """
    if cascade_file is None:
        cascade_file = Paths.CASCADES
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
    print(f"Median delta: {median:.2f} seconds")
    return median


def alpha_centers_from_delta(delta_seconds: float) -> dict:
    """
    Derive alpha centre values for each diffusion model from a delta (seconds).

    Returns a nested dict with keys ``exponential`` and ``rayleigh``, each
    containing sub-keys ``alpha0_seconds``, ``alpha0_days``,
    ``alpha0_years``.

    Note: The power-law model uses a different parametrisation (exponent),
    so no alpha centre is derived for it here.

    Args:
        delta_seconds: Median inter-event time in seconds.

    Returns:
        Dict mapping model name → dict of alpha0 values in multiple time units.
    """
    seconds_per_day = 86_400.0
    seconds_per_year = 365.25 * seconds_per_day

    # Exponential: median m = ln(2)/α  ⟹  α = ln(2)/m
    alpha0_exp_sec = np.log(2.0) / delta_seconds
    # Rayleigh: median m = sqrt(2·ln(2)/α)  ⟹  α = 2·ln(2)/m²
    alpha0_ray_sec = 2.0 * np.log(2.0) / (delta_seconds**2)

    return {
        "exponential": {
            "alpha0_seconds": alpha0_exp_sec,
            "alpha0_days": alpha0_exp_sec * seconds_per_day,
            "alpha0_years": alpha0_exp_sec * seconds_per_year,
        },
        "rayleigh": {
            "alpha0_seconds": alpha0_ray_sec,
            "alpha0_days": alpha0_ray_sec * (seconds_per_day**2),
            "alpha0_years": alpha0_ray_sec * (seconds_per_year**2),
        },
    }


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
        default=str(Paths.CASCADES),
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
        print(f"\n{model.capitalize()} model alpha centres:")
        for unit, value in vals.items():
            print(f"  {unit}: {value:.6e}")

    print("\nSample log alpha grids (first 5 values):")
    for model in ("exponential", "rayleigh"):
        alpha0 = centers[model]["alpha0_seconds"]
        grid = log_alpha_grid(alpha0, r=args.range_r, n=args.n_alphas)
        print(f"  {model}: {grid[:5]} ...")


if __name__ == "__main__":
    main()
