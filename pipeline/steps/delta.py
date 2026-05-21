"""Delta computation step."""

from __future__ import annotations

import argparse


def run_delta(args: argparse.Namespace) -> None:
    from config import DatasetPaths
    from networks.delta import compute_median_delta, alpha_centers_from_delta

    delta = compute_median_delta(DatasetPaths(args.dataset).CASCADES)
    print(f"Median delta: {delta:.4f} days")
    centers = alpha_centers_from_delta(delta)
    for model, info in centers.items():
        print(f"  {model}: alpha0 = {info['alpha0']:.4e} days⁻¹")
