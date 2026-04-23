"""
Network inference via the NetInf algorithm.

For each alpha value in a log-spaced grid, the NetInf binary is invoked as a
subprocess.  The binary infers a directed influence network from cascade data
using one of three probabilistic diffusion models:

    0 – exponential
    1 – powerlaw
    2 – rayleigh

The alpha magnitude controls the decay rate of the model.  A symmetric grid
centred on a data-driven value (derived from the median inter-event delta in
``cascades.txt``) is used so that the sweep covers meaningfully diverse
network densities.

Outputs saved in ``data/<dataset>/inferred_networks/<model>/``:

* ``inferred-network-<short>-<index>.txt`` – NetInf network file per alpha
* ``edge_info/``                            – edge detail files from NetInf
* ``inferred_edges_<short>.csv``           – alpha | edges table (pipe-sep)
* ``alpha_grid_info_<short>.csv``          – grid provenance metadata

Usage (CLI)::

    python -m networks.inference --model exponential
    python -m networks.inference --all
    python -m networks.inference --help
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets, Paths, Models, Defaults
from networks.delta import (
    compute_median_delta,
    alpha_centers_from_delta,
    log_alpha_grid,
    count_cascade_nodes,
    compute_k_from_nodes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------


def infer_networks(
    cascades_file: str | Path | None = None,
    n: int = Defaults.N_ALPHAS,
    model: int = 0,
    max_iter: int = Defaults.MAX_ITER,
    k_avg_degree: float | None = Defaults.K_AVG_DEGREE,
    name_output: str = "inferred-network",
    r: float = Defaults.RANGE_R,
    networks_dir: Path | None = None,
) -> bool:
    """
    Run NetInf for every alpha in a log-spaced grid and save the results.

    The NetInf binary is expected at ``Paths.NETINF_BIN``.  All output files
    are written under ``networks_dir / <model_name>``.

    The edge budget k (NetInf ``-e`` flag) is derived dynamically from the
    cascade file when *k_avg_degree* is set::

        k = round(k_avg_degree * N)

    where N is the number of nodes in the cascade file.  This matches the
    paper's real-data setup (Gomez-Rodriguez et al. 2011, Section 5.2) where
    the inferred graph is sparse: roughly ``k ≈ avg_degree × N`` edges,
    corresponding to an average out-degree of 1–4.  If *k_avg_degree* is
    ``None``, *max_iter* is used instead.

    Args:
        cascades_file:  Path to the cascades input file.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).CASCADES``.
        n:              Number of alpha grid points.
        model:          Model index — 0 (exponential), 1 (powerlaw), 2 (rayleigh).
        max_iter:       Fallback edge budget k when *k_avg_degree* is ``None``.
        k_avg_degree:   Target average out-degree used to compute k.  Defaults
            to ``Defaults.K_AVG_DEGREE`` (2).  Set to ``None`` to use
            *max_iter* directly.
        name_output:    Base name for per-alpha output files.
        r:              Multiplicative range factor for the alpha grid.
        networks_dir:   Root directory for output networks.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).NETWORKS``.

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    if model not in (0, 1, 2):
        print(f"Error: invalid model {model!r}. Must be 0, 1, or 2.")
        return False

    model_name = Models.ALL[model]
    model_suffix = Models.SHORT[model_name]

    if cascades_file is None:
        cascades_file = DatasetPaths(Datasets.DEFAULT).CASCADES
    cascades_file = Path(cascades_file)

    if not cascades_file.exists():
        print(f"Error: cascades file not found: {cascades_file}")
        print("Run 'python -m networks.cascades' first.")
        return False

    if not Paths.NETINF_BIN.exists():
        print(f"Error: NetInf binary not found at {Paths.NETINF_BIN}")
        return False

    # -- Determine edge budget k (NetInf -e flag) ----------------------------
    if k_avg_degree is not None:
        try:
            n_nodes = count_cascade_nodes(cascades_file)
            k = compute_k_from_nodes(n_nodes, avg_degree=k_avg_degree)
            print(f"Edge budget k = {k_avg_degree} × {n_nodes} nodes = {k} edges")
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"Warning: could not compute dynamic k ({exc}); falling back to max_iter={max_iter}"
            )
            k = max_iter
    else:
        k = max_iter

    # -- Compute alpha grid --------------------------------------------------
    print("Computing median delta from cascades …")
    try:
        delta_days = compute_median_delta(cascades_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return False

    print(f"Median Δ = {delta_days:.2f} days")

    alpha_centers = alpha_centers_from_delta(delta_days)
    alpha_center: float | None = None

    if model == 0:  # exponential
        alpha_center = alpha_centers["exponential"]["alpha0"]
        alpha_values = log_alpha_grid(float(alpha_center), r=r, n=n)  # type: ignore[arg-type]
        print(f"Exponential α_center = {alpha_center:.6e} days⁻¹")
    elif model == 1:
        # Power-law model: f(Δt; α) = (α − 1) · Δt^{−α}  (Pareto distribution).
        # α is a dimensionless shape exponent — it does NOT scale with the time
        # unit, so unlike exponential/Rayleigh no data-driven centre is needed.
        #
        # Lower bound is 1.1 (not 1.0): at α = 1 the Pareto density is
        # f(t) ∝ t^{-1}, whose integral diverges — the distribution is
        # non-normalizable, so the likelihood is undefined.  NetInf may accept
        # α = 1 without error, but the result is numerically meaningless.
        #
        # Upper bound 5.0: social-influence cascades rarely have exponents above
        # 3–4 (Gomez-Rodriguez et al. 2011, Section 5.2 real-data experiments
        # use α ∈ {1.5, 2.0, 2.5}).  Sweeping to 5.0 gives margin while keeping
        # runtime tractable.
        alpha_values = np.linspace(1.1, 5.0, n)
    else:  # rayleigh
        alpha_center = alpha_centers["rayleigh"]["alpha0"]
        alpha_values = log_alpha_grid(float(alpha_center), r=r, n=n)  # type: ignore[arg-type]
        print(f"Rayleigh α_center = {alpha_center:.6e} days⁻²")

    alpha_min, alpha_max = float(alpha_values.min()), float(alpha_values.max())
    print(f"Alpha grid: [{alpha_min:.2e}, {alpha_max:.2e}], {n} points")

    # -- Prepare output directories  -----------------------------------------
    model_dir, edge_info_dir = _create_output_dirs(model_name, networks_dir)

    print(f"\nStarting inference — model: {model_name}, max_iter: {max_iter}")
    print(f"Output directory: {model_dir}\n")

    edges_count: list[int] = []
    successful_runs: int = 0

    from tqdm import tqdm

    pbar = tqdm(
        enumerate(alpha_values),
        total=n,
        desc=f"{model_name[:4].upper()} alpha sweep",
        unit="net",
        dynamic_ncols=True,
    )
    for idx, alpha in pbar:
        output_stem = f"{name_output}-{model_suffix}-{idx:03d}"
        pbar.set_postfix(alpha=f"{alpha:.2e}")

        cmd = [
            str(Paths.NETINF_BIN),
            f"-i:{cascades_file}",
            f"-o:{output_stem}",
            f"-m:{model}",
            f"-e:{k}",
            f"-a:{alpha}",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(
                    Paths.NETINF_BIN.parent
                ),  # run from networks/ where binary lives
                check=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"subprocess error: {exc}")
            edges_count.append(0)
            continue

        netinf_cwd = Paths.NETINF_BIN.parent
        output_file = netinf_cwd / f"{output_stem}.txt"

        if result.returncode != 0 or not output_file.exists():
            pbar.write(f"  [{idx:03d}] FAILED (rc={result.returncode})")
            edges_count.append(0)
            continue

        # Count inferred edges (second block of the output file)
        try:
            edge_count = 0
            in_edges = False
            with open(output_file, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    stripped = raw_line.strip()
                    if not stripped:
                        in_edges = True
                        continue
                    if in_edges:
                        edge_count += 1
            edges_count.append(edge_count)
            pbar.set_postfix(alpha=f"{alpha:.2e}", edges=edge_count)
        except Exception as exc:  # pylint: disable=broad-except
            pbar.write(f"  [{idx:03d}] parse error: {exc}")
            edges_count.append(0)

        # Relocate network file
        shutil.move(str(output_file), model_dir / output_file.name)

        # Relocate edge info file if present
        edge_info_src = netinf_cwd / f"{output_stem}-edge.info"
        if edge_info_src.exists():
            shutil.move(str(edge_info_src), edge_info_dir / edge_info_src.name)

        successful_runs += 1

    # Clean up any stragglers
    _cleanup_leftover_edge_info(model_suffix, edge_info_dir)

    # -- Save summary CSV files ----------------------------------------------
    results_df = pd.DataFrame(
        {
            "alpha": alpha_values,
            f"inferred_edges_{model_suffix}": edges_count,
        }
    )
    results_file = model_dir / f"inferred_edges_{model_suffix}.csv"
    results_df.to_csv(results_file, sep="|", index=False)

    grid_info_df = pd.DataFrame(
        {
            "median_delta_days": [delta_days],
            "alpha_center": [alpha_center],
            "alpha_min": [alpha_min],
            "alpha_max": [alpha_max],
            "r_factor": [r],
            "model_type": [model_name],
        }
    )
    grid_info_file = model_dir / f"alpha_grid_info_{model_suffix}.csv"
    grid_info_df.to_csv(grid_info_file, index=False)

    print(f"\nDone — {successful_runs}/{n} successful runs")
    print(f"Results   : {results_file}")
    print(f"Grid info : {grid_info_file}")
    return True


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer diffusion networks from cascade data using NetInf.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=Models.ALL,
        help="Run inference for a single diffusion model.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run inference for all three models.",
    )
    parser.add_argument(
        "--dataset",
        choices=Datasets.ALL,
        default=Datasets.DEFAULT,
        help="Dataset whose cascades/networks to use (default: %(default)s).",
    )
    parser.add_argument(
        "--cascades",
        default=None,
        help="Override path to the cascades file.",
    )
    parser.add_argument(
        "--n-alphas",
        type=int,
        default=Defaults.N_ALPHAS,
        help="Number of alpha values in the grid.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=Defaults.MAX_ITER,
        help="Fallback edge budget k when --k-fraction is disabled.",
    )
    parser.add_argument(
        "--k-avg-degree",
        type=float,
        default=Defaults.K_AVG_DEGREE,
        help="k = avg_degree × N edges per network (0 to disable; paper default: 2).",
    )
    parser.add_argument(
        "--range-r",
        type=float,
        default=Defaults.RANGE_R,
        help="Multiplicative range factor for the log alpha grid.",
    )
    parser.add_argument(
        "--name-output",
        default="inferred-network",
        help="Base name for per-alpha output files.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dp = DatasetPaths(args.dataset)
    cascades_file = args.cascades or dp.CASCADES
    networks_dir = dp.NETWORKS
    k_avg_degree = args.k_avg_degree if args.k_avg_degree > 0 else None

    if args.all:
        results = infer_networks_all_models(
            cascades_file=cascades_file,
            n=args.n_alphas,
            max_iter=args.max_iter,
            k_avg_degree=k_avg_degree,
            name_output=args.name_output,
            r=args.range_r,
            networks_dir=networks_dir,
        )
        any_failed = any(not v for v in results.values())
        sys.exit(1 if any_failed else 0)
    else:
        model_idx = Models.ALL.index(args.model)
        success = infer_networks(
            cascades_file=cascades_file,
            n=args.n_alphas,
            model=model_idx,
            max_iter=args.max_iter,
            k_avg_degree=k_avg_degree,
            name_output=args.name_output,
            r=args.range_r,
            networks_dir=networks_dir,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
