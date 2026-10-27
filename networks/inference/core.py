"""
Core NetInf invocation: run the binary for a single model across an alpha grid.
"""

from __future__ import annotations

import shutil
import subprocess
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
from networks.inference.subprocess_utils import (
    _create_output_dirs,
    _cleanup_leftover_edge_info,
)


def _run_one_netinf(
    *,
    idx: int,
    alpha: float,
    cascades_file: Path,
    output_stem: str,
    model: int,
    k: int,
    model_dir: Path,
    edge_info_dir: Path,
) -> tuple[int, float, int, bool]:
    """Run one NetInf alpha; return (idx, alpha, edge_count, ok)."""
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
            cwd=str(Paths.NETINF_BIN.parent),
            check=False,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  [{idx:03d}] subprocess error: {exc}")
        return idx, float(alpha), 0, False

    netinf_cwd = Paths.NETINF_BIN.parent
    output_file = netinf_cwd / f"{output_stem}.txt"
    if result.returncode != 0 or not output_file.exists():
        err = (result.stderr or "").strip()
        out = (result.stdout or "").strip()
        hint = ""
        if "Can not open file" in out or "Can not open file" in err:
            hint = " (cascade path unreadable from NetInf cwd=networks/)"
        print(f"  [{idx:03d}] FAILED (rc={result.returncode}){hint}")
        return idx, float(alpha), 0, False

    edge_count = 0
    try:
        in_edges = False
        with open(output_file, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if not stripped:
                    in_edges = True
                    continue
                if in_edges:
                    edge_count += 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  [{idx:03d}] parse error: {exc}")
        return idx, float(alpha), 0, False

    shutil.move(str(output_file), model_dir / output_file.name)
    edge_info_src = netinf_cwd / f"{output_stem}-edge.info"
    if edge_info_src.exists():
        shutil.move(str(edge_info_src), edge_info_dir / edge_info_src.name)
    return idx, float(alpha), edge_count, True


def infer_networks(
    cascades_file: str | Path | None = None,
    n: int = Defaults.N_ALPHAS,
    model: int = 0,
    max_iter: int = Defaults.MAX_ITER,
    k_avg_degree: float | None = Defaults.K_AVG_DEGREE,
    name_output: str = "inferred-network",
    r: float = Defaults.RANGE_R,
    networks_dir: Path | None = None,
    n_jobs: int = 1,
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
        n_jobs:         Parallel NetInf workers (ThreadPool; each job is a
            subprocess). ``1`` = sequential.

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
    # Absolute path: NetInf subprocess uses cwd=networks/, so relative -i: breaks.
    cascades_file = Path(cascades_file).expanduser().resolve()

    if not cascades_file.exists():
        print(f"Error: cascades file not found: {cascades_file}")
        print("Run 'python -m networks.cascades' first.")
        return False

    if not Paths.NETINF_BIN.exists():
        print(f"Error: NetInf binary not found at {Paths.NETINF_BIN}")
        return False

    if networks_dir is not None:
        networks_dir = Path(networks_dir).expanduser().resolve()

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

    workers = max(1, int(n_jobs))
    print(f"\nStarting inference — model: {model_name}, max_iter: {max_iter}")
    print(f"Output directory: {model_dir}")
    print(f"Parallel NetInf workers: {workers}\n")

    edges_count = [0] * n
    successful_runs = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    jobs = [
        {
            "idx": idx,
            "alpha": float(alpha),
            "cascades_file": cascades_file,
            "output_stem": f"{name_output}-{model_suffix}-{idx:03d}",
            "model": model,
            "k": k,
            "model_dir": model_dir,
            "edge_info_dir": edge_info_dir,
        }
        for idx, alpha in enumerate(alpha_values)
    ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one_netinf, **job) for job in jobs]
        pbar = tqdm(
            as_completed(futures),
            total=n,
            desc=f"{model_name[:4].upper()} alpha sweep",
            unit="net",
            dynamic_ncols=True,
        )
        for fut in pbar:
            idx, alpha, edge_count, ok = fut.result()
            edges_count[idx] = edge_count
            if ok:
                successful_runs += 1
            pbar.set_postfix(alpha=f"{alpha:.2e}", edges=edge_count, ok=ok)

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
