"""
Batch runners and persistence for centrality metrics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DatasetPaths, Datasets, Models
from networks.network_io import load_as_snap, parse_network_filename
from networks.centrality.metrics import (
    calculate_degree,
    calculate_in_degree,
    calculate_out_degree,
    calculate_hits,
    calculate_betweenness,
    calculate_closeness,
    calculate_eigenvector,
    calculate_pagerank,
    calculate_clustering,
    calculate_eccentricity,
)
from networks.centrality.pagerank_lph import compute_pagerank_lph


def compute_all_centrality(G) -> dict[str, dict[int, float]]:
    """
    Compute all centrality measures for graph *G*.

    Returns:
        Dict mapping metric name → {node_id: value}.
    """
    hub_scores, auth_scores = calculate_hits(G)
    return {
        "degree": calculate_degree(G),
        "in_degree": calculate_in_degree(G),
        "out_degree": calculate_out_degree(G),
        "betweenness": calculate_betweenness(G),
        "closeness": calculate_closeness(G),
        "eigenvector": calculate_eigenvector(G),
        "pagerank": calculate_pagerank(G),
        "clustering": calculate_clustering(G),
        "eccentricity": calculate_eccentricity(G),
        "hub_score": hub_scores,
        "auth_score": auth_scores,
    }


def save_centrality_results(
    user_ids: list[int],
    metrics: dict[str, dict[int, float]],
    model_name: str,
    network_id: str,
    output_dir: str | Path | None = None,
    pagerank_lph: dict[int, float] | None = None,
) -> Path:
    """
    Write centrality metrics to a CSV file.

    Args:
        user_ids:     Ordered list of node identifiers.
        metrics:      Dict produced by :func:`compute_all_centrality`.
        model_name:   Diffusion model name (exponential / powerlaw / rayleigh).
        network_id:   Zero-padded index string (e.g. ``"007"``).
        output_dir:   Directory to write into.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).CENTRALITY / model_name``.
        pagerank_lph: Optional LPH-weighted custom PageRank scores.  When
            provided, a ``pagerank_lph`` column is appended to the CSV.

    Returns:
        Path of the written CSV file.
    """
    if output_dir is None:
        output_dir = DatasetPaths(Datasets.DEFAULT).CENTRALITY / model_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for uid in user_ids:
        row = {
            "UserId": uid,
            "degree": metrics["degree"].get(uid, 0.0),
            "in_degree": metrics["in_degree"].get(uid, 0.0),
            "out_degree": metrics["out_degree"].get(uid, 0.0),
            "betweenness": metrics["betweenness"].get(uid, 0.0),
            "closeness": metrics["closeness"].get(uid, 0.0),
            "eigenvector": metrics["eigenvector"].get(uid, 0.0),
            "pagerank": metrics["pagerank"].get(uid, 0.0),
            "clustering": metrics["clustering"].get(uid, 0.0),
            "eccentricity": metrics["eccentricity"].get(uid, 0.0),
            "hub_score": metrics["hub_score"].get(uid, 0.0),
            "auth_score": metrics["auth_score"].get(uid, 0.0),
        }
        if pagerank_lph is not None:
            row["pagerank_lph"] = pagerank_lph.get(uid, 0.0)
        records.append(row)

    df = pd.DataFrame(records)
    out_file = output_dir / f"centrality_metrics_{model_name}_{network_id}.csv"
    df.to_csv(out_file, index=False)
    print(f"  Saved: {out_file}")
    return out_file


def calculate_centrality_for_network(
    network_file: str | Path,
    communities_dir: Path | None = None,
    centrality_dir: Path | None = None,
) -> bool:
    """
    Compute and save centrality metrics for a single network file.

    Args:
        network_file:    Path to a NetInf output ``.txt`` file.
        communities_dir: Root communities directory for pagerank_lph lookup.
            Defaults to ``DatasetPaths(Datasets.DEFAULT).COMMUNITIES``.
        centrality_dir:  Root centrality output directory.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).CENTRALITY``.

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    network_file = Path(network_file)
    if not network_file.exists():
        print(f"Error: network file not found: {network_file}")
        return False

    try:
        model_name, network_id = parse_network_filename(network_file.name)
    except ValueError as exc:
        print(f"Error parsing filename: {exc}")
        return False

    print(f"Processing {network_file.name} (model={model_name}, id={network_id})")
    G, user_ids = load_as_snap(network_file)

    # Guard: skip degenerate (edgeless) networks — centrality on an edgeless
    # graph produces all-zero features that dilute the side-information signal.
    if G.GetEdges() == 0:  # type: ignore[attr-defined]
        print(f"  Warning: no edges in {network_file.name}, skipping.")
        return False

    metrics = compute_all_centrality(G)

    pr_lph = compute_pagerank_lph(
        network_file, model_name, network_id, communities_dir=communities_dir
    )
    if pr_lph is not None:
        print("  Computing pagerank_lph (LPH-weighted custom PageRank) ✓")
    else:
        print(
            "  pagerank_lph skipped (no community CSV found; run communities step first)"
        )

    save_centrality_results(
        user_ids,
        metrics,
        model_name,
        network_id,
        output_dir=centrality_dir,
        pagerank_lph=pr_lph,
    )
    return True


def calculate_centrality_for_all_models(
    dataset: str | None = None,
) -> dict[str, int]:
    """
    Process every inferred network for all three diffusion models.

    Args:
        dataset: Dataset name.  Defaults to ``Datasets.DEFAULT``.  Used to
            locate inferred networks and write centrality outputs into the
            correct dataset-scoped subdirectory.

    Returns:
        Dict mapping model name → number of successfully processed files.
    """
    dp = DatasetPaths(dataset or Datasets.DEFAULT)
    summary: dict[str, int] = {}
    for model_name in Models.ALL:
        model_dir = dp.NETWORKS / model_name
        if not model_dir.exists():
            print(f"  Skipping {model_name}: directory not found ({model_dir})")
            summary[model_name] = 0
            continue

        network_files = sorted(model_dir.glob("inferred-network-*.txt"))
        if not network_files:
            print(f"  Skipping {model_name}: no network files found")
            summary[model_name] = 0
            continue

        from tqdm import tqdm

        print(f"\n{'='*50}")
        print(f"Model: {model_name.upper()} — {len(network_files)} networks")
        print("=" * 50)

        success_count = 0
        pbar = tqdm(
            network_files,
            desc=f"{model_name[:4].upper()} centrality",
            unit="net",
            dynamic_ncols=True,
        )
        for nf in pbar:
            pbar.set_postfix(file=nf.stem[-6:])
            if calculate_centrality_for_network(
                nf,
                communities_dir=dp.COMMUNITIES,
                centrality_dir=dp.CENTRALITY / model_name,
            ):
                success_count += 1
        pbar.close()

        summary[model_name] = success_count
        print(f"Completed: {success_count}/{len(network_files)}")

    return summary
