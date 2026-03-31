"""
Centrality metrics computation using SNAP.

For each inferred network (a NetInf ``.txt`` output file), seven node-level
centrality measures are computed and saved as a CSV file.

Centrality measures
-------------------
degree, betweenness, closeness, eigenvector, pagerank, clustering_coefficient,
eccentricity

Each measure is computed with the SNAP library (snap-stanford).  Because SNAP
returns node-keyed dictionaries via its own container types, the results are
extracted with ``snap.ConvertToFltNIdH`` / ``snap.ConvertToIntNIdH`` helpers.

Outputs saved in ``data/centrality_metrics/<model>/``:

* ``centrality_metrics_<model>_<index>.csv``
  Columns: ``UserId, degree, betweenness, closeness, eigenvector,
  pagerank, clustering, eccentricity``

Usage (CLI)::

    python -m networks.centrality --model exponential
    python -m networks.centrality --all
    python -m networks.centrality --network path/to/network.txt
    python -m networks.centrality --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from snap import snap  # type: ignore[import-untyped]
except ImportError as _snap_err:
    raise ImportError(
        "snap-stanford is required: pip install snap-stanford==6.0.0"
    ) from _snap_err

from config import Paths, Models
from networks.network_io import load_as_snap, parse_network_filename


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def _snap_hash_to_dict(snap_hash) -> dict[int, float]:
    """Convert a SNAP NIdFltH / NIdIntH hash to a plain Python dict."""
    result: dict[int, float] = {}
    item = snap_hash.BegI()
    end = snap_hash.EndI()
    while item < end:
        result[item.GetKey()] = float(item.GetDat())
        item.Next()
    return result


def calculate_degree(G) -> dict[int, float]:
    """Degree centrality (raw degree count) for every node in *G*."""
    deg: dict[int, float] = {}
    for node in G.Nodes():
        deg[node.GetId()] = float(node.GetDeg())
    return deg


def calculate_betweenness(G) -> dict[int, float]:
    """
    Betweenness centrality.

    Uses ``snap.GetBetweennessCentr``.  Only node betweenness is returned;
    edge betweenness is discarded.
    """
    nodes_btwn = snap.TIntFltH()  # type: ignore[attr-defined]
    edges_btwn = snap.TIntPrFltH()  # type: ignore[attr-defined]
    snap.GetBetweennessCentr(G, nodes_btwn, edges_btwn, 1.0)  # type: ignore[attr-defined]
    return _snap_hash_to_dict(nodes_btwn)


def calculate_closeness(G) -> dict[int, float]:
    """Closeness centrality via ``snap.GetClosenessCentr`` (per-node)."""
    closeness: dict[int, float] = {}
    for node in G.Nodes():
        nid = node.GetId()
        closeness[nid] = snap.GetClosenessCentr(G, nid)  # type: ignore[attr-defined]
    return closeness


def calculate_eigenvector(G) -> dict[int, float]:
    """Eigenvector centrality via ``snap.GetEigenVectorCentr``."""
    eig_hash = snap.TIntFltH()  # type: ignore[attr-defined]
    snap.GetEigenVectorCentr(G, eig_hash)  # type: ignore[attr-defined]
    return _snap_hash_to_dict(eig_hash)


def calculate_pagerank(G) -> dict[int, float]:
    """PageRank via ``snap.GetPageRank`` (damping = 0.85)."""
    pr_hash = snap.TIntFltH()  # type: ignore[attr-defined]
    snap.GetPageRank(G, pr_hash)  # type: ignore[attr-defined]
    return _snap_hash_to_dict(pr_hash)


def calculate_clustering(G) -> dict[int, float]:
    """Local clustering coefficient via ``snap.GetNodeClustCf``."""
    cc_hash = snap.TIntFltH()  # type: ignore[attr-defined]
    snap.GetNodeClustCf(G, cc_hash)  # type: ignore[attr-defined]
    return _snap_hash_to_dict(cc_hash)


def calculate_eccentricity(G) -> dict[int, float]:
    """Eccentricity (maximum shortest-path length) per node."""
    ecc: dict[int, float] = {}
    for node in G.Nodes():
        nid = node.GetId()
        ecc[nid] = float(snap.GetNodeEcc(G, nid, False))  # type: ignore[attr-defined]
    return ecc


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_all_centrality(G) -> dict[str, dict[int, float]]:
    """
    Compute all seven centrality measures for graph *G*.

    Returns:
        Dict mapping metric name → {node_id: value}.
    """
    return {
        "degree": calculate_degree(G),
        "betweenness": calculate_betweenness(G),
        "closeness": calculate_closeness(G),
        "eigenvector": calculate_eigenvector(G),
        "pagerank": calculate_pagerank(G),
        "clustering": calculate_clustering(G),
        "eccentricity": calculate_eccentricity(G),
    }


def save_centrality_results(
    user_ids: list[int],
    metrics: dict[str, dict[int, float]],
    model_name: str,
    network_id: str,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Write centrality metrics to a CSV file.

    Args:
        user_ids:    Ordered list of node identifiers.
        metrics:     Dict produced by :func:`compute_all_centrality`.
        model_name:  Diffusion model name (exponential / powerlaw / rayleigh).
        network_id:  Zero-padded index string (e.g. ``"007"``).
        output_dir:  Directory to write into.  Defaults to
            ``Paths.CENTRALITY / model_name``.

    Returns:
        Path of the written CSV file.
    """
    if output_dir is None:
        output_dir = Paths.CENTRALITY / model_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for uid in user_ids:
        records.append(
            {
                "UserId": uid,
                "degree": metrics["degree"].get(uid, 0.0),
                "betweenness": metrics["betweenness"].get(uid, 0.0),
                "closeness": metrics["closeness"].get(uid, 0.0),
                "eigenvector": metrics["eigenvector"].get(uid, 0.0),
                "pagerank": metrics["pagerank"].get(uid, 0.0),
                "clustering": metrics["clustering"].get(uid, 0.0),
                "eccentricity": metrics["eccentricity"].get(uid, 0.0),
            }
        )

    df = pd.DataFrame(records)
    out_file = output_dir / f"centrality_metrics_{model_name}_{network_id}.csv"
    df.to_csv(out_file, index=False)
    print(f"  Saved: {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Per-network and batch runners
# ---------------------------------------------------------------------------


def calculate_centrality_for_network(network_file: str | Path) -> bool:
    """
    Compute and save centrality metrics for a single network file.

    Args:
        network_file: Path to a NetInf output ``.txt`` file.

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
    metrics = compute_all_centrality(G)
    save_centrality_results(user_ids, metrics, model_name, network_id)
    return True


def calculate_centrality_for_all_models() -> dict[str, int]:
    """
    Process every inferred network for all three diffusion models.

    Returns:
        Dict mapping model name → number of successfully processed files.
    """
    summary: dict[str, int] = {}
    for model_name in Models.ALL:
        model_dir = Paths.NETWORKS / model_name
        if not model_dir.exists():
            print(f"  Skipping {model_name}: directory not found ({model_dir})")
            summary[model_name] = 0
            continue

        network_files = sorted(model_dir.glob("inferred-network-*.txt"))
        if not network_files:
            print(f"  Skipping {model_name}: no network files found")
            summary[model_name] = 0
            continue

        print(f"\n{'='*50}")
        print(f"Model: {model_name.upper()} — {len(network_files)} networks")
        print("=" * 50)

        success_count = 0
        for nf in network_files:
            if calculate_centrality_for_network(nf):
                success_count += 1

        summary[model_name] = success_count
        print(f"Completed: {success_count}/{len(network_files)}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute centrality metrics for inferred networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=Models.ALL,
        help="Process all networks for a single diffusion model.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all models.",
    )
    group.add_argument(
        "--network",
        metavar="FILE",
        help="Process a single network file.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.network:
        success = calculate_centrality_for_network(args.network)
        sys.exit(0 if success else 1)

    elif args.all:
        summary = calculate_centrality_for_all_models()
        total = sum(summary.values())
        print(f"\nTotal networks processed: {total}")
        sys.exit(0)

    else:
        model_dir = Paths.NETWORKS / args.model
        if not model_dir.exists():
            print(f"Error: directory not found: {model_dir}")
            sys.exit(1)
        network_files = sorted(model_dir.glob("inferred-network-*.txt"))
        success_count = sum(
            1 for nf in network_files if calculate_centrality_for_network(nf)
        )
        print(f"\nProcessed: {success_count}/{len(network_files)}")
        sys.exit(0)


if __name__ == "__main__":
    main()
