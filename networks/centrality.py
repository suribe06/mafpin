"""
Centrality metrics computation using SNAP.

For each inferred network (a NetInf ``.txt`` output file), eleven node-level
centrality measures are computed and saved as a CSV file.

Centrality measures
-------------------
degree, in_degree, out_degree, betweenness, closeness, eigenvector,
pagerank, clustering_coefficient, eccentricity, hub_score, auth_score

Each measure is computed with the SNAP library (snap-stanford).  Because SNAP
returns node-keyed dictionaries via its own container types, the results are
extracted with ``snap.ConvertToFltNIdH`` / ``snap.ConvertToIntNIdH`` helpers.

Additionally, if the ``communities`` step has already been run for the same
network, a ``pagerank_lph`` column is appended: a generalised PageRank
(Newman Eq. 7.18) where the intrinsic centrality vector β is set to the
normalised LPH scores from Barraza et al. (2025).

Outputs saved in ``data/<dataset>/centrality_metrics/<model>/``:

* ``centrality_metrics_<model>_<index>.csv``
  Columns: ``UserId, degree, in_degree, out_degree, betweenness, closeness,
  eigenvector, pagerank, clustering, eccentricity, hub_score, auth_score
  [, pagerank_lph]``

  ``pagerank_lph`` is present only when the corresponding community CSV exists.

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

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from snap import snap  # type: ignore[import-untyped]
except ImportError as _snap_err:
    raise ImportError(
        "snap-stanford is required: pip install snap-stanford==6.0.0"
    ) from _snap_err

from config import DatasetPaths, Datasets, Models
from networks.network_io import load_as_networkx, load_as_snap, parse_network_filename


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
    """Normalised degree centrality (degree / (N-1)) for every node in *G*."""
    n_nodes = G.GetNodes()
    denom = max(n_nodes - 1, 1)
    deg: dict[int, float] = {}
    for node in G.Nodes():
        deg[node.GetId()] = float(node.GetDeg()) / denom
    return deg


def calculate_in_degree(G) -> dict[int, float]:
    """Normalised in-degree centrality (in_degree / (N-1)) for every node in *G*.

    In a directed NetInf influence graph, in-degree counts incoming influence
    edges — the number of users who tend to influence this node.  High
    in-degree marks **influence sinks** (followers / late adopters).
    """
    n_nodes = G.GetNodes()
    denom = max(n_nodes - 1, 1)
    in_deg: dict[int, float] = {}
    for node in G.Nodes():
        in_deg[node.GetId()] = float(node.GetInDeg()) / denom
    return in_deg


def calculate_out_degree(G) -> dict[int, float]:
    """Normalised out-degree centrality (out_degree / (N-1)) for every node in *G*.

    In a directed NetInf influence graph, out-degree counts outgoing influence
    edges — the number of users this node tends to influence.  High out-degree
    marks **influence sources** (taste-makers / early adopters).
    """
    n_nodes = G.GetNodes()
    denom = max(n_nodes - 1, 1)
    out_deg: dict[int, float] = {}
    for node in G.Nodes():
        out_deg[node.GetId()] = float(node.GetOutDeg()) / denom
    return out_deg


def calculate_hits(G) -> tuple[dict[int, float], dict[int, float]]:
    """HITS hub and authority scores via ``snap.GetHits``.

    HITS (Hyperlink-Induced Topic Search, Kleinberg 1999) computes two
    complementary scores for each node in a directed graph:

    * **Hub score** — high when the node points to many authoritative nodes.
      In a diffusion network a high-hub user propagates influence toward
      authoritative taste-makers; hubs are *aggregators* of influence.
    * **Authority score** — high when the node is pointed to by many high-hub
      nodes.  In a diffusion network a high-authority user is a *canonical
      taste-maker* whose preferences many others follow.

    Returns:
        hub_scores:   Dict mapping node_id → hub score (H).
        auth_scores:  Dict mapping node_id → authority score (A).
    """
    hub_hash = snap.TIntFltH()  # type: ignore[attr-defined]
    auth_hash = snap.TIntFltH()  # type: ignore[attr-defined]
    snap.GetHits(G, hub_hash, auth_hash)  # type: ignore[attr-defined]
    return _snap_hash_to_dict(hub_hash), _snap_hash_to_dict(auth_hash)


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
    """Eigenvector centrality via ``snap.GetEigenVectorCentr``.

    ``GetEigenVectorCentr`` requires an undirected graph (``PUNGraph``), so
    the directed NetInf graph is symmetrised first.  Node IDs are preserved;
    edge directions are dropped.
    """
    ug = snap.ConvertGraph(snap.PUNGraph, G)  # type: ignore[attr-defined]
    eig_hash = snap.TIntFltH()  # type: ignore[attr-defined]
    snap.GetEigenVectorCentr(ug, eig_hash)  # type: ignore[attr-defined]
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
# Custom PageRank with LPH-weighted intrinsic centralities
# ---------------------------------------------------------------------------


def pagerank_custom_beta(
    G: nx.DiGraph,
    alpha: float,
    beta: dict[int, float],
) -> dict[int, float]:
    """
    Generalised PageRank with per-node intrinsic centrality vector β.

    Solves the linear system derived from Newman Eq. 7.18:

        x_i = α Σ_j A_{ij} (x_j / k_j^out) + β_i

    which in matrix form is ``(D - α A) x = β``, where *D* is the diagonal
    out-degree matrix.  The caller is responsible for providing a *beta* dict
    that already satisfies ``Σ β_i = 1 - α`` (use
    :func:`compute_pagerank_lph` to get the correctly normalised vector).

    After solving, any negative entries (numerical artefacts) are clipped to
    zero and the result is re-normalised to sum to 1.

    Args:
        G:      Directed NetworkX graph (``nx.DiGraph``), as returned by
                :func:`networks.network_io.load_as_networkx`.  ``A.sum(axis=1)``
                gives the out-degree of each node, consistent with the Newman
                Eq. 7.18 formulation.
        alpha:  Damping factor (same role as in standard PageRank).
        beta:   Mapping node_id → intrinsic centrality value.  Nodes missing
                from the dict receive ``(1 − alpha) / n`` as default.

    Returns:
        Dict mapping node_id → PageRank score.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=float)

    out_degrees = np.array(A.sum(axis=1)).flatten()
    out_degrees[out_degrees == 0] = 1.0  # dangling nodes: avoid division by zero
    D = sp.diags(out_degrees, format="csr")
    M = D - alpha * A

    default_beta = (1.0 - alpha) / n
    beta_vec = np.array([beta.get(node, default_beta) for node in nodes])

    x = np.asarray(spla.spsolve(M, beta_vec), dtype=float)
    x = np.clip(x, 0.0, None)
    total = x.sum()
    if total > 0.0:
        x = x / total

    return {node: float(x[idx[node]]) for node in nodes}


def compute_pagerank_lph(
    network_file: str | Path,
    model_name: str,
    network_id: str,
    alpha: float = 0.85,
    communities_dir: Path | None = None,
) -> dict[int, float] | None:
    """
    Compute LPH-weighted custom PageRank for a single network.

    Loads the corresponding community CSV to retrieve ``lph_score`` (h̃v,
    Barraza et al. 2025), applies a softmax transform to map the real-valued
    scores to a valid intrinsic-centrality vector β with ``Σ β_i = 1 − α``,
    then calls :func:`pagerank_custom_beta`.

    Using softmax rather than a linear shift avoids making β_i = 0 dependent
    on the most extreme outlier in each network, and is the maximum-entropy
    transformation that preserves the relative ordering of h̃v scores.

    Returns ``None`` if the community CSV is missing (falls back gracefully so
    the standard centrality metrics are still saved without the extra column).

    Args:
        network_file:    Path to the NetInf ``.txt`` network file.
        model_name:      Diffusion model name (exponential / powerlaw / rayleigh).
        network_id:      Zero-padded index string (e.g. ``"007"``).
        alpha:           PageRank damping factor.
        communities_dir: Root communities directory.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).COMMUNITIES``.

    Returns:
        Dict mapping node_id → pagerank_lph score, or ``None``.
    """
    root = communities_dir or DatasetPaths(Datasets.DEFAULT).COMMUNITIES
    community_csv = root / model_name / f"communities_{model_name}_{network_id}.csv"
    if not community_csv.exists():
        return None

    com_df = pd.read_csv(community_csv)
    if "lph_score" not in com_df.columns:
        return None

    # Use h̃v (lph_score, Barraza et al. 2025) as the intrinsic centrality
    # vector β.  h̃v can be negative, so a linear shift would make β=0 depend
    # on the most extreme outlier in each network.  Instead, use softmax:
    #
    #   β_i ∝ exp(h̃v_i)
    #
    # which maps any real-valued score to a strictly positive probability
    # vector without introducing an arbitrary zero, and preserves the relative
    # ordering (more homophilic nodes receive higher intrinsic weight).
    # Centring by max(h̃v) before exponentiating avoids numerical overflow.
    lph_series = com_df.set_index("UserId")["lph_score"]
    lph_shifted = lph_series - lph_series.max()  # numerical stability
    exp_lph = np.exp(lph_shifted.values.astype(float))
    exp_sum = exp_lph.sum()
    if exp_sum == 0.0:
        return None

    # Normalise: Σ β_i = 1 − α
    beta: dict[int, float] = {
        int(uid): float(w) * (1.0 - alpha) / exp_sum  # type: ignore[arg-type]
        for uid, w in zip(lph_series.index, exp_lph)
    }

    G_nx, _ = load_as_networkx(network_file)
    return pagerank_custom_beta(G_nx, alpha, beta)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-network and batch runners
# ---------------------------------------------------------------------------


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
        dp = DatasetPaths(Datasets.DEFAULT)
        model_dir = dp.NETWORKS / args.model
        if not model_dir.exists():
            print(f"Error: directory not found: {model_dir}")
            sys.exit(1)
        network_files = sorted(model_dir.glob("inferred-network-*.txt"))
        success_count = sum(
            1
            for nf in network_files
            if calculate_centrality_for_network(
                nf,
                communities_dir=dp.COMMUNITIES,
                centrality_dir=dp.CENTRALITY,
            )
        )
        print(f"\nProcessed: {success_count}/{len(network_files)}")
        sys.exit(0)


if __name__ == "__main__":
    main()
