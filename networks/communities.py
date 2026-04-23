"""
Community detection and Local Pluralistic Homophily (LPH) computation.

For each inferred network, overlapping communities are detected using one of
two algorithms from cdlib, then the *Local Pluralistic Homophily* (LPH) score
is computed for every node and saved to CSV.

Community detection algorithms
-------------------------------
Demon
    Local community detection that uses an ego-network expansion strategy
    (``cdlib.algorithms.demon``).
ASLPAw
    Adaptive Label Propagation Algorithm for overlapping community finding
    (``cdlib.algorithms.aslpaw``).

Local Pluralistic Homophily (LPH)
----------------------------------
For a node *v* with community set *C(v)* and neighbour set *N(v)*:

    LPH(v) = mean_{u ∈ N(v)} Jaccard(C(v), C(u))

where ``Jaccard(A, B) = |A ∩ B| / |A ∪ B|``.

A high LPH value indicates that *v* and its direct neighbours tend to share
community memberships — a measure of neighbourhood cohesion in overlapping
structures.

Outputs saved in ``data/<dataset>/communities/<model>/``:

* ``communities_<model>_<index>.csv``
  Columns: ``UserId, num_communities, community_ids, local_pluralistic_hom, lph_score``

  - ``local_pluralistic_hom``: mean Jaccard similarity with neighbours ∈ [0, 1].
  - ``lph_score``: normalized h̃v boundary score (Barraza et al. 2025); most
    negative values indicate boundary-spanning positions.

Usage (CLI)::

    python -m networks.communities --model exponential
    python -m networks.communities --all
    python -m networks.communities --network path/to/network.txt
    python -m networks.communities --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
from cdlib import algorithms  # type: ignore[import-untyped]

from config import DatasetPaths, Datasets, Models, Defaults
from networks.network_io import load_as_networkx, parse_network_filename


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------


def detect_overlapping_communities(
    G: nx.Graph,
    algorithm: str = "demon",
    epsilon: float = Defaults.EPSILON,
    min_community: int = Defaults.MIN_COM,
) -> list[list[int]]:
    """
    Detect overlapping communities in *G*.

    Args:
        G:             An undirected NetworkX graph.
        algorithm:     ``"demon"`` or ``"aslpaw"``.
        epsilon:       Merging threshold for Demon.
        min_community: Minimum community size for Demon.

    Returns:
        List of communities, each community being a list of node IDs.

    Raises:
        ValueError: If *algorithm* is not ``"demon"`` or ``"aslpaw"``.
    """
    if algorithm == "demon":
        result = algorithms.demon(G, epsilon=epsilon, min_com_size=min_community)
    elif algorithm == "aslpaw":
        result = algorithms.aslpaw(G)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Use 'demon' or 'aslpaw'.")

    return [list(community) for community in result.communities]


def compute_node_community_membership(
    node_ids: list[int],
    communities: list[list[int]],
) -> dict[int, set[int]]:
    """
    Map each node to the set of community indices it belongs to.

    Args:
        node_ids:    All node IDs (used to guarantee every node appears).
        communities: List of community node lists.

    Returns:
        Dict mapping node_id → set of zero-based community indices.
    """
    membership: dict[int, set[int]] = {nid: set() for nid in node_ids}
    for com_idx, community in enumerate(communities):
        for node in community:
            if node in membership:
                membership[node].add(com_idx)
    return membership


# ---------------------------------------------------------------------------
# LPH computation — Jaccard-based
# ---------------------------------------------------------------------------


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets.  Returns 0.0 when both are empty."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_local_pluralistic_homophily(
    G: nx.Graph,
    membership: dict[int, set[int]],
) -> dict[int, float]:
    """
    Compute LPH for every node in *G*.

    LPH(v) = mean_{u ∈ N(v)} Jaccard(C(v), C(u))

    A node with no neighbours receives LPH = 0.0.

    Args:
        G:          Undirected NetworkX graph.
        membership: Dict returned by :func:`compute_node_community_membership`.

    Returns:
        Dict mapping node_id → LPH score.
    """
    lph: dict[int, float] = {}
    for node in G.nodes():
        neighbours = list(G.neighbors(node))
        if not neighbours:
            lph[node] = 0.0
            continue
        jaccard_values = [
            _jaccard(membership.get(node, set()), membership.get(nb, set()))
            for nb in neighbours
        ]
        lph[node] = sum(jaccard_values) / len(jaccard_values)
    return lph


# ---------------------------------------------------------------------------
# LPH computation — Paper metric (h̃v, Barraza et al. 2025)
# ---------------------------------------------------------------------------


def _compute_neighborhood_alignment(
    G: nx.Graph,
    membership: dict[int, set[int]],
) -> dict[int, int]:
    """
    Compute s(v) = |⋃_{i∈N(v)} (C(v) ∩ C(i))| for every node.

    s(v) counts how many of v's own communities are represented at least once
    among its neighbours (Eq. 3 of Barraza et al. 2025).

    Args:
        G:          Undirected NetworkX graph.
        membership: Dict mapping node_id → set of community indices.

    Returns:
        Dict mapping node_id → integer s value.
    """
    s: dict[int, int] = {}
    for node in G.nodes():
        c_v = membership.get(node, set())
        s_neigh: set[int] = set()
        for nb in G.neighbors(node):
            s_neigh |= c_v & membership.get(nb, set())
        s[node] = len(s_neigh)
    return s


def _compute_network_homophily(G: nx.Graph, s: dict[int, int]) -> float:
    """
    Compute the network-level pluralistic homophily coefficient h.

    Uses the edge-based Pearson form (equivalent form, SI Eq. 12):

        h = Σ_{(i,j)∈E} (s(i) − µq)(s(j) − µq)
            ─────────────────────────────────────
            Σ_{(i,j)∈E} (s(i) − µq)²

    where µq = (1/2M) Σ_{(i,j)∈E} (s(i) + s(j)).

    Returns 0.0 when the graph has no edges or zero variance.
    """
    edges = list(G.edges())
    if not edges:
        return 0.0

    M = len(edges)
    mu_q = sum(s.get(u, 0) + s.get(v, 0) for u, v in edges) / (2 * M)
    numerator = sum((s.get(u, 0) - mu_q) * (s.get(v, 0) - mu_q) for u, v in edges)
    denominator = sum((s.get(u, 0) - mu_q) ** 2 for u, v in edges)

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_lph_paper(
    G: nx.Graph,
    membership: dict[int, set[int]],
) -> dict[int, float]:
    """
    Compute the normalized Local Pluralistic Homophily score h̃v for every node
    (Algorithm 1, Barraza et al. 2025).

    Steps:

    1. s(v) = |⋃_{i∈N(v)} (C(v) ∩ C(i))|  — neighborhood alignment.
    2. h = network-level Pearson assortativity using s(·).
    3. δv = (1/dv) Σ_{i∈N(v)} |s(v) − s(i)|  — local dissimilarity (0 if isolated).
    4. λ = (h + Σu δu) / N
    5. h̃v = λ − δv

    Nodes with strongly negative h̃v are boundary-spanning candidates: their
    community profile diverges from those of their neighbours.
    Sum property: Σv h̃v = h (global-local consistency).

    Args:
        G:          Undirected NetworkX graph.
        membership: Dict returned by :func:`compute_node_community_membership`.

    Returns:
        Dict mapping node_id → h̃v score.
    """
    s = _compute_neighborhood_alignment(G, membership)
    h = _compute_network_homophily(G, s)

    delta: dict[int, float] = {}
    for node in G.nodes():
        neighbours = list(G.neighbors(node))
        dv = len(neighbours)
        if dv == 0:
            delta[node] = 0.0
        else:
            delta[node] = (
                sum(abs(s.get(node, 0) - s.get(nb, 0)) for nb in neighbours) / dv
            )

    N = G.number_of_nodes()
    lam = (h + sum(delta.values())) / N if N > 0 else 0.0

    return {node: lam - delta[node] for node in G.nodes()}


def save_community_results(
    user_ids: list[int],
    lph: dict[int, float],
    lph_paper: dict[int, float],
    membership: dict[int, set[int]],
    model_name: str,
    network_id: str,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Write community and LPH results to CSV.

    Args:
        user_ids:    Ordered list of node identifiers.
        lph:         Dict of Jaccard-based LPH scores per node.
        lph_paper:   Dict of normalized h̃v scores per node (Barraza et al. 2025).
        membership:  Dict of community-index sets per node.
        model_name:  Diffusion model name.
        network_id:  Zero-padded index string (e.g. ``"007"``).
        output_dir:  Target directory.  Defaults to
            ``DatasetPaths(Datasets.DEFAULT).COMMUNITIES / model_name``.

    Returns:
        Path of the written CSV file.
    """
    if output_dir is None:
        output_dir = DatasetPaths(Datasets.DEFAULT).COMMUNITIES / model_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for uid in user_ids:
        coms = sorted(membership.get(uid, set()))
        records.append(
            {
                "UserId": uid,
                "num_communities": len(coms),
                "community_ids": ";".join(map(str, coms)),
                "local_pluralistic_hom": lph.get(uid, 0.0),
                "lph_score": lph_paper.get(uid, 0.0),
            }
        )

    df = pd.DataFrame(records)
    out_file = output_dir / f"communities_{model_name}_{network_id}.csv"
    df.to_csv(out_file, index=False)
    print(f"  Saved: {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Per-network and batch runners
# ---------------------------------------------------------------------------


def calculate_communities_for_network(
    network_file: str | Path,
    algorithm: str = "demon",
    epsilon: float = Defaults.EPSILON,
    min_community: int = Defaults.MIN_COM,
    output_dir: str | Path | None = None,
) -> bool:
    """
    Detect communities and compute LPH for a single network file.

    Args:
        network_file:  Path to a NetInf output ``.txt`` file.
        algorithm:     Community detection algorithm (``"demon"`` or ``"aslpaw"``).
        epsilon:       Merging threshold for Demon.
        min_community: Minimum community size for Demon.

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

    G, user_ids = load_as_networkx(network_file)

    # Guard: skip degenerate (edgeless) networks — community detection on an
    # edgeless graph produces meaningless results and all-zero LPH scores.
    if G.number_of_edges() == 0:
        print(f"  Warning: no edges in {network_file.name}, skipping.")
        return False

    try:
        communities = detect_overlapping_communities(
            G, algorithm=algorithm, epsilon=epsilon, min_community=min_community
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  Warning: community detection failed: {exc}")
        communities = []

    membership = compute_node_community_membership(user_ids, communities)
    lph = compute_local_pluralistic_homophily(G, membership)
    lph_paper = compute_lph_paper(G, membership)

    save_community_results(
        user_ids,
        lph,
        lph_paper,
        membership,
        model_name,
        network_id,
        output_dir=output_dir,
    )
    return True


def calculate_communities_for_all_models(
    algorithm: str = "demon",
    epsilon: float = Defaults.EPSILON,
    min_community: int = Defaults.MIN_COM,
    dataset: str | None = None,
) -> dict[str, int]:
    """
    Detect communities and compute LPH for all inferred networks across all models.

    Args:
        algorithm:     Community detection algorithm.
        epsilon:       Demon merge threshold.
        min_community: Minimum community size for Demon.
        dataset:       Dataset name.  Defaults to ``Datasets.DEFAULT``.  Used
            to locate inferred networks and write community outputs into the
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
            desc=f"{model_name[:4].upper()} communities",
            unit="net",
            dynamic_ncols=True,
        )
        for nf in pbar:
            pbar.set_postfix(file=nf.stem[-6:])
            if calculate_communities_for_network(
                nf,
                algorithm=algorithm,
                epsilon=epsilon,
                min_community=min_community,
                output_dir=dp.COMMUNITIES / model_name,
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
        description="Detect communities and compute LPH for inferred networks.",
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
    parser.add_argument(
        "--algorithm",
        choices=["demon", "aslpaw"],
        default="demon",
        help="Community detection algorithm.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=Defaults.EPSILON,
        help="Merging threshold for Demon.",
    )
    parser.add_argument(
        "--min-community",
        type=int,
        default=Defaults.MIN_COM,
        help="Minimum community size for Demon.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    kwargs = {
        "algorithm": args.algorithm,
        "epsilon": args.epsilon,
        "min_community": args.min_community,
    }

    if args.network:
        success = calculate_communities_for_network(args.network, **kwargs)
        sys.exit(0 if success else 1)

    elif args.all:
        summary = calculate_communities_for_all_models(**kwargs)
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
            if calculate_communities_for_network(
                nf,
                **kwargs,
                output_dir=dp.COMMUNITIES / args.model,
            )
        )
        print(f"\nProcessed: {success_count}/{len(network_files)}")
        sys.exit(0)


if __name__ == "__main__":
    main()
