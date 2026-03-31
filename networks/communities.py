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

Outputs saved in ``data/communities/<model>/``:

* ``communities_<model>_<index>.csv``
  Columns: ``UserId, num_communities, community_ids, local_pluralistic_hom``

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

from config import Paths, Models, Defaults
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
# LPH computation
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
# File I/O
# ---------------------------------------------------------------------------


def save_community_results(
    user_ids: list[int],
    lph: dict[int, float],
    membership: dict[int, set[int]],
    model_name: str,
    network_id: str,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Write community and LPH results to CSV.

    Args:
        user_ids:    Ordered list of node identifiers.
        lph:         Dict of LPH scores per node.
        membership:  Dict of community-index sets per node.
        model_name:  Diffusion model name.
        network_id:  Zero-padded index string (e.g. ``"007"``).
        output_dir:  Target directory.  Defaults to
            ``Paths.COMMUNITIES / model_name``.

    Returns:
        Path of the written CSV file.
    """
    if output_dir is None:
        output_dir = Paths.COMMUNITIES / model_name
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

    try:
        communities = detect_overlapping_communities(
            G, algorithm=algorithm, epsilon=epsilon, min_community=min_community
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  Warning: community detection failed: {exc}")
        communities = []

    membership = compute_node_community_membership(user_ids, communities)
    lph = compute_local_pluralistic_homophily(G, membership)

    save_community_results(user_ids, lph, membership, model_name, network_id)
    return True


def calculate_communities_for_all_models(
    algorithm: str = "demon",
    epsilon: float = Defaults.EPSILON,
    min_community: int = Defaults.MIN_COM,
) -> dict[str, int]:
    """
    Detect communities and compute LPH for all inferred networks across all models.

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

        success_count = sum(
            1
            for nf in network_files
            if calculate_communities_for_network(
                nf, algorithm=algorithm, epsilon=epsilon, min_community=min_community
            )
        )
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
        model_dir = Paths.NETWORKS / args.model
        if not model_dir.exists():
            print(f"Error: directory not found: {model_dir}")
            sys.exit(1)
        network_files = sorted(model_dir.glob("inferred-network-*.txt"))
        success_count = sum(
            1 for nf in network_files if calculate_communities_for_network(nf, **kwargs)
        )
        print(f"\nProcessed: {success_count}/{len(network_files)}")
        sys.exit(0)


if __name__ == "__main__":
    main()
