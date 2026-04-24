"""
Overlapping community detection algorithms (cdlib wrappers).
"""

from __future__ import annotations

import networkx as nx
from cdlib import algorithms  # type: ignore[import-untyped]

from config import Defaults


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
