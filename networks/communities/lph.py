"""
Local Pluralistic Homophily (LPH) computation.

Implements both the Jaccard-based LPH and the paper metric h̃v
(Barraza et al. 2025).
"""

from __future__ import annotations

import networkx as nx


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
        membership: Dict returned by :func:`~networks.communities.detection.compute_node_community_membership`.

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
) -> tuple[
    dict[int, float],  # h̃v scores
    dict[int, int],  # s(v) neighborhood alignment
    dict[int, float],  # δv local dissimilarity
]:
    """
    Compute the normalized Local Pluralistic Homophily score h̃v and its
    intermediate quantities for every node (Algorithm 1, Barraza et al. 2025).

    Steps:

    1. s(v) = |⋃_{i∈N(v)} (C(v) ∩ C(i))|  — neighborhood alignment (Eq. 3).
    2. h = network-level Pearson assortativity using s(·) (Eq. 1).
    3. δv = (1/dv) Σ_{i∈N(v)} |s(v) − s(i)|  — local dissimilarity (Eq. 4, 0 if isolated).
    4. λ = (h + Σu δu) / N  (Eq. 5).
    5. h̃v = λ − δv  (Eq. 6).

    Nodes with strongly negative h̃v are boundary-spanning candidates: their
    community profile diverges from those of their neighbours.
    Sum property: Σv h̃v = h (global-local consistency).

    Args:
        G:          Undirected NetworkX graph.
        membership: Dict returned by :func:`~networks.communities.detection.compute_node_community_membership`.

    Returns:
        A 3-tuple:
        * ``lph_scores`` — Dict mapping node_id → h̃v (normalized boundary score).
        * ``s_values``   — Dict mapping node_id → s(v) (neighborhood alignment count).
        * ``delta_values`` — Dict mapping node_id → δv (local dissimilarity).
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

    lph_scores = {node: lam - delta[node] for node in G.nodes()}
    return lph_scores, s, delta
