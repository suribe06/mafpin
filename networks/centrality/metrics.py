"""
SNAP-based node-level centrality metric functions.
"""

from __future__ import annotations

try:
    from snap import snap  # type: ignore[import-untyped]
except ImportError as _snap_err:
    raise ImportError(
        "snap-stanford is required: pip install snap-stanford==6.0.0"
    ) from _snap_err


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
