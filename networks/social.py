"""
Explicit trust-graph loader and alignment analysis for Ciao and Epinions.

The Ciao and Epinions datasets provide an explicit user–trust graph (who
trusts whom) in addition to the rating data used to infer the implicit
influence network.  This module loads those trust graphs and computes:

1. **Trust-graph centrality features** — in-degree (number of followers),
   out-degree (number of followees), and PageRank on the trust graph.
2. **Graph-overlap score** — Jaccard similarity between a user's trust
   neighbourhood and their inferred-network neighbourhood.
3. **Alignment ratio** — fraction of inferred edges that also appear in
   the trust graph, measured per inferred network.

These features can be merged into the CMF user-attribute matrix via
:func:`recommender.enhanced.load_network_features`.

Usage::

    from networks.social import load_trust_graph, compute_trust_features
    G_trust = load_trust_graph("ciao")
    df = compute_trust_features(G_trust)
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from config import Datasets

# ---------------------------------------------------------------------------
# Trust-graph datasets
# ---------------------------------------------------------------------------

# Datasets that ship with an explicit trust graph.
_TRUST_DATASETS = {"ciao", "epinions"}

# Trust-file format: two whitespace-separated columns (truster, trustee).
# Values are stored as scientific notation floats (e.g. 1.0000000e+000).
_TRUST_FILE = "trust.txt"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_trust_graph(dataset: str) -> nx.DiGraph:
    """
    Load the explicit user trust graph for *dataset*.

    The trust file contains rows ``truster  trustee`` in scientific-notation
    float format.  Each row is read as a directed edge truster → trustee
    (i.e. "truster follows/trusts trustee").

    Args:
        dataset: Dataset name — must be one of ``{"ciao", "epinions"}``.

    Returns:
        Directed NetworkX graph with integer node IDs.

    Raises:
        ValueError: If *dataset* has no trust graph.
        FileNotFoundError: If the trust file is missing from ``datasets/``.
    """
    if dataset not in _TRUST_DATASETS:
        raise ValueError(
            f"Dataset '{dataset}' has no explicit trust graph. "
            f"Supported: {sorted(_TRUST_DATASETS)}"
        )

    trust_file = Datasets.ROOT / dataset / _TRUST_FILE
    if not trust_file.exists():
        raise FileNotFoundError(
            f"Trust file not found: {trust_file}\n"
            "Check that the dataset directory is present under datasets/."
        )

    edges_df = pd.read_csv(trust_file, sep=r"\s+", header=None, names=["src", "dst"])
    # Convert scientific-notation floats to int node IDs
    edges_df["src"] = edges_df["src"].apply(lambda x: int(round(float(x))))
    edges_df["dst"] = edges_df["dst"].apply(lambda x: int(round(float(x))))

    G = nx.DiGraph()
    G.add_edges_from(zip(edges_df["src"], edges_df["dst"]))
    return G


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_trust_features(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute per-user centrality features derived from the trust graph.

    Computed features:

    * ``trust_in_degree``  — number of users who trust this user (followers).
    * ``trust_out_degree`` — number of users this user trusts (followees).
    * ``trust_pagerank``   — PageRank score on the trust graph.

    Args:
        G: Directed trust graph returned by :func:`load_trust_graph`.

    Returns:
        DataFrame indexed by ``UserId`` with the three feature columns.
    """
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    pr = nx.pagerank(G, alpha=0.85)

    nodes = sorted(G.nodes())
    df = pd.DataFrame(
        {
            "UserId": nodes,
            "trust_in_degree": [in_deg.get(n, 0) for n in nodes],
            "trust_out_degree": [out_deg.get(n, 0) for n in nodes],
            "trust_pagerank": [pr.get(n, 0.0) for n in nodes],
        }
    ).set_index("UserId")
    return df


def compute_neighbourhood_overlap(
    G_trust: nx.DiGraph,
    G_inferred: nx.DiGraph,
) -> pd.DataFrame:
    """
    Compute Jaccard similarity between each user's trust and inferred
    neighbourhood.

    For each user present in both graphs the Jaccard coefficient is:

    .. math::

        J(u) = \\frac{|N_{\\text{trust}}(u) \\cap N_{\\text{inferred}}(u)|}
                     {|N_{\\text{trust}}(u) \\cup N_{\\text{inferred}}(u)|}

    where :math:`N(u)` is the set of *successors* (out-neighbours) of *u*.

    Args:
        G_trust:    Directed trust graph.
        G_inferred: Directed inferred-influence network.

    Returns:
        DataFrame indexed by ``UserId`` with column ``jaccard_overlap``
        (float in [0, 1], NaN when both neighbourhoods are empty).
    """
    common_nodes = sorted(set(G_trust.nodes()) & set(G_inferred.nodes()))
    records = []
    for u in common_nodes:
        trust_nb = set(G_trust.successors(u))
        inferred_nb = set(G_inferred.successors(u))
        union = trust_nb | inferred_nb
        if not union:
            jaccard = float("nan")
        else:
            jaccard = len(trust_nb & inferred_nb) / len(union)
        records.append({"UserId": u, "jaccard_overlap": jaccard})

    return pd.DataFrame(records).set_index("UserId")


def compute_alignment_ratio(
    G_trust: nx.DiGraph,
    G_inferred: nx.DiGraph,
) -> float:
    """
    Compute the global alignment ratio between the inferred and trust graphs.

    The alignment ratio is the fraction of inferred edges that also appear
    in the trust graph:

    .. math::

        \\text{alignment} = \\frac{|E_{\\text{inferred}} \\cap E_{\\text{trust}}|}
                                  {|E_{\\text{inferred}}|}

    A high alignment ratio indicates that NetInf has recovered many explicit
    social-trust relationships, supporting the claim that co-rating cascades
    carry social influence signal.  A low ratio suggests the inferred network
    captures latent taste similarity rather than stated trust.

    Args:
        G_trust:    Directed trust graph.
        G_inferred: Directed inferred-influence network.

    Returns:
        Float in [0, 1].  Returns ``float("nan")`` when *G_inferred* has no
        edges.
    """
    inferred_edges = set(G_inferred.edges())
    if not inferred_edges:
        return float("nan")
    trust_edges = set(G_trust.edges())
    return len(inferred_edges & trust_edges) / len(inferred_edges)


# ---------------------------------------------------------------------------
# Convenience: load features + overlap for one (dataset, model, network) pair
# ---------------------------------------------------------------------------


def load_social_features(
    dataset: str,
    G_inferred: nx.DiGraph,
) -> pd.DataFrame | None:
    """
    Load trust-graph features and neighbourhood overlap for a single inferred
    network.

    Returns a DataFrame indexed by ``UserId`` with columns:

    * ``trust_in_degree``
    * ``trust_out_degree``
    * ``trust_pagerank``
    * ``jaccard_overlap``

    Missing values (users not in the trust graph) are filled with 0.

    Args:
        dataset:    Dataset name (must support a trust graph).
        G_inferred: The inferred influence network for one model/network pair.

    Returns:
        Feature DataFrame, or ``None`` if the dataset has no trust graph or
        the trust file is missing.
    """
    if dataset not in _TRUST_DATASETS:
        return None

    try:
        G_trust = load_trust_graph(dataset)
    except FileNotFoundError as exc:
        print(f"  Warning: {exc}")
        return None

    centrality_df = compute_trust_features(G_trust)
    overlap_df = compute_neighbourhood_overlap(G_trust, G_inferred)

    combined = centrality_df.join(overlap_df, how="outer").fillna(0.0)
    return combined
