"""
Unified loader for NetInf output network files.

NetInf encodes the inferred graph in a plain-text file where:
  - Self-loop lines  ``i,i``  declare that node *i* exists in the network.
  - Non-loop lines   ``i,j``  (i ≠ j) declare a directed/undirected edge.

Both ``calculate_centrality_metrics.py`` and ``calculate_communities.py``
previously contained identical parsing logic paired with different graph
backends (SNAP and NetworkX respectively).  This module is the single source
of truth: parse once, build the backend-specific graph on top.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from snap import snap

from config import Paths


# ---------------------------------------------------------------------------
# Low-level parser (backend-agnostic)
# ---------------------------------------------------------------------------


def parse_network_file(
    network_file: str | Path,
) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Parse a NetInf ``.txt`` network file into raw node and edge lists.

    Args:
        network_file: Path to the network file.

    Returns:
        nodes: Original node IDs (from self-loop lines).
        edges: List of ``(src, dst)`` pairs (from non-loop lines).

    Raises:
        FileNotFoundError: If *network_file* does not exist.
    """
    network_file = Path(network_file)
    if not network_file.exists():
        raise FileNotFoundError(f"Network file not found: {network_file}")

    nodes: list[int] = []
    edges: list[tuple[int, int]] = []

    with open(network_file, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip().rstrip("\r")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            i, j = int(parts[0]), int(parts[1])
            if i == j:
                nodes.append(i)
            else:
                edges.append((i, j))

    return nodes, edges


def _build_mapper(nodes: list[int]) -> dict[int, int]:
    """Return a mapping from original IDs to compact 0-based integers."""
    return {old: new for new, old in enumerate(sorted(set(nodes)), start=0)}


# ---------------------------------------------------------------------------
# SNAP backend  (used by centrality metrics)
# ---------------------------------------------------------------------------


def load_as_snap(
    network_file: str | Path,
) -> tuple[object, list[int]]:
    """
    Load a NetInf network file as a **directed** SNAP graph.

    NetInf infers directed influence edges (u → v means u tends to
    influence v).  Using a directed graph (``snap.TNGraph``) preserves
    the semantic meaning of PageRank, betweenness, and in-degree metrics.

    Args:
        network_file: Path to the ``.txt`` network file.

    Returns:
        G:        Directed SNAP graph with compacted node IDs.
        user_ids: Ordered list of node IDs present in the graph.
    """
    nodes, edges = parse_network_file(network_file)
    mapper = _build_mapper(nodes)

    G = snap.TNGraph.New()  # type: ignore[attr-defined]  # directed graph
    user_ids = list(mapper.values())
    for u in user_ids:
        G.AddNode(u)
    for i, j in edges:
        if i in mapper and j in mapper:
            G.AddEdge(mapper[i], mapper[j])

    return G, user_ids


# ---------------------------------------------------------------------------
# NetworkX backend  (used by community detection)
# ---------------------------------------------------------------------------


def load_as_networkx(
    network_file: str | Path,
) -> tuple[nx.DiGraph, list[int]]:
    """
    Load a NetInf network file as a **directed** NetworkX graph.

    NetInf infers directed influence edges (u → v means u tends to
    influence v).  Using ``nx.DiGraph`` preserves edge directionality for
    PageRank, betweenness, and community algorithms that support directed
    graphs.

    Args:
        network_file: Path to the ``.txt`` network file.

    Returns:
        G:        Directed NetworkX graph with compacted node IDs.
        user_ids: Ordered list of node IDs present in the graph.
    """
    nodes, edges = parse_network_file(network_file)
    mapper = _build_mapper(nodes)

    G = nx.DiGraph()  # directed graph
    user_ids = list(mapper.values())
    G.add_nodes_from(user_ids)
    for i, j in edges:
        if i in mapper and j in mapper:
            G.add_edge(mapper[i], mapper[j])

    return G, user_ids


# ---------------------------------------------------------------------------
# Filename metadata helper
# ---------------------------------------------------------------------------


def parse_network_filename(filename: str) -> tuple[str, str]:
    """
    Extract model name and zero-padded network ID from a NetInf output filename.

    Expected format: ``inferred-network-<short>-<NNN>.txt``
    where ``<short>`` is one of ``expo``, ``power``, or ``ray``.

    Args:
        filename: Basename of the network file (with or without ``.txt``).

    Returns:
        model_name: Full model name (e.g. ``"exponential"``).
        network_id: Zero-padded numeric string (e.g. ``"007"``).

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    from config import Models

    stem = Path(filename).stem  # strip .txt
    parts = stem.split("-")
    if len(parts) < 4:
        raise ValueError(
            f"Cannot parse network filename '{filename}'. "
            "Expected format: inferred-network-<short>-<NNN>"
        )
    model_short = parts[2]
    network_id = parts[3]

    model_name = Models.FROM_SHORT.get(model_short, model_short)
    return model_name, network_id
