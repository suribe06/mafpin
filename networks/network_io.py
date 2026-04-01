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
    Load a NetInf network file as an undirected SNAP graph.

    Args:
        network_file: Path to the ``.txt`` network file.

    Returns:
        G:        Undirected SNAP graph with compacted node IDs.
        user_ids: Ordered list of node IDs present in the graph.
    """
    nodes, edges = parse_network_file(network_file)
    mapper = _build_mapper(nodes)

    G = snap.TUNGraph.New()  # type: ignore[attr-defined]
    user_ids = list(mapper.values())
    # SNAP TUNGraph requires node IDs >= 0; 0-based mapper satisfies this.
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
) -> tuple[nx.Graph, list[int]]:
    """
    Load a NetInf network file as an undirected NetworkX graph.

    Args:
        network_file: Path to the ``.txt`` network file.

    Returns:
        G:        Undirected NetworkX graph with compacted node IDs.
        user_ids: Ordered list of node IDs present in the graph.
    """
    nodes, edges = parse_network_file(network_file)
    mapper = _build_mapper(nodes)

    G = nx.Graph()
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
