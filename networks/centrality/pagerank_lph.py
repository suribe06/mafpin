"""
LPH-weighted custom PageRank (Newman Eq. 7.18).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from config import DatasetPaths, Datasets
from networks.network_io import load_as_networkx


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
