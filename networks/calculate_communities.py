"""
A module for detecting overlapping communities in inferred networks and computing
Local Pluralistic Homophily (LPH) for each node.

Overlapping community detection is performed using the Demon algorithm (cdlib),
which is well-suited for sparse social networks. ASLPAw is provided as a fast
alternative. For each node the following are saved:

    - community_ids          : semicolon-separated list of community IDs the node belongs to
    - num_communities        : number of communities the node belongs to
    - local_pluralistic_hom  : mean Jaccard similarity of community sets with each neighbor

Local Pluralistic Homophily (LPH) for node v:

    LPH(v) = (1 / |N(v)|) * sum_{u in N(v)} J(C(v), C(u))

where J(A, B) = |A ∩ B| / |A ∪ B| (Jaccard similarity of community-membership sets).

Results are saved to:
    ../data/communities/<model_name>/communities_<model_name>_<network_id>.csv

Directory structure mirrors the centrality_metrics layout so that cmf_centrality.py
can load both files and merge them on UserId.

Dependencies:
    - os
    - glob
    - pandas
    - networkx
    - cdlib
    - centrality_metrics (local, only for the network loader helper)
"""

import os
import glob
import pandas as pd
import networkx as nx

# cdlib is required for Demon and ASLPAw
try:
    from cdlib import algorithms as cdlib_alg  # type: ignore[import-untyped]

    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False
    print("Warning: cdlib not installed. Run: pip install cdlib")


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def create_output_directory(base_path, model_name):
    """Create and return the model-specific output directory."""
    model_dir = os.path.join(base_path, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


# ---------------------------------------------------------------------------
# Network loading
# ---------------------------------------------------------------------------


def load_inferred_network_nx(network_file):
    """
    Load an inferred network file and return a NetworkX undirected graph.

    The file format encodes node declarations as self-loops (i,i) and edges
    as (i,j) with i != j, using comma as delimiter and Windows line endings.

    Args:
        network_file (str): Path to the .txt network file.

    Returns:
        tuple: (nx.Graph, list of original user IDs) or (None, []) on error.
    """
    if not os.path.exists(network_file):
        print(f"Error: Network file '{network_file}' not found.")
        return None, []

    try:
        nodes = []
        edges = []
        with open(network_file, "r", encoding="utf-8") as fh:
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

        # Map original IDs to compact 1-based integers (mirrors centrality module)
        mapper = {old: new for new, old in enumerate(sorted(set(nodes)), start=1)}

        G = nx.Graph()
        user_ids = list(mapper.values())
        G.add_nodes_from(user_ids)
        for i, j in edges:
            if i in mapper and j in mapper:
                G.add_edge(mapper[i], mapper[j])

        print(f"  Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, user_ids

    except Exception as e:  # pylint: disable=broad-except
        print(f"Error loading '{network_file}': {e}")
        return None, []


# ---------------------------------------------------------------------------
# Overlapping community detection
# ---------------------------------------------------------------------------


def detect_overlapping_communities(G, algorithm="demon", epsilon=0.25, min_community=3):
    """
    Run overlapping community detection on a NetworkX graph.

    Args:
        G (nx.Graph): Input graph.
        algorithm (str): 'demon' (default, best overlap quality) or 'aslpaw' (faster,
            requires: pip install ASLPAw pyclustering).
        epsilon (float): Demon merge threshold — lower → more communities (default 0.25).
        min_community (int): Minimum community size to keep (default 3).

    Returns:
        list[set[int]]: List of communities, each a set of node IDs.
                        Returns [] on failure.
    """
    if not CDLIB_AVAILABLE:
        print("Error: cdlib is required. Install with: pip install cdlib")
        return []

    if G.number_of_nodes() == 0:
        print("Warning: empty graph, skipping community detection.")
        return []

    try:
        if algorithm == "demon":
            result = cdlib_alg.demon(G, epsilon=epsilon, min_com_size=min_community)
        elif algorithm == "aslpaw":
            result = cdlib_alg.aslpaw(G)
        else:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Use 'demon' or 'aslpaw'."
            )

        communities = [set(c) for c in result.communities]
        print(f"  Communities found: {len(communities)}")
        return communities

    except Exception as e:  # pylint: disable=broad-except
        print(f"Error in community detection ({algorithm}): {e}")
        return []


# ---------------------------------------------------------------------------
# Local Pluralistic Homophily
# ---------------------------------------------------------------------------


def compute_node_community_membership(node_ids, communities):
    """
    Build a dict mapping each node to the set of community indices it belongs to.

    Args:
        node_ids (list[int]): All node IDs in the graph.
        communities (list[set[int]]): Overlapping communities.

    Returns:
        dict[int, set[int]]: node → {community_index, ...}
    """
    membership = {v: set() for v in node_ids}
    for c_idx, community in enumerate(communities):
        for v in community:
            if v in membership:
                membership[v].add(c_idx)
    return membership


def _jaccard(set_a, set_b):
    """Jaccard similarity between two sets; returns 0 for empty union."""
    if not set_a and not set_b:
        return 1.0  # both isolated from all communities → treat as identical
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_local_pluralistic_homophily(G, membership):
    """
    Compute Local Pluralistic Homophily (LPH) for every node.

    LPH(v) = mean Jaccard( C(v), C(u) ) over all neighbors u of v.
    Nodes with no neighbors receive LPH = 0.0.

    Args:
        G (nx.Graph): Input graph.
        membership (dict[int, set[int]]): node → set of community indices.

    Returns:
        dict[int, float]: node → LPH score in [0, 1].
    """
    lph = {}
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if not neighbors:
            lph[v] = 0.0
            continue
        cv = membership.get(v, set())
        scores = [_jaccard(cv, membership.get(u, set())) for u in neighbors]
        lph[v] = sum(scores) / len(scores)
    return lph


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_community_results(
    user_ids, lph, membership, model_name, network_id, output_dir
):
    """
    Save community membership and LPH scores to a CSV file.

    Output columns:
        UserId, num_communities, community_ids, local_pluralistic_hom

    Args:
        user_ids (list[int]): Ordered list of node IDs.
        lph (dict[int, float]): Precomputed LPH per node.
        membership (dict[int, set[int]]): node → set of community indices.
        model_name (str): Model name (exponential / powerlaw / rayleigh).
        network_id (str): Zero-padded network identifier.
        output_dir (str): Directory where CSV will be written.
    """
    rows = []
    for v in user_ids:
        cids = sorted(membership.get(v, set()))
        rows.append(
            {
                "UserId": v,
                "num_communities": len(cids),
                "community_ids": ";".join(map(str, cids)) if cids else "",
                "local_pluralistic_hom": lph.get(v, 0.0),
            }
        )

    df = pd.DataFrame(rows)
    filename = f"communities_{model_name}_{network_id}.csv"
    output_file = os.path.join(output_dir, filename)
    df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")


# ---------------------------------------------------------------------------
# Per-network entry point
# ---------------------------------------------------------------------------


def calculate_communities_for_network(
    network_file, algorithm="demon", epsilon=0.25, min_community=3
):
    """
    Detect overlapping communities and compute LPH for a single inferred network.

    Args:
        network_file (str): Path to the .txt network file.
        algorithm (str): Community detection algorithm ('demon' or 'aslpaw').
        epsilon (float): Demon merge threshold (ignored for aslpaw).
        min_community (int): Minimum community size (Demon only).

    Returns:
        bool: True on success, False on failure.
    """
    filename = os.path.basename(network_file)
    parts = filename.replace(".txt", "").split("-")
    if len(parts) < 4:
        print(f"Error: unexpected filename format: {filename}")
        return False

    model_short = parts[2]  # expo / power / ray
    network_id = parts[3]  # zero-padded index

    model_mapping = {"expo": "exponential", "power": "powerlaw", "ray": "rayleigh"}
    model_name = model_mapping.get(model_short, model_short)

    print(f"\n{'='*60}")
    print(f"Communities — {model_name} | network {network_id}")
    print(f"  File: {network_file}")
    print(f"{'='*60}")

    # Output directory
    base_output_dir = os.path.join("..", "data", "communities")
    model_output_dir = create_output_directory(base_output_dir, model_name)

    # Load graph
    G, user_ids = load_inferred_network_nx(network_file)
    if G is None or G.number_of_nodes() == 0:
        print("Error: empty or unloadable network.")
        return False

    # Detect communities
    communities = detect_overlapping_communities(
        G, algorithm=algorithm, epsilon=epsilon, min_community=min_community
    )

    # Compute membership and LPH (works even with 0 communities)
    membership = compute_node_community_membership(user_ids, communities)
    lph = compute_local_pluralistic_homophily(G, membership)

    # Save
    save_community_results(
        user_ids, lph, membership, model_name, network_id, model_output_dir
    )

    avg_lph = sum(lph.values()) / len(lph) if lph else 0.0
    avg_coms = (
        sum(len(v) for v in membership.values()) / len(membership)
        if membership
        else 0.0
    )
    print(f"  avg communities/node: {avg_coms:.3f} | avg LPH: {avg_lph:.4f}")
    return True


# ---------------------------------------------------------------------------
# Batch entry point (all models)
# ---------------------------------------------------------------------------


def calculate_communities_for_all_models(
    algorithm="demon", epsilon=0.25, min_community=3
):
    """
    Run overlapping community detection for every inferred network across all models.

    Args:
        algorithm (str): 'demon' (default) or 'aslpaw'.
        epsilon (float): Demon merge threshold.
        min_community (int): Minimum community size (Demon only).

    Returns:
        dict: {'model': {'processed': int, 'failed': int}, ...}
    """
    base_network_path = os.path.join("..", "data", "inferred_networks")

    if not os.path.exists(base_network_path):
        print(f"Error: directory not found: {base_network_path}")
        return {}

    results = {}
    models = ["exponential", "powerlaw", "rayleigh"]

    for model in models:
        model_path = os.path.join(base_network_path, model)

        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping.")
            results[model] = {"processed": 0, "failed": 0}
            continue

        network_files = sorted(
            f
            for f in glob.glob(os.path.join(model_path, "*.txt"))
            if not f.endswith(".csv")
        )

        if not network_files:
            print(f"Warning: no network files for {model}.")
            results[model] = {"processed": 0, "failed": 0}
            continue

        print(f"\n{'='*80}")
        print(
            f"{model.upper()} — {len(network_files)} networks | algorithm: {algorithm}"
        )
        print(f"{'='*80}")

        processed, failed = 0, 0
        for nf in network_files:
            try:
                ok = calculate_communities_for_network(
                    nf,
                    algorithm=algorithm,
                    epsilon=epsilon,
                    min_community=min_community,
                )
                if ok:
                    processed += 1
                else:
                    failed += 1
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error on {nf}: {e}")
                failed += 1

        results[model] = {"processed": processed, "failed": failed}
        print(f"\n  {model}: {processed} processed, {failed} failed")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    total_ok = sum(r["processed"] for r in results.values())
    total_fail = sum(r["failed"] for r in results.values())
    for model, r in results.items():
        print(f"  {model}: {r['processed']} OK, {r['failed']} failed")
    print(f"  Total: {total_ok} processed, {total_fail} failed")

    return results
