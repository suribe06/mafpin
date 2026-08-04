"""Trust-graph side features for Ciao/Epinions zero-shot cold-start."""

from __future__ import annotations

import networkx as nx
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from networks.social import compute_trust_features, load_trust_graph


def encoded_trust_user_ids(user_enc: LabelEncoder, G: nx.DiGraph) -> set[int]:
    """Map raw trust-graph node IDs onto LabelEncoder user IDs."""
    classes = set(map(int, user_enc.classes_))
    out: set[int] = set()
    for raw in G.nodes():
        raw_i = int(raw)
        if raw_i not in classes:
            continue
        out.add(int(user_enc.transform([raw_i])[0]))
    return out


def map_trust_features_to_encoded(
    trust_df: pd.DataFrame,
    user_enc: LabelEncoder,
) -> pd.DataFrame:
    """Reindex trust centrality features onto encoded UserId."""
    classes = set(map(int, user_enc.classes_))
    rows = []
    for raw_id, row in trust_df.iterrows():
        raw_i = int(raw_id)
        if raw_i not in classes:
            continue
        encoded = int(user_enc.transform([raw_i])[0])
        rows.append({"UserId": encoded, **row.to_dict()})
    if not rows:
        return pd.DataFrame(
            columns=["trust_in_degree", "trust_out_degree", "trust_pagerank"]
        ).rename_axis("UserId")
    frame = pd.DataFrame(rows).set_index("UserId")
    return frame


def _community_boundary_features(G: nx.DiGraph) -> pd.DataFrame:
    """Simple community / boundary features on the undirected trust graph.

    Uses greedy modularity communities (stdlib NetworkX) — not NetInf LPH —
    so M3_trust stays independent of cascade inference.
    """
    undirected = G.to_undirected()
    if undirected.number_of_edges() == 0:
        return pd.DataFrame(
            columns=["trust_community_size", "trust_boundary_frac"]
        ).rename_axis("UserId")

    communities = list(nx.community.greedy_modularity_communities(undirected))
    node_to_com: dict[int, int] = {}
    com_sizes: dict[int, int] = {}
    for cid, members in enumerate(communities):
        com_sizes[cid] = len(members)
        for node in members:
            node_to_com[int(node)] = cid

    rows = []
    for node in undirected.nodes():
        nid = int(node)
        cid = node_to_com.get(nid)
        if cid is None:
            rows.append(
                {
                    "UserId": nid,
                    "trust_community_size": 1,
                    "trust_boundary_frac": 0.0,
                }
            )
            continue
        neighbors = list(undirected.neighbors(nid))
        if not neighbors:
            boundary = 0.0
        else:
            outside = sum(1 for n in neighbors if node_to_com.get(int(n)) != cid)
            boundary = outside / len(neighbors)
        rows.append(
            {
                "UserId": nid,
                "trust_community_size": com_sizes[cid],
                "trust_boundary_frac": boundary,
            }
        )
    return pd.DataFrame(rows).set_index("UserId")


def build_trust_attribute_tables(
    dataset: str,
    user_enc: LabelEncoder,
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    """Return (M2_trust attrs, M3_trust attrs, encoded trust user set)."""
    G = load_trust_graph(dataset)
    trust_raw = compute_trust_features(G)
    m2 = map_trust_features_to_encoded(trust_raw, user_enc)
    # Community features computed on raw graph, then remapped.
    com_raw = _community_boundary_features(G)
    com_enc = map_trust_features_to_encoded(com_raw, user_enc)
    m3 = m2.join(com_enc, how="left").fillna(0.0)
    encoded_users = set(map(int, m2.index))
    return m2, m3, encoded_users
