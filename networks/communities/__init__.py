"""
networks.communities — community detection and Local Pluralistic Homophily.

Public API (mirrors the original communities.py public surface):
"""

from networks.communities.detection import (
    detect_overlapping_communities,
    compute_node_community_membership,
)
from networks.communities.lph import (
    _jaccard,
    compute_local_pluralistic_homophily,
    _compute_neighborhood_alignment,
    _compute_network_homophily,
    compute_lph_paper,
)
from networks.communities.boundary import (
    compute_boundary_indicator,
    save_community_results,
)
from networks.communities.batch import (
    calculate_communities_for_network,
    calculate_communities_for_all_models,
)

__all__ = [
    "detect_overlapping_communities",
    "compute_node_community_membership",
    "_jaccard",
    "compute_local_pluralistic_homophily",
    "_compute_neighborhood_alignment",
    "_compute_network_homophily",
    "compute_lph_paper",
    "compute_boundary_indicator",
    "save_community_results",
    "calculate_communities_for_network",
    "calculate_communities_for_all_models",
]
