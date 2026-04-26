"""
networks.cascades — cascade generation and statistics for NetInf.

Public API
----------
- :func:`list_available_datasets`
- :func:`generate_cascades_from_df`
- :func:`compute_cascade_user_stats`
"""

from networks.cascades.generation import (
    generate_cascades_from_df,
    list_available_datasets,
)
from networks.cascades.stats import compute_cascade_user_stats

__all__ = [
    "list_available_datasets",
    "generate_cascades_from_df",
    "compute_cascade_user_stats",
]
