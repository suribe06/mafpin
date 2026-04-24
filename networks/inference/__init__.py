"""
networks.inference — NetInf-based network inference.

Public API (mirrors the original inference.py public surface):
"""

from networks.inference.subprocess_utils import (
    _create_output_dirs,
    _cleanup_leftover_edge_info,
)
from networks.inference.core import infer_networks
from networks.inference.batch import infer_networks_all_models

__all__ = [
    "_create_output_dirs",
    "_cleanup_leftover_edge_info",
    "infer_networks",
    "infer_networks_all_models",
]
