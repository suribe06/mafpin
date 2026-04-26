"""
recommender.enhanced — Enhanced CMF with network and community side-information.

Public API (mirrors the original enhanced.py public surface):
"""

from recommender.enhanced.features import _SCALERS, load_network_features
from recommender.enhanced.model import evaluate_cmf_with_user_attributes
from recommender.enhanced.search import (
    search_enhanced_params,
    save_enhanced_search_results,
)
from recommender.enhanced.workers import (
    _WORKER_DATA,
    _WORKER_SHARED,
    _worker_init,
    _eval_network_worker,
)
from recommender.enhanced.network_eval import (
    evaluate_single_network,
    _save_rmses,
    run_network_evaluation,
)

__all__ = [
    "_SCALERS",
    "load_network_features",
    "evaluate_cmf_with_user_attributes",
    "search_enhanced_params",
    "save_enhanced_search_results",
    "_WORKER_DATA",
    "_WORKER_SHARED",
    "_worker_init",
    "_eval_network_worker",
    "evaluate_single_network",
    "_save_rmses",
    "run_network_evaluation",
]
