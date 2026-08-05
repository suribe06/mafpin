"""Model variant definitions for the core experiment ladder."""

from __future__ import annotations

from typing import Any

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "M1": {
        "run_id": "m1_baseline",
        "log_name": None,
        "needs_network": False,
        "social_regularization": False,
        "include_communities": True,
        "social_mode": None,
        "social_normalization": None,
    },
    "M2": {
        "run_id": "m2_recommend",
        "log_name": "m2_recommend.log",
        "needs_network": True,
        "social_regularization": False,
        "include_communities": False,
        "social_mode": None,
        "social_normalization": None,
    },
    "M3": {
        "run_id": "m3_recommend",
        "log_name": "m3_recommend.log",
        "needs_network": True,
        "social_regularization": False,
        "include_communities": True,
        "social_mode": None,
        "social_normalization": None,
    },
    "M4a": {
        "run_id": "m4a_recommend",
        "log_name": "m4a_recommend.log",
        "needs_network": True,
        "social_regularization": True,
        "include_communities": True,
        "social_mode": "uniform",
        "social_normalization": "mean_weight",
    },
    "M4b": {
        "run_id": "m4b_recommend",
        "log_name": "m4b_recommend.log",
        "needs_network": True,
        "social_regularization": True,
        "include_communities": True,
        "social_mode": "community_jaccard",
        "social_normalization": "mean_weight",
    },
    "M4c": {
        "run_id": "m4c_recommend",
        "log_name": "m4c_recommend.log",
        "needs_network": True,
        "social_regularization": True,
        "include_communities": True,
        "social_mode": "boundary_downweight",
        "social_normalization": "mean_weight",
    },
    "M4d": {
        "run_id": "m4d_recommend",
        "log_name": "m4d_recommend.log",
        "needs_network": True,
        "social_regularization": True,
        "include_communities": True,
        "social_mode": "bridge_preserve",
        "social_normalization": "mean_weight",
    },
    "M4c_robustness": {
        "run_id": "m4c_robustness_laplacian",
        "log_name": "m4c_robustness_laplacian.log",
        "needs_network": True,
        "social_regularization": True,
        "include_communities": True,
        "social_mode": "boundary_downweight",
        "social_normalization": "normalized_laplacian",
    },
    "M2_trust": {
        "run_id": "m2_trust",
        "log_name": None,
        "needs_network": False,
        "social_regularization": False,
        "include_communities": False,
        "social_mode": None,
        "social_normalization": None,
        "trust_features": True,
        "trust_include_communities": False,
    },
    "M3_trust": {
        "run_id": "m3_trust",
        "log_name": None,
        "needs_network": False,
        "social_regularization": False,
        "include_communities": True,
        "social_mode": None,
        "social_normalization": None,
        "trust_features": True,
        "trust_include_communities": True,
    },
    "M3_soft": {
        "run_id": "m3_soft_communities",
        "log_name": None,
        "needs_network": True,
        "social_regularization": False,
        "include_communities": True,
        "social_mode": None,
        "social_normalization": None,
        "soft_communities": True,
    },
}

ALL_VARIANT_IDS: list[str] = list(VARIANT_SPECS.keys())
CORE_VARIANT_IDS: list[str] = ["M1", "M2", "M3", "M4a", "M4b", "M4c", "M4d"]
COLD_START_VARIANT_IDS: list[str] = ["M1", "M2", "M3", "M4c", "M4d"]
SOFT_COLD_START_VARIANT_IDS: list[str] = ["M1", "M2", "M3", "M3_soft"]
TRUST_VARIANT_IDS: list[str] = ["M1", "M2_trust", "M3_trust"]


def variant_cli_flags(variant_id: str) -> dict[str, Any]:
    """Map a model variant id to recommend-step CLI-equivalent flags."""
    spec = VARIANT_SPECS[variant_id]
    flags: dict[str, Any] = {
        "include_communities": spec["include_communities"],
        "social_regularization": spec["social_regularization"],
    }
    if spec["social_regularization"]:
        flags["social_mode"] = spec["social_mode"]
        flags["social_normalization"] = spec["social_normalization"]
    return flags
