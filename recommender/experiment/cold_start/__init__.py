"""Cold-start experiment package (diagnostic, controlled, trust zero-shot)."""

from recommender.experiment.cold_start.paths import ColdStartPaths
from recommender.experiment.cold_start.splits import (
    global_strata_split,
    per_user_chrono_split,
    per_user_leave_k_split,
    zero_shot_trust_split,
)
from recommender.experiment.cold_start.strata import assign_stratum, build_user_strata

__all__ = [
    "ColdStartPaths",
    "assign_stratum",
    "build_user_strata",
    "global_strata_split",
    "per_user_chrono_split",
    "per_user_leave_k_split",
    "zero_shot_trust_split",
]
