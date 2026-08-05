"""Compatibility re-exports — deltas live in ``strata`` after locality merge."""

from recommender.experiment.cold_start.strata import (  # noqa: F401
    bootstrap_delta_table,
    bootstrap_mean_ci,
    build_user_deltas,
    per_user_rmse,
)
