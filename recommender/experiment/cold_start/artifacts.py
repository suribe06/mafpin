"""Compatibility re-exports — writers live in ``paths`` after locality merge."""

from recommender.experiment.cold_start.paths import (  # noqa: F401
    upsert_frame,
    upsert_results,
    write_csv,
    write_json,
    write_readme,
    write_split_tables,
)
