"""User stratum assignment for cold-start evaluation."""

from __future__ import annotations

from typing import Any

import pandas as pd

STRATA_ORDER = ("0", "1-3", "4-10", ">10")


def assign_stratum(n_train_ratings: int) -> str:
    """Map train rating count to a cold-start stratum label."""
    if n_train_ratings <= 0:
        return "0"
    if n_train_ratings <= 3:
        return "1-3"
    if n_train_ratings <= 10:
        return "4-10"
    return ">10"


def build_user_strata(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-user stratum table from train/test rating counts.

    Optional *feature_coverage* must be indexed by ``UserId`` with boolean
    columns among ``appears_in_netinf_graph``, ``has_centrality_features``,
    ``has_lph_features``, ``has_trust_features``.
    """
    train_counts = train_df.groupby("UserId").size().rename("n_train_ratings")
    test_counts = test_df.groupby("UserId").size().rename("n_test_ratings")
    users = sorted(set(train_counts.index) | set(test_counts.index))
    frame = pd.DataFrame({"user_id": users}).set_index("user_id")
    frame["n_train_ratings"] = train_counts.reindex(users).fillna(0).astype(int)
    frame["n_test_ratings"] = test_counts.reindex(users).fillna(0).astype(int)
    frame["stratum"] = frame["n_train_ratings"].map(assign_stratum)

    for col in (
        "appears_in_netinf_graph",
        "has_centrality_features",
        "has_lph_features",
        "has_trust_features",
    ):
        if feature_coverage is not None and col in feature_coverage.columns:
            frame[col] = (
                feature_coverage.reindex(users)[col].fillna(False).astype(bool)
            )
        else:
            frame[col] = False

    # Only users with at least one test rating are evaluation targets.
    frame = frame[frame["n_test_ratings"] > 0].copy()
    frame = frame.reset_index().rename(columns={"user_id": "user_id"})
    return frame


def strata_user_map(user_strata: pd.DataFrame) -> dict[str, set[int]]:
    """Return stratum → set of user ids (with test ratings)."""
    out: dict[str, set[int]] = {s: set() for s in STRATA_ORDER}
    for row in user_strata.itertuples(index=False):
        out.setdefault(str(row.stratum), set()).add(int(row.user_id))
    return out


def coverage_from_feature_index(
    user_ids: list[int],
    *,
    centrality_users: set[int] | None = None,
    lph_users: set[int] | None = None,
    trust_users: set[int] | None = None,
) -> pd.DataFrame:
    """Build a feature-coverage frame indexed by UserId."""
    centrality_users = centrality_users or set()
    lph_users = lph_users or set()
    trust_users = trust_users or set()
    rows: list[dict[str, Any]] = []
    for uid in user_ids:
        rows.append(
            {
                "appears_in_netinf_graph": uid in centrality_users,
                "has_centrality_features": uid in centrality_users,
                "has_lph_features": uid in lph_users,
                "has_trust_features": uid in trust_users,
            }
        )
    return pd.DataFrame(rows, index=pd.Index(user_ids, name="UserId"))
