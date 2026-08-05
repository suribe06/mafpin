"""Soft community assignment for few-shot cold-start users."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from recommender.experiment.route_b.communities_freeze import parse_community_ids


def build_community_item_profiles(
    train_df: pd.DataFrame,
    communities: pd.DataFrame,
) -> dict[int, Counter]:
    """For each community, count item ratings from its member users in train."""
    com = communities.copy()
    if "UserId" in com.columns:
        com = com.set_index("UserId")
    if "community_ids" in com.columns and "community_set" not in com.columns:
        com["community_set"] = com["community_ids"].map(parse_community_ids)
    user_coms = (
        com["community_set"] if "community_set" in com.columns else pd.Series(dtype=object)
    )
    profiles: dict[int, Counter] = defaultdict(Counter)
    for row in train_df.itertuples(index=False):
        uid = int(row.UserId)
        iid = int(row.ItemId)
        coms = user_coms.get(uid, set()) if uid in user_coms.index else set()
        if not isinstance(coms, set) or not coms:
            continue
        for c in coms:
            profiles[int(c)][iid] += 1
    return dict(profiles)


def soft_assign_user(
    user_items: set[int],
    profiles: dict[int, Counter],
    *,
    top_k: int = 5,
) -> dict[int, float]:
    """Overlap-based soft membership over community item profiles.

    ``score(c) = |I_u ∩ support(c)| / |I_u|`` then L1-normalise the top_k.
    Users with empty ``user_items`` get {}.
    """
    if not user_items:
        return {}
    raw: list[tuple[int, float]] = []
    n = float(len(user_items))
    for cid, counter in profiles.items():
        overlap = sum(1 for i in user_items if i in counter)
        if overlap:
            raw.append((cid, overlap / n))
    if not raw:
        return {}
    raw.sort(key=lambda t: t[1], reverse=True)
    raw = raw[:top_k]
    total = sum(s for _, s in raw)
    if total <= 0:
        return {}
    return {cid: s / total for cid, s in raw}


def soft_community_feature_frame(
    train_df: pd.DataFrame,
    communities: pd.DataFrame,
    user_ids: list[int],
    *,
    top_k: int = 5,
    min_train_ratings: int = 1,
    max_train_ratings: int = 10,
    prefix: str = "soft_community_",
) -> pd.DataFrame:
    """Build soft community columns for users in the few-shot band.

    Warm users (``n_train > max_train_ratings``) and pure cold (0 ratings)
    keep zeros for soft columns — warm already have hard membership in M3;
    pure cold has no item evidence (use trust / external signals instead).
    """
    profiles = build_community_item_profiles(train_df, communities)
    community_ids = sorted(profiles.keys())
    train_items = train_df.groupby("UserId")["ItemId"].apply(set).to_dict()
    train_counts = train_df.groupby("UserId").size().to_dict()

    cols = [f"{prefix}{cid}" for cid in community_ids]
    data = {c: np.zeros(len(user_ids), dtype=float) for c in cols}
    data["soft_community_entropy"] = np.zeros(len(user_ids), dtype=float)
    data["soft_community_assigned"] = np.zeros(len(user_ids), dtype=float)
    index = list(map(int, user_ids))

    for i, uid in enumerate(index):
        n = int(train_counts.get(uid, 0))
        if n < min_train_ratings or n > max_train_ratings:
            continue
        items = train_items.get(uid, set())
        weights = soft_assign_user(set(map(int, items)), profiles, top_k=top_k)
        if not weights:
            continue
        data["soft_community_assigned"][i] = 1.0
        probs = np.array(list(weights.values()), dtype=float)
        # Shannon entropy in bits
        data["soft_community_entropy"][i] = float(-(probs * np.log2(probs + 1e-12)).sum())
        for cid, w in weights.items():
            col = f"{prefix}{cid}"
            if col in data:
                data[col][i] = w

    return pd.DataFrame(data, index=pd.Index(index, name="UserId"))


def merge_soft_into_user_attributes(
    base_attrs: pd.DataFrame,
    soft_attrs: pd.DataFrame,
    *,
    drop_hard_community_bins: bool = True,
) -> pd.DataFrame:
    """Combine NetInf attrs with soft community features.

    When *drop_hard_community_bins* is True, remove ``community_<id>`` one-hots
    so M3_soft is not dominated by hard DEMON bins the cold user never earned.
    """
    out = base_attrs.copy()
    if drop_hard_community_bins:
        drop = [c for c in out.columns if str(c).startswith("community_")]
        out = out.drop(columns=drop, errors="ignore")
    soft = soft_attrs.reindex(out.index).fillna(0.0)
    for col in soft.columns:
        out[col] = soft[col]
    return out.fillna(0.0)
