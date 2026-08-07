"""Beyond-accuracy metrics for Route B WP1 (CCE, ILD, novelty, coverage, Gini)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from recommender.data import predict_ratings
from recommender.experiment.route_b.communities_freeze import (
    item_dominant_communities,
    load_frozen_communities,
)


def _top_k_for_users(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    k: int = 10,
) -> dict[int, list[int]]:
    """Same candidate construction as ``evaluate_ranking``."""
    all_items = set(train_df["ItemId"].unique()) | set(test_df["ItemId"].unique())
    top_by_user: dict[int, list[int]] = {}
    for user_id in test_df["UserId"].unique():
        uid = int(user_id)
        train_items = set(train_df.loc[train_df["UserId"] == uid, "ItemId"])
        candidates = list(all_items - train_items)
        if not candidates:
            continue
        candidate_df = pd.DataFrame(
            {"UserId": [uid] * len(candidates), "ItemId": candidates}
        )
        candidate_df["_score"] = predict_ratings(model, candidate_df)
        ranked = candidate_df.sort_values("_score", ascending=False)["ItemId"].tolist()
        top_by_user[uid] = [int(x) for x in ranked[:k]]
    return top_by_user


def _item_popularity(train_df: pd.DataFrame) -> pd.Series:
    n_users = max(int(train_df["UserId"].nunique()), 1)
    counts = train_df.groupby("ItemId")["UserId"].nunique()
    return (counts / n_users).rename("pop")


def coverage_at_k(top_by_user: dict[int, list[int]], n_items: int) -> float:
    if n_items <= 0:
        return float("nan")
    recommended = {i for items in top_by_user.values() for i in items}
    return float(len(recommended) / n_items)


def gini_at_k(top_by_user: dict[int, list[int]], all_items: set[int]) -> float:
    freq = {i: 0 for i in all_items}
    for items in top_by_user.values():
        for i in items:
            if i in freq:
                freq[i] += 1
    values = np.array(sorted(freq.values()), dtype=float)
    total = values.sum()
    if total <= 0 or len(values) == 0:
        return float("nan")
    n = len(values)
    idx = np.arange(1, n + 1, dtype=float)
    return float(((2 * idx - n - 1) * values).sum() / (n * total))


def ild_latent(
    top_items: list[int],
    item_factors: np.ndarray | None,
    item_index: dict[int, int],
) -> float:
    """Intra-list diversity from item latent factors (diagnostic / fallback)."""
    if item_factors is None or len(top_items) < 2:
        return float("nan")
    vecs = []
    for iid in top_items:
        j = item_index.get(iid)
        if j is None:
            continue
        v = item_factors[j]
        norm = np.linalg.norm(v)
        if norm > 0:
            vecs.append(v / norm)
    if len(vecs) < 2:
        return float("nan")
    sims = []
    for a in range(len(vecs)):
        for b in range(a + 1, len(vecs)):
            sims.append(float(np.dot(vecs[a], vecs[b])))
    return float(1.0 - np.mean(sims)) if sims else float("nan")


def novelty_at_k(top_items: list[int], popularity: pd.Series) -> float:
    vals = []
    for iid in top_items:
        raw = popularity.get(iid, 0.0)
        pop = float(0.0 if raw is None or (isinstance(raw, float) and np.isnan(raw)) else raw)
        pop = max(pop, 1e-12)
        vals.append(-np.log2(pop))
    return float(np.mean(vals)) if vals else float("nan")


def cce_at_k(
    top_items: list[int],
    user_coms: set[int],
    item_dominant: dict[int, set[int]],
) -> float:
    """Cross-community exposure @K: (1/K) Σ 1[D(i) ∩ C(u) = ∅].

    Items without a known train dominant community contribute 0 (not counted
    as cross-community). Users without communities return NaN.
    """
    if not top_items or not user_coms:
        return float("nan")
    k = len(top_items)
    hits = 0
    for iid in top_items:
        d = item_dominant.get(int(iid))
        if d is None:
            continue
        if d.isdisjoint(user_coms):
            hits += 1
    return float(hits / k)


def extract_item_factors(model) -> tuple[np.ndarray | None, dict[int, int]]:
    """Best-effort item factor matrix from a fitted cmfrec model."""
    for attr in ("B", "B_", "item_factors", "Qi"):
        mat = getattr(model, attr, None)
        if mat is None:
            continue
        arr = np.asarray(mat, dtype=float)
        if arr.ndim != 2:
            continue
        # cmfrec often stores items as rows
        index = {i: i for i in range(arr.shape[0])}
        return arr, index
    return None, {}


def compute_beyond_accuracy_metrics(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    dataset: str,
    k: int = 10,
    communities: pd.DataFrame | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return (aggregate dict, per-user DataFrame)."""
    com = communities if communities is not None else load_frozen_communities(dataset)
    user_coms = pd.Series(com["community_set"])
    item_dom = item_dominant_communities(train_df, user_coms)
    top_by_user = _top_k_for_users(model, train_df, test_df, k=k)
    pop = _item_popularity(train_df)
    factors, factor_index = extract_item_factors(model)
    all_items = set(train_df["ItemId"].unique()) | set(test_df["ItemId"].unique())

    rows: list[dict[str, Any]] = []
    for uid, top_items in top_by_user.items():
        coms = user_coms.get(uid, set()) if uid in user_coms.index else set()
        rows.append(
            {
                "UserId": uid,
                "cce_at_k": cce_at_k(top_items, coms if isinstance(coms, set) else set(), item_dom),
                "ild_latent_at_k": ild_latent(top_items, factors, factor_index),
                "novelty_at_k": novelty_at_k(top_items, pop),
                "n_communities": len(coms) if isinstance(coms, set) else 0,
            }
        )
    per_user = pd.DataFrame(rows)
    agg = {
        "item_coverage_at_k": coverage_at_k(top_by_user, len(all_items)),
        "gini_at_k": gini_at_k(top_by_user, all_items),
        "cce_at_k_mean": float(per_user["cce_at_k"].mean(skipna=True))
        if len(per_user)
        else float("nan"),
        "ild_latent_at_k_mean": float(per_user["ild_latent_at_k"].mean(skipna=True))
        if len(per_user)
        else float("nan"),
        "novelty_at_k_mean": float(per_user["novelty_at_k"].mean(skipna=True))
        if len(per_user)
        else float("nan"),
        "n_users_ranked": float(len(per_user)),
        "n_users_with_cce": float(per_user["cce_at_k"].notna().sum())
        if len(per_user)
        else 0.0,
    }
    return agg, per_user
