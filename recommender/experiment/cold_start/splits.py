"""Train/test split helpers for cold-start experiments."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import Datasets
from recommender.data import load_and_split_dataset

# Stratified leave-k caps: 0 → stratum 0; 2 → 1-3; 7 → 4-10; None → keep all early (warm).
DEFAULT_LEAVE_K_TARGETS: tuple[int | None, ...] = (0, 2, 7, None)


def dedupe_ratings(data: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate (UserId, ItemId, timestamp) rows; keep first."""
    cols = ["UserId", "ItemId"]
    if "timestamp" in data.columns:
        cols.append("timestamp")
    before = len(data)
    out = data.drop_duplicates(subset=cols, keep="first").copy()
    dropped = before - len(out)
    if dropped:
        print(f"Dropped {dropped} duplicate rating rows on {cols}")
    return out


def _with_stable_order(data: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable row order for timestamp-tie breaks."""
    out = data.copy()
    out["_ord"] = np.arange(len(out), dtype=np.int64)
    return out


def global_strata_split(
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reuse the core global temporal/random split (§6.1 diagnostic)."""
    return load_and_split_dataset(dataset=dataset)


def per_user_chrono_split(
    data: pd.DataFrame,
    *,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user chronological leave-last split (§6.2 leave-last).

    Dedupes exact rows, sorts by ``(timestamp, row_order)``, and holds out the
    last ``max(1, ceil(test_frac * N))`` ratings as test.
    """
    if "timestamp" not in data.columns:
        raise ValueError("per_user_chrono_split requires a timestamp column")
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    data = _with_stable_order(dedupe_ratings(data))
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in data.groupby("UserId", sort=False):
        ordered = group.sort_values(["timestamp", "_ord"], kind="mergesort")
        n = len(ordered)
        n_test = max(1, int(math.ceil(test_frac * n)))
        if n_test >= n:
            test_parts.append(ordered)
            continue
        test_parts.append(ordered.iloc[-n_test:])
        train_parts.append(ordered.iloc[:-n_test])

    train_df = _concat_drop_ord(train_parts, data)
    test_df = _concat_drop_ord(test_parts, data)
    return train_df, test_df


def per_user_leave_k_split(
    data: pd.DataFrame,
    *,
    test_frac: float = 0.2,
    seed: int = 42,
    targets: Sequence[int | None] = DEFAULT_LEAVE_K_TARGETS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Per-user leave-k split that populates cold strata on dense datasets.

    Protocol:
    1. Dedupe + chrono sort with stable tie-break.
    2. Hold out the last ``max(1, ceil(test_frac * N))`` ratings as test.
    3. Assign each user a train-cap from *targets* (round-robin after shuffle):
       ``0``, ``2`` (→1-3), ``7`` (→4-10), ``None`` (keep all early → usually >10).
    4. Train = first ``cap`` ratings of the early block; unused early ratings
       are dropped (neither train nor test) to avoid leakage.
    """
    if "timestamp" not in data.columns:
        raise ValueError("per_user_leave_k_split requires a timestamp column")
    if not targets:
        raise ValueError("targets must be non-empty")

    data = _with_stable_order(dedupe_ratings(data))
    users = np.array(sorted(data["UserId"].unique()), dtype=int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(users))
    target_list = list(targets)
    assignment: dict[int, int | None] = {
        int(users[i]): target_list[j % len(target_list)] for j, i in enumerate(order)
    }

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    dropped_early = 0
    for uid_key, group in data.groupby("UserId", sort=False):
        uid = int(uid_key)  # type: ignore[arg-type]
        ordered = group.sort_values(["timestamp", "_ord"], kind="mergesort")
        n = len(ordered)
        n_test = max(1, int(math.ceil(test_frac * n)))
        if n_test >= n:
            test_parts.append(ordered)
            continue
        early = ordered.iloc[:-n_test]
        late = ordered.iloc[-n_test:]
        test_parts.append(late)
        cap = assignment[uid]
        if cap is None:
            train_parts.append(early)
        elif cap <= 0:
            dropped_early += len(early)
        else:
            keep = min(int(cap), len(early))
            train_parts.append(early.iloc[:keep])
            dropped_early += len(early) - keep

    train_df = _concat_drop_ord(train_parts, data)
    test_df = _concat_drop_ord(test_parts, data)
    meta = {
        "split_protocol": "leave_k",
        "leave_k_targets": [t if t is not None else "all" for t in target_list],
        "leave_k_seed": seed,
        "test_frac": test_frac,
        "dropped_early_ratings": int(dropped_early),
        "n_users_cap_0": sum(1 for v in assignment.values() if v == 0),
        "n_users_cap_2": sum(1 for v in assignment.values() if v == 2),
        "n_users_cap_7": sum(1 for v in assignment.values() if v == 7),
        "n_users_cap_all": sum(1 for v in assignment.values() if v is None),
    }
    return train_df, test_df, meta


def _concat_drop_ord(
    parts: list[pd.DataFrame], empty_like: pd.DataFrame
) -> pd.DataFrame:
    if not parts:
        out = empty_like.iloc[0:0].copy()
    else:
        out = pd.concat(parts, ignore_index=True)
    if "_ord" in out.columns:
        out = out.drop(columns=["_ord"])
    return out


def load_encoded_ratings(
    dataset: str,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Load ratings with LabelEncoders retained for trust ID alignment."""
    if dataset not in Datasets.CONFIG:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {Datasets.ALL}")
    cfg = Datasets.CONFIG[dataset]
    path = Datasets.ROOT / dataset / cfg["file"]
    cols: list[int] = [cfg["col_user"], cfg["col_item"], cfg["col_rating"]]
    col_names = ["UserId", "ItemId", "Rating"]
    if cfg.get("col_time") is not None:
        cols.append(cfg["col_time"])
        col_names.append("timestamp")
    data = pd.read_csv(
        path,
        sep=cfg["sep"],
        header=cfg["header"],
        usecols=cols,  # type: ignore[call-overload]
        engine="python",
    )
    data.columns = pd.Index(col_names)
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    data["UserId"] = user_enc.fit_transform(data["UserId"])
    data["ItemId"] = item_enc.fit_transform(data["ItemId"])
    return data, user_enc, item_enc


def zero_shot_trust_split(
    data: pd.DataFrame,
    *,
    encoded_trust_users: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Hold out all ratings for users present in the trust graph.

    Train = ratings from users *not* in the trust-overlap set.
    Test = all ratings from trust-overlap users (n_train = 0).
    """
    data = dedupe_ratings(data)
    zero_shot_users = sorted(
        uid for uid in encoded_trust_users if uid in set(data["UserId"].unique())
    )
    if not zero_shot_users:
        raise ValueError("No overlap between trust-graph users and rating users")
    zero_set = set(zero_shot_users)
    zero_list = list(zero_set)
    test_df = data[data["UserId"].isin(zero_list)].copy()
    train_df = data[~data["UserId"].isin(zero_list)].copy()
    if isinstance(train_df, pd.Series) or train_df.empty:
        raise ValueError("Zero-shot split left an empty train set")
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(train_df, pd.DataFrame)
    return train_df, test_df, zero_shot_users


def split_manifest_payload(
    *,
    mode: str,
    dataset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable split manifest."""
    payload: dict[str, Any] = {
        "mode": mode,
        "dataset": dataset,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_train_users": int(train_df["UserId"].nunique()) if len(train_df) else 0,
        "n_test_users": int(test_df["UserId"].nunique()) if len(test_df) else 0,
    }
    if "timestamp" in train_df.columns and len(train_df):
        payload["train_timestamp_max"] = int(train_df["timestamp"].max())
    if "timestamp" in test_df.columns and len(test_df):
        payload["test_timestamp_min"] = int(test_df["timestamp"].min())
        payload["test_timestamp_max"] = int(test_df["timestamp"].max())
    if extra:
        payload.update(extra)
    return payload


def assert_finite_ids(df: pd.DataFrame) -> None:
    """Sanity-check rating frame columns."""
    for col in ("UserId", "ItemId", "Rating"):
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")
    if not np.isfinite(df["Rating"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite ratings in frame")
