"""
Dataset utilities for the MAFPIN recommender pipeline.

Provides helpers for loading and encoding rating datasets, splitting data into
training and test sets, running inference with a fitted CMF model, and
computing evaluation metrics.

Functions
---------
load_dataset
    Load a CSV ratings file and return an encoded DataFrame.
load_and_split_dataset
    Load a dataset and apply the global train/test split from :class:`config.Split`.
split_data_single
    Perform a single random train/test split.
split_data_temporal
    Perform a temporal train/test split ordered by timestamp.
predict_ratings
    Call a fitted CMF model to obtain predicted ratings.
evaluate_single_split
    Evaluate a fitted model on a test set, returning RMSE, MAE, and R².
evaluate_ranking
    Compute ranking metrics (NDCG@K, Precision@K, Recall@K, MRR) on a test set.
make_recommendations
    Generate the top-N item recommendations for a given user.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import Datasets


# ---------------------------------------------------------------------------
# Loading and encoding
# ---------------------------------------------------------------------------


def load_dataset(
    dataset: str | None = None,
    filename: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load a ratings dataset and return a DataFrame with encoded integer IDs.

    Pass *dataset* (e.g. ``"movielens"``, ``"ciao"``, ``"epinions"``) to
    read from the ``datasets/<name>/`` folder using the format defined in
    :class:`config.Datasets`.  Alternatively, pass an explicit *filename*
    for a CSV file (legacy behaviour, columns 0-1-2 are assumed to be
    UserId, ItemId, Rating respectively).  When neither is supplied the
    default dataset (``config.Datasets.DEFAULT``) is used.

    Args:
        dataset:  Dataset name.  Defaults to ``Datasets.DEFAULT``.
        filename: Explicit path to a CSV file (legacy, overrides *dataset*).

    Returns:
        DataFrame with columns ``UserId`` (int), ``ItemId`` (int),
        ``Rating`` (float), and ``timestamp`` (int) when available.
    """
    if filename is not None:
        path = Path(filename)
        data = pd.read_csv(path, usecols=[0, 1, 2])  # type: ignore[call-overload]
        data.columns = pd.Index(["UserId", "ItemId", "Rating"])
    else:
        ds_name = dataset or Datasets.DEFAULT
        if ds_name not in Datasets.CONFIG:
            raise ValueError(
                f"Unknown dataset '{ds_name}'. " f"Choose from: {Datasets.ALL}"
            )
        cfg = Datasets.CONFIG[ds_name]
        path = Datasets.ROOT / ds_name / cfg["file"]
        cols: list[int] = [cfg["col_user"], cfg["col_item"], cfg["col_rating"]]
        col_names = ["UserId", "ItemId", "Rating"]
        if cfg.get("col_time") is not None:
            cols.append(cfg["col_time"])
            col_names.append("timestamp")
        data = pd.read_csv(
            path,
            sep=cfg["sep"],
            header=cfg["header"],
            usecols=cols,
            engine="python",
        )  # type: ignore[call-overload]
        data.columns = pd.Index(col_names)

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    data["UserId"] = user_enc.fit_transform(data["UserId"])
    data["ItemId"] = item_enc.fit_transform(data["ItemId"])

    print(
        f"Dataset loaded: {len(data)} ratings, "
        f"{data['UserId'].nunique()} users, "
        f"{data['ItemId'].nunique()} items"
    )
    return data


def load_and_split_dataset(
    dataset: str | None = None,
    filename: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset and apply the single global train/test split defined in
    :class:`config.Split`.

    Using the same seed and test fraction everywhere in the pipeline guarantees
    that cascade generation, feature scaling, and model evaluation all operate
    on the same held-out partition.

    Args:
        dataset:  Dataset name (e.g. ``"movielens"``, ``"ciao"``,
                  ``"epinions"``).  Defaults to ``Datasets.DEFAULT``.
        filename: Explicit path to a CSV file (legacy, overrides *dataset*).

    Returns:
        Tuple of ``(full_df, train_df, test_df)`` with LabelEncoder-encoded
        user and item IDs.
    """
    from config import Split  # local import to avoid circular dependency

    data = load_dataset(dataset=dataset, filename=filename)

    strategy = getattr(Split, "STRATEGY", "random")
    if strategy == "temporal":
        if "timestamp" not in data.columns:
            raise ValueError(
                "Temporal split requires a 'timestamp' column in the dataset. "
                "Load the dataset with timestamp support or set Split.STRATEGY='random'."
            )
        train_df, test_df = split_data_temporal(data, test_size=Split.TEST_SIZE)
        print(
            f"Global split (temporal, test_size={Split.TEST_SIZE}): "
            f"{len(train_df)} train / {len(test_df)} test ratings"
        )
    else:
        train_df, test_df = split_data_single(
            data, test_size=Split.TEST_SIZE, random_state=Split.RANDOM_STATE
        )
        print(
            f"Global split (random, seed={Split.RANDOM_STATE}): "
            f"{len(train_df)} train / {len(test_df)} test ratings"
        )
    return data, train_df, test_df


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------


def split_data_single(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a single random train/test split.

    Args:
        data:         Full ratings DataFrame.
        test_size:    Fraction of data to hold out for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    return train_test_split(data, test_size=test_size, random_state=random_state)  # type: ignore[return-value]


def split_data_temporal(
    data: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a temporal train/test split.

    Rows are sorted globally by ``timestamp`` and the last *test_size* fraction
    is held out as the test set.  This guarantees that every test interaction
    occurred *after* every training interaction in the cascade timeline,
    eliminating the temporal-leakage issue that arises with a random shuffle.

    Args:
        data:      Full ratings DataFrame.  Must contain a ``timestamp`` column.
        test_size: Fraction of rows to hold out (taken from the *end* of the
                   sorted sequence).

    Returns:
        Tuple of (train_df, test_df), both sorted by timestamp.
    """
    data_sorted = data.sort_values("timestamp").reset_index(drop=True)
    cutoff = int(len(data_sorted) * (1 - test_size))
    return data_sorted.iloc[:cutoff].copy(), data_sorted.iloc[cutoff:].copy()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def predict_ratings(model, test_data: pd.DataFrame) -> np.ndarray:
    """
    Generate predicted ratings for all rows in *test_data*.

    Args:
        model:     A fitted object that exposes a ``predict(UserIds, ItemIds)``
                   method (e.g. a :class:`cmfrec.CMF` instance).
        test_data: DataFrame with ``UserId`` and ``ItemId`` columns.

    Returns:
        1-D numpy array of predicted ratings aligned with *test_data* rows.
    """
    return model.predict(
        user=test_data["UserId"].values,
        item=test_data["ItemId"].values,
    )


def evaluate_single_split(
    model,
    test_data: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate a fitted model on *test_data*.

    Args:
        model:     Fitted recommender model.
        test_data: DataFrame with ``UserId``, ``ItemId``, ``Rating`` columns.

    Returns:
        Dict with keys ``rmse``, ``mae``, ``r2``.
    """
    predictions = predict_ratings(model, test_data)
    ground_truth = test_data["Rating"].values

    rmse = float(np.sqrt(mean_squared_error(ground_truth, predictions)))
    mae = float(mean_absolute_error(ground_truth, predictions))
    r2 = float(r2_score(ground_truth, predictions))

    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_ranking(
    model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    k: int = 10,
    rating_threshold: float | None = None,
) -> dict[str, float]:
    """
    Compute ranking metrics over the test set.

    For each user in *test_data*, all items **not** seen during training are
    scored.  Items in the test set with rating ≥ *rating_threshold* are
    treated as *relevant*.  When *rating_threshold* is ``None``, all test
    items for a user are treated as relevant (standard implicit-feedback
    setting).

    Metrics returned
    ----------------
    ndcg_at_k
        Normalised Discounted Cumulative Gain at *k*.
    precision_at_k
        Fraction of the top-*k* recommended items that are relevant.
    recall_at_k
        Fraction of the user's relevant items that appear in the top-*k*.
    mrr
        Mean Reciprocal Rank of the first relevant item in the ranked list.

    Args:
        model:             Fitted recommender model.
        train_data:        Training DataFrame (used to exclude already-rated
                           items from the candidate set).
        test_data:         Test DataFrame with ``UserId``, ``ItemId``,
                           ``Rating`` columns.
        k:                 Cut-off for rank-based metrics.
        rating_threshold:  Minimum rating to consider an item relevant.
                           ``None`` treats every test item as relevant.

    Returns:
        Dict with keys ``ndcg_at_k``, ``precision_at_k``, ``recall_at_k``,
        ``mrr``.
    """
    all_items = set(train_data["ItemId"].unique()) | set(test_data["ItemId"].unique())
    test_users = test_data["UserId"].unique()

    ndcg_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    rr_scores: list[float] = []

    for user_id in test_users:
        user_test = test_data[test_data["UserId"] == user_id]
        train_items = set(train_data.loc[train_data["UserId"] == user_id, "ItemId"])
        candidates = list(all_items - train_items)

        if not candidates:
            continue

        # Score all candidate items
        candidate_df = pd.DataFrame(
            {"UserId": [user_id] * len(candidates), "ItemId": candidates}
        )
        candidate_df["_score"] = predict_ratings(model, candidate_df)
        ranked = candidate_df.sort_values("_score", ascending=False)["ItemId"].tolist()

        # Determine relevant items
        if rating_threshold is not None:
            relevant = set(
                user_test.loc[user_test["Rating"] >= rating_threshold, "ItemId"]
            )
        else:
            relevant = set(user_test["ItemId"])

        if not relevant:
            continue

        top_k = ranked[:k]

        # Precision@K
        hits = [1 if item in relevant else 0 for item in top_k]
        precision_scores.append(sum(hits) / k)

        # Recall@K
        recall_scores.append(sum(hits) / len(relevant))

        # NDCG@K — binary relevance, ideal DCG assumes all relevant items at top
        dcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(hits))
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        # MRR — first hit position across the full ranked list
        rr = 0.0
        for rank, item in enumerate(ranked, start=1):
            if item in relevant:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    def _safe_mean(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    return {
        "ndcg_at_k": _safe_mean(ndcg_scores),
        "precision_at_k": _safe_mean(precision_scores),
        "recall_at_k": _safe_mean(recall_scores),
        "mrr": _safe_mean(rr_scores),
    }


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------


def make_recommendations(
    model,
    user_id: int,
    data: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the top-*n* item recommendations for *user_id*.

    Items already rated by the user are excluded from the candidates.  Items
    are scored with :func:`predict_ratings` and sorted descending.

    Args:
        model:   Fitted recommender model.
        user_id: Encoded user identifier.
        data:    Full ratings DataFrame (used to exclude known items).
        n:       Number of recommendations to return.

    Returns:
        DataFrame with columns ``ItemId`` and ``predicted_rating``, sorted by
        ``predicted_rating`` descending, length ≤ *n*.
    """
    rated_items = set(data.loc[data["UserId"] == user_id, "ItemId"])
    all_items = set(data["ItemId"].unique())
    candidates = list(all_items - rated_items)

    if not candidates:
        return pd.DataFrame(columns=["ItemId", "predicted_rating"])  # type: ignore[arg-type]

    candidate_df = pd.DataFrame(
        {
            "UserId": [user_id] * len(candidates),
            "ItemId": candidates,
        }
    )
    candidate_df["predicted_rating"] = predict_ratings(model, candidate_df)
    top_n = (
        candidate_df[["ItemId", "predicted_rating"]]  # type: ignore[call-overload]
        .sort_values("predicted_rating", ascending=False)  # type: ignore[call-overload]
        .head(n)
        .reset_index(drop=True)
    )
    return top_n
