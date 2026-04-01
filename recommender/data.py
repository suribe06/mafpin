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
predict_ratings
    Call a fitted CMF model to obtain predicted ratings.
evaluate_single_split
    Evaluate a fitted model on a test set, returning RMSE, MAE, and R².
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

from config import Paths


# ---------------------------------------------------------------------------
# Loading and encoding
# ---------------------------------------------------------------------------


def load_dataset(filename: str | Path | None = None) -> pd.DataFrame:
    """
    Load a ratings CSV and return a DataFrame with encoded integer IDs.

    The CSV must have at least three columns in order: UserId, ItemId, Rating.
    Both UserId and ItemId are re-encoded to zero-based consecutive integers
    via :class:`sklearn.preprocessing.LabelEncoder`.

    Args:
        filename: Path to the CSV file.  Defaults to
            ``Paths.DATA / "ratings_small.csv"``.

    Returns:
        DataFrame with columns ``UserId`` (int), ``ItemId`` (int),
        ``Rating`` (float).
    """
    if filename is None:
        filename = Paths.DATA / "ratings_small.csv"
    filename = Path(filename)

    data = pd.read_csv(filename, usecols=[0, 1, 2])  # type: ignore[call-overload]
    data.columns = pd.Index(["UserId", "ItemId", "Rating"])

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
    filename: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset and apply the single global train/test split defined in
    :class:`config.Split`.

    Using the same seed and test fraction everywhere in the pipeline guarantees
    that cascade generation, feature scaling, and model evaluation all operate
    on the same held-out partition.

    Args:
        filename: Path to the CSV file.  Defaults to
            ``Paths.DATA / "ratings_small.csv"``.

    Returns:
        Tuple of ``(full_df, train_df, test_df)`` with LabelEncoder-encoded
        user and item IDs.
    """
    from config import Split  # local import to avoid circular dependency

    data = load_dataset(filename)
    train_df, test_df = split_data_single(
        data, test_size=Split.TEST_SIZE, random_state=Split.RANDOM_STATE
    )
    print(
        f"Global split (seed={Split.RANDOM_STATE}): "
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
