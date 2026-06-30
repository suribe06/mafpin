"""Non-finite prediction handling in evaluate_single_split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data import evaluate_single_split, rating_reasonableness_limit


class _InfPredictor:
    def predict(self, user, item):  # noqa: ANN001
        return np.full(len(user), np.inf, dtype=float)


def test_evaluate_single_split_non_finite_predictions() -> None:
    test_df = pd.DataFrame({"UserId": [0, 1], "ItemId": [0, 1], "Rating": [3.0, 4.0]})
    metrics = evaluate_single_split(_InfPredictor(), test_df)
    assert metrics["rmse"] == float("inf")
    assert metrics["mae"] == float("inf")
    assert metrics["r2"] == float("-inf")


def test_rating_reasonableness_limit() -> None:
    assert rating_reasonableness_limit(pd.Series([0.5, 5.0])) == 45.0


if __name__ == "__main__":
    test_evaluate_single_split_non_finite_predictions()
    test_rating_reasonableness_limit()
    print("ok")
