"""Non-finite prediction handling in evaluate_single_split."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from recommender.data import evaluate_single_split, metrics_are_reasonable, rating_reasonableness_limit


class _InfPredictor:
    def predict(self, user, item):  # noqa: ANN001
        return np.full(len(user), np.inf, dtype=float)


class EvaluateFiniteTests(unittest.TestCase):
    def test_evaluate_single_split_non_finite_predictions(self) -> None:
        test_df = pd.DataFrame({"UserId": [0, 1], "ItemId": [0, 1], "Rating": [3.0, 4.0]})
        metrics = evaluate_single_split(_InfPredictor(), test_df)
        self.assertEqual(metrics["rmse"], float("inf"))
        self.assertEqual(metrics["mae"], float("inf"))
        self.assertEqual(metrics["r2"], float("-inf"))

    def test_rating_reasonableness_limit(self) -> None:
        self.assertEqual(rating_reasonableness_limit(pd.Series([0.5, 5.0])), 45.0)

    def test_metrics_are_reasonable_rejects_degenerate_rmse(self) -> None:
        ratings = pd.Series([1.0, 5.0])
        self.assertTrue(metrics_are_reasonable({"rmse": 0.9}, ratings))
        self.assertFalse(metrics_are_reasonable({"rmse": 1e28}, ratings))
        self.assertFalse(metrics_are_reasonable({"rmse": float("inf")}, ratings))


if __name__ == "__main__":
    unittest.main()
