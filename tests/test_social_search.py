"""P1-4..P1-5: Optuna search guards and data-leakage prevention."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fixtures import ratings_frame, user_attributes_frame
from recommender.enhanced.social_search import _metrics_are_usable, _prepare_search_data


class SocialSearchTests(unittest.TestCase):
    def test_metrics_are_usable_accepts_finite_reasonable_rmse(self) -> None:
        self.assertTrue(_metrics_are_usable({"rmse": 0.9, "mae": 0.7, "r2": 0.1}, 10.0))

    def test_metrics_are_usable_rejects_non_finite(self) -> None:
        self.assertFalse(
            _metrics_are_usable({"rmse": float("inf"), "mae": 1.0, "r2": 0.0}, 10.0)
        )

    def test_metrics_are_usable_rejects_unreasonable_rmse(self) -> None:
        self.assertFalse(_metrics_are_usable({"rmse": 50.0, "mae": 1.0, "r2": 0.0}, 10.0))

    def test_metrics_are_usable_rejects_empty_metrics(self) -> None:
        self.assertFalse(_metrics_are_usable({}, 10.0))

    def test_prepare_search_data_uses_external_train_split_only(self) -> None:
        global_train = ratings_frame(
            [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0), (2, 0, 5.0), (2, 1, 6.0)]
        )
        leaked_test_only = ratings_frame([(99, 99, 5.0)])
        user_attrs = user_attributes_frame([0, 1, 2])

        with (
            patch(
                "recommender.enhanced.social_search.load_network_features",
                return_value=user_attrs,
            ),
            patch(
                "recommender.enhanced.social_search.load_dataset",
                return_value=leaked_test_only,
            ) as load_mock,
        ):
            inner_train, inner_test, attrs = _prepare_search_data(
                dataset="movielens",
                model_name="exponential",
                network_index=0,
                max_ratings=0,
                test_size=0.34,
                random_state=0,
                train_df=global_train,
            )

        load_mock.assert_not_called()
        self.assertIs(attrs, user_attrs)
        self.assertTrue(set(inner_train["UserId"]).issubset({0, 1, 2}))
        self.assertTrue(set(inner_test["UserId"]).issubset({0, 1, 2}))
        self.assertNotIn(99, set(inner_test["UserId"]))

    def test_prepare_search_data_without_external_train_loads_dataset(self) -> None:
        full_data = ratings_frame(
            [
                (0, 0, 1.0),
                (0, 1, 2.0),
                (0, 2, 3.0),
                (1, 0, 4.0),
                (1, 1, 5.0),
                (1, 2, 6.0),
            ]
        )
        user_attrs = user_attributes_frame([0, 1])

        with (
            patch(
                "recommender.enhanced.social_search.load_network_features",
                return_value=user_attrs,
            ),
            patch(
                "recommender.enhanced.social_search.load_dataset",
                return_value=full_data,
            ) as load_mock,
        ):
            inner_train, inner_test, _ = _prepare_search_data(
                dataset="movielens",
                model_name="exponential",
                network_index=0,
                max_ratings=0,
                test_size=0.34,
                random_state=0,
                train_df=None,
            )

        load_mock.assert_called_once()
        self.assertFalse(inner_train.empty)
        self.assertFalse(inner_test.empty)


if __name__ == "__main__":
    unittest.main()
