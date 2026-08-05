"""Unit tests for Route B soft assignment and CCE helpers."""

from __future__ import annotations

import unittest
from collections import Counter

import pandas as pd

from recommender.experiment.route_b.beyond_accuracy import cce_at_k, coverage_at_k
from recommender.experiment.route_b.soft_assignment import (
    soft_assign_user,
    soft_community_feature_frame,
)


class SoftAssignmentTests(unittest.TestCase):
    def test_soft_assign_prefers_overlapping_community(self) -> None:
        profiles = {
            1: Counter({10: 5, 11: 3}),
            2: Counter({20: 5, 21: 3}),
        }
        weights = soft_assign_user({10, 11, 99}, profiles, top_k=2)
        self.assertIn(1, weights)
        self.assertGreater(weights[1], weights.get(2, 0.0))
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_soft_frame_only_few_shot_users(self) -> None:
        train = pd.DataFrame(
            {
                "UserId": [0, 0, 1, 1, 1, 2, 2],
                "ItemId": [10, 11, 10, 20, 21, 10, 11],
                "Rating": [4.0] * 7,
            }
        )
        com = pd.DataFrame(
            {
                "UserId": [0, 1],
                "community_ids": ["1", "1;2"],
            }
        )
        # user 2 has 2 ratings → few-shot; users 0/1 are also ≤10
        frame = soft_community_feature_frame(
            train, com, user_ids=[0, 1, 2, 3], max_train_ratings=10
        )
        self.assertEqual(frame.loc[3, "soft_community_assigned"], 0.0)
        self.assertGreaterEqual(frame.loc[2, "soft_community_assigned"], 0.0)


class BeyondAccuracyHelperTests(unittest.TestCase):
    def test_cce_and_coverage(self) -> None:
        cce = cce_at_k(
            [1, 2, 3],
            user_coms={10},
            item_dominant={1: {10}, 2: {99}, 3: {99}},
        )
        self.assertAlmostEqual(cce, 2 / 3)
        cov = coverage_at_k({0: [1, 2], 1: [2, 3]}, n_items=5)
        self.assertAlmostEqual(cov, 3 / 5)


if __name__ == "__main__":
    unittest.main()
