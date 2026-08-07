"""P0-6: temporal and random split invariants."""

from __future__ import annotations

import unittest

from fixtures import ratings_frame
from recommender.data import (
    split_data_single,
    split_data_temporal,
    split_data_temporal_global,
    warm_test_slice,
)


class DataSplitTests(unittest.TestCase):
    def test_split_data_temporal_global_holds_out_latest_rows(self) -> None:
        data = ratings_frame(
            [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0), (2, 0, 5.0)],
            timestamps=[10, 20, 30, 40, 50],
        )
        train, test = split_data_temporal_global(data, test_size=0.4)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 2)
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())

    def test_split_data_temporal_is_per_user_leave_last(self) -> None:
        # Two users, 5 ratings each → ceil(0.4*5)=2 test each → 6 train / 4 test
        rows = [(0, i, float(i)) for i in range(5)] + [(1, i, float(i)) for i in range(5)]
        ts = list(range(10))
        data = ratings_frame(rows, timestamps=ts)
        train, test = split_data_temporal(data, test_size=0.4)
        self.assertEqual(len(train), 6)
        self.assertEqual(len(test), 4)
        for uid in (0, 1):
            u_train = train.loc[train["UserId"] == uid, "timestamp"]
            u_test = test.loc[test["UserId"] == uid, "timestamp"]
            self.assertEqual(len(u_test), 2)
            self.assertLess(u_train.max(), u_test.min())

    def test_split_data_temporal_preserves_user_temporal_order(self) -> None:
        data = ratings_frame([(0, i, float(i)) for i in range(6)], timestamps=list(range(6)))
        train, test = split_data_temporal(data, test_size=1 / 3)
        self.assertTrue(train["timestamp"].is_monotonic_increasing)
        self.assertTrue(test["timestamp"].is_monotonic_increasing)
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())

    def test_warm_test_slice_keeps_seen_user_item(self) -> None:
        train = ratings_frame([(0, 0, 1.0), (1, 1, 2.0)])
        test = ratings_frame([(0, 0, 3.0), (0, 2, 4.0), (2, 1, 5.0)])
        warm = warm_test_slice(train, test)
        self.assertEqual(len(warm), 1)
        self.assertEqual(int(warm.iloc[0]["UserId"]), 0)
        self.assertEqual(int(warm.iloc[0]["ItemId"]), 0)

    def test_split_data_single_respects_test_size(self) -> None:
        data = ratings_frame([(i, 0, 1.0) for i in range(100)])
        train, test = split_data_single(data, test_size=0.25, random_state=0)
        self.assertEqual(len(train), 75)
        self.assertEqual(len(test), 25)


if __name__ == "__main__":
    unittest.main()
