"""P0-6: temporal and random split invariants."""

from __future__ import annotations

import unittest

from fixtures import ratings_frame
from recommender.data import split_data_single, split_data_temporal


class DataSplitTests(unittest.TestCase):
    def test_split_data_temporal_holds_out_latest_rows(self) -> None:
        data = ratings_frame(
            [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0), (2, 0, 5.0)],
            timestamps=[10, 20, 30, 40, 50],
        )
        train, test = split_data_temporal(data, test_size=0.4)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 2)
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())

    def test_split_data_temporal_preserves_order_within_splits(self) -> None:
        data = ratings_frame([(0, i, float(i)) for i in range(6)], timestamps=list(range(6)))
        train, test = split_data_temporal(data, test_size=1 / 3)
        self.assertTrue(train["timestamp"].is_monotonic_increasing)
        self.assertTrue(test["timestamp"].is_monotonic_increasing)

    def test_split_data_single_respects_test_size(self) -> None:
        data = ratings_frame([(i, 0, 1.0) for i in range(100)])
        train, test = split_data_single(data, test_size=0.25, random_state=0)
        self.assertEqual(len(train), 75)
        self.assertEqual(len(test), 25)


if __name__ == "__main__":
    unittest.main()
