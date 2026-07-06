"""P0-7: ranking metrics with a deterministic mock predictor."""

from __future__ import annotations

import unittest

import numpy as np

from fixtures import ratings_frame
from recommender.data import evaluate_ranking


class _ScoreByItemPredictor:
    def predict(self, user, item):  # noqa: ANN001
        return np.asarray(item, dtype=float)


class DataRankingTests(unittest.TestCase):
    def test_evaluate_ranking_perfect_ndcg_when_top_item_is_relevant(self) -> None:
        train = ratings_frame([(0, 0, 5.0), (0, 1, 4.0)])
        test = ratings_frame([(0, 2, 5.0)])
        metrics = evaluate_ranking(_ScoreByItemPredictor(), train, test, k=2)
        self.assertEqual(metrics["ndcg_at_k"], 1.0)
        self.assertEqual(metrics["precision_at_k"], 0.5)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_evaluate_ranking_mrr_finds_first_relevant_in_full_list(self) -> None:
        train = ratings_frame([(0, 0, 5.0)])
        test = ratings_frame([(0, 1, 5.0), (0, 3, 5.0)])
        metrics = evaluate_ranking(_ScoreByItemPredictor(), train, test, k=1)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_evaluate_ranking_returns_zeros_when_no_candidates(self) -> None:
        train = ratings_frame([(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0)])
        test = ratings_frame([(0, 0, 1.0)])
        metrics = evaluate_ranking(_ScoreByItemPredictor(), train, test, k=5)
        self.assertEqual(metrics["ndcg_at_k"], 0.0)
        self.assertEqual(metrics["precision_at_k"], 0.0)


if __name__ == "__main__":
    unittest.main()
