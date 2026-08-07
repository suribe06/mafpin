from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from recommender.enhanced import social_regularization as sr


class _FakeGraph:
    def __init__(self, edges: list[tuple[int, int]]) -> None:
        self._edges = edges

    def edges(self) -> list[tuple[int, int]]:
        return self._edges


class SocialEdgeNormalizationTests(unittest.TestCase):
    def _build_edges(self, normalization: sr.SocialNormalization) -> sr.SocialEdges:
        community_frame = pd.DataFrame(
            {
                "UserId": [1, 2, 3],
                "community_ids": ["1", "1", "1"],
                "is_boundary": [0, 0, 0],
            }
        ).set_index("UserId")
        fake_graph = _FakeGraph([(1, 2), (2, 3)])

        with (
            patch.object(sr, "load_community_frame", return_value=community_frame),
            patch.object(sr, "load_as_networkx", return_value=(fake_graph, None)),
            patch.object(sr, "directed_to_undirected", return_value=fake_graph),
        ):
            return sr.build_social_edges(
                dataset="movielens",
                model_name="exponential",
                network_index=0,
                user_index=[1, 2, 3],
                mode="uniform",
                normalization=normalization,
                dtype=np.float64,
            )

    def test_mean_alias_matches_mean_weight(self) -> None:
        mean_edges = self._build_edges("mean")
        mean_weight_edges = self._build_edges("mean_weight")

        self.assertEqual(mean_edges.normalization, "mean_weight")
        np.testing.assert_allclose(mean_edges.val, mean_weight_edges.val)
        np.testing.assert_allclose(mean_edges.val, np.array([1.0, 1.0]))

    def test_edges_alias_matches_n_edges(self) -> None:
        edges_alias = self._build_edges("edges")
        n_edges = self._build_edges("n_edges")

        self.assertEqual(edges_alias.normalization, "n_edges")
        np.testing.assert_allclose(edges_alias.val, n_edges.val)
        np.testing.assert_allclose(edges_alias.val, np.array([0.5, 0.5]))

    def test_sum_weight_normalization_sums_to_one(self) -> None:
        edges = self._build_edges("sum_weight")

        self.assertEqual(edges.normalization, "sum_weight")
        self.assertAlmostEqual(float(edges.val.sum()), 1.0)
        np.testing.assert_allclose(edges.val, np.array([0.5, 0.5]))

    def test_normalized_laplacian_uses_endpoint_degrees(self) -> None:
        edges = self._build_edges("normalized_laplacian")

        expected = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])

        self.assertEqual(edges.normalization, "normalized_laplacian")
        np.testing.assert_allclose(edges.val, expected)

    def test_none_keeps_raw_weights(self) -> None:
        edges = self._build_edges("none")

        self.assertEqual(edges.normalization, "none")
        np.testing.assert_allclose(edges.val, np.array([1.0, 1.0]))

    def test_invalid_normalization_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown normalization"):
            sr.build_social_edges(
                dataset="movielens",
                model_name="exponential",
                network_index=0,
                user_index=[1, 2, 3],
                mode="uniform",
                normalization="bad",  # type: ignore[arg-type]
            )

    def test_boundary_mechanism_stats_detects_downweight(self) -> None:
        # Users 1-2 share a community; user 2 is a strong boundary → edge downweighted.
        community_frame = pd.DataFrame(
            {
                "UserId": [1, 2, 3],
                "community_ids": ["1", "1", ""],
                "lph_score": [1.0, -5.0, 1.0],
            }
        ).set_index("UserId")
        fake_graph = _FakeGraph([(1, 2), (1, 3)])

        with (
            patch.object(sr, "load_community_frame", return_value=community_frame),
            patch.object(sr, "load_as_networkx", return_value=(fake_graph, None)),
            patch.object(sr, "directed_to_undirected", return_value=fake_graph),
        ):
            stats = sr.compute_boundary_mechanism_stats(
                dataset="movielens",
                model_name="exponential",
                network_index=0,
                user_index=[1, 2, 3],
                beta=1.0,
            )

        self.assertEqual(stats["n_edges"], 2)
        self.assertGreater(float(stats["frac_endpoint_has_community"]), 0.0)
        self.assertGreaterEqual(float(stats["frac_downweighted_given_jaccard"]), 1.0)
        self.assertEqual(int(stats["n_jaccard_positive"]), 1)


if __name__ == "__main__":
    unittest.main()
