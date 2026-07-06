"""P1-1..P1-3: social edge weighting modes and community parsing."""

from __future__ import annotations

import unittest

import pandas as pd

from recommender.enhanced import social_regularization as sr


class SocialModeTests(unittest.TestCase):
    def test_edge_weight_uniform_is_one(self) -> None:
        weight = sr._edge_weight(1, 2, {1: {1}, 2: {2}}, {1: 0.0, 2: 1.0}, "uniform", 0.5, 1.0)
        self.assertEqual(weight, 1.0)

    def test_edge_weight_community_jaccard_uses_overlap(self) -> None:
        communities = {1: {1, 2}, 2: {1, 2}, 3: {3}}
        weight = sr._edge_weight(1, 2, communities, {}, "community_jaccard", 0.5, 1.0)
        self.assertAlmostEqual(weight, 1.0)
        weight_disjoint = sr._edge_weight(1, 3, communities, {}, "community_jaccard", 0.5, 1.0)
        self.assertAlmostEqual(weight_disjoint, 0.0)

    def test_edge_weight_boundary_downweight_reduces_boundary_pairs(self) -> None:
        communities = {1: {1, 2}, 2: {1, 3}}
        boundary = {1: 0.0, 2: 1.0}
        shared = sr._jaccard(communities[1], communities[2])
        self.assertGreater(shared, 0.0)
        weight = sr._edge_weight(
            1, 2, communities, boundary, "boundary_downweight", beta=0.5, gamma=1.0
        )
        self.assertLess(weight, shared)
        self.assertAlmostEqual(weight, shared * 0.5)

    def test_edge_weight_bridge_preserve_is_sigmoid(self) -> None:
        communities = {1: {1, 2}, 2: {1, 2}}
        boundary = {1: 0.0, 2: 0.0}
        weight = sr._edge_weight(
            1, 2, communities, boundary, "bridge_preserve", beta=0.5, gamma=1.0
        )
        self.assertGreater(weight, 0.0)
        self.assertLess(weight, 1.0)

    def test_edge_weight_unknown_mode_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown social mode"):
            sr._edge_weight(1, 2, {}, {}, "bad", 0.5, 1.0)  # type: ignore[arg-type]


class BoundaryIntensityTests(unittest.TestCase):
    def test_boundary_intensity_prefers_lph_score(self) -> None:
        frame = pd.DataFrame(
            {"lph_score": [0.0, -2.0]},
            index=pd.Index([1, 2]),
        )
        scores = sr._boundary_intensity(frame)
        self.assertAlmostEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], 1.0)

    def test_boundary_intensity_falls_back_to_is_boundary(self) -> None:
        frame = pd.DataFrame({"is_boundary": [0.0, 1.0]}, index=pd.Index([1, 2]))
        scores = sr._boundary_intensity(frame)
        self.assertAlmostEqual(scores[2], 1.0)

    def test_boundary_intensity_defaults_to_zero(self) -> None:
        frame = pd.DataFrame({"other": [1.0]}, index=pd.Index([1]))
        scores = sr._boundary_intensity(frame)
        self.assertAlmostEqual(scores[1], 0.0)


class CommunityParsingTests(unittest.TestCase):
    def test_parse_community_sets_splits_semicolons(self) -> None:
        frame = pd.DataFrame({"community_ids": ["1;2", "3", ""]}, index=pd.Index([10, 20, 30]))
        parsed = sr._parse_community_sets(frame)
        self.assertEqual(parsed[10], {1, 2})
        self.assertEqual(parsed[20], {3})
        self.assertEqual(parsed[30], set())

    def test_parse_community_sets_empty_when_column_missing(self) -> None:
        frame = pd.DataFrame({"x": [1]}, index=pd.Index([5]))
        parsed = sr._parse_community_sets(frame)
        self.assertEqual(parsed[5], set())


if __name__ == "__main__":
    unittest.main()
