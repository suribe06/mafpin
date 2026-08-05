"""Tests for seeded overlapping community detection."""

from __future__ import annotations

import random
import unittest

import networkx as nx

from networks.communities.detection import detect_overlapping_communities


def _two_cliques_bridged() -> nx.Graph:
    """Two 8-cliques joined by a bridge: enough triangles for Demon to find them."""
    graph = nx.disjoint_union(nx.complete_graph(8), nx.complete_graph(8))
    graph.add_edge(0, 8)
    return graph


class CommunityDetectionSeedTests(unittest.TestCase):
    def test_same_seed_gives_identical_communities(self) -> None:
        graph = _two_cliques_bridged()
        first = detect_overlapping_communities(graph, seed=7)
        second = detect_overlapping_communities(graph, seed=7)
        self.assertEqual(
            sorted(sorted(c) for c in first), sorted(sorted(c) for c in second)
        )

    def test_detection_leaves_global_rng_untouched(self) -> None:
        random.seed(123)
        expected = [random.random() for _ in range(3)]

        random.seed(123)
        detect_overlapping_communities(_two_cliques_bridged(), seed=7)
        self.assertEqual([random.random() for _ in range(3)], expected)


if __name__ == "__main__":
    unittest.main()
