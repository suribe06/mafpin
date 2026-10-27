"""Unit tests for the paper LPH implementation."""

from __future__ import annotations

import unittest

import networkx as nx

from networks.communities.lph import (
    _compute_network_homophily,
    compute_lph_paper,
)


class LocalPluralisticHomophilyTests(unittest.TestCase):
    def test_global_h_is_invariant_to_edge_insertion_order(self) -> None:
        edges = [(0, 1), (0, 2), (1, 3)]
        s = {0: 2, 1: 1, 2: 0, 3: 2}

        first = nx.Graph()
        first.add_edges_from(edges)
        second = nx.Graph()
        second.add_edges_from((v, u) for u, v in reversed(edges))

        self.assertAlmostEqual(
            _compute_network_homophily(first, s),
            _compute_network_homophily(second, s),
        )

    def test_global_h_matches_networkx_numeric_assortativity(self) -> None:
        graph = nx.Graph([(0, 1), (0, 2), (1, 3)])
        s = {0: 2, 1: 1, 2: 0, 3: 2}
        nx.set_node_attributes(graph, s, "s")

        self.assertAlmostEqual(
            _compute_network_homophily(graph, s),
            nx.numeric_assortativity_coefficient(graph, "s"),
        )

    def test_h_tilde_sums_to_global_h(self) -> None:
        graph = nx.Graph([(0, 1), (0, 2), (1, 3)])
        membership = {
            0: {0, 1},
            1: {0},
            2: {1},
            3: {0, 2},
        }

        h_tilde, s, _ = compute_lph_paper(graph, membership)

        self.assertAlmostEqual(
            sum(h_tilde.values()),
            _compute_network_homophily(graph, s),
        )


if __name__ == "__main__":
    unittest.main()
