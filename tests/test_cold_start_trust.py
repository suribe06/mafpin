"""Trust feature ID alignment tests."""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from recommender.experiment.cold_start.trust_variants import (
    encoded_trust_user_ids,
    map_trust_features_to_encoded,
)


class TrustIdAlignTests(unittest.TestCase):
    def test_map_trust_features_to_encoded(self) -> None:
        user_enc = LabelEncoder()
        user_enc.fit([10, 20, 30])
        trust = pd.DataFrame(
            {
                "trust_in_degree": [2, 0],
                "trust_out_degree": [1, 3],
                "trust_pagerank": [0.2, 0.1],
            },
            index=pd.Index([10, 99], name="UserId"),
        )
        mapped = map_trust_features_to_encoded(trust, user_enc)
        self.assertEqual(list(mapped.index), [0])  # 10 → 0
        self.assertEqual(int(mapped.loc[0, "trust_in_degree"]), 2)

    def test_encoded_trust_user_ids(self) -> None:
        user_enc = LabelEncoder()
        user_enc.fit([5, 7])
        G = nx.DiGraph()
        G.add_edge(5, 7)
        G.add_edge(7, 99)  # 99 not in ratings encoder
        ids = encoded_trust_user_ids(user_enc, G)
        self.assertEqual(ids, {0, 1})


if __name__ == "__main__":
    unittest.main()
