"""Unit tests for cold-start splits, strata, and deltas."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from recommender.experiment.cold_start.deltas import (
    bootstrap_mean_ci,
    per_user_rmse,
)
from recommender.experiment.cold_start.paths import ColdStartPaths
from recommender.experiment.cold_start.splits import (
    dedupe_ratings,
    per_user_chrono_split,
    per_user_leave_k_split,
    zero_shot_trust_split,
)
from recommender.experiment.cold_start.strata import (
    assign_stratum,
    build_user_strata,
)
from recommender.experiment.variants import (
    COLD_START_VARIANT_IDS,
    TRUST_VARIANT_IDS,
    VARIANT_SPECS,
)
from tests.fixtures import ratings_frame


class StratumTests(unittest.TestCase):
    def test_assign_stratum_boundaries(self) -> None:
        self.assertEqual(assign_stratum(0), "0")
        self.assertEqual(assign_stratum(1), "1-3")
        self.assertEqual(assign_stratum(3), "1-3")
        self.assertEqual(assign_stratum(4), "4-10")
        self.assertEqual(assign_stratum(10), "4-10")
        self.assertEqual(assign_stratum(11), ">10")

    def test_build_user_strata_counts(self) -> None:
        train = ratings_frame(
            [(0, 1, 4.0), (0, 2, 3.0), (1, 1, 5.0)],
            timestamps=[1, 2, 3],
        )
        test = ratings_frame(
            [(0, 3, 2.0), (1, 2, 1.0), (2, 1, 3.0)],
            timestamps=[4, 5, 6],
        )
        strata = build_user_strata(train, test)
        by_user = strata.set_index("user_id")
        self.assertEqual(int(by_user.loc[0, "n_train_ratings"]), 2)
        self.assertEqual(by_user.loc[0, "stratum"], "1-3")
        self.assertEqual(int(by_user.loc[2, "n_train_ratings"]), 0)
        self.assertEqual(by_user.loc[2, "stratum"], "0")


class SplitTests(unittest.TestCase):
    def test_per_user_chrono_leave_last(self) -> None:
        data = ratings_frame(
            [
                (0, 1, 1.0),
                (0, 2, 2.0),
                (0, 3, 3.0),
                (0, 4, 4.0),
                (0, 5, 5.0),
                (1, 1, 1.0),
            ],
            timestamps=[10, 20, 30, 40, 50, 1],
        )
        train, test = per_user_chrono_split(data, test_frac=0.2)
        # user 0: N=5 → n_test=ceil(1)=1 → last rating in test
        self.assertEqual(len(test[test["UserId"] == 0]), 1)
        self.assertEqual(int(test[test["UserId"] == 0]["ItemId"].iloc[0]), 5)
        self.assertEqual(len(train[train["UserId"] == 0]), 4)
        # user 1: N=1 → all in test
        self.assertTrue(train[train["UserId"] == 1].empty)
        self.assertEqual(len(test[test["UserId"] == 1]), 1)

    def test_zero_shot_trust_split(self) -> None:
        data = ratings_frame(
            [(0, 1, 4.0), (1, 1, 3.0), (2, 1, 2.0)],
            timestamps=[1, 2, 3],
        )
        train, test, zero_users = zero_shot_trust_split(
            data, encoded_trust_users={1, 2}
        )
        self.assertEqual(sorted(zero_users), [1, 2])
        self.assertEqual(set(train["UserId"]), {0})
        self.assertEqual(set(test["UserId"]), {1, 2})

    def test_dedupe_exact_user_item_ts(self) -> None:
        data = ratings_frame(
            [(0, 1, 4.0), (0, 1, 5.0), (0, 2, 3.0)],
            timestamps=[10, 10, 20],
        )
        out = dedupe_ratings(data)
        self.assertEqual(len(out), 2)
        self.assertEqual(float(out.iloc[0]["Rating"]), 4.0)

    def test_chrono_stable_tiebreak(self) -> None:
        # Same timestamp: later row order goes later in chrono → last = test.
        data = ratings_frame(
            [(0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0), (0, 4, 4.0), (0, 5, 5.0)],
            timestamps=[10, 10, 10, 10, 10],
        )
        train, test = per_user_chrono_split(data, test_frac=0.2)
        self.assertEqual(int(test["ItemId"].iloc[0]), 5)
        self.assertEqual(list(train["ItemId"]), [1, 2, 3, 4])

    def test_leave_k_populates_cold_strata(self) -> None:
        # 8 users × 20 ratings → after leave-last 20%, early=16; caps 0/2/7/all.
        rows = []
        ts = []
        t = 0
        for uid in range(8):
            for item in range(20):
                rows.append((uid, item, 3.0))
                ts.append(t)
                t += 1
        data = ratings_frame(rows, timestamps=ts)
        train, test, meta = per_user_leave_k_split(
            data, test_frac=0.2, seed=0
        )
        self.assertEqual(meta["n_users_cap_0"], 2)
        self.assertEqual(meta["n_users_cap_2"], 2)
        self.assertEqual(meta["n_users_cap_7"], 2)
        self.assertEqual(meta["n_users_cap_all"], 2)
        strata = build_user_strata(train, test)
        counts = strata["stratum"].value_counts().to_dict()
        self.assertEqual(counts.get("0", 0), 2)
        self.assertEqual(counts.get("1-3", 0), 2)
        self.assertEqual(counts.get("4-10", 0), 2)
        self.assertEqual(counts.get(">10", 0), 2)
        # Unused early ratings are dropped, not leaked into test.
        n_full = len(data)
        self.assertLess(len(train) + len(test), n_full)
        self.assertGreater(meta["dropped_early_ratings"], 0)


class DeltaTests(unittest.TestCase):
    def test_per_user_rmse(self) -> None:
        test = ratings_frame([(0, 1, 4.0), (0, 2, 2.0), (1, 1, 5.0)])
        y_true = np.array([4.0, 2.0, 5.0])
        y_pred = np.array([3.0, 1.0, 5.0])
        rmse = per_user_rmse(test, y_true, y_pred)
        self.assertAlmostEqual(float(rmse.loc[0]), math.sqrt(1.0))
        self.assertAlmostEqual(float(rmse.loc[1]), 0.0)

    def test_bootstrap_mean_ci_nonempty(self) -> None:
        stats = bootstrap_mean_ci(np.array([1.0, 2.0, 3.0]), n_samples=200, seed=0)
        self.assertEqual(stats["n"], 3.0)
        self.assertTrue(stats["ci_low"] <= stats["mean"] <= stats["ci_high"])


class PathsAndVariantTests(unittest.TestCase):
    def test_cold_start_paths_root(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            paths = ColdStartPaths("movielens", root=Path(tmp) / "cs")
            paths.ensure_dirs()
            self.assertTrue(paths.BASE.exists())
            self.assertEqual(paths.RESULTS.name, "cold_start_results.csv")

    def test_cold_start_paths_resolves_relative_output_dir(self) -> None:
        paths = ColdStartPaths("movielens", root="data/movielens/cold_start")
        self.assertTrue(paths.CASCADES.is_absolute())
        self.assertTrue(str(paths.CASCADES).endswith("data/movielens/cold_start/cascades.txt"))

    def test_trust_variant_specs_exist(self) -> None:
        self.assertIn("M2_trust", VARIANT_SPECS)
        self.assertTrue(VARIANT_SPECS["M2_trust"]["trust_features"])
        self.assertEqual(COLD_START_VARIANT_IDS[0], "M1")
        self.assertIn("M3_trust", TRUST_VARIANT_IDS)

    def test_remap_selected_network_picks_closest_alpha(self) -> None:
        import tempfile
        from pathlib import Path

        from recommender.experiment.cold_start.features import (
            remap_selected_network_to_paths,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "inferred_networks" / "rayleigh"
            model_dir.mkdir(parents=True)
            (model_dir / "inferred_edges_ray.csv").write_text(
                "alpha|inferred_edges_ray\n"
                "0.0001|10\n"
                "0.001|20\n"
                "0.002|30\n"
                "0.01|5\n"
                "0.1|1\n",
                encoding="utf-8",
            )
            for i in range(5):
                (model_dir / f"inferred-network-ray-{i:03d}.txt").write_text(
                    "0,0\n", encoding="utf-8"
                )

            class _P:
                NETWORKS = root / "inferred_networks"

            remapped = remap_selected_network_to_paths(
                {
                    "diffusion_model": "rayleigh",
                    "alpha_index": 63,
                    "alpha_value": 0.002014,
                },
                _P(),  # type: ignore[arg-type]
            )
            self.assertEqual(remapped["alpha_index"], 2)
            self.assertAlmostEqual(float(remapped["alpha_value"]), 0.002)


class ReportLogicTests(unittest.TestCase):
    def _rows(self, cells: list[tuple]) -> list[dict]:
        rows = []
        for stratum, n, m1, m3 in cells:
            for variant, rmse in (("M1", m1), ("M2", m1), ("M3", m3)):
                rows.append(
                    {
                        "model_variant": variant,
                        "stratum": stratum,
                        "n_users": n,
                        "n_ratings": n * 5,
                        "rmse": rmse,
                    }
                )
        return rows

    def test_h1_fails_when_m3_worse_everywhere(self) -> None:
        from recommender.experiment.cold_start.report import build_success_summary

        text = build_success_summary(
            pd.DataFrame(
                self._rows(
                    [
                        ("0", 50, 1.0, 1.1),
                        ("1-3", 40, 1.0, 1.05),
                        ("4-10", 40, 1.0, 1.02),
                        (">10", 40, 1.0, 1.2),
                    ]
                )
            ),
            None,
            dataset="toy",
            mode="diagnostic",
        )
        self.assertIn("H1-gain", text)
        self.assertIn("H1-stronger", text)
        self.assertIn(
            "H1-gain** (M3 beats M1 in cold, Δ>0): FAIL/INCONCLUSIVE", text
        )

    def test_h1_gain_pass_stronger_fail(self) -> None:
        from recommender.experiment.cold_start.report import build_success_summary

        # Cold Δ>0 but warm Δ larger → gain PASS, stronger FAIL.
        text = build_success_summary(
            pd.DataFrame(
                self._rows(
                    [
                        ("1-3", 40, 1.2, 1.0),
                        ("4-10", 40, 1.2, 1.0),
                        (">10", 40, 1.5, 1.0),
                    ]
                )
            ),
            None,
            dataset="toy",
            mode="controlled",
        )
        self.assertIn("H1-gain** (M3 beats M1 in cold, Δ>0): PASS", text)
        self.assertIn("H1-stronger** (cold gain > warm gain): FAIL/INCONCLUSIVE", text)
        self.assertIn("general side-info benefit", text)
        self.assertIn("force cold strata", text)

    def test_h1_skips_tiny_strata(self) -> None:
        from recommender.experiment.cold_start.report import build_success_summary

        text = build_success_summary(
            pd.DataFrame(
                self._rows(
                    [
                        ("1-3", 1, 1.0, 0.5),
                        ("4-10", 1, 1.0, 0.5),
                        (">10", 50, 1.0, 0.95),
                    ]
                )
            ),
            None,
            dataset="toy",
            mode="diagnostic",
        )
        self.assertIn("N=1", text)
        self.assertIn("FAIL/INCONCLUSIVE", text)

    def test_controlled_empty_stratum_mentions_leave_k(self) -> None:
        from recommender.experiment.cold_start.report import build_success_summary

        text = build_success_summary(
            pd.DataFrame(
                self._rows(
                    [
                        ("1-3", 0, 1.0, 0.9),
                        ("4-10", 0, 1.0, 0.9),
                        (">10", 50, 1.0, 0.95),
                    ]
                )
            ),
            None,
            dataset="toy",
            mode="controlled",
        )
        self.assertIn("leave_k", text)
        self.assertNotIn("global temporal diagnostic", text)

    def test_zero_shot_h4_uses_trust_variants(self) -> None:
        from recommender.experiment.cold_start.report import build_success_summary

        rows = []
        for variant, rmse in (
            ("M1", 1.2),
            ("M2_trust", 1.0),
            ("M3_trust", 1.25),
        ):
            rows.append(
                {
                    "model_variant": variant,
                    "stratum": "0",
                    "n_users": 100,
                    "n_ratings": 500,
                    "rmse": rmse,
                }
            )
        text = build_success_summary(
            pd.DataFrame(rows), None, dataset="ciao", mode="zero_shot_trust"
        )
        self.assertIn("H4-M2_trust", text)
        self.assertIn("H4-M2_trust** (beats M1 on stratum `0`): PASS", text)
        self.assertIn("H4-M3_trust** (beats M1 on stratum `0`): FAIL/INCONCLUSIVE", text)
        self.assertIn("M2_trust", text)
        self.assertNotIn("leave_k", text)



if __name__ == "__main__":
    unittest.main()
