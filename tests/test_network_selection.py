"""P0-1..P0-3: network selection and alpha grid helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fixtures import FakeDatasetPaths, sample_manifest_entry, write_alpha_grid_csv
from recommender.experiment.network_selection import (
    _pick_from_network_best,
    _pick_from_run_csv,
    resolve_alpha_index,
    run_network_selection,
)


class NetworkSelectionTests(unittest.TestCase):
    def test_resolve_alpha_index_picks_nearest_grid_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dp = FakeDatasetPaths(root, "movielens")
            csv_path = dp.NETWORKS / "exponential" / "inferred_edges_expo.csv"
            write_alpha_grid_csv(csv_path, [0.1, 0.2, 0.3])

            def _paths(dataset: str) -> FakeDatasetPaths:
                return FakeDatasetPaths(root, dataset)

            with patch("recommender.experiment.network_selection.DatasetPaths", _paths):
                idx, alpha = resolve_alpha_index("movielens", "exponential", 0.19)
            self.assertEqual(idx, 1)
            self.assertEqual(alpha, 0.2)

    def test_pick_from_network_best_selects_lowest_cv_rmse(self) -> None:
        picked = _pick_from_network_best(
            {
                "exponential": {"alpha": 0.15, "cv_rmse": 0.89},
                "powerlaw": {"alpha": 2.1, "cv_rmse": 0.91},
            }
        )
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked["diffusion_model"], "exponential")
        self.assertEqual(picked["cv_rmse"], 0.89)

    def test_pick_from_run_csv_ignores_invalid_rmse_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "movielens" / "runs" / "m3_recommend" / "network_metrics"
            write_alpha_grid_csv(
                metrics_dir / "inferred_edges_expo.csv",
                [0.1, 0.2],
                rmse_col="enhanced_rmse_mean",
                rmse_values=[0.0, 0.85],
            )
            write_alpha_grid_csv(
                metrics_dir / "inferred_edges_power.csv",
                [1.0, 2.0],
                rmse_col="enhanced_rmse_mean",
                rmse_values=[15.0, 0.95],
            )

            def _paths(dataset: str) -> FakeDatasetPaths:
                return FakeDatasetPaths(root, dataset)

            with patch("recommender.experiment.network_selection.DatasetPaths", _paths):
                picked = _pick_from_run_csv("movielens", "m3_recommend", social=False)
            self.assertIsNotNone(picked)
            assert picked is not None
            self.assertEqual(picked["diffusion_model"], "exponential")
            self.assertEqual(picked["alpha_index"], 1)
            self.assertEqual(picked["cv_rmse"], 0.85)

    def test_run_network_selection_freezes_manifest_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dp = FakeDatasetPaths(root, "movielens")
            grid = dp.NETWORKS / "exponential" / "inferred_edges_expo.csv"
            write_alpha_grid_csv(grid, [0.1, 0.15, 0.2], rmse_values=[0.95, 0.88, 0.90])

            metrics_dir = dp.RUNS / "m3_recommend" / "network_metrics"
            write_alpha_grid_csv(
                metrics_dir / "inferred_edges_expo.csv",
                [0.1, 0.15, 0.2],
                rmse_values=[0.95, 0.87, 0.90],
            )

            manifest = {
                "dataset": "movielens",
                "variants": {"M3": sample_manifest_entry()},
            }

            def _paths(dataset: str) -> FakeDatasetPaths:
                return FakeDatasetPaths(root, dataset)

            with patch("recommender.experiment.network_selection.DatasetPaths", _paths):
                selections = run_network_selection(
                    "movielens",
                    variant_ids=["M3"],
                    manifest=manifest,
                )

            self.assertIn("M3", selections["variants"])
            picked = selections["variants"]["M3"]
            self.assertEqual(picked["diffusion_model"], "exponential")
            self.assertEqual(picked["alpha_index"], 1)
            self.assertEqual(manifest["variants"]["M3"]["selected_network"], picked)
            self.assertTrue(dp.NETWORK_SELECTION.exists())
            saved = json.loads(dp.NETWORK_SELECTION.read_text(encoding="utf-8"))
            self.assertEqual(saved["variants"]["M3"]["cv_rmse"], 0.87)


if __name__ == "__main__":
    unittest.main()
