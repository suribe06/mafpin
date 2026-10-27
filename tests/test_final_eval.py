"""P0-5: global held-out evaluation orchestration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fixtures import ratings_frame
from recommender.experiment.final_eval import (
    apply_final_eval_deltas,
    evaluate_variant_global_test,
    upsert_beyond_accuracy_per_user,
    upsert_beyond_accuracy_results,
)


class _ConstantPredictor:
    def predict(self, user, item):  # noqa: ANN001
        return np.full(len(user), 3.5, dtype=float)


class FinalEvalTests(unittest.TestCase):
    def test_evaluate_variant_m1_uses_baseline_params_and_returns_metrics(self) -> None:
        train = ratings_frame([(0, 0, 3.0), (0, 1, 4.0), (1, 0, 5.0)])
        test = ratings_frame([(0, 2, 3.5), (1, 1, 4.5)])
        fake_model = _ConstantPredictor()

        with patch(
            "recommender.experiment.final_eval.train_final_model",
            return_value=fake_model,
        ) as train_mock:
            row = evaluate_variant_global_test(
                "M1",
                train,
                test,
                dataset="movielens",
                hyperparameters={},
                baseline_params={"k": 10, "lambda_reg": 0.1},
                selected_network=None,
                ranking_k=2,
            )

        train_mock.assert_called_once()
        self.assertEqual(row["model_variant"], "M1")
        self.assertEqual(row["k"], 10)
        self.assertEqual(row["lambda_reg"], 0.1)
        self.assertTrue(row["valid_metric_row"])
        self.assertGreaterEqual(row["rmse"], 0.0)
        self.assertIn("ndcg_at_10", row)
        self.assertIn("rmse_warm", row)
        self.assertIn("n_warm_test", row)

    def test_evaluate_variant_social_requires_selected_network(self) -> None:
        train = ratings_frame([(0, 0, 3.0)])
        test = ratings_frame([(0, 1, 4.0)])
        with self.assertRaisesRegex(ValueError, "requires selected_network"):
            evaluate_variant_global_test(
                "M4c",
                train,
                test,
                dataset="movielens",
                hyperparameters={"k": 5, "lambda_reg": 0.1, "lambda_social": 0.01},
                baseline_params={"k": 10, "lambda_reg": 0.1},
                selected_network=None,
            )

    def test_evaluate_variant_m2_enhanced_path_with_mocked_features(self) -> None:
        train = ratings_frame([(0, 0, 3.0), (0, 1, 4.0)])
        test = ratings_frame([(0, 2, 3.5)])
        user_attrs = pd.DataFrame({"f0": [0.0, 1.0]}, index=pd.Index([0, 1])).rename_axis("UserId")
        fake_model = _ConstantPredictor()

        with (
            patch(
                "recommender.experiment.final_eval.load_network_features",
                return_value=user_attrs,
            ),
            patch(
                "recommender.experiment.final_eval.train_enhanced_final",
                return_value=fake_model,
            ) as train_mock,
        ):
            row = evaluate_variant_global_test(
                "M2",
                train,
                test,
                dataset="movielens",
                hyperparameters={"k": 8, "lambda_reg": 0.2, "w_main": 1.0, "w_user": 0.1},
                baseline_params={"k": 10, "lambda_reg": 0.1},
                selected_network={
                    "diffusion_model": "exponential",
                    "alpha_index": 0,
                    "alpha_value": 0.15,
                },
            )

        train_mock.assert_called_once()
        self.assertEqual(row["model_variant"], "M2")
        self.assertEqual(row["diffusion_model"], "exponential")
        self.assertEqual(row["k"], 8)

    def test_apply_final_eval_deltas_uses_m1_row_not_canonical_json(self) -> None:
        rows = [
            {"model_variant": "M1", "rmse": 0.93},
            {"model_variant": "M3", "rmse": 0.92},
        ]
        apply_final_eval_deltas(
            rows,
            canonical_baseline_rmse=1e28,
            ratings=pd.Series([1.0, 5.0]),
        )
        self.assertAlmostEqual(rows[0]["rmse_delta_vs_baseline"], 0.0)
        self.assertAlmostEqual(rows[1]["rmse_delta_vs_baseline"], 0.01)
        self.assertAlmostEqual(rows[1]["rmse_delta_vs_m3"], 0.0)
        self.assertAlmostEqual(rows[0]["rmse_delta_vs_m3"], -0.01)

    def test_apply_final_eval_deltas_ignores_diverged_reference_rows(self) -> None:
        rows = [
            {"model_variant": "M1", "rmse": 0.93, "valid_metric_row": True},
            {"model_variant": "M3", "rmse": 4.1e20, "valid_metric_row": False},
            {"model_variant": "M4a", "rmse": 0.92, "valid_metric_row": True},
        ]
        apply_final_eval_deltas(rows, ratings=pd.Series([1.0, 5.0]))
        self.assertAlmostEqual(rows[2]["rmse_delta_vs_baseline"], 0.01)
        for row in rows:
            if row["model_variant"] != "M3":
                self.assertTrue(np.isnan(row["rmse_delta_vs_m3"]))

    def test_upsert_beyond_accuracy_keeps_prior_variants(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "beyond_accuracy_results.csv"
            upsert_beyond_accuracy_results(
                [{"dataset": "movielens", "model_variant": "M1", "item_coverage_at_k": 0.01}],
                path,
            )
            upsert_beyond_accuracy_results(
                [{"dataset": "movielens", "model_variant": "M4c", "item_coverage_at_k": 0.02}],
                path,
            )
            upsert_beyond_accuracy_results(
                [{"dataset": "movielens", "model_variant": "M1", "item_coverage_at_k": 0.015}],
                path,
            )
            df = pd.read_csv(path)
            self.assertEqual(set(df["model_variant"]), {"M1", "M4c"})
            m1 = float(df.loc[df["model_variant"] == "M1", "item_coverage_at_k"].iloc[0])
            self.assertAlmostEqual(m1, 0.015)

    def test_upsert_beyond_accuracy_per_user_replaces_only_touched_variants(
        self,
    ) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "beyond_accuracy_per_user.parquet"
            first = pd.DataFrame(
                {
                    "dataset": ["movielens", "movielens"],
                    "model_variant": ["M1", "M1"],
                    "UserId": [0, 1],
                    "cce_at_k": [0.1, 0.2],
                }
            )
            second = pd.DataFrame(
                {
                    "dataset": ["movielens"],
                    "model_variant": ["M4c"],
                    "UserId": [0],
                    "cce_at_k": [0.5],
                }
            )
            upsert_beyond_accuracy_per_user(first, path)
            upsert_beyond_accuracy_per_user(second, path)
            out = pd.read_parquet(path)
            self.assertEqual(set(out["model_variant"]), {"M1", "M4c"})
            self.assertEqual(int((out["model_variant"] == "M1").sum()), 2)


if __name__ == "__main__":
    unittest.main()
