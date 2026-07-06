"""Experiment manifest import and network selection helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fixtures import recommend_log_m3, recommend_log_m4c
from recommender.experiment.manifest import build_manifest_from_logs
from recommender.experiment.network_selection import _pick_from_network_best


class ExperimentManifestTests(unittest.TestCase):
    def test_pick_from_network_best(self) -> None:
        network_best = {
            "exponential": {"alpha": 0.15, "cv_rmse": 0.89},
            "powerlaw": {"alpha": 2.1, "cv_rmse": 0.91},
            "rayleigh": {"alpha": 0.02, "cv_rmse": 0.95},
        }
        picked = _pick_from_network_best(network_best)
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked["diffusion_model"], "exponential")
        self.assertEqual(picked["cv_rmse"], 0.89)

    def test_build_manifest_from_logs_enhanced_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            (log_dir / "m3_recommend.log").write_text(recommend_log_m3(), encoding="utf-8")
            manifest = build_manifest_from_logs(
                "movielens",
                log_dir=log_dir,
                variant_ids=["M3"],
            )
        self.assertIn("M3", manifest["variants"])
        self.assertEqual(manifest["variants"]["M3"]["hyperparameters"]["k"], 9)
        self.assertIn("exponential", manifest["variants"]["M3"]["network_best"])

    def test_build_manifest_from_logs_social_variant_uses_social_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            (log_dir / "m4c_recommend.log").write_text(recommend_log_m4c(), encoding="utf-8")
            manifest = build_manifest_from_logs(
                "movielens",
                log_dir=log_dir,
                variant_ids=["M4c"],
            )
        params = manifest["variants"]["M4c"]["hyperparameters"]
        self.assertEqual(params["lambda_social"], 0.01)
        self.assertEqual(params["k"], 7)
        self.assertEqual(params["lambda_reg"], 0.3)


if __name__ == "__main__":
    unittest.main()
