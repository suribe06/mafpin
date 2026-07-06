"""P1-6: experiment log parser parity with manifest import."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fixtures import recommend_log_m3, recommend_log_m4c
from scripts.parse_experiment_logs import (
    _load_csv_summary,
    parse_hypertune,
    parse_recommend_log,
)


class ParseExperimentLogsTests(unittest.TestCase):
    def test_parse_recommend_log_extracts_enhanced_and_network_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m3_recommend.log"
            path.write_text(recommend_log_m3(), encoding="utf-8")
            parsed = parse_recommend_log(path)
        self.assertTrue(parsed["completed"])
        self.assertEqual(parsed["enhanced_params"]["k"], 9)
        self.assertIn("exponential", parsed["network_best"])

    def test_parse_recommend_log_keeps_social_and_enhanced_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m4c_recommend.log"
            path.write_text(recommend_log_m4c(), encoding="utf-8")
            parsed = parse_recommend_log(path)
        self.assertEqual(parsed["social_params"]["lambda_social"], 0.01)
        self.assertEqual(parsed["enhanced_params"]["k"], 9)

    def test_parse_hypertune_reads_enhanced_cv_line(self) -> None:
        text = "Best enhanced params: {'k': 5, 'lambda_reg': 0.2} RMSE=0.91\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m2_hypertune.log"
            path.write_text(text, encoding="utf-8")
            parsed = parse_hypertune(path)
        self.assertEqual(parsed["enhanced_cv"]["params"]["k"], 5)
        self.assertEqual(parsed["enhanced_cv"]["rmse"], 0.91)

    def test_load_csv_summary_picks_minimum_valid_rmse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_dir = root / "data" / "movielens" / "inferred_networks" / "exponential"
            csv_dir.mkdir(parents=True)
            csv_path = csv_dir / "inferred_edges_expo.csv"
            csv_path.write_text(
                "alpha|enhanced_rmse_mean\n0.1|0.95\n0.2|0.82\n0.3|15.0\n",
                encoding="utf-8",
            )
            with patch("scripts.parse_experiment_logs.ROOT", root):
                summary = _load_csv_summary()
        self.assertEqual(summary["enhanced_exponential"]["min_rmse"], 0.82)
        self.assertEqual(summary["enhanced_exponential"]["best_alpha_index"], 1)


if __name__ == "__main__":
    unittest.main()
