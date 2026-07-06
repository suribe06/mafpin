"""P2-2: artifact manifest staleness checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Split
from fixtures import FakeDatasetPaths
from pipeline._artifacts import _check_artifact_manifest, _write_artifact_manifest


class PipelineArtifactTests(unittest.TestCase):
    def test_check_artifact_manifest_missing_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def _paths(dataset: str) -> FakeDatasetPaths:
                return FakeDatasetPaths(root, dataset)

            with patch("config.DatasetPaths", _paths), patch("builtins.print"):
                ok = _check_artifact_manifest("movielens", context="test")
        self.assertFalse(ok)

    def test_check_artifact_manifest_detects_split_strategy_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dp = FakeDatasetPaths(root, "movielens")
            dp.BASE.mkdir(parents=True)
            manifest = {
                "dataset": "movielens",
                "split_strategy": "not-the-current-strategy",
                "test_size": Split.TEST_SIZE,
                "random_state": Split.RANDOM_STATE,
            }
            (dp.BASE / "artifact_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            def _paths(dataset: str) -> FakeDatasetPaths:
                return FakeDatasetPaths(root, dataset)

            with patch("config.DatasetPaths", _paths), patch("builtins.print"):
                ok = _check_artifact_manifest("movielens", context="recommend")
        self.assertFalse(ok)

    def test_write_and_check_artifact_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dp = FakeDatasetPaths(root, "movielens")

            def _paths(dataset: str) -> FakeDatasetPaths:
                return FakeDatasetPaths(root, dataset)

            with patch("config.DatasetPaths", _paths), patch("builtins.print"):
                _write_artifact_manifest(
                    "movielens",
                    train_rows=80,
                    test_rows=20,
                    total_rows=100,
                )
                ok = _check_artifact_manifest("movielens", context="recommend")
            self.assertTrue(ok)
            self.assertTrue((dp.BASE / "artifact_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
