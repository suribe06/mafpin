"""P2-4: core experiment batch runner shell contract."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_core_experiment.sh"


class RunCoreExperimentScriptTests(unittest.TestCase):
    def test_run_core_experiment_requires_dataset(self) -> None:
        result = subprocess.run(
            [str(SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--dataset is required", result.stderr)

    def test_run_core_experiment_rejects_unknown_dataset(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--dataset", "not-a-dataset"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid dataset", result.stderr)

    def test_run_core_experiment_dry_run_writes_summary_without_pipeline(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--dataset", "movielens", "--dry-run"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("DRY-RUN", result.stdout)
        summary = ROOT / "data" / "movielens" / "logs" / "run_summary.tsv"
        self.assertTrue(summary.exists())
        header = summary.read_text(encoding="utf-8").splitlines()[0]
        self.assertTrue(header.startswith("step\texit_code"))


if __name__ == "__main__":
    unittest.main()
