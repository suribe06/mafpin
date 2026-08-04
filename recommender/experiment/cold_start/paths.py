"""Path bundle for cold-start artifacts (isolated from core Phase-2 outputs)."""

from __future__ import annotations

from pathlib import Path

from config import DatasetPaths, Paths


class ColdStartPaths(DatasetPaths):
    """DatasetPaths rooted at ``data/<dataset>/cold_start/``."""

    def __init__(self, dataset: str, root: Path | None = None) -> None:
        # DatasetPaths.__init__ sets core attrs from data/<dataset>/; override after.
        super().__init__(dataset)
        # Absolute paths are required: NetInf runs with cwd=networks/, so a
        # relative --output-dir would make -i:cascades unreadable (FAILED rc=0).
        if root is not None:
            base = Path(root).expanduser().resolve()
        else:
            base = (Paths.DATA / dataset / "cold_start").resolve()
        self.BASE = base
        self.CASCADES = base / "cascades.txt"
        self.CASCADE_USER_STATS = base / "cascade_user_stats.csv"
        self.NETWORKS = base / "inferred_networks"
        self.CENTRALITY = base / "centrality_metrics"
        self.COMMUNITIES = base / "communities"
        self.SHAP_MATRICES = base / "shap_matrices"
        self.PLOTS = Paths.PLOTS / dataset / "cold_start"
        self.RUNS = base / "runs"
        self.LOGS = base / "logs"
        self.COLD_START = base
        self.SPLIT_MANIFEST = base / "split_manifest.json"
        self.USER_STRATA = base / "user_strata.csv"
        self.TRAIN_CSV = base / "train.csv"
        self.TEST_CSV = base / "test.csv"
        self.RESULTS = base / "cold_start_results.csv"
        self.USER_DELTAS = base / "cold_start_user_deltas.csv"
        self.BOOTSTRAP_CIS = base / "bootstrap_confidence_intervals.csv"
        self.SUCCESS_SUMMARY = base / "success_summary.md"
        self.README = base / "README.md"
        self.ZERO_SHOT = base / "zero_shot_trust"

    def ensure_dirs(self) -> None:
        for path in (
            self.BASE,
            self.NETWORKS,
            self.CENTRALITY,
            self.COMMUNITIES,
            self.RUNS,
            self.LOGS,
            self.ZERO_SHOT,
        ):
            path.mkdir(parents=True, exist_ok=True)
