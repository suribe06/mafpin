"""Path bundle and artifact writers for cold-start outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import DatasetPaths, Paths


class ColdStartPaths(DatasetPaths):
    """DatasetPaths rooted at ``data/<dataset>/cold_start/``."""

    def __init__(self, dataset: str, root: Path | str | None = None) -> None:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def write_split_tables(
    paths: ColdStartPaths,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    write_csv(paths.TRAIN_CSV, train_df)
    write_csv(paths.TEST_CSV, test_df)


def write_readme(paths: ColdStartPaths, *, mode: str, dataset: str) -> None:
    text = f"""# Cold-start artifacts ({dataset})

Mode: `{mode}`

See `docs/experiments/cold_start_commands.md` and
`docs/experiments/cold_start_experiment_proposal.md`.

Primary tables:
- `user_strata.csv`
- `cold_start_results.csv`
- `cold_start_user_deltas.csv`
- `bootstrap_confidence_intervals.csv`
- `split_manifest.json`
"""
    paths.README.write_text(text, encoding="utf-8")


def upsert_results(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    """Append rows and dedupe on *keys* (keep last)."""
    upsert_frame(path, pd.DataFrame(rows), keys)


def upsert_frame(path: Path, frame: pd.DataFrame, keys: list[str]) -> None:
    """Merge *frame* into an existing CSV, deduping on *keys* (keep last).

    Prevents diagnostic/controlled/zero-shot runs from wiping each other's
    deltas or bootstrap rows when they share ``--output-dir``.
    """
    if path.exists() and not frame.empty:
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
        missing = [k for k in keys if k not in frame.columns]
        if missing:
            raise ValueError(f"upsert keys missing from frame: {missing}")
        frame = frame.drop_duplicates(subset=keys, keep="last")
    write_csv(path, frame)
