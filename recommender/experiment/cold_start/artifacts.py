"""Artifact writers for cold-start experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from recommender.experiment.cold_start.paths import ColdStartPaths


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
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_old = pd.read_csv(path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new = df_new.drop_duplicates(subset=keys, keep="last")
    write_csv(path, df_new)
