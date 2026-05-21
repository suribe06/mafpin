"""Artifact manifest helpers (write and staleness check)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from typing import Any


def _write_artifact_manifest(dataset: str) -> None:
    """Record current split configuration in an artifact manifest file.

    Written after the cascade step so downstream steps can warn if the
    manifest is stale relative to the current :class:`~config.Split` settings.
    """
    from config import Split, DatasetPaths as _DP

    manifest: dict[str, Any] = {
        "split_strategy": Split.STRATEGY,
        "test_size": Split.TEST_SIZE,
        "random_state": Split.RANDOM_STATE,
        "created_at": datetime.now().isoformat(),
    }
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        manifest["git_commit"] = git_hash
    except Exception:  # pylint: disable=broad-except
        pass

    dest = _DP(dataset).BASE / "artifact_manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(dest)
    print(f"Artifact manifest written → {dest}")


def _check_artifact_manifest(dataset: str) -> None:
    """Warn if cached artifacts were built with a different split config."""
    from config import Split, DatasetPaths as _DP

    manifest_path = _DP(dataset).BASE / "artifact_manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split_strategy") != Split.STRATEGY:
            print(
                f"  WARNING: artifacts were generated with split_strategy="
                f"'{manifest['split_strategy']}' but the current config has "
                f"'{Split.STRATEGY}'. Re-run --steps cascade to regenerate."
            )
        if (
            abs(float(manifest.get("test_size", Split.TEST_SIZE)) - Split.TEST_SIZE)
            > 1e-6
        ):
            print(
                f"  WARNING: artifacts were generated with test_size="
                f"{manifest['test_size']} but the current config has "
                f"{Split.TEST_SIZE}."
            )
    except Exception:  # pylint: disable=broad-except
        pass
