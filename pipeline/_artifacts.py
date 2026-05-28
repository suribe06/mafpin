"""Artifact manifest helpers (write and staleness check)."""

from __future__ import annotations

import json
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_artifact_manifest(
    dataset: str,
    source_path: str | Path | None = None,
    train_rows: int | None = None,
    test_rows: int | None = None,
    total_rows: int | None = None,
    n_users: int | None = None,
    n_items: int | None = None,
    temporal_cutoff: Any | None = None,
) -> None:
    """Record current split configuration in an artifact manifest file.

    Written after the cascade step so downstream steps can warn if the
    manifest is stale relative to the current :class:`~config.Split` settings.
    """
    from config import Split, DatasetPaths as _DP

    manifest: dict[str, Any] = {
        "dataset": dataset,
        "split_strategy": Split.STRATEGY,
        "test_size": Split.TEST_SIZE,
        "random_state": Split.RANDOM_STATE,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "total_rows": total_rows,
        "n_users": n_users,
        "n_items": n_items,
        "temporal_cutoff": temporal_cutoff,
        "created_at": datetime.now().isoformat(),
    }
    if source_path is not None:
        source = Path(source_path)
        manifest["source_file"] = str(source)
        if source.exists():
            manifest["source_sha256"] = _sha256_file(source)
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


def _check_artifact_manifest(
    dataset: str,
    *,
    context: str = "artifacts",
    validate_source_hash: bool = False,
) -> bool:
    """Warn if cached artifacts are missing or stale for the current config."""
    from config import Split, DatasetPaths as _DP

    manifest_path = _DP(dataset).BASE / "artifact_manifest.json"
    if not manifest_path.exists():
        print(
            f"  WARNING: no artifact manifest found for {dataset} before {context}. "
            "Run --steps cascade to record the current split contract."
        )
        return False
    ok = True
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") not in (None, dataset):
            ok = False
            print(
                f"  WARNING: artifact manifest dataset={manifest.get('dataset')!r} "
                f"does not match requested dataset={dataset!r}."
            )
        if manifest.get("split_strategy") != Split.STRATEGY:
            ok = False
            print(
                f"  WARNING: artifacts were generated with split_strategy="
                f"'{manifest['split_strategy']}' but the current config has "
                f"'{Split.STRATEGY}'. Re-run --steps cascade to regenerate."
            )
        if (
            abs(float(manifest.get("test_size", Split.TEST_SIZE)) - Split.TEST_SIZE)
            > 1e-6
        ):
            ok = False
            print(
                f"  WARNING: artifacts were generated with test_size="
                f"{manifest['test_size']} but the current config has "
                f"{Split.TEST_SIZE}."
            )
        if manifest.get("random_state") not in (None, Split.RANDOM_STATE):
            ok = False
            print(
                f"  WARNING: artifacts were generated with random_state="
                f"{manifest['random_state']} but the current config has "
                f"{Split.RANDOM_STATE}."
            )
        if validate_source_hash and manifest.get("source_file"):
            source = Path(manifest["source_file"])
            if source.exists() and manifest.get("source_sha256"):
                current_hash = _sha256_file(source)
                if current_hash != manifest["source_sha256"]:
                    ok = False
                    print(
                        "  WARNING: dataset source hash differs from the artifact "
                        f"manifest before {context}. Re-run --steps cascade."
                    )
        return ok
    except Exception:  # pylint: disable=broad-except
        print(
            f"  WARNING: could not read artifact manifest at {manifest_path} "
            f"before {context}. Re-run --steps cascade."
        )
        return False
