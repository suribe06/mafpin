"""Experiment manifest: import hyperparameters from logs, archive per-run artifacts."""

from __future__ import annotations

import ast
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DatasetPaths, Models
from recommender.experiment.variants import CORE_VARIANT_IDS, VARIANT_SPECS


def _parse_logged_params(raw: str) -> dict[str, Any]:
    return ast.literal_eval(raw)


def _parse_recommend_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, Any] = {
        "log_file": str(path),
        "completed": "[RECOMMEND] Done." in text,
    }
    if m := re.search(
        r"Baseline \(global test\) — RMSE: ([\d.]+)\s+MAE: ([\d.]+)\s+R²: ([\d.-]+)",
        text,
    ):
        out["baseline_global"] = {
            "rmse": float(m.group(1)),
            "mae": float(m.group(2)),
            "r2": float(m.group(3)),
        }
    if m := re.search(r"Best baseline params: (\{[^}]+\})\s+RMSE=([\d.]+)", text):
        out["baseline_cv"] = {
            "params": _parse_logged_params(m.group(1)),
            "rmse": float(m.group(2)),
        }
    if m := re.search(r"Best enhanced params: (\{[^}]+\})\s+RMSE=([\d.]+)", text):
        out["enhanced_cv"] = {
            "params": _parse_logged_params(m.group(1)),
            "rmse": float(m.group(2)),
        }
    if m := re.search(r"Social CMF best RMSE: ([\d.]+)", text):
        out["social_cv_rmse"] = float(m.group(1))
    if m := re.search(r"Social CMF best hyperparameters:\n(\{.*?\n\})", text, re.S):
        out["social_params"] = json.loads(m.group(1))
    if m := re.search(r"Enhanced CMF best hyperparameters:\n(\{.*?\n\})", text, re.S):
        out["enhanced_params"] = json.loads(m.group(1))
    out["network_best"] = {}
    for pat, key in [
        (
            r"Exponential\s*[—\-]\s*Best α=([\d.e+-]+)\s+RMSE=([\d.]+)",
            "exponential",
        ),
        (
            r"Powerlaw\s*[—\-]\s*Best α=([\d.e+-]+)\s+RMSE=([\d.]+)",
            "powerlaw",
        ),
        (
            r"Rayleigh\s*[—\-]\s*Best α=([\d.e+-]+)\s+RMSE=([\d.]+)",
            "rayleigh",
        ),
    ]:
        if m := re.search(pat, text, re.I):
            out["network_best"][key] = {
                "alpha": float(m.group(1)),
                "cv_rmse": float(m.group(2)),
            }
    return out


def _variant_hyperparams(variant_id: str, parsed: dict[str, Any]) -> dict[str, Any]:
    spec = VARIANT_SPECS[variant_id]
    if spec["social_regularization"]:
        params = dict(parsed.get("social_params") or {})
        if not params and parsed.get("enhanced_cv"):
            params = dict(parsed["enhanced_cv"]["params"])
        return params
    params = dict(parsed.get("enhanced_params") or {})
    if not params and parsed.get("enhanced_cv"):
        params = dict(parsed["enhanced_cv"]["params"])
    return params


def build_manifest_from_logs(
    dataset: str,
    *,
    log_dir: Path | None = None,
    variant_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build experiment manifest from existing recommend logs (Phase 2 bootstrap)."""
    dp = DatasetPaths(dataset)
    logs = log_dir or dp.LOGS
    targets = variant_ids or [v for v in CORE_VARIANT_IDS if v != "M1"]
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "import_manifest",
        "variants": {},
    }

    for variant_id in targets:
        spec = VARIANT_SPECS[variant_id]
        log_name = spec.get("log_name")
        if not log_name:
            continue
        log_path = logs / log_name
        if not log_path.exists():
            continue
        parsed = _parse_recommend_log(log_path)
        hyperparams = _variant_hyperparams(variant_id, parsed)
        entry: dict[str, Any] = {
            "variant_id": variant_id,
            "run_id": spec["run_id"],
            "log_file": str(log_path),
            "completed": parsed.get("completed", False),
            "needs_network": spec["needs_network"],
            "social_regularization": spec["social_regularization"],
            "include_communities": spec["include_communities"],
            "social_mode": spec["social_mode"],
            "social_normalization": spec["social_normalization"],
            "hyperparameters": hyperparams,
            "network_best": parsed.get("network_best", {}),
            "baseline_cv": parsed.get("baseline_cv"),
            "baseline_global": parsed.get("baseline_global"),
            "selected_network": None,
        }
        if parsed.get("social_cv_rmse") is not None:
            entry["social_cv_rmse"] = parsed["social_cv_rmse"]
        manifest["variants"][variant_id] = entry

    return manifest


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)
    print(f"Experiment manifest saved → {path}")


def import_manifest_from_logs(
    dataset: str,
    *,
    variant_ids: list[str] | None = None,
    all_variants: bool = False,
) -> dict[str, Any]:
    """Build and persist the experiment manifest from recommend logs."""
    dp = DatasetPaths(dataset)
    log_dir = dp.LOGS
    if not log_dir.exists():
        print(f"WARNING: log directory missing: {log_dir}")

    if variant_ids is not None:
        resolved = variant_ids
    elif all_variants:
        resolved = [v for v in VARIANT_SPECS if v != "M1"]
    else:
        resolved = [v for v in CORE_VARIANT_IDS if v != "M1"]

    manifest = build_manifest_from_logs(
        dataset,
        log_dir=log_dir,
        variant_ids=resolved,
    )
    n = len(manifest.get("variants", {}))
    print(f"Imported {n} variant(s) from {log_dir}")
    if n == 0:
        print(
            "No logs found. Expected files like data/<dataset>/logs/m3_recommend.log"
        )
    save_manifest(manifest, dp.EXPERIMENT_MANIFEST)
    return manifest


def load_manifest(dataset: str) -> dict[str, Any]:
    path = DatasetPaths(dataset).EXPERIMENT_MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"Missing experiment manifest at {path}. "
            "Run --steps import_manifest first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def archive_recommend_run(
    dataset: str,
    run_id: str,
    *,
    baseline_path: Path | None = None,
    enhanced_path: Path | None = None,
    social_path: Path | None = None,
) -> Path:
    """Copy recommend JSON artifacts and network CSV snapshots under runs/<run_id>/."""
    from networks.artifacts import NetworkArtifacts

    dp = DatasetPaths(dataset)
    dest = dp.RUNS / run_id
    dest.mkdir(parents=True, exist_ok=True)

    for src in [baseline_path, enhanced_path, social_path]:
        if src and src.exists():
            shutil.copy2(src, dest / src.name)

    metrics_dir = dest / "network_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    arts = NetworkArtifacts(dataset, paths=dp)
    for model_name in Models.ALL:
        src_csv = arts.inferred_edges_csv(model_name)
        if src_csv.exists():
            shutil.copy2(src_csv, metrics_dir / src_csv.name)

    meta = {
        "run_id": run_id,
        "dataset": dataset,
        "archived_at": datetime.now().isoformat(timespec="seconds"),
    }
    (dest / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Archived recommend artifacts → {dest}")
    return dest
