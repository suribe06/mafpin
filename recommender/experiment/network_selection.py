"""Freeze one (diffusion_model, alpha_index) per variant from CV metrics."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, cast

import pandas as pd

from config import DatasetPaths, Models
from recommender.experiment.manifest import load_manifest, save_manifest
from recommender.experiment.variants import CORE_VARIANT_IDS, VARIANT_SPECS


def resolve_alpha_index(
    dataset: str,
    diffusion_model: str,
    alpha_value: float,
) -> tuple[int, float]:
    """Map a continuous alpha to the nearest grid row index."""
    short = Models.SHORT[diffusion_model]
    csv_path = (
        DatasetPaths(dataset).NETWORKS
        / diffusion_model
        / f"inferred_edges_{short}.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing alpha grid file: {csv_path}")
    df = pd.read_csv(csv_path, sep="|")
    idx = int((df["alpha"] - alpha_value).abs().idxmin())
    return idx, float(df.loc[idx, "alpha"])


def _pick_from_network_best(
    network_best: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not network_best:
        return None
    best_model = None
    best_rmse = float("inf")
    for model_name, row in network_best.items():
        cv_rmse = float(row.get("cv_rmse", float("inf")))
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_model = model_name
    if best_model is None:
        return None
    row = network_best[best_model]
    return {
        "diffusion_model": best_model,
        "alpha_value": float(row["alpha"]),
        "cv_rmse": float(row["cv_rmse"]),
        "selection_source": "manifest_network_best",
    }


def _pick_from_run_csv(
    dataset: str,
    run_id: str,
    *,
    social: bool,
) -> dict[str, Any] | None:
    metrics_dir = DatasetPaths(dataset).RUNS / run_id / "network_metrics"
    if not metrics_dir.exists():
        return None
    rmse_col = "social_rmse_mean" if social else "enhanced_rmse_mean"
    best: dict[str, Any] | None = None
    best_rmse = float("inf")
    for model_name in Models.ALL:
        short = Models.SHORT[model_name]
        csv_path = metrics_dir / f"inferred_edges_{short}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, sep="|")
        if rmse_col not in df.columns:
            continue
        valid = cast(pd.Series, df[rmse_col].dropna())
        valid = cast(pd.Series, valid[(valid > 0) & (valid < 10)])
        if len(valid) == 0:
            continue
        idx = int(cast(int, valid.idxmin()))
        cv_rmse = float(valid.loc[idx])
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best = {
                "diffusion_model": model_name,
                "alpha_index": idx,
                "alpha_value": float(df.loc[idx, "alpha"]),
                "cv_rmse": cv_rmse,
                "selection_source": f"runs/{run_id}/network_metrics",
            }
    return best


def run_network_selection(
    dataset: str,
    *,
    variant_ids: list[str] | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select and freeze the best network per variant; update manifest + JSON."""
    manifest = manifest or load_manifest(dataset)
    if variant_ids is None:
        targets = [v for v in CORE_VARIANT_IDS if VARIANT_SPECS[v]["needs_network"]]
    else:
        targets = [v for v in variant_ids if VARIANT_SPECS[v]["needs_network"]]

    selections: dict[str, Any] = {
        "dataset": dataset,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variants": {},
    }

    for variant_id in targets:
        entry = manifest["variants"].get(variant_id)
        if not entry:
            print(f"  Skipping {variant_id}: not in manifest")
            continue
        social = bool(entry.get("social_regularization"))
        picked = _pick_from_run_csv(
            dataset, entry["run_id"], social=social
        ) or _pick_from_network_best(entry.get("network_best", {}))
        if picked is None:
            print(f"  WARNING: no network candidate for {variant_id}")
            continue
        if "alpha_index" not in picked:
            idx, alpha = resolve_alpha_index(
                dataset,
                picked["diffusion_model"],
                picked["alpha_value"],
            )
            picked["alpha_index"] = idx
            picked["alpha_value"] = alpha
        picked["variant_id"] = variant_id
        selections["variants"][variant_id] = picked
        entry["selected_network"] = picked
        print(
            f"  {variant_id}: {picked['diffusion_model']} "
            f"α_idx={picked['alpha_index']} α={picked['alpha_value']:.6g} "
            f"CV RMSE={picked['cv_rmse']:.6f}"
        )

    dp = DatasetPaths(dataset)
    tmp = dp.NETWORK_SELECTION.with_suffix(".tmp")
    tmp.write_text(json.dumps(selections, indent=2), encoding="utf-8")
    tmp.replace(dp.NETWORK_SELECTION)
    print(f"Network selection saved → {dp.NETWORK_SELECTION}")

    manifest["network_selection_at"] = selections["created_at"]
    save_manifest(manifest, dp.EXPERIMENT_MANIFEST)
    return selections
