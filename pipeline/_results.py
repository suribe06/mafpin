"""Helpers for displaying hyperparameter search results."""

from __future__ import annotations

import json
import math
from typing import Any


def _best_rmse_from_results(search_result: dict[str, Any]) -> float | None:
    values: list[float] = []
    for row in search_result.get("all_results", []):
        rmse = row.get("rmse") if isinstance(row, dict) else None
        if isinstance(rmse, (int, float)) and math.isfinite(float(rmse)):
            values.append(float(rmse))
    return min(values) if values else None


def _print_best_hyperparams(label: str, search_result: dict[str, Any]) -> None:
    best_params = search_result.get("best_params") or {}
    print(f"\n{label} best hyperparameters:", flush=True)
    if best_params:
        print(json.dumps(best_params, indent=2, sort_keys=True), flush=True)
    else:
        print("{}", flush=True)

    best_value = search_result.get("best_value")
    if best_value is None:
        best_value = _best_rmse_from_results(search_result)
    if isinstance(best_value, (int, float)) and math.isfinite(float(best_value)):
        print(f"{label} best RMSE: {float(best_value):.6f}", flush=True)

    best_metrics = search_result.get("best_metrics") or {}
    if best_metrics:
        print(f"{label} best metrics:", flush=True)
        print(json.dumps(best_metrics, indent=2, sort_keys=True), flush=True)
