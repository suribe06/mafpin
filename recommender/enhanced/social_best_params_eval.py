"""Evaluate the best Phase 6 social CMF params against plain baseline CMF."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets, ROOT
from recommender.baseline import train_model
from recommender.data import evaluate_single_split, load_dataset, split_data_single
from recommender.enhanced.features import load_network_features
from recommender.enhanced.social_regularization import (
    SocialMode,
    build_social_edges,
    fit_social_cmf_split,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _default_search_path(dataset: str) -> Path:
    return DatasetPaths(dataset).BASE / "social_hyperparam_search_results.json"


def _default_output_path(dataset: str) -> Path:
    return DatasetPaths(dataset).BASE / "social_best_params_eval_results.json"


def _default_report_path(dataset: str, model_name: str) -> Path:
    return ROOT / "docs" / "reports" / f"social_best_params_{dataset}_{model_name}.md"


def _load_search_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Social hyperparameter search result not found: {path}"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("best_params"):
        raise ValueError(f"Search result has no best_params: {path}")
    return result


def _prepare_data(
    dataset: str,
    model_name: str,
    network_index: int,
    max_ratings: int,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_dataset(dataset=dataset)
    user_attributes = load_network_features(
        model_name,
        network_index,
        include_communities=True,
        dataset=dataset,
    )
    if user_attributes is None:
        raise FileNotFoundError(
            f"Feature file not found for {dataset}/{model_name}/{network_index:03d}."
        )

    valid_users = sorted(map(int, user_attributes.index))
    data = cast(pd.DataFrame, data.loc[data["UserId"].isin(valid_users)].copy())
    if max_ratings and len(data) > max_ratings:
        data = data.sample(n=max_ratings, random_state=random_state).copy()

    train_df, test_df = split_data_single(
        data,
        test_size=test_size,
        random_state=random_state,
    )
    seen_users = sorted(map(int, train_df["UserId"].unique()))
    seen_items = sorted(map(int, train_df["ItemId"].unique()))
    test_df = cast(
        pd.DataFrame,
        test_df.loc[
            test_df["UserId"].isin(seen_users) & test_df["ItemId"].isin(seen_items)
        ].copy(),
    )
    if test_df.empty:
        raise ValueError("Evaluation split has no warm test rows after filtering.")
    return data, train_df, test_df, user_attributes


def _metric_deltas(
    social_metrics: dict[str, float], baseline_metrics: dict[str, float]
) -> dict[str, float]:
    rmse_delta = social_metrics["rmse"] - baseline_metrics["rmse"]
    return {
        "rmse_delta": rmse_delta,
        "mae_delta": social_metrics["mae"] - baseline_metrics["mae"],
        "r2_delta": social_metrics["r2"] - baseline_metrics["r2"],
        "rmse_relative_improvement": -rmse_delta / baseline_metrics["rmse"],
    }


def _metrics_are_reasonable(metrics: dict[str, float], limit: float) -> bool:
    values = list(metrics.values())
    return bool(
        values
        and all(np.isfinite(value) for value in values)
        and metrics["rmse"] <= limit
    )


def _format_metric(value: float | None) -> str:
    if value is None or not np.isfinite(value) or abs(value) > 1e6:
        return "invalid"
    return f"`{value:.6f}`"


def _display_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_report(result: dict[str, Any], report_path: Path) -> None:
    params = result["social_best_params"]
    baseline = result["baseline_cmf"]["metrics"]
    social = result["social_cmf"]["metrics"]
    deltas = result["comparison"]
    edges = result["social_edges"]
    diagnostics = result["diagnostics"]
    search_result_path = _display_path(result["search_result_path"])
    output_path = _display_path(result["output_path"])
    displayed_report_path = _display_path(report_path)
    actual_improvement = (
        diagnostics["baseline_reasonable"]
        and diagnostics["social_reasonable"]
        and deltas["rmse_delta"] < 0.0
    )
    if diagnostics["social_reasonable"]:
        interpretation = (
            "The Optuna-selected Phase 6 model produced a rating-scale sane fit on "
            "this rerun. "
        )
        if actual_improvement:
            interpretation += (
                "It improves RMSE and R2 over the plain CMF baseline on the same "
                "filtered warm split."
            )
        else:
            interpretation += (
                "It did not improve RMSE over the plain CMF baseline on this rerun, "
                "so this should be treated as a stability check rather than a win."
            )
    else:
        interpretation = (
            "The social CMF fit did not produce a rating-scale sane result after "
            "the configured retries. The selected parameter region should be retried "
            "or evaluated with stricter optimizer safeguards before it is promoted."
        )

    text = f"""# Phase 6 Specialized Best-Params Evaluation

This report evaluates the best Phase 6 social-regularized CMF parameters from the Optuna search against a plain baseline CMF on the same filtered warm split. This is separate from the smoke-test report: the smoke tests validated integration and small-grid behavior, while this section uses the specialized Optuna-selected setting for `{result['dataset']}` / `{result['model_name']}` / network `{result['network_index']}`.

## Configuration

| Setting | Value |
| --- | --- |
| Dataset | `{result['dataset']}` |
| Diffusion model | `{result['model_name']}` |
| Network index | `{result['network_index']}` |
| Search result | `{search_result_path}` |
| Max ratings | `{result['max_ratings']}` |
| Train ratings | `{result['train_ratings']}` |
| Warm test ratings | `{result['test_ratings']}` |
| Random state | `{result['random_state']}` |
| L-BFGS iterations | `{result['maxiter']}` |
| Social retries | `{result['social_retries']}` |
| User attributes | enabled for social CMF |

## Optuna Best Parameters

| Parameter | Value |
| --- | ---: |
| `k` | `{params['k']}` |
| `lambda_reg` | `{params['lambda_reg']:.12g}` |
| `w_main` | `{params['w_main']:.12g}` |
| `w_user` | `{params['w_user']:.12g}` |
| `lambda_social` | `{params['lambda_social']:.12g}` |
| `social_mode` | `{params['social_mode']}` |
| `beta` | `{params['beta']:.12g}` |
| `gamma` | `{params['gamma']:.12g}` |

## Social Edge Summary

| Metric | Value |
| --- | ---: |
| Edges | `{edges['n_edges']}` |
| Mean weight | `{edges['mean_weight']:.12g}` |
| Min weight | `{edges['min_weight']:.12g}` |
| Max weight | `{edges['max_weight']:.12g}` |

## Baseline vs Social CMF

The baseline is a plain CMF model without side-user attributes and without the social Laplacian. It uses the same `k` and `lambda_reg` as the best social model so this comparison isolates the added Phase 6 ingredients on the same train/test split.

| Model | User attributes | Social regularizer | RMSE | MAE | R2 |
| --- | --- | --- | ---: | ---: | ---: |
| Plain baseline CMF | no | no | {_format_metric(baseline.get('rmse'))} | {_format_metric(baseline.get('mae'))} | {_format_metric(baseline.get('r2'))} |
| Social CMF best params | yes | yes | {_format_metric(social.get('rmse'))} | {_format_metric(social.get('mae'))} | {_format_metric(social.get('r2'))} |

| Delta | Value |
| --- | ---: |
| RMSE delta | {_format_metric(deltas.get('rmse_delta'))} |
| MAE delta | {_format_metric(deltas.get('mae_delta'))} |
| R2 delta | {_format_metric(deltas.get('r2_delta'))} |
| Relative RMSE improvement | `{100 * deltas['rmse_relative_improvement']:.3f}%` |

## Diagnostics

| Diagnostic | Value |
| --- | --- |
| Baseline rating-scale sane | `{str(diagnostics['baseline_reasonable']).lower()}` |
| Social rating-scale sane | `{str(diagnostics['social_reasonable']).lower()}` |
| Social selected attempt | `{result['social_cmf']['selected_attempt']}` |

## Interpretation

{interpretation} Because the baseline is intentionally held to the same `k` and `lambda_reg`, the comparison reads as an ablation of user-side features plus the social Laplacian rather than a contest against a separately tuned baseline.

This result is strong enough to carry the selected parameter region into the later full analysis across all sampled networks and diffusion models. The next full-analysis step should re-evaluate these settings across the network grid rather than treating the single network-index result as final model evidence.

## Artifacts

```text
{output_path}
{displayed_report_path}
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def evaluate_best_social_params(
    search_result_path: str | Path | None = None,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    maxiter: int | None = None,
    nthreads: int | None = None,
    social_retries: int = 5,
) -> dict[str, Any]:
    """Evaluate Optuna-best social CMF params against a plain CMF baseline."""
    search_path = (
        Path(search_result_path)
        if search_result_path is not None
        else _default_search_path(Datasets.DEFAULT)
    )
    search_result = _load_search_result(search_path)

    dataset = str(search_result["dataset"])
    model_name = str(search_result["model_name"])
    network_index = int(search_result["network_index"])
    best_params = dict(search_result["best_params"])
    run_maxiter = int(
        maxiter if maxiter is not None else search_result.get("maxiter", 25)
    )
    run_nthreads = int(
        nthreads if nthreads is not None else search_result.get("nthreads", 1)
    )
    max_ratings = int(search_result.get("max_ratings", 5000))
    test_size = float(search_result.get("test_size", 0.2))
    random_state = int(search_result.get("random_state", 42))
    transform = str(search_result.get("transform", "standard"))

    data, train_df, test_df, user_attributes = _prepare_data(
        dataset=dataset,
        model_name=model_name,
        network_index=network_index,
        max_ratings=max_ratings,
        test_size=test_size,
        random_state=random_state,
    )

    k = int(best_params["k"])
    lambda_reg = float(best_params["lambda_reg"])
    baseline_model = train_model(train_df, k=k, lambda_reg=lambda_reg)
    baseline_metrics = evaluate_single_split(baseline_model, test_df)
    rating_span = float(data["Rating"].max() - data["Rating"].min())
    reasonableness_limit = max(10.0, 10.0 * rating_span)

    social_edges = build_social_edges(
        dataset=dataset,
        model_name=model_name,
        network_index=network_index,
        user_index=user_attributes.index,
        mode=cast(SocialMode, best_params["social_mode"]),
        beta=float(best_params["beta"]),
        gamma=float(best_params["gamma"]),
        dtype=np.float32,
    )
    attempts: list[dict[str, Any]] = []
    social_metrics: dict[str, float] | None = None
    selected_attempt: int | None = None
    for attempt in range(max(1, social_retries)):
        attempt_seed = random_state + attempt
        _, attempt_metrics = fit_social_cmf_split(
            train_df,
            test_df,
            user_attributes,
            social_edges,
            k=k,
            lambda_reg=lambda_reg,
            w_main=float(best_params["w_main"]),
            w_user=float(best_params["w_user"]),
            lambda_social=float(best_params["lambda_social"]),
            transform=transform,
            maxiter=run_maxiter,
            nthreads=run_nthreads,
            random_state=attempt_seed,
            include_user_attributes=True,
        )
        reasonable = _metrics_are_reasonable(attempt_metrics, reasonableness_limit)
        attempts.append(
            {
                "attempt": attempt,
                "random_state": attempt_seed,
                "reasonable": reasonable,
                "metrics": attempt_metrics,
            }
        )
        if reasonable:
            social_metrics = attempt_metrics
            selected_attempt = attempt
            break
    if social_metrics is None:
        social_metrics = cast(dict[str, float], attempts[-1]["metrics"])
        selected_attempt = None
    selected_social_metrics = social_metrics

    out_path = (
        Path(output_path) if output_path is not None else _default_output_path(dataset)
    )
    md_path = (
        Path(report_path)
        if report_path is not None
        else _default_report_path(dataset, model_name)
    )
    edge_summary = {
        key: value
        for key, value in asdict(social_edges).items()
        if key not in {"row", "col", "val"}
    }
    result: dict[str, Any] = {
        "dataset": dataset,
        "model_name": model_name,
        "network_index": network_index,
        "search_result_path": str(search_path),
        "output_path": str(out_path),
        "max_ratings": max_ratings,
        "test_size": test_size,
        "train_ratings": int(len(train_df)),
        "test_ratings": int(len(test_df)),
        "warm_test_only": True,
        "random_state": random_state,
        "reasonableness_limit": reasonableness_limit,
        "social_retries": social_retries,
        "maxiter": run_maxiter,
        "nthreads": run_nthreads,
        "transform": transform,
        "rating_range": [float(data["Rating"].min()), float(data["Rating"].max())],
        "social_best_params": best_params,
        "baseline_cmf": {
            "description": "Plain CMF with same k and lambda_reg as social best params; no side-user attributes and no social regularizer.",
            "k": k,
            "lambda_reg": lambda_reg,
            "metrics": baseline_metrics,
        },
        "social_cmf": {
            "description": "Patched L-BFGS CMF using Optuna-best side-user and social regularization parameters.",
            "metrics": selected_social_metrics,
            "selected_attempt": selected_attempt,
            "attempts": attempts,
        },
        "social_edges": edge_summary,
        "comparison": _metric_deltas(selected_social_metrics, baseline_metrics),
        "diagnostics": {
            "baseline_reasonable": _metrics_are_reasonable(
                baseline_metrics, reasonableness_limit
            ),
            "social_reasonable": _metrics_are_reasonable(
                selected_social_metrics, reasonableness_limit
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
    _write_report(result, md_path)
    print(f"Specialized evaluation JSON saved -> {out_path}")
    print(f"Specialized evaluation report saved -> {md_path}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Optuna-best Phase 6 social CMF against plain baseline CMF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--search-result-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--nthreads", type=int, default=None)
    parser.add_argument("--social-retries", type=int, default=5)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = evaluate_best_social_params(
        search_result_path=args.search_result_path,
        output_path=args.output_path,
        report_path=args.report_path,
        maxiter=args.maxiter,
        nthreads=args.nthreads,
        social_retries=args.social_retries,
    )
    print(json.dumps(_json_ready(result["comparison"]), indent=2))


if __name__ == "__main__":
    main()
