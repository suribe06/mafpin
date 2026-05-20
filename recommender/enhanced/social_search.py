"""Optuna search for Phase 6 social-regularized CMF hyperparameters."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets, Models
from recommender.data import load_dataset, split_data_single
from recommender.enhanced.features import load_network_features
from recommender.enhanced.social_regularization import (
    SocialMode,
    build_social_edges,
    fit_social_cmf_split,
)

SOCIAL_MODES: tuple[SocialMode, ...] = (
    "uniform",
    "community_jaccard",
    "boundary_downweight",
    "bridge_preserve",
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


def _default_output_path(dataset: str) -> Path:
    return DatasetPaths(dataset).BASE / "social_hyperparam_search_results.json"


def _prepare_search_data(
    dataset: str,
    model_name: str,
    network_index: int,
    max_ratings: int,
    test_size: float,
    random_state: int,
    train_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare (inner_train, inner_test, user_attributes) for a search trial.

    When *train_df* is supplied (the pipeline's global training split) this
    function draws an internal validation split from it instead of loading the
    full dataset independently, eliminating the data-leakage path where the
    global test set would otherwise be visible to Optuna trials.
    """
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

    if train_df is not None:
        # Use the externally supplied training split to avoid leaking the
        # global test set into the hyperparameter search.
        data = cast(
            pd.DataFrame,
            train_df.loc[
                train_df["UserId"].isin(sorted(map(int, user_attributes.index)))
            ].copy(),
        )
    else:
        data = load_dataset(dataset=dataset)
        valid_users = sorted(map(int, user_attributes.index))
        data = cast(pd.DataFrame, data.loc[data["UserId"].isin(valid_users)].copy())

    if max_ratings and len(data) > max_ratings:
        data = data.sample(n=max_ratings, random_state=random_state).copy()

    inner_train, inner_test = split_data_single(
        data,
        test_size=test_size,
        random_state=random_state,
    )
    seen_users = sorted(map(int, inner_train["UserId"].unique()))
    seen_items = sorted(map(int, inner_train["ItemId"].unique()))
    inner_test = cast(
        pd.DataFrame,
        inner_test.loc[
            inner_test["UserId"].isin(seen_users)
            & inner_test["ItemId"].isin(seen_items)
        ].copy(),
    )
    if inner_test.empty:
        raise ValueError("Search split has no warm test rows after filtering.")
    return inner_train, inner_test, user_attributes


def _trial_params(
    trial: Any,
    social_modes: Iterable[SocialMode],
    k_min: int,
    k_max: int,
    lambda_reg_min: float,
    lambda_reg_max: float,
    w_main_min: float,
    w_main_max: float,
    w_user_min: float,
    w_user_max: float,
    lambda_social_min: float,
    lambda_social_max: float,
    beta_min: float,
    beta_max: float,
    gamma_min: float,
    gamma_max: float,
) -> dict[str, Any]:
    social_mode = trial.suggest_categorical("social_mode", list(social_modes))
    params: dict[str, Any] = {
        "k": trial.suggest_int("k", k_min, k_max),
        "lambda_reg": trial.suggest_float(
            "lambda_reg", lambda_reg_min, lambda_reg_max, log=True
        ),
        "w_main": trial.suggest_float("w_main", w_main_min, w_main_max),
        "w_user": trial.suggest_float("w_user", w_user_min, w_user_max, log=True),
        "lambda_social": trial.suggest_float(
            "lambda_social", lambda_social_min, lambda_social_max, log=True
        ),
        "social_mode": social_mode,
        # beta / gamma only apply to certain modes; sample conditionally so
        # Optuna's TPE doesn't waste budget on irrelevant parameters.
        "beta": 0.0,
        "gamma": 1.0,
    }
    if social_mode in ("boundary_downweight", "bridge_preserve"):
        params["beta"] = trial.suggest_float("beta", beta_min, beta_max)
    if social_mode == "bridge_preserve":
        params["gamma"] = trial.suggest_float("gamma", gamma_min, gamma_max)
    return params


def _metrics_are_usable(metrics: dict[str, float], reasonableness_limit: float) -> bool:
    values = list(metrics.values())
    return bool(
        values
        and all(np.isfinite(value) for value in values)
        and metrics["rmse"] <= reasonableness_limit
    )


def save_social_search_results(
    search_result: dict, path: str | Path | None = None
) -> None:
    """Persist Phase 6 social hyperparameter search results to JSON atomically."""
    dest = (
        Path(path)
        if path is not None
        else _default_output_path(search_result["dataset"])
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json_ready(search_result), indent=2), encoding="utf-8")
    tmp.replace(dest)
    print(f"Social hyperparameter search results saved -> {dest}")


def search_social_regularized_params(
    dataset: str = Datasets.DEFAULT,
    model_name: str = "exponential",
    network_index: int = 0,
    n_trials: int = 200,
    timeout: int | None = None,
    max_ratings: int = 5000,
    test_size: float = 0.2,
    maxiter: int = 25,
    random_state: int = 42,
    nthreads: int = 1,
    include_user_attributes: bool = True,
    social_modes: Iterable[SocialMode] = SOCIAL_MODES,
    k_min: int = 5,
    k_max: int = 50,
    lambda_reg_min: float = 0.01,
    lambda_reg_max: float = 10.0,
    w_main_min: float = 0.1,
    w_main_max: float = 1.0,
    w_user_min: float = 0.01,
    w_user_max: float = 1.0,
    lambda_social_min: float = 1e-4,
    lambda_social_max: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 1.0,
    gamma_min: float = 0.1,
    gamma_max: float = 3.0,
    transform: str = "standard",
    output_path: str | Path | None = None,
    train_df: pd.DataFrame | None = None,
) -> dict:
    """Search CMF and social-regularization hyperparameters with Optuna TPE.

    Args:
        train_df: Optional pre-split training DataFrame from the pipeline's
                  global split.  When supplied, the internal validation split
                  is drawn from it rather than the full dataset, preventing
                  the global test set from leaking into the search.
    """
    import optuna

    if dataset not in Datasets.ALL:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {Datasets.ALL}.")
    if model_name not in Models.ALL:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {Models.ALL}.")

    selected_social_modes = tuple(social_modes)
    if not selected_social_modes:
        raise ValueError("At least one social mode must be supplied.")
    unknown_modes = set(selected_social_modes) - set(SOCIAL_MODES)
    if unknown_modes:
        raise ValueError(f"Unknown social modes: {sorted(unknown_modes)}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_df, test_df, user_attributes = _prepare_search_data(
        dataset=dataset,
        model_name=model_name,
        network_index=network_index,
        max_ratings=max_ratings,
        test_size=test_size,
        random_state=random_state,
        train_df=train_df,
    )
    rating_span = float(
        max(train_df["Rating"].max(), test_df["Rating"].max())
        - min(train_df["Rating"].min(), test_df["Rating"].min())
    )
    reasonableness_limit = max(10.0, 10.0 * rating_span)
    all_results: list[dict[str, Any]] = []

    def _objective(trial: optuna.Trial) -> float:
        params = _trial_params(
            trial=trial,
            social_modes=selected_social_modes,
            k_min=k_min,
            k_max=k_max,
            lambda_reg_min=lambda_reg_min,
            lambda_reg_max=lambda_reg_max,
            w_main_min=w_main_min,
            w_main_max=w_main_max,
            w_user_min=w_user_min,
            w_user_max=w_user_max,
            lambda_social_min=lambda_social_min,
            lambda_social_max=lambda_social_max,
            beta_min=beta_min,
            beta_max=beta_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        row: dict[str, Any] = {"trial": trial.number, "status": "ok", **params}
        try:
            social_edges = build_social_edges(
                dataset=dataset,
                model_name=model_name,
                network_index=network_index,
                user_index=user_attributes.index,
                mode=cast(SocialMode, params["social_mode"]),
                beta=float(params["beta"]),
                gamma=float(params["gamma"]),
                dtype=np.float32,
            )
            if social_edges.n_edges == 0:
                row.update({"status": "pruned", "error": "no usable social edges"})
                all_results.append(row)
                raise optuna.exceptions.TrialPruned("no usable social edges")

            _, metrics = fit_social_cmf_split(
                train_df,
                test_df,
                user_attributes,
                social_edges,
                k=int(params["k"]),
                lambda_reg=float(params["lambda_reg"]),
                w_main=float(params["w_main"]),
                w_user=float(params["w_user"]),
                lambda_social=float(params["lambda_social"]),
                transform=transform,
                maxiter=maxiter,
                nthreads=nthreads,
                random_state=random_state,
                include_user_attributes=include_user_attributes,
            )
            edge_summary = {
                key: value
                for key, value in asdict(social_edges).items()
                if key not in {"row", "col", "val"}
            }
            row.update(
                {
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "metrics": metrics,
                    "social_edges": edge_summary,
                }
            )
            if not _metrics_are_usable(metrics, reasonableness_limit):
                row["status"] = "pruned"
                row["error"] = "non-finite or unreasonable-scale metrics"
                all_results.append(row)
                raise optuna.exceptions.TrialPruned(row["error"])

            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("social_edges", edge_summary)
            all_results.append(row)
            return float(metrics["rmse"])
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            row.update({"status": "error", "error": str(exc)})
            all_results.append(row)
            raise optuna.exceptions.TrialPruned(str(exc)) from exc

    def _print_trial(_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = f"{trial.value:.6f}" if trial.value is not None else "None"
        print(
            f"  [trial {trial.number + 1:3d}/{n_trials}] "
            f"state={trial.state.name} rmse={value}"
        )
        import mlflow as _mlflow

        if trial.value is not None and _mlflow.active_run():
            _mlflow.log_metric("social_trial_rmse", trial.value, step=trial.number)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        _objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[_print_trial],
    )

    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    best_trial = study.best_trial if complete_trials else None
    best_value = best_trial.value if best_trial is not None else None
    best_params = dict(best_trial.params) if best_trial is not None else {}
    # beta/gamma are sampled conditionally: if the best trial's social_mode
    # did not require them, they won't be in best_trial.params.  Fill in the
    # same fixed defaults used in _trial_params so callers always see both keys.
    best_params.setdefault("beta", 0.0)
    best_params.setdefault("gamma", 1.0)
    best_metrics = (
        dict(best_trial.user_attrs.get("metrics", {})) if best_trial is not None else {}
    )
    best_social_edges = (
        dict(best_trial.user_attrs.get("social_edges", {}))
        if best_trial is not None
        else {}
    )

    result = {
        "dataset": dataset,
        "model_name": model_name,
        "network_index": network_index,
        "objective": "rmse",
        "best_value": float(best_value) if best_value is not None else None,
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_social_edges": best_social_edges,
        "n_trials_requested": n_trials,
        "n_trials_complete": len(complete_trials),
        "include_user_attributes": include_user_attributes,
        "max_ratings": max_ratings,
        "test_size": test_size,
        "train_ratings": int(len(train_df)),
        "test_ratings": int(len(test_df)),
        "warm_test_only": True,
        "maxiter": maxiter,
        "random_state": random_state,
        "nthreads": nthreads,
        "transform": transform,
        "reasonableness_limit": reasonableness_limit,
        "search_space": {
            "k": [k_min, k_max],
            "lambda_reg": [lambda_reg_min, lambda_reg_max],
            "w_main": [w_main_min, w_main_max],
            "w_user": [w_user_min, w_user_max],
            "lambda_social": [lambda_social_min, lambda_social_max],
            "social_mode": list(selected_social_modes),
            "beta": [beta_min, beta_max],
            "gamma": [gamma_min, gamma_max],
        },
        "all_results": all_results,
    }
    save_social_search_results(result, output_path)

    # Log best params and edge diagnostics to MLflow when an active run exists.
    import mlflow as _mlflow

    if _mlflow.active_run() and best_params:
        _mlflow.log_params(
            {
                "social_best_k": best_params.get("k"),
                "social_best_lambda_reg": best_params.get("lambda_reg"),
                "social_best_w_main": best_params.get("w_main"),
                "social_best_w_user": best_params.get("w_user"),
                "social_best_lambda_social": best_params.get("lambda_social"),
                "social_best_mode": best_params.get("social_mode"),
                "social_best_beta": best_params.get("beta"),
                "social_best_gamma": best_params.get("gamma"),
            }
        )
        if best_social_edges:
            _mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in best_social_edges.items()
                    if isinstance(v, (int, float))
                }
            )
        if best_value is not None:
            _mlflow.log_metric("social_best_rmse", float(best_value))

    return result


def search_social_per_mode(
    dataset: str = Datasets.DEFAULT,
    model_name: str = "exponential",
    network_index: int = 0,
    n_trials: int = 200,
    timeout: int | None = None,
    max_ratings: int = 5000,
    test_size: float = 0.2,
    maxiter: int = 25,
    random_state: int = 42,
    nthreads: int = 1,
    include_user_attributes: bool = True,
    modes: Iterable[SocialMode] = SOCIAL_MODES,
    k_min: int = 5,
    k_max: int = 50,
    lambda_reg_min: float = 0.01,
    lambda_reg_max: float = 10.0,
    w_main_min: float = 0.1,
    w_main_max: float = 1.0,
    w_user_min: float = 0.01,
    w_user_max: float = 1.0,
    lambda_social_min: float = 1e-4,
    lambda_social_max: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 1.0,
    gamma_min: float = 0.1,
    gamma_max: float = 3.0,
    transform: str = "standard",
    output_dir: str | Path | None = None,
    train_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Run a separate Optuna study for each social mode and compare results.

    This is the per-mode study variant recommended for publication and reviewer
    contexts, where ``boundary_downweight`` has been identified as the best mode
    in prior smoke tests.  Each mode is searched independently so that Optuna's
    TPE concentrates its budget on the relevant parameter subspace.

    Args:
        modes:      Social modes to evaluate.  Defaults to all four.
        output_dir: Directory to write per-mode JSON results.  ``None`` writes
                    each result alongside the default dataset results path.
        train_df:   Optional pre-split training DataFrame from the pipeline's
                    global split (passed through to avoid data leakage).

    Returns:
        Dict mapping ``social_mode`` → best-result dict from
        :func:`search_social_regularized_params`, with an extra
        ``"comparison"`` key summarising best RMSE per mode.
    """
    mode_list: list[SocialMode] = list(modes)
    per_mode_results: dict[str, dict] = {}

    for mode in mode_list:
        print(f"\n{'='*55}")
        print(f"Per-mode search: {mode.upper()}")
        print("=" * 55)
        out_path: Path | None = None
        if output_dir is not None:
            out_path = (
                Path(output_dir)
                / f"social_search_{model_name}_{network_index:03d}_{mode}.json"
            )

        result = search_social_regularized_params(
            dataset=dataset,
            model_name=model_name,
            network_index=network_index,
            n_trials=n_trials,
            timeout=timeout,
            max_ratings=max_ratings,
            test_size=test_size,
            maxiter=maxiter,
            random_state=random_state,
            nthreads=nthreads,
            include_user_attributes=include_user_attributes,
            social_modes=[mode],
            k_min=k_min,
            k_max=k_max,
            lambda_reg_min=lambda_reg_min,
            lambda_reg_max=lambda_reg_max,
            w_main_min=w_main_min,
            w_main_max=w_main_max,
            w_user_min=w_user_min,
            w_user_max=w_user_max,
            lambda_social_min=lambda_social_min,
            lambda_social_max=lambda_social_max,
            beta_min=beta_min,
            beta_max=beta_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            transform=transform,
            output_path=out_path,
            train_df=train_df,
        )
        per_mode_results[mode] = result

    # Build comparison summary.
    comparison = {mode: res.get("best_value") for mode, res in per_mode_results.items()}
    per_mode_results["comparison"] = {
        "best_rmse_per_mode": comparison,
        "best_mode": min(
            (m for m, v in comparison.items() if v is not None),
            key=lambda m: comparison[m],  # type: ignore[arg-type]
            default=None,
        ),
    }
    print("\nPer-mode comparison (best RMSE):")
    for mode, val in sorted(comparison.items(), key=lambda kv: (kv[1] is None, kv[1])):
        marker = (
            " <-- best" if mode == per_mode_results["comparison"]["best_mode"] else ""
        )
        print(f"  {mode:<25s}: {val}{marker}")

    return per_mode_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Phase 6 social-regularized CMF hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=Datasets.DEFAULT, choices=Datasets.ALL)
    parser.add_argument(
        "--model", dest="model_name", default="exponential", choices=Models.ALL
    )
    parser.add_argument("--network-index", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-ratings", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--transform", default="standard")
    parser.add_argument(
        "--social-modes",
        nargs="+",
        default=list(SOCIAL_MODES),
        choices=list(SOCIAL_MODES),
    )
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=50)
    parser.add_argument("--lambda-reg-min", type=float, default=0.01)
    parser.add_argument("--lambda-reg-max", type=float, default=10.0)
    parser.add_argument("--w-main-min", type=float, default=0.1)
    parser.add_argument("--w-main-max", type=float, default=1.0)
    parser.add_argument("--w-user-min", type=float, default=0.01)
    parser.add_argument("--w-user-max", type=float, default=1.0)
    parser.add_argument("--lambda-social-min", type=float, default=1e-4)
    parser.add_argument("--lambda-social-max", type=float, default=1.0)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--gamma-min", type=float, default=0.1)
    parser.add_argument("--gamma-max", type=float, default=3.0)
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--no-user-attributes",
        action="store_true",
        help="Search without passing side-user attributes as U.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = search_social_regularized_params(
        dataset=args.dataset,
        model_name=args.model_name,
        network_index=args.network_index,
        n_trials=args.n_trials,
        timeout=args.timeout,
        max_ratings=args.max_ratings,
        test_size=args.test_size,
        maxiter=args.maxiter,
        random_state=args.random_state,
        nthreads=args.nthreads,
        include_user_attributes=not args.no_user_attributes,
        social_modes=cast(Iterable[SocialMode], args.social_modes),
        k_min=args.k_min,
        k_max=args.k_max,
        lambda_reg_min=args.lambda_reg_min,
        lambda_reg_max=args.lambda_reg_max,
        w_main_min=args.w_main_min,
        w_main_max=args.w_main_max,
        w_user_min=args.w_user_min,
        w_user_max=args.w_user_max,
        lambda_social_min=args.lambda_social_min,
        lambda_social_max=args.lambda_social_max,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        transform=args.transform,
        output_path=args.output_path,
    )
    if result["best_params"]:
        print(json.dumps(_json_ready(result["best_params"]), indent=2))
    else:
        print("No complete trials produced usable metrics.")


if __name__ == "__main__":
    main()
