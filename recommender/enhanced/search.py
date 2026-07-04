"""
Optuna-based hyperparameter search and result persistence for enhanced CMF.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets
from config import Defaults
from recommender.data import rating_reasonableness_limit
from recommender.enhanced.model import evaluate_cmf_with_user_attributes


def _enhanced_trial_rmse_is_usable(mean_rmse: float, limit: float) -> bool:
    return np.isfinite(mean_rmse) and mean_rmse <= limit


def search_enhanced_params(
    data: pd.DataFrame,
    user_attributes: pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 3,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    cmf_nthreads: int = -1,
    random_state: int = 42,
) -> dict:
    """
    Bayesian hyperparameter search (Optuna TPE) over ``k``, ``lambda_reg``,
    ``w_main``, and ``w_user`` for the enhanced CMF model.

    Args:
        data:            Full (training) ratings DataFrame.
        user_attributes: Raw (unscaled) feature DataFrame indexed by 0-based
                         ``UserId``.
        n_trials:        Number of Optuna trials (default 50).
        n_splits:        CV splits per trial.

    Returns:
        Dict with ``best_params`` (``k``, ``lambda_reg``, ``w_main``,
        ``w_user``) and ``all_results`` (list, one dict per trial).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_results: list[dict] = []
    rmse_limit = rating_reasonableness_limit(cast(pd.Series, data["Rating"]))

    def _objective(trial: optuna.Trial) -> float:
        k_val = trial.suggest_int("k", 5, 50)
        lambda_val = trial.suggest_float("lambda_reg", 0.01, 10.0, log=True)
        w_main_val = trial.suggest_float("w_main", 0.1, 1.0)
        w_user_val = trial.suggest_float("w_user", 0.01, 1.0, log=True)

        try:
            split_results = evaluate_cmf_with_user_attributes(
                data,
                user_attributes,
                k=k_val,
                lambda_reg=lambda_val,
                w_main=w_main_val,
                w_user=w_user_val,
                n_splits=n_splits,
                method=method,
                maxiter=maxiter,
                cmf_nthreads=cmf_nthreads,
            )
        except (RuntimeError, ValueError) as exc:
            raise optuna.exceptions.TrialPruned(str(exc)) from exc

        if not split_results:
            raise optuna.exceptions.TrialPruned("no CV splits")

        mean_rmse = float(np.mean([r["rmse_enhanced"] for r in split_results]))
        if not _enhanced_trial_rmse_is_usable(mean_rmse, rmse_limit):
            raise optuna.exceptions.TrialPruned(
                "non-finite or unreasonable-scale RMSE"
            )

        all_results.append(
            {
                "k": k_val,
                "lambda_reg": lambda_val,
                "w_main": w_main_val,
                "w_user": w_user_val,
                "rmse": mean_rmse,
            }
        )
        import mlflow as _mlflow

        if _mlflow.active_run():
            _mlflow.log_metric("enhanced_trial_rmse", mean_rmse, step=trial.number)
        return mean_rmse

    def _print_trial(_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        params = trial.params
        value = f"{trial.value:.4f}" if trial.value is not None else "None"
        print(
            f"  [trial {trial.number + 1:2d}/{n_trials}] "
            f"state={trial.state.name} "
            f"k={params.get('k')}  "
            f"lambda={params.get('lambda_reg', 0.0):.4f}  "
            f"w_main={params.get('w_main', 0.0):.3f}  "
            f"w_user={params.get('w_user', 0.0):.3f}  "
            f"RMSE={value}"
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        _objective,
        n_trials=n_trials,
        callbacks=[_print_trial],
    )

    complete = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise RuntimeError("Enhanced hyperparameter search produced no usable trials.")

    best = study.best_params
    best_params = {
        "k": best["k"],
        "lambda_reg": best["lambda_reg"],
        "w_main": best["w_main"],
        "w_user": best["w_user"],
    }
    print(f"\nBest enhanced params: {best_params}  RMSE={study.best_value:.4f}")

    import mlflow as _mlflow

    if _mlflow.active_run():
        _mlflow.log_params(
            {
                "enhanced_best_k": best_params["k"],
                "enhanced_best_lambda_reg": best_params["lambda_reg"],
                "enhanced_best_w_main": best_params["w_main"],
                "enhanced_best_w_user": best_params["w_user"],
            }
        )
        _mlflow.log_metric("enhanced_best_rmse", study.best_value)

    return {"best_params": best_params, "all_results": all_results}


def save_enhanced_search_results(
    search_result: dict,
    path: Path | None = None,
) -> None:
    """
    Persist *search_result* (from :func:`search_enhanced_params`) to a JSON file.

    Saved at ``data/enhanced_search_results.json`` by default so that
    :mod:`analysis.shap_analysis` can load the best hyperparameters without
    re-running the search.

    Args:
        search_result: Dict with ``best_params`` and ``all_results``.
        path:          Override destination path.
    """
    import json

    dest = path or DatasetPaths(Datasets.DEFAULT).ENHANCED_RESULTS
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(search_result, indent=2), encoding="utf-8")
    tmp.replace(dest)
    print(f"Enhanced search results saved → {dest}")
