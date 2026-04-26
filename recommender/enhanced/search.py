"""
Optuna-based hyperparameter search and result persistence for enhanced CMF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets
from recommender.enhanced.model import evaluate_cmf_with_user_attributes


def search_enhanced_params(
    data: pd.DataFrame,
    user_attributes: pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 3,
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

    def _objective(trial: optuna.Trial) -> float:
        k_val = trial.suggest_int("k", 5, 50)
        lambda_val = trial.suggest_float("lambda_reg", 0.01, 10.0, log=True)
        w_main_val = trial.suggest_float("w_main", 0.1, 1.0)
        w_user_val = trial.suggest_float("w_user", 0.01, 1.0, log=True)

        split_results = evaluate_cmf_with_user_attributes(
            data,
            user_attributes,
            k=k_val,
            lambda_reg=lambda_val,
            w_main=w_main_val,
            w_user=w_user_val,
            n_splits=n_splits,
        )
        if not split_results:
            raise optuna.exceptions.TrialPruned()

        mean_rmse = float(np.mean([r["rmse_enhanced"] for r in split_results]))
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

    study = optuna.create_study(direction="minimize")
    study.optimize(
        _objective,
        n_trials=n_trials,
        callbacks=[
            lambda study, trial: print(
                f"  [trial {trial.number + 1:2d}/{n_trials}] "
                f"k={trial.params.get('k')}  "
                f"lambda={trial.params.get('lambda_reg'):.4f}  "
                f"w_main={trial.params.get('w_main'):.3f}  "
                f"w_user={trial.params.get('w_user'):.3f}  "
                f"RMSE={trial.value:.4f}"
            )
        ],
    )

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
    path: "Path | None" = None,
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
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(search_result, fh, indent=2)
    print(f"Enhanced search results saved → {dest}")
