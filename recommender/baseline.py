"""
Baseline Collaborative Matrix Factorisation (CMF) recommender.

Wraps the :class:`cmfrec.CMF` model with helpers for cross-validated
evaluation, hyperparameter search, and final model training.

The key idea is a standard matrix-factorisation approach in which a latent
factor matrix is learned jointly from user-item interactions.  The pipeline
defaults to L-BFGS so baseline and social-regularized CMF fits use the same
optimizer family.  No side information is used at this stage; side information
is introduced in :mod:`recommender.enhanced`.

Functions
---------
train_model
    Fit a CMF model to a training DataFrame and return it.
evaluate_with_cv
    Evaluate a parameter combination via repeated random splits.
search_best_params
    Randomised hyperparameter search over *k* and *lambda_reg*.
train_final_model
    Fit the final model on the full dataset with the best parameters.
run_complete_example
    End-to-end demonstration pipeline.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipy.stats as stats

from config import DatasetPaths, Datasets, Defaults
from recommender._cmfrec import CMF

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_search_results(search_result: dict, path: Path | None = None) -> None:
    """
    Persist *search_result* (from :func:`search_best_params`) to a JSON file.

    Saved at ``data/baseline_search_results.json`` by default so that
    :mod:`visualization.model_plots` can generate plots without re-running
    the search.

    Args:
        search_result: Dict with ``best_params`` and ``all_results``.
        path:          Override destination path.
    """
    dest = path or DatasetPaths(Datasets.DEFAULT).BASELINE_RESULTS
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(search_result, indent=2), encoding="utf-8")
    tmp.replace(dest)
    print(f"Search results saved → {dest}")


from recommender.data import (
    load_dataset,
    rating_reasonableness_limit,
    split_data_single,
    evaluate_single_split,
)

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_model(
    train_data: pd.DataFrame,
    k: int = Defaults.K,
    lambda_reg: float = Defaults.LAMBDA_REG,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    nthreads: int = -1,
    random_state: int = Defaults.CMF_RANDOM_STATE,
) -> CMF:
    """
    Fit a CMF model on *train_data*.

    Args:
        train_data:  DataFrame with ``UserId``, ``ItemId``, ``Rating``
                     columns.
        k:           Number of latent factors.
        lambda_reg:  L2 regularisation strength.
        method:      CMF optimizer. Defaults to the pipeline-wide method.
        maxiter:     L-BFGS iteration budget.
        nthreads:    BLAS threads for cmfrec.
        random_state: Random seed for L-BFGS initialization.

    Returns:
        Fitted :class:`cmfrec.CMF` instance.
    """
    kwargs = {
        "method": method,
        "k": k,
        "lambda_": lambda_reg,
        "verbose": False,
        "nthreads": nthreads,
    }
    if method == "lbfgs":
        kwargs.update({"maxiter": maxiter, "random_state": random_state})
    model = CMF(**kwargs)
    model.fit(X=train_data)
    return model


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------


def evaluate_with_cv(
    data: pd.DataFrame,
    k: int = Defaults.K,
    lambda_reg: float = Defaults.LAMBDA_REG,
    n_splits: int = 5,
    test_size: float = 0.2,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    nthreads: int = -1,
    random_state: int = Defaults.CMF_RANDOM_STATE,
) -> dict[str, float]:
    """
    Evaluate a parameter combination via *n_splits* random train/test splits.

    Args:
        data:       Full ratings DataFrame.
        k:          Number of latent factors.
        lambda_reg: L2 regularisation strength.
        n_splits:   Number of random splits.
        test_size:  Fraction of data to use as test set.
        method:     CMF optimizer.
        maxiter:    L-BFGS iteration budget.
        nthreads:   BLAS threads for cmfrec.
        random_state: Base seed for L-BFGS initialization.

    Returns:
        Dict with keys ``rmse``, ``mae``, ``r2`` containing the mean value
        across all splits.
    """
    rmses, maes, r2s = [], [], []

    for split_idx in range(n_splits):
        train_df, test_df = split_data_single(
            data, test_size=test_size, random_state=split_idx
        )
        model = train_model(
            train_df,
            k=k,
            lambda_reg=lambda_reg,
            method=method,
            maxiter=maxiter,
            nthreads=nthreads,
            random_state=random_state + split_idx,
        )
        result = evaluate_single_split(model, test_df)
        rmses.append(result["rmse"])
        maes.append(result["mae"])
        r2s.append(result["r2"])

    return {
        "rmse": float(np.mean(rmses)),
        "mae": float(np.mean(maes)),
        "r2": float(np.mean(r2s)),
    }


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------


def search_best_params(
    data: pd.DataFrame,
    param_grid: dict | None = None,
    n_iter: int = 20,
    n_splits: int = 3,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    nthreads: int = -1,
) -> dict:
    """
    Randomised search over *k* and *lambda_reg*.

    When *param_grid* is ``None``, a default grid is used:
    ``k ∈ [5, 50]`` (randint) and ``lambda_reg ∈ [0.01, 10]`` (uniform).

    Args:
        data:       Full ratings DataFrame.
        param_grid: Dict with keys ``k`` and ``lambda_reg`` whose values are
                    scipy distribution objects (or integers / floats for fixed
                    values).  Defaults to None (uses built-in distributions).
        n_iter:     Number of random parameter combinations to try.
        n_splits:   Number of CV splits per combination.

    Returns:
        Dict with keys ``best_params`` (dict) and ``all_results``
        (list of dicts, one per combination).
    """
    warnings.warn(
        "search_best_params is deprecated and will be removed in a future version. "
        "Use search_baseline_params (Optuna, 50 trials) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if param_grid is None:
        param_grid = {
            "k": stats.randint(5, 51),
            "lambda_reg": stats.uniform(0.01, 10.0),
        }

    all_results = []
    best_rmse = float("inf")
    best_params: dict = {}

    for iteration in range(n_iter):
        k_val = int(param_grid["k"].rvs())
        lambda_val = float(param_grid["lambda_reg"].rvs())
        metrics = evaluate_with_cv(
            data,
            k=k_val,
            lambda_reg=lambda_val,
            n_splits=n_splits,
            method=method,
            maxiter=maxiter,
            nthreads=nthreads,
        )
        result_record = {"k": k_val, "lambda_reg": lambda_val, **metrics}
        all_results.append(result_record)

        print(
            f"  [{iteration+1:2d}/{n_iter}] k={k_val:3d}  "
            f"lambda={lambda_val:.4f}  RMSE={metrics['rmse']:.4f}"
        )

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_params = {"k": k_val, "lambda_reg": lambda_val}

    print(f"\nBest params: {best_params}  RMSE={best_rmse:.4f}")
    return {"best_params": best_params, "all_results": all_results}


def search_baseline_params(
    data: pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 3,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    nthreads: int = -1,
    random_state: int = 42,
) -> dict:
    """
    Optuna TPE hyperparameter search over *k* and *lambda_reg*.

    Uses the same TPE sampler as
    :func:`recommender.enhanced.search_enhanced_params` so the two searches
    are directly comparable.  Only the two parameters relevant to a plain CMF
    (no side-information) are tuned; ``w_main`` and ``w_user`` are not
    involved.

    Args:
        data:      Full ratings DataFrame.
        n_trials:  Number of Optuna trials.
        n_splits:  Number of CV splits per trial.

    Returns:
        Dict with keys ``best_params`` (``k``, ``lambda_reg``) and
        ``all_results`` (one dict per trial).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    all_results: list[dict] = []
    rmse_limit = rating_reasonableness_limit(cast(pd.Series, data["Rating"]))

    def _objective(trial: "optuna.Trial") -> float:
        k_val = trial.suggest_int("k", 5, 50)
        lambda_val = trial.suggest_float("lambda_reg", 0.01, 10.0, log=True)
        try:
            metrics = evaluate_with_cv(
                data,
                k=k_val,
                lambda_reg=lambda_val,
                n_splits=n_splits,
                method=method,
                maxiter=maxiter,
                nthreads=nthreads,
            )
        except (RuntimeError, ValueError) as exc:
            raise optuna.exceptions.TrialPruned(str(exc)) from exc

        if not np.isfinite(metrics["rmse"]) or metrics["rmse"] > rmse_limit:
            raise optuna.exceptions.TrialPruned(
                "non-finite or unreasonable-scale RMSE"
            )

        all_results.append({"k": k_val, "lambda_reg": lambda_val, **metrics})
        import mlflow

        if mlflow.active_run():
            mlflow.log_metric("baseline_trial_rmse", metrics["rmse"], step=trial.number)
        return metrics["rmse"]

    def _print_trial(_study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        value = f"{trial.value:.6f}" if trial.value is not None else "None"
        print(
            f"  [baseline trial {trial.number + 1:3d}/{n_trials}] "
            f"state={trial.state.name} rmse={value}",
            flush=True,
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        _objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_print_trial],
    )

    complete = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise RuntimeError("Baseline hyperparameter search produced no usable trials.")

    best_params = {
        "k": study.best_params["k"],
        "lambda_reg": study.best_params["lambda_reg"],
    }
    print(f"Best baseline params: {best_params}  RMSE={study.best_value:.4f}")

    import mlflow

    if mlflow.active_run():
        mlflow.log_params(
            {
                "baseline_best_k": best_params["k"],
                "baseline_best_lambda_reg": best_params["lambda_reg"],
            }
        )
        mlflow.log_metric("baseline_best_rmse", study.best_value)

    return {"best_params": best_params, "all_results": all_results}


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------


def train_final_model(
    data: pd.DataFrame,
    k: int = Defaults.K,
    lambda_reg: float = Defaults.LAMBDA_REG,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    nthreads: int = -1,
    random_state: int = Defaults.CMF_RANDOM_STATE,
) -> CMF:
    """
    Fit the final CMF model on the complete dataset.

    Args:
        data:       Full ratings DataFrame.
        k:          Number of latent factors.
        lambda_reg: L2 regularisation strength.
        method:     CMF optimizer.
        maxiter:    L-BFGS iteration budget.
        nthreads:   BLAS threads for cmfrec.
        random_state: Random seed for L-BFGS initialization.

    Returns:
        Fitted :class:`cmfrec.CMF` instance trained on all data.
    """
    print(f"Training final model: method={method}, k={k}, " f"lambda_reg={lambda_reg}")
    return train_model(
        data,
        k=k,
        lambda_reg=lambda_reg,
        method=method,
        maxiter=maxiter,
        nthreads=nthreads,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# End-to-end demo
# ---------------------------------------------------------------------------


def run_complete_example() -> None:
    """
    Demonstrate the full baseline training and evaluation pipeline.

    Loads the default dataset, runs a hyperparameter search, trains the final
    model and prints a brief evaluation summary.
    """
    print("=== Baseline CMF — complete example ===\n")
    data = load_dataset()

    print("\nRunning hyperparameter search …")
    search_result = search_best_params(data, n_iter=50, n_splits=3)
    best = search_result["best_params"]

    print("\nEvaluating best parameters with 5-fold CV …")
    cv_metrics = evaluate_with_cv(data, k=best["k"], lambda_reg=best["lambda_reg"])
    print(f"  RMSE: {cv_metrics['rmse']:.4f}")
    print(f"  MAE : {cv_metrics['mae']:.4f}")
    print(f"  R²  : {cv_metrics['r2']:.4f}")

    save_search_results(search_result)

    train_final_model(data, k=best["k"], lambda_reg=best["lambda_reg"])
    print("\nFinal model trained successfully.")


if __name__ == "__main__":
    run_complete_example()
