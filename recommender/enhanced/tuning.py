"""Shared hyperparameter campaign for enhanced / social CMF (Optuna)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config import DatasetPaths, Defaults, Models
from recommender.baseline import search_baseline_params
from recommender.enhanced.features import load_network_features
from recommender.enhanced.search import (
    save_enhanced_search_results,
    search_enhanced_params,
)


@dataclass
class HyperparamCampaignResult:
    """Structured result of a recommend/hypertune Optuna campaign."""

    sample_features: pd.DataFrame | None
    sample_model_name: str | None
    baseline_search: dict[str, Any]
    enhanced_search: dict[str, Any]
    social_regularization: bool
    best_k_b: int
    best_lambda_b: float
    best_k_e: int
    best_lambda_e: float
    best_w_main: float
    best_w_user: float
    social_mode: str
    lambda_social: float
    social_beta: float
    social_gamma: float


def _default_searches(
    *,
    social_regularization: bool,
    social_mode: str,
    lambda_social: float,
    social_beta: float,
    social_gamma: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    best_k_b = Defaults.K
    best_lambda_b = Defaults.LAMBDA_REG
    baseline_search = {
        "best_params": {"k": best_k_b, "lambda_reg": best_lambda_b},
        "all_results": [],
    }
    enhanced_params: dict[str, Any] = {
        "k": Defaults.K,
        "lambda_reg": Defaults.LAMBDA_REG,
        "w_main": Defaults.W_MAIN,
        "w_user": Defaults.W_USER,
    }
    if social_regularization:
        enhanced_params.update(
            {
                "lambda_social": lambda_social,
                "social_mode": social_mode,
                "beta": social_beta,
                "gamma": social_gamma,
            }
        )
    enhanced_search = {"best_params": enhanced_params, "all_results": []}
    return baseline_search, enhanced_search


def find_sample_features(
    selected_models: list[str],
    *,
    dataset: str,
    include_communities: bool,
) -> tuple[pd.DataFrame | None, str | None]:
    for model_name in selected_models:
        features = load_network_features(
            model_name,
            0,
            include_communities=include_communities,
            dataset=dataset,
        )
        if features is not None:
            return features, model_name
    return None, None


def run_hyperparam_campaign(
    train_df: pd.DataFrame,
    *,
    dataset: str,
    selected_models: list[str] | None = None,
    include_communities: bool = True,
    social_regularization: bool = False,
    social_mode: str = "boundary_downweight",
    social_normalization: str = "mean_weight",
    lambda_social: float = 0.001,
    social_beta: float = 0.5,
    social_gamma: float = 1.0,
    social_n_trials: int = Defaults.SOCIAL_N_TRIALS,
    social_search_max_ratings: int | None = None,
    baseline_n_trials: int = 50,
    enhanced_n_trials: int = 50,
    n_splits: int = 3,
    cmf_method: str = Defaults.CMF_METHOD,
    cmf_maxiter: int = Defaults.CMF_MAXITER,
    cmf_nthreads: int = -1,
    random_state: int = Defaults.CMF_RANDOM_STATE,
    search_baseline: bool = True,
    require_features: bool = False,
) -> HyperparamCampaignResult:
    """Run Optuna searches shared by recommend and hypertune steps.

    When *search_baseline* is False (hypertune), only enhanced/social search runs.
    When no feature files exist and *require_features* is False, returns defaults.
    """
    models = selected_models or list(Models.ALL)
    dp = DatasetPaths(dataset)
    sample_features, sample_model_name = find_sample_features(
        models,
        dataset=dataset,
        include_communities=include_communities,
    )

    if sample_features is None:
        if require_features:
            raise FileNotFoundError(
                "No feature files found. Run --steps centrality first."
            )
        baseline_search, enhanced_search = _default_searches(
            social_regularization=social_regularization,
            social_mode=social_mode,
            lambda_social=lambda_social,
            social_beta=social_beta,
            social_gamma=social_gamma,
        )
    else:
        if search_baseline:
            print(
                "Searching best baseline hyperparameters "
                "(Optuna TPE — k, lambda_reg) …"
            )
            baseline_search = search_baseline_params(
                train_df,
                n_trials=baseline_n_trials,
                n_splits=n_splits,
                method=cmf_method,
                maxiter=cmf_maxiter,
                nthreads=cmf_nthreads,
                random_state=random_state,
            )
        else:
            baseline_search = {
                "best_params": {"k": Defaults.K, "lambda_reg": Defaults.LAMBDA_REG},
                "all_results": [],
            }

        if social_regularization:
            from recommender.enhanced.social_search import (
                search_social_regularized_params,
            )

            print(
                f"Searching best social CMF hyperparameters (Optuna TPE) "
                f"using {sample_model_name} network #000 "
                f"({social_n_trials} trials) …"
            )
            enhanced_search = search_social_regularized_params(
                dataset=dataset,
                model_name=sample_model_name or models[0],
                network_index=0,
                n_trials=social_n_trials,
                max_ratings=social_search_max_ratings,
                maxiter=cmf_maxiter,
                random_state=random_state,
                nthreads=cmf_nthreads,
                include_user_attributes=True,
                social_modes=(social_mode,),  # type: ignore[arg-type]
                social_normalization=social_normalization,  # type: ignore[arg-type]
                output_path=dp.SOCIAL_RESULTS,
                train_df=train_df,
            )
            if not enhanced_search.get("best_params"):
                raise RuntimeError(
                    "Social hyperparameter search produced no usable trials."
                )
        else:
            print(
                f"Searching best enhanced hyperparameters (Optuna TPE — k, "
                f"lambda_reg, w_main, w_user) using first "
                f"{sample_model_name} network …"
            )
            enhanced_search = search_enhanced_params(
                train_df,
                sample_features,
                n_trials=enhanced_n_trials,
                n_splits=n_splits,
                method=cmf_method,
                maxiter=cmf_maxiter,
                cmf_nthreads=cmf_nthreads,
                random_state=random_state,
            )
            save_enhanced_search_results(enhanced_search, path=dp.ENHANCED_RESULTS)

    bp = baseline_search["best_params"]
    ep = enhanced_search["best_params"]
    return HyperparamCampaignResult(
        sample_features=sample_features,
        sample_model_name=sample_model_name,
        baseline_search=baseline_search,
        enhanced_search=enhanced_search,
        social_regularization=social_regularization,
        best_k_b=int(bp["k"]),
        best_lambda_b=float(bp["lambda_reg"]),
        best_k_e=int(ep["k"]),
        best_lambda_e=float(ep["lambda_reg"]),
        best_w_main=float(ep.get("w_main", Defaults.W_MAIN)),
        best_w_user=float(ep.get("w_user", Defaults.W_USER)),
        social_mode=str(ep.get("social_mode", social_mode)),
        lambda_social=float(ep.get("lambda_social", lambda_social)),
        social_beta=float(ep.get("beta", social_beta)),
        social_gamma=float(ep.get("gamma", social_gamma)),
    )
