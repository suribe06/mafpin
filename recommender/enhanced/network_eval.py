"""
Single-network and batch network evaluation for the enhanced CMF model.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets, Models, Defaults
from networks.artifacts import NetworkArtifacts
from recommender.baseline import train_model
from recommender.data import evaluate_ranking, evaluate_single_split, rating_reasonableness_limit
from recommender.enhanced.features import load_network_features
from recommender.enhanced.model import (
    evaluate_cmf_with_user_attributes,
    filter_to_feature_users,
    iter_warm_splits,
)
from recommender.enhanced.social_regularization import (
    SocialNormalization,
    SocialMode,
    build_social_edges,
    fit_social_cmf_split,
)
from recommender.enhanced.workers import _worker_init, _eval_network_worker


def _available_centrality_indices(dp: DatasetPaths, model_name: str) -> list[int]:
    return NetworkArtifacts(dp.BASE.name, paths=dp).list_centrality_indices(model_name)


def _available_social_indices(dp: DatasetPaths, model_name: str) -> list[int]:
    return NetworkArtifacts(dp.BASE.name, paths=dp).list_complete_indices(model_name)


def evaluate_single_network(
    data: pd.DataFrame,
    model_name: str,
    network_index: int,
    k: int = 20,
    lambda_reg: float = 1.0,
    w_main: float = Defaults.W_MAIN,
    w_user: float = Defaults.W_USER,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    transform: str = "standard",
    include_communities: bool = True,
    n_splits: int = 5,
    baseline_k: int | None = None,
    baseline_lambda: float | None = None,
    compute_ranking: bool = False,
    ranking_k: int = 10,
    dataset: str | None = None,
    cmf_nthreads: int = -1,
    use_social_regularization: bool = False,
    social_mode: SocialMode = "boundary_downweight",
    lambda_social: float = 0.001,
    social_beta: float = 0.5,
    social_gamma: float = 1.0,
    social_normalization: SocialNormalization = "mean_weight",
) -> list[dict]:
    """
    Load features and evaluate CMF for one (model, index) pair.

    Args:
        data:                Full ratings DataFrame.
        model_name:          Diffusion model name.
        network_index:       Zero-based network index.
        k:                   Number of latent factors.
        lambda_reg:          L2 regularisation strength.
        w_main:              Weight for main rating-matrix loss.
        w_user:              Weight for user side-information loss.
        method:              CMF optimizer for the non-social path and paired baseline.
        maxiter:             L-BFGS iteration budget.
        transform:           Feature normalisation method.
        include_communities: Whether to include community features.
        n_splits:            Number of cross-validation splits.
        baseline_k:          Latent factors for the paired plain-CMF baseline.
        baseline_lambda:     L2 regularisation for the paired plain-CMF baseline.
        compute_ranking:     When ``True``, compute NDCG@K, Precision@K, Recall@K,
                             and MRR.
        ranking_k:           Cut-off for rank-based metrics.
        dataset:             Dataset name.  Defaults to ``Datasets.DEFAULT``.
        cmf_nthreads:        BLAS threads for cmfrec.
        use_social_regularization: Fit with Phase 6 social regularization.
        social_mode:         Social edge weighting mode.
        lambda_social:       Social regularization strength.
        social_beta:         Boundary penalty parameter for social edge weights.
        social_gamma:        Shared-community gain parameter for social edge weights.

    Returns:
        List of per-split result dicts, or empty list on failure.
    """
    features = load_network_features(
        model_name,
        network_index,
        include_communities=include_communities,
        dataset=dataset,
    )
    if features is None:
        print(f"  Skipping {model_name} #{network_index:03d}: features not found.")
        return []

    if use_social_regularization:
        return evaluate_social_cmf_with_user_attributes(
            data=data,
            user_attributes=features,
            model_name=model_name,
            network_index=network_index,
            k=k,
            lambda_reg=lambda_reg,
            w_main=w_main,
            w_user=w_user,
            n_splits=n_splits,
            transform=transform,
            baseline_k=baseline_k,
            baseline_lambda=baseline_lambda,
            compute_ranking=compute_ranking,
            ranking_k=ranking_k,
            dataset=dataset,
            cmf_nthreads=cmf_nthreads,
            social_mode=social_mode,
            lambda_social=lambda_social,
            social_beta=social_beta,
            social_gamma=social_gamma,
            social_normalization=social_normalization,
            maxiter=maxiter,
            method=method,
        )

    return evaluate_cmf_with_user_attributes(
        data,
        features,
        k=k,
        lambda_reg=lambda_reg,
        w_main=w_main,
        w_user=w_user,
        method=method,
        maxiter=maxiter,
        n_splits=n_splits,
        transform=transform,
        baseline_k=baseline_k,
        baseline_lambda=baseline_lambda,
        compute_ranking=compute_ranking,
        ranking_k=ranking_k,
        cmf_nthreads=cmf_nthreads,
    )


def evaluate_social_cmf_with_user_attributes(
    data: pd.DataFrame,
    user_attributes: pd.DataFrame,
    model_name: str,
    network_index: int,
    k: int = 20,
    lambda_reg: float = 1.0,
    w_main: float = Defaults.W_MAIN,
    w_user: float = Defaults.W_USER,
    n_splits: int = 5,
    test_size: float = 0.2,
    transform: str = "standard",
    baseline_k: int | None = None,
    baseline_lambda: float | None = None,
    compute_ranking: bool = False,
    ranking_k: int = 10,
    dataset: str | None = None,
    cmf_nthreads: int = -1,
    social_mode: SocialMode = "boundary_downweight",
    lambda_social: float = 0.001,
    social_beta: float = 0.5,
    social_gamma: float = 1.0,
    social_normalization: SocialNormalization = "mean_weight",
    maxiter: int = Defaults.CMF_MAXITER,
    method: str = Defaults.CMF_METHOD,
) -> list[dict]:
    """Evaluate social-regularized CMF on repeated warm train/test splits."""
    filtered = filter_to_feature_users(data, user_attributes)
    if filtered is None:
        return []

    social_edges = build_social_edges(
        dataset=dataset or Datasets.DEFAULT,
        model_name=model_name,
        network_index=network_index,
        user_index=user_attributes.index,
        mode=social_mode,
        beta=social_beta,
        gamma=social_gamma,
        normalization=social_normalization,
        dtype=np.float32,
    )
    if social_edges.n_edges == 0:
        print(f"  Skipping {model_name} #{network_index:03d}: no usable social edges.")
        return []

    results: list[dict] = []
    for split_idx, train_df, warm_test in iter_warm_splits(
        filtered, n_splits=n_splits, test_size=test_size
    ):
        social_model, social_metrics = fit_social_cmf_split(
            train_df,
            warm_test,
            user_attributes,
            social_edges,
            k=k,
            lambda_reg=lambda_reg,
            w_main=w_main,
            w_user=w_user,
            lambda_social=lambda_social,
            transform=transform,
            maxiter=maxiter,
            nthreads=cmf_nthreads,
            random_state=Defaults.CMF_RANDOM_STATE + split_idx,
            include_user_attributes=True,
        )

        if baseline_k is not None and baseline_lambda is not None:
            baseline_model = train_model(
                train_df,
                k=baseline_k,
                lambda_reg=baseline_lambda,
                method=method,
                maxiter=maxiter,
                nthreads=cmf_nthreads,
                random_state=Defaults.CMF_RANDOM_STATE + split_idx,
            )
            baseline_metrics = evaluate_single_split(baseline_model, warm_test)
            baseline_rmse = baseline_metrics["rmse"]
        else:
            baseline_metrics = {
                "rmse": float("nan"),
                "mae": float("nan"),
                "r2": float("nan"),
            }
            baseline_rmse = float("nan")

        result: dict = {
            "rmse_enhanced": social_metrics["rmse"],
            "rmse_baseline": baseline_rmse,
            "improvement": baseline_rmse - social_metrics["rmse"],
            "mae_enhanced": social_metrics["mae"],
            "mae_baseline": baseline_metrics["mae"],
            "r2_enhanced": social_metrics["r2"],
            "r2_baseline": baseline_metrics["r2"],
            "social_edges": social_edges.n_edges,
            "social_mode": social_mode,
            "lambda_social": lambda_social,
            "social_normalization": social_edges.normalization,
        }

        if compute_ranking:
            ranking = evaluate_ranking(social_model, train_df, warm_test, k=ranking_k)
            result.update(ranking)

        results.append(result)

    return results


def _save_rmses(
    model_name: str,
    network_index: int,
    split_results: list[dict],
    dataset: str | None = None,
    run_mode: str = "enhanced",
    rmse_limit: float | None = None,
) -> None:
    """
    Append per-mode mean RMSE, std, and improvement vs paired baseline.

    Args:
        model_name:    Diffusion model name.
        network_index: Zero-based network index.
        split_results: List of per-split dicts.
        dataset:       Dataset name.  Defaults to ``Datasets.DEFAULT``.
        run_mode:      ``"enhanced"`` or ``"social"`` — used as column prefix so
                       both modes can coexist in the same results CSV.
    """
    dp = DatasetPaths(dataset or Datasets.DEFAULT)
    results_file = NetworkArtifacts(
        dataset or Datasets.DEFAULT, paths=dp
    ).inferred_edges_csv(model_name)

    if not results_file.exists():
        return

    df = pd.read_csv(results_file, sep="|")
    ranking_cols = tuple(f"{run_mode}_{c}" for c in (
        "ndcg_at_k",
        "precision_at_k",
        "recall_at_k",
        "mrr",
    ))
    mode_cols = (
        f"{run_mode}_rmse_mean",
        f"{run_mode}_rmse_std",
        f"{run_mode}_baseline_rmse_mean",
        f"{run_mode}_improvement_pct",
    )
    for col in mode_cols + ranking_cols:
        if col not in df.columns:
            df[col] = np.nan

    if network_index < len(df):
        enhanced_rmses = [r["rmse_enhanced"] for r in split_results]
        baseline_rmses = [r["rmse_baseline"] for r in split_results]
        mean_enhanced = float(np.mean(enhanced_rmses))
        mean_baseline = float(np.mean(baseline_rmses))
        df.loc[network_index, f"{run_mode}_rmse_mean"] = mean_enhanced
        df.loc[network_index, f"{run_mode}_rmse_std"] = float(np.std(enhanced_rmses))
        df.loc[network_index, f"{run_mode}_baseline_rmse_mean"] = mean_baseline
        if (
            mean_baseline > 0
            and np.isfinite(mean_baseline)
            and np.isfinite(mean_enhanced)
            and (rmse_limit is None or mean_baseline <= rmse_limit)
        ):
            df.loc[network_index, f"{run_mode}_improvement_pct"] = (
                (mean_baseline - mean_enhanced) / mean_baseline
            ) * 100.0

        bare = ("ndcg_at_k", "precision_at_k", "recall_at_k", "mrr")
        for bare_col, col in zip(bare, ranking_cols):
            col_vals = [r[bare_col] for r in split_results if bare_col in r]
            if col_vals:
                df.loc[network_index, col] = float(np.mean(col_vals))

        # Atomic write: avoids partial files on crash / concurrent access.
        tmp = results_file.with_suffix(".tmp")
        df.to_csv(tmp, sep="|", index=False)
        tmp.replace(results_file)


def run_network_evaluation(
    data: pd.DataFrame,
    model_names: list[str] | None = None,
    sample_networks: int = 5,
    transform: str = "standard",
    include_communities: bool = True,
    n_splits: int = 5,
    k: int | None = None,
    lambda_reg: float | None = None,
    w_main: float | None = None,
    w_user: float | None = None,
    baseline_k: int | None = None,
    baseline_lambda: float | None = None,
    compute_ranking: bool = False,
    ranking_k: int = 10,
    dataset: str | None = None,
    seed: int = 42,
    n_jobs: int = 1,
    cmf_nthreads: int = 1,
    method: str = Defaults.CMF_METHOD,
    maxiter: int = Defaults.CMF_MAXITER,
    use_social_regularization: bool = False,
    social_mode: SocialMode = "boundary_downweight",
    lambda_social: float = 0.001,
    social_beta: float = 0.5,
    social_gamma: float = 1.0,
    social_normalization: SocialNormalization = "mean_weight",
) -> dict[str, dict[str, list[float]]]:
    """
    Evaluate a random sample of networks for all three diffusion models.

    For each sampled network the mean enhanced RMSE, paired baseline RMSE, and
    improvement percentage are saved back to ``inferred_edges_<short>.csv``.

    Args:
        data:                Ratings DataFrame (global train split).
        model_names:         Diffusion models to evaluate. Defaults to all.
        sample_networks:     Number of networks to randomly sample per model.
        transform:           Feature normalisation method.
        include_communities: Whether to include community features.
        n_splits:            Cross-validation splits per network.
        k:                   Number of latent factors.  If ``None``, searched via Optuna.
        lambda_reg:          L2 regularisation strength.  If ``None``, searched via Optuna.
        w_main:              Weight for main rating-matrix loss.  If ``None``, searched.
        w_user:              Weight for user side-information loss.  If ``None``, searched.
        baseline_k:          Latent factors for the paired plain-CMF baseline.
        baseline_lambda:     L2 regularisation for the paired baseline.
        compute_ranking:     When ``True``, compute and store NDCG@K, Precision@K,
                             Recall@K, and MRR for each evaluated network.
        ranking_k:           Cut-off for rank-based metrics.
        dataset:             Dataset name.  Defaults to ``Datasets.DEFAULT``.
        seed:                Random seed for reproducible network sampling.
        n_jobs:              Number of parallel worker processes.  ``1`` (default)
                             runs sequentially.  ``-1`` uses all available CPU cores
                             unless the caller maps it to a lower cap.
        cmf_nthreads:        BLAS threads per CMF fit.
        method:              CMF optimizer for non-social enhanced fits.
        maxiter:             L-BFGS iteration budget.
        use_social_regularization: Fit Phase 6 social-regularized CMF.
        social_mode:         Social edge weighting mode.
        lambda_social:       Social regularization strength.
        social_beta:         Boundary penalty parameter for social edge weights.
        social_gamma:        Shared-community gain parameter for social edge weights.
        social_normalization: Social edge normalization strategy.

    Returns:
        Dict mapping model name → ``{"enhanced": list[float], "baseline": list[float]}``
        where each list holds per-network mean RMSE values in the same order as
        the evaluated networks.
    """
    from recommender.enhanced.search import search_enhanced_params

    dp = DatasetPaths(dataset or Datasets.DEFAULT)
    selected_models = model_names or Models.ALL
    rmse_limit = rating_reasonableness_limit(cast(pd.Series, data["Rating"]))
    all_results: dict[str, dict[str, list[float]]] = {
        m: {"enhanced": [], "baseline": []} for m in selected_models
    }

    if any(p is None for p in (k, lambda_reg, w_main, w_user)):
        sample_features: pd.DataFrame | None = None
        sample_model_name: str | None = None
        sample_network_index = 0
        for _mn in selected_models:
            _indices = (
                _available_social_indices(dp, _mn)
                if use_social_regularization
                else _available_centrality_indices(dp, _mn)
            )
            if _indices:
                sample_network_index = _indices[0]
                sample_features = load_network_features(
                    _mn,
                    sample_network_index,
                    include_communities=include_communities,
                    dataset=dataset,
                )
                sample_model_name = _mn
                if sample_features is not None:
                    break

        if sample_features is not None:
            print(
                f"\nSearching best hyperparameters (Optuna TPE — k, lambda_reg, "
                f"w_main, w_user) using first {sample_model_name} network …"
            )
            import mlflow as _mlflow_tune

            if _mlflow_tune.active_run():
                _mlflow_tune.log_param(
                    "enhanced_search_tuning_model", sample_model_name or "unknown"
                )
                _mlflow_tune.log_param(
                    "enhanced_search_tuning_network_index", sample_network_index
                )
            enhanced_search = search_enhanced_params(
                data,
                sample_features,
                n_trials=50,
                n_splits=3,
                method=method,
                maxiter=maxiter,
                cmf_nthreads=cmf_nthreads,
            )
            best_k = enhanced_search["best_params"]["k"]
            best_lambda = enhanced_search["best_params"]["lambda_reg"]
            best_w_main = enhanced_search["best_params"]["w_main"]
            best_w_user = enhanced_search["best_params"]["w_user"]
        else:
            print("No feature files found — using default enhanced params.")
            best_k = Defaults.K
            best_lambda = Defaults.LAMBDA_REG
            best_w_main = Defaults.W_MAIN
            best_w_user = Defaults.W_USER
    else:
        assert k is not None and lambda_reg is not None
        assert w_main is not None and w_user is not None
        best_k, best_lambda, best_w_main, best_w_user = k, lambda_reg, w_main, w_user

    for model_name in selected_models:
        model_dir = dp.CENTRALITY / model_name
        if not model_dir.exists():
            print(f"  Skipping {model_name}: centrality directory not found.")
            continue

        indices = (
            _available_social_indices(dp, model_name)
            if use_social_regularization
            else _available_centrality_indices(dp, model_name)
        )
        if not indices:
            reason = (
                "no complete centrality/network/community artifact triplets found"
                if use_social_regularization
                else "no centrality CSVs found"
            )
            print(f"  Skipping {model_name}: {reason}.")
            continue
        rng = np.random.default_rng(seed)
        sampled = (
            indices[:sample_networks]
            if sample_networks >= len(indices)
            else sorted(rng.choice(indices, sample_networks, replace=False).tolist())
        )

        print(f"\n{'='*55}")
        print(f"Model: {model_name.upper()} — sampling {len(sampled)} networks")
        print("=" * 55)

        _shared: dict = {
            "model_name": model_name,
            "k": best_k,
            "lambda_reg": best_lambda,
            "w_main": best_w_main,
            "w_user": best_w_user,
            "method": method,
            "maxiter": maxiter,
            "transform": transform,
            "include_communities": include_communities,
            "n_splits": n_splits,
            "baseline_k": baseline_k,
            "baseline_lambda": baseline_lambda,
            "compute_ranking": compute_ranking,
            "ranking_k": ranking_k,
            "dataset": dataset,
            "cmf_nthreads": cmf_nthreads if n_jobs == 1 else 1,
            "use_social_regularization": use_social_regularization,
            "social_mode": social_mode,
            "lambda_social": lambda_social,
            "social_beta": social_beta,
            "social_gamma": social_gamma,
            "social_normalization": social_normalization,
        }

        from tqdm import tqdm

        if n_jobs == 1:
            network_results: dict[int, list[dict]] = {}
            pbar = tqdm(
                sampled,
                desc=f"{model_name[:4].upper()} networks",
                unit="net",
                dynamic_ncols=True,
            )
            for net_idx in pbar:
                pbar.set_postfix(net=f"{net_idx:03d}")
                _, split_results = _eval_network_worker(
                    (
                        net_idx,
                        {**_shared, "data": data, "network_index": net_idx},
                    )
                )
                network_results[net_idx] = split_results
                if split_results:
                    mean_e = float(np.mean([r["rmse_enhanced"] for r in split_results]))
                    pbar.set_postfix(net=f"{net_idx:03d}", rmse=f"{mean_e:.4f}")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import os
            import signal

            cpu_count = os.cpu_count() or 1
            max_workers = cpu_count if n_jobs == -1 else min(n_jobs, cpu_count)
            max_workers = min(max_workers, len(sampled))

            worker_args = [(net_idx, {"network_index": net_idx}) for net_idx in sampled]
            network_results = {}
            pbar = tqdm(
                total=len(sampled),
                desc=f"{model_name[:4].upper()} networks ({max_workers}p)",
                unit="net",
                dynamic_ncols=True,
            )

            def _kill_pool_children(pool: "ProcessPoolExecutor") -> None:
                procs = getattr(pool, "_processes", None) or {}
                for proc in list(procs.values()):
                    try:
                        proc.kill()
                    except Exception:  # pylint: disable=broad-except
                        pass

            pool = ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_worker_init,
                initargs=(data, _shared),
            )
            futures: dict = {}
            try:
                futures = {
                    pool.submit(_eval_network_worker, arg): arg[0]
                    for arg in worker_args
                }
                for future in as_completed(futures):
                    net_idx, split_results = future.result()
                    network_results[net_idx] = split_results
                    pbar.update(1)
                    if split_results:
                        mean_e = float(
                            np.mean([r["rmse_enhanced"] for r in split_results])
                        )
                        pbar.set_postfix(
                            last_net=f"{net_idx:03d}", rmse=f"{mean_e:.4f}"
                        )
            except KeyboardInterrupt:
                pbar.close()
                print(
                    "\n[interrupt] Ctrl+C received — terminating worker pool …",
                    flush=True,
                )
                for fut in futures:
                    fut.cancel()
                _kill_pool_children(pool)
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                pool.shutdown(wait=True)
                signal.signal(signal.SIGINT, signal.default_int_handler)
            pbar.close()

        for net_idx in sampled:
            split_results = network_results.get(net_idx, [])
            if split_results:
                mean_enhanced = float(
                    np.mean([r["rmse_enhanced"] for r in split_results])
                )
                mean_baseline = float(
                    np.mean([r["rmse_baseline"] for r in split_results])
                )
                improvement = mean_baseline - mean_enhanced
                sign = "+" if improvement > 0 else ""
                pct_str = ""
                if (
                    mean_baseline > 0
                    and np.isfinite(mean_baseline)
                    and mean_baseline <= rmse_limit
                ):
                    pct_str = f"({sign}{improvement / mean_baseline * 100:.2f}%)"
                print(
                    f"  Enhanced RMSE = {mean_enhanced:.4f}  "
                    f"Baseline RMSE = {mean_baseline:.4f}  "
                    f"improvement={sign}{improvement:.4f} {pct_str}"
                )
                _save_rmses(
                    model_name,
                    net_idx,
                    split_results,
                    dataset=dataset,
                    run_mode="social" if use_social_regularization else "enhanced",
                    rmse_limit=rmse_limit,
                )
                all_results[model_name]["enhanced"].append(mean_enhanced)
                all_results[model_name]["baseline"].append(mean_baseline)

                import mlflow as _mlflow

                if _mlflow.active_run():
                    _mlflow.log_metric(
                        f"{model_name}_rmse_enhanced", mean_enhanced, step=net_idx
                    )
                    _mlflow.log_metric(
                        f"{model_name}_rmse_baseline", mean_baseline, step=net_idx
                    )
                    if (
                        mean_baseline > 0
                        and np.isfinite(mean_baseline)
                        and mean_baseline <= rmse_limit
                    ):
                        _mlflow.log_metric(
                            f"{model_name}_improvement_pct",
                            improvement / mean_baseline * 100,
                            step=net_idx,
                        )

    import mlflow as _mlflow

    if _mlflow.active_run():
        for _model_name, _net_results in all_results.items():
            _rmse_list = _net_results["enhanced"]
            if _rmse_list:
                _mlflow.log_metric(
                    f"{_model_name}_mean_rmse_enhanced", float(np.mean(_rmse_list))
                )
                _mlflow.log_metric(
                    f"{_model_name}_n_networks_evaluated", len(_rmse_list)
                )

    return all_results
