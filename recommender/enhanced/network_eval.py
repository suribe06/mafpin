"""
Single-network and batch network evaluation for the enhanced CMF model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets, Models, Defaults
from recommender.enhanced.features import load_network_features
from recommender.enhanced.model import evaluate_cmf_with_user_attributes
from recommender.enhanced.workers import _worker_init, _eval_network_worker


def evaluate_single_network(
    data: pd.DataFrame,
    model_name: str,
    network_index: int,
    k: int = 20,
    lambda_reg: float = 1.0,
    w_main: float = Defaults.W_MAIN,
    w_user: float = Defaults.W_USER,
    transform: str = "standard",
    include_communities: bool = True,
    n_splits: int = 5,
    baseline_k: int | None = None,
    baseline_lambda: float | None = None,
    compute_ranking: bool = False,
    ranking_k: int = 10,
    dataset: str | None = None,
    cmf_nthreads: int = -1,
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

    return evaluate_cmf_with_user_attributes(
        data,
        features,
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
        cmf_nthreads=cmf_nthreads,
    )


def _save_rmses(
    model_name: str,
    network_index: int,
    split_results: list[dict],
    dataset: str | None = None,
) -> None:
    """
    Append mean RMSE, std, and improvement vs paired baseline to the results file.

    Args:
        model_name:    Diffusion model name.
        network_index: Zero-based network index.
        split_results: List of per-split dicts from
                       :func:`~recommender.enhanced.model.evaluate_cmf_with_user_attributes`.
        dataset:       Dataset name.  Defaults to ``Datasets.DEFAULT``.
    """
    dp = DatasetPaths(dataset or Datasets.DEFAULT)
    model_short = Models.SHORT[model_name]
    results_file = dp.NETWORKS / model_name / f"inferred_edges_{model_short}.csv"

    if not results_file.exists():
        return

    df = pd.read_csv(results_file, sep="|")
    _ranking_cols = ("ndcg_at_k", "precision_at_k", "recall_at_k", "mrr")
    for col in (
        "rmse_mean",
        "rmse_std",
        "baseline_rmse_mean",
        "improvement_pct",
    ) + _ranking_cols:
        if col not in df.columns:
            df[col] = np.nan

    if network_index < len(df):
        enhanced_rmses = [r["rmse_enhanced"] for r in split_results]
        baseline_rmses = [r["rmse_baseline"] for r in split_results]
        mean_enhanced = float(np.mean(enhanced_rmses))
        mean_baseline = float(np.mean(baseline_rmses))
        df.loc[network_index, "rmse_mean"] = mean_enhanced
        df.loc[network_index, "rmse_std"] = float(np.std(enhanced_rmses))
        df.loc[network_index, "baseline_rmse_mean"] = mean_baseline
        if mean_baseline > 0:
            df.loc[network_index, "improvement_pct"] = (
                (mean_baseline - mean_enhanced) / mean_baseline
            ) * 100.0

        for col in _ranking_cols:
            col_vals = [r[col] for r in split_results if col in r]
            if col_vals:
                df.loc[network_index, col] = float(np.mean(col_vals))

        df.to_csv(results_file, sep="|", index=False)


def run_network_evaluation(
    data: pd.DataFrame,
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
) -> dict[str, list[float]]:
    """
    Evaluate a random sample of networks for all three diffusion models.

    For each sampled network the mean enhanced RMSE, paired baseline RMSE, and
    improvement percentage are saved back to ``inferred_edges_<short>.csv``.

    Args:
        data:                Ratings DataFrame (global train split).
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
                             runs sequentially.  ``-1`` uses all available CPU cores.

    Returns:
        Dict mapping model name → list of mean enhanced RMSE values.
    """
    from recommender.enhanced.search import search_enhanced_params

    dp = DatasetPaths(dataset or Datasets.DEFAULT)
    all_results: dict[str, list[float]] = {m: [] for m in Models.ALL}

    if any(p is None for p in (k, lambda_reg, w_main, w_user)):
        sample_features: pd.DataFrame | None = None
        sample_model_name: str | None = None
        for _mn in Models.ALL:
            _csvs = sorted(
                (dp.CENTRALITY / _mn).glob(f"centrality_metrics_{_mn}_*.csv")
            )
            if _csvs:
                sample_features = load_network_features(
                    _mn, 0, include_communities=include_communities, dataset=dataset
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
                _mlflow_tune.log_param("enhanced_search_tuning_network_index", 0)
            enhanced_search = search_enhanced_params(
                data, sample_features, n_trials=50, n_splits=3
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

    for model_name in Models.ALL:
        model_dir = dp.CENTRALITY / model_name
        if not model_dir.exists():
            print(f"  Skipping {model_name}: centrality directory not found.")
            continue

        csvs = sorted(model_dir.glob(f"centrality_metrics_{model_name}_*.csv"))
        if not csvs:
            print(f"  Skipping {model_name}: no centrality CSVs found.")
            continue

        indices = list(range(len(csvs)))
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
            "transform": transform,
            "include_communities": include_communities,
            "n_splits": n_splits,
            "baseline_k": baseline_k,
            "baseline_lambda": baseline_lambda,
            "compute_ranking": compute_ranking,
            "ranking_k": ranking_k,
            "dataset": dataset,
            "cmf_nthreads": -1 if n_jobs == 1 else 1,
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
                print(
                    f"  Enhanced RMSE = {mean_enhanced:.4f}  "
                    f"Baseline RMSE = {mean_baseline:.4f}  "
                    f"improvement={sign}{improvement:.4f} "
                    f"({sign}{improvement / mean_baseline * 100:.2f}%)"
                )
                _save_rmses(model_name, net_idx, split_results, dataset=dataset)
                all_results[model_name].append(mean_enhanced)

                import mlflow as _mlflow

                if _mlflow.active_run():
                    _mlflow.log_metric(
                        f"{model_name}_rmse_enhanced", mean_enhanced, step=net_idx
                    )
                    _mlflow.log_metric(
                        f"{model_name}_rmse_baseline", mean_baseline, step=net_idx
                    )
                    if mean_baseline > 0:
                        _mlflow.log_metric(
                            f"{model_name}_improvement_pct",
                            improvement / mean_baseline * 100,
                            step=net_idx,
                        )

    import mlflow as _mlflow

    if _mlflow.active_run():
        for _model_name, _rmse_list in all_results.items():
            if _rmse_list:
                _mlflow.log_metric(
                    f"{_model_name}_mean_rmse_enhanced", float(np.mean(_rmse_list))
                )
                _mlflow.log_metric(
                    f"{_model_name}_n_networks_evaluated", len(_rmse_list)
                )

    return all_results
