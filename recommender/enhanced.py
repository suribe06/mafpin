"""
Enhanced CMF recommender using network and community features as side information.

This module extends the baseline CMF model by enriching user representations with
node-level features derived from the inferred diffusion networks:

* **Centrality metrics** — degree, betweenness, closeness, eigenvector, PageRank,
  clustering coefficient, eccentricity
* **Community features** — number of communities, local pluralistic homophily (LPH)

These features are loaded from the pre-computed CSV files in
``data/centrality_metrics/`` and ``data/communities/``, normalised, and passed as
the ``U`` (user-side attribute) matrix to :class:`cmfrec.CMF`.

Comparison Design
-----------------
The evaluation compares **plain CMF** (baseline) against **CMF with network
side-information** (enhanced) under the same conditions.

Two independent Optuna searches are run before the per-network evaluation:

1. **Baseline search** — optimises ``k`` and ``lambda_reg`` for plain CMF only.
   The resulting (k*, λ*) represent the best achievable RMSE without any
   side-information.
2. **Enhanced search** — optimises ``k``, ``lambda_reg``, ``w_main``, and
   ``w_user`` for the enhanced model.  The optimal λ here is systematically
   different because the side-information reconstruction term shifts the
   effective regularisation landscape; reusing the baseline λ would
   mis-configure the enhanced model.

For each inferred network a **paired comparison** is used: both models are
trained and evaluated on the same user subset (only users present in that
network have features), the same CV folds, and the same data.  The paired
baseline always uses the baseline-optimal (k*, λ*) from the baseline search —
not the enhanced model's hyperparameters — so that any improvement in RMSE is
attributable solely to the network side-information and not to differences in
regularisation strength.

Functions
---------
load_network_features
    Load and optionally merge centrality and community CSV files for a specific
    network, then apply a normalisation transform.
evaluate_cmf_with_user_attributes
    Train and evaluate CMF with user-side attributes via random splits.
evaluate_single_network
    Evaluate a single (model, index) combination and return its RMSE scores.
run_network_evaluation
    Batch-evaluate a sample of networks for all three diffusion models.
"""

from __future__ import annotations

from pathlib import Path as _Path  # noqa: F401

import numpy as np
import pandas as pd
from cmfrec import CMF  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from config import DatasetPaths, Datasets, Models, Defaults
from recommender.data import evaluate_single_split, split_data_single

# Scaler type alias used inside the per-split loop.
# WARNING: "normalizer" uses sklearn.preprocessing.Normalizer, which normalises
# each *sample* (row) to unit norm, NOT each *feature* (column).  This is
# semantically incorrect for feature scaling and should not be the first choice.
# Prefer "standard" (StandardScaler) or "minmax" (MinMaxScaler).
_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "normalizer": Normalizer,  # row-normalisation — see warning above
}


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_network_features(
    model_name: str,
    network_index: int,
    include_communities: bool = True,
    dataset: str | None = None,
) -> pd.DataFrame | None:
    """
    Load centrality (and optionally community) features for one inferred network.

    Returns a raw (unscaled) DataFrame indexed by ``UserId``.  Scaling is
    intentionally deferred to :func:`evaluate_cmf_with_user_attributes` where
    it is fitted on training users only, preventing test-set leakage (M-2).

    Args:
        model_name:          Diffusion model name (exponential / powerlaw / rayleigh).
        network_index:       Zero-based network index (selects the CSV by filename).
        include_communities: If ``True``, merge LPH and ``num_communities`` from the
                             corresponding community CSV.
        dataset:             Dataset name.  Defaults to ``Datasets.DEFAULT``.

    Returns:
        Raw feature DataFrame indexed by ``UserId``, or ``None`` if the file is
        missing.
    """
    dp = DatasetPaths(dataset or Datasets.DEFAULT)
    index_str = f"{network_index:03d}"
    centrality_dir = dp.CENTRALITY / model_name
    centrality_csv = centrality_dir / f"centrality_metrics_{model_name}_{index_str}.csv"

    if not centrality_csv.exists():
        return None

    df = pd.read_csv(centrality_csv)

    if include_communities:
        community_dir = dp.COMMUNITIES / model_name
        community_csv = community_dir / f"communities_{model_name}_{index_str}.csv"
        if community_csv.exists():
            com_raw = pd.read_csv(community_csv)
            com_cols = ["UserId", "local_pluralistic_hom", "num_communities"]
            if "lph_score" in com_raw.columns:
                com_cols.append("lph_score")
            df = df.merge(com_raw[com_cols], on="UserId", how="left")

            # Binary community-membership features: one column per community in
            # the top-K most-populated communities.  Each column is 1 when the
            # user belongs to that community, 0 otherwise.  This allows the CMF
            # model to distinguish which community a user belongs to, not just
            # how many they belong to (which is captured by num_communities).
            if "community_ids" in com_raw.columns:
                TOP_K_COMMUNITIES = 20  # encode at most the 20 largest communities
                # Parse semicolon-separated lists into sets of community ids
                parsed = (
                    com_raw["community_ids"]
                    .fillna("")
                    .apply(
                        lambda s: set(map(int, filter(None, s.split(";"))))
                    )
                )
                # Count occurrences to find the most populated communities
                from collections import Counter
                community_counts: Counter = Counter()
                for cids in parsed:
                    community_counts.update(cids)
                top_communities = [cid for cid, _ in community_counts.most_common(TOP_K_COMMUNITIES)]

                # Build binary feature columns
                for cid in top_communities:
                    col = f"community_{cid}"
                    com_raw[col] = parsed.apply(lambda cids, c=cid: int(c in cids))

                binary_cols = [f"community_{cid}" for cid in top_communities]
                df = df.merge(com_raw[["UserId"] + binary_cols], on="UserId", how="left")

    return df.set_index("UserId").fillna(0.0)


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------


def evaluate_cmf_with_user_attributes(
    data: pd.DataFrame,
    user_attributes: pd.DataFrame,
    k: int = 20,
    lambda_reg: float = 1.0,
    w_main: float = Defaults.W_MAIN,
    w_user: float = Defaults.W_USER,
    n_splits: int = 5,
    test_size: float = 0.2,
    transform: str = "standard",
    baseline_k: int | None = None,
    baseline_lambda: float | None = None,
) -> list[dict]:
    """
    Evaluate enhanced CMF via repeated random train/test splits.

    The user-attribute matrix is passed as the ``U`` parameter to
    :class:`cmfrec.CMF`.  Feature scaling is fitted on training users only
    within each fold to prevent leakage (M-2).  A paired baseline CMF (no
    side information) is evaluated on the same split and user subset so that
    improvement is measured fairly (M-3).

    Args:
        data:             Full ratings DataFrame (0-based UserId from LabelEncoder).
        user_attributes:  Raw (unscaled) feature DataFrame indexed by 0-based
                          ``UserId`` (aligned with *data*).
        k:                Number of latent factors.
        lambda_reg:       L2 regularisation strength.
        w_main:           Weight for the main rating-matrix reconstruction loss.
                          Relative to *w_user*; higher values retain more signal
                          from the rating matrix.
        w_user:           Weight for the user side-information reconstruction
                          loss.  Lower values reduce the influence of network
                          features on the learned user embeddings.
        n_splits:         Number of random splits.
        test_size:        Test fraction.
        transform:        Scaler to apply per fold — ``"standard"``, ``"minmax"``,
                          or ``"normalizer"``.
        baseline_k:       Number of latent factors for the paired plain-CMF baseline.
                          Should come from the independently-tuned baseline search.
                          When ``None`` (e.g. during hyperparameter search) the
                          baseline is skipped and ``rmse_baseline`` is ``nan``.
        baseline_lambda:  L2 regularisation for the paired baseline.  Same origin
                          as *baseline_k*.

    Returns:
        List of per-split result dicts with keys ``rmse_enhanced``,
        ``rmse_baseline``, and ``improvement`` (baseline − enhanced).
    """
    if transform not in _SCALERS:
        raise ValueError(
            f"Unknown transform: {transform!r}. Use one of {list(_SCALERS)}."
        )

    # Keep only users for whom network features are available.
    # user_attributes is 0-based — same space as data["UserId"].
    valid_users = list(user_attributes.index)
    filtered = data[data["UserId"].isin(valid_users)]

    if filtered.empty:
        print("  Warning: no overlap between rating users and network users.")
        return []

    results: list[dict] = []
    for split_idx in range(n_splits):
        train_df, test_df = split_data_single(
            filtered, test_size=test_size, random_state=split_idx  # type: ignore[arg-type]
        )

        # --- M-2: fit scaler on training users only -------------------------
        train_users = sorted(train_df["UserId"].unique())
        train_feats = user_attributes.loc[train_users]
        scaler = _SCALERS[transform]()
        scaler.fit(train_feats.values)
        # Apply the train-fitted scaler to ALL users (train + test).
        # Passing every user's features to cmfrec lets it use network
        # side-information for test-user embeddings too.  There is no
        # leakage because these features are derived from the training
        # network (C-3), not from test ratings.
        scaled_all = pd.DataFrame(
            scaler.transform(user_attributes.values),
            index=user_attributes.index,
            columns=user_attributes.columns,
        )

        # Build U matrix: UserId as a column (required by cmfrec).
        u_matrix = scaled_all.reset_index()  # all users, not just training

        # --- Enhanced model -------------------------------------------------
        enhanced_model = CMF(
            method="als",
            k=k,
            lambda_=lambda_reg,
            w_main=w_main,
            w_user=w_user,
            verbose=False,
        )
        enhanced_model.fit(X=train_df, U=u_matrix)
        enhanced_rmse = evaluate_single_split(enhanced_model, test_df)["rmse"]

        # --- M-3: paired baseline on the same filtered subset ---------------
        # Use the independently-tuned baseline params (baseline_k, baseline_lambda)
        # so the comparison is fair: same users, same fold, optimal plain-CMF
        # hyperparameters rather than hyperparameters calibrated for the enhanced
        # model.  When baseline params are not provided (e.g. during the Optuna
        # search over enhanced hyperparameters) the paired baseline is skipped.
        if baseline_k is not None and baseline_lambda is not None:
            baseline_model = CMF(
                method="als", k=baseline_k, lambda_=baseline_lambda, verbose=False
            )
            baseline_model.fit(X=train_df)
            baseline_rmse = evaluate_single_split(baseline_model, test_df)["rmse"]
        else:
            baseline_rmse = float("nan")

        results.append(
            {
                "rmse_enhanced": enhanced_rmse,
                "rmse_baseline": baseline_rmse,
                "improvement": baseline_rmse - enhanced_rmse,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Enhanced hyperparameter search (k, lambda_reg, w_main, w_user) — Optuna TPE
# ---------------------------------------------------------------------------


def search_enhanced_params(
    data: pd.DataFrame,
    user_attributes: pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 3,
) -> dict:
    """
    Bayesian hyperparameter search (Optuna TPE) over ``k``, ``lambda_reg``,
    ``w_main``, and ``w_user`` for the enhanced CMF model.

    Random search over 4 interacting parameters is sample-inefficient; Optuna's
    Tree-structured Parzen Estimator (TPE) models the objective surface and
    proposes candidates that are likely to improve over previous trials, making
    ~50 trials roughly equivalent to ~150 uniform random draws for this
    parameter count.

    ``w_main`` and ``w_user`` control the split of the total loss between the
    rating matrix and the user side-information matrix.  They must be searched
    jointly with ``k`` and ``lambda_reg`` because high ``w_user`` demands lower
    regularisation to avoid killing the side-info signal.

    Args:
        data:            Full (training) ratings DataFrame.
        user_attributes: Raw (unscaled) feature DataFrame indexed by 0-based
                         ``UserId``.  Scaling is applied inside
                         :func:`evaluate_cmf_with_user_attributes`.
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
        # w_main: keep ≥ 0.1 so the model cannot fully ignore ratings.
        w_main_val = trial.suggest_float("w_main", 0.1, 1.0)
        # w_user: log-scale concentrates trials on small values where
        # side-information typically helps without overwhelming ratings.
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
    path: "_Path | None" = None,
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
    import json as _json

    dest = path or DatasetPaths(Datasets.DEFAULT).ENHANCED_RESULTS
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        _json.dump(search_result, fh, indent=2)
    print(f"Enhanced search results saved → {dest}")


# ---------------------------------------------------------------------------
# Single-network evaluation
# ---------------------------------------------------------------------------


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
    dataset: str | None = None,
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

    Returns:
        List of per-split result dicts (keys: ``rmse_enhanced``,
        ``rmse_baseline``, ``improvement``), or empty list on failure.
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
                       :func:`evaluate_cmf_with_user_attributes`.
        dataset:       Dataset name.  Defaults to ``Datasets.DEFAULT``.
    """
    dp = DatasetPaths(dataset or Datasets.DEFAULT)
    model_short = Models.SHORT[model_name]
    results_file = dp.NETWORKS / model_name / f"inferred_edges_{model_short}.csv"

    if not results_file.exists():
        return

    df = pd.read_csv(results_file, sep="|")
    for col in ("rmse_mean", "rmse_std", "baseline_rmse_mean", "improvement_pct"):
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
        df.to_csv(results_file, sep="|", index=False)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


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
    dataset: str | None = None,
    seed: int = 42,
) -> dict[str, list[float]]:
    """
    Evaluate a random sample of networks for all three diffusion models.

    For each sampled network the mean enhanced RMSE, paired baseline RMSE, and
    improvement percentage are saved back to ``inferred_edges_<short>.csv``.

    Hyperparameters (*k*, *lambda_reg*, *w_main*, *w_user*) can be supplied
    directly (e.g. from a prior Optuna search in the pipeline) or left as
    ``None``, in which case :func:`search_enhanced_params` is called once
    here using the first available feature file.

    Args:
        data:                Ratings DataFrame to use for training (should be the
                             global train split so test ratings are never seen).
                             Obtained via :func:`recommender.data.load_and_split_dataset`.
        sample_networks:     Number of networks to randomly sample per model.
        transform:           Feature normalisation method.
        include_communities: Whether to include community features.
        n_splits:            Cross-validation splits per network.
        k:                   Number of latent factors.  If ``None``, searched
                             via Optuna.
        lambda_reg:          L2 regularisation strength.  If ``None``, searched
                             via Optuna.
        w_main:              Weight for main rating-matrix loss.  If ``None``,
                             searched via Optuna.
        w_user:              Weight for user side-information loss.  If ``None``,
                             searched via Optuna.
        baseline_k:          Latent factors for the paired plain-CMF baseline per
                             network.  Should be the result of the independent
                             baseline Optuna search.  When ``None``, the paired
                             baseline is skipped (``rmse_baseline`` = nan).
        baseline_lambda:     L2 regularisation for the paired plain-CMF baseline.
        dataset:             Dataset name.  Defaults to ``Datasets.DEFAULT``.
        seed:                Random seed for reproducible network sampling.
                             Uses ``np.random.default_rng(seed)`` so results
                             are identical across re-runs with the same seed.

    Returns:
        Dict mapping model name → list of mean enhanced RMSE values (one per
        sampled network).
    """
    dp = DatasetPaths(dataset or Datasets.DEFAULT)
    all_results: dict[str, list[float]] = {m: [] for m in Models.ALL}

    # --- Optuna hyperparameter search (only when params are not pre-supplied) --
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
        # All four params were supplied; assert to narrow types for type checker.
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

        for net_idx in sampled:
            print(f"\n  Network {net_idx:03d}")
            split_results = evaluate_single_network(
                data,
                model_name,
                net_idx,
                k=best_k,
                lambda_reg=best_lambda,
                w_main=best_w_main,
                w_user=best_w_user,
                transform=transform,
                include_communities=include_communities,
                n_splits=n_splits,
                baseline_k=baseline_k,
                baseline_lambda=baseline_lambda,
            )
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


if __name__ == "__main__":
    import argparse
    import sys

    _parser = argparse.ArgumentParser(
        description="Evaluate enhanced CMF with network features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every available network for all models.",
    )
    _parser.add_argument(
        "--sample-networks",
        type=int,
        default=5,
        metavar="N",
        help="Number of networks to randomly sample per model.",
    )
    _parser.add_argument(
        "--transform",
        default="standard",
        choices=["standard", "minmax", "normalizer"],
        help="Feature normalisation method.",
    )
    _parser.add_argument(
        "--no-communities",
        action="store_true",
        help="Exclude community features (LPH, num_communities).",
    )
    _parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Cross-validation splits per network.",
    )
    _args = _parser.parse_args()

    _sample = 999_999 if _args.all else _args.sample_networks

    from recommender.data import load_and_split_dataset as _load_split

    _, _train_df, _ = _load_split()
    _eval_results = run_network_evaluation(
        data=_train_df,
        sample_networks=_sample,
        transform=_args.transform,
        include_communities=not _args.no_communities,
        n_splits=_args.n_splits,
    )
    print("\n" + "=" * 55)
    print("Summary")
    print("=" * 55)
    for _model_name, _rmse_list in _eval_results.items():
        if _rmse_list:
            print(
                f"{_model_name}: n={len(_rmse_list)}, "
                f"mean={np.mean(_rmse_list):.4f}, "
                f"min={np.min(_rmse_list):.4f}, "
                f"max={np.max(_rmse_list):.4f}"
            )
        else:
            print(f"{_model_name}: no results")
    sys.exit(0)
