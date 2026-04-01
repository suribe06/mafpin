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

from config import Paths, Models, Defaults
from recommender.data import evaluate_single_split, split_data_single

# Scaler type alias used inside the per-split loop.
_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "normalizer": Normalizer,
}


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_network_features(
    model_name: str,
    network_index: int,
    include_communities: bool = True,
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

    Returns:
        Raw feature DataFrame indexed by ``UserId``, or ``None`` if the file is
        missing.
    """
    index_str = f"{network_index:03d}"
    centrality_dir = Paths.CENTRALITY / model_name
    centrality_csv = centrality_dir / f"centrality_metrics_{model_name}_{index_str}.csv"

    if not centrality_csv.exists():
        return None

    df = pd.read_csv(centrality_csv)

    if include_communities:
        community_dir = Paths.COMMUNITIES / model_name
        community_csv = community_dir / f"communities_{model_name}_{index_str}.csv"
        if community_csv.exists():
            com_df = pd.read_csv(community_csv)[
                ["UserId", "local_pluralistic_hom", "num_communities"]
            ]
            df = df.merge(com_df, on="UserId", how="left")

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
    valid_users = set(user_attributes.index)
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
        baseline_model = CMF(method="als", k=k, lambda_=lambda_reg, verbose=False)
        baseline_model.fit(X=train_df)
        baseline_rmse = evaluate_single_split(baseline_model, test_df)["rmse"]

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
    return {"best_params": best_params, "all_results": all_results}


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

    Returns:
        List of per-split result dicts (keys: ``rmse_enhanced``,
        ``rmse_baseline``, ``improvement``), or empty list on failure.
    """
    features = load_network_features(
        model_name,
        network_index,
        include_communities=include_communities,
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
    )


def _save_rmses(
    model_name: str,
    network_index: int,
    split_results: list[dict],
) -> None:
    """
    Append mean RMSE, std, and improvement vs paired baseline to the results file.

    Args:
        model_name:    Diffusion model name.
        network_index: Zero-based network index.
        split_results: List of per-split dicts from
                       :func:`evaluate_cmf_with_user_attributes`.
    """
    model_short = Models.SHORT[model_name]
    results_file = Paths.NETWORKS / model_name / f"inferred_edges_{model_short}.csv"

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

    Returns:
        Dict mapping model name → list of mean enhanced RMSE values (one per
        sampled network).
    """
    all_results: dict[str, list[float]] = {m: [] for m in Models.ALL}

    # --- Optuna hyperparameter search (only when params are not pre-supplied) --
    if any(p is None for p in (k, lambda_reg, w_main, w_user)):
        sample_features: pd.DataFrame | None = None
        sample_model_name: str | None = None
        for _mn in Models.ALL:
            _csvs = sorted(
                (Paths.CENTRALITY / _mn).glob(f"centrality_metrics_{_mn}_*.csv")
            )
            if _csvs:
                sample_features = load_network_features(
                    _mn, 0, include_communities=include_communities
                )
                sample_model_name = _mn
                if sample_features is not None:
                    break

        if sample_features is not None:
            print(
                f"\nSearching best hyperparameters (Optuna TPE — k, lambda_reg, "
                f"w_main, w_user) using first {sample_model_name} network …"
            )
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
        model_dir = Paths.CENTRALITY / model_name
        if not model_dir.exists():
            print(f"  Skipping {model_name}: centrality directory not found.")
            continue

        csvs = sorted(model_dir.glob(f"centrality_metrics_{model_name}_*.csv"))
        if not csvs:
            print(f"  Skipping {model_name}: no centrality CSVs found.")
            continue

        indices = list(range(len(csvs)))
        sampled = (
            indices[:sample_networks]
            if sample_networks >= len(indices)
            else sorted(np.random.choice(indices, sample_networks, replace=False))
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
                _save_rmses(model_name, net_idx, split_results)
                all_results[model_name].append(mean_enhanced)

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
