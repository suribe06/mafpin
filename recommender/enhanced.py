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

from config import Paths, Models
from recommender.baseline import save_search_results, search_best_params
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
        train_feats = user_attributes.loc[
            [u for u in train_users if u in user_attributes.index]
        ]
        scaler = _SCALERS[transform]()
        scaler.fit(train_feats.values)
        # Apply the train-fitted scaler to all users (for inference on test).
        scaled_all = pd.DataFrame(
            scaler.transform(user_attributes.values),
            index=user_attributes.index,
            columns=user_attributes.columns,
        )

        # Build U matrix: UserId as a column (required by cmfrec).
        u_matrix = scaled_all.loc[
            [u for u in train_users if u in scaled_all.index]
        ].reset_index()  # brings UserId back as a column

        # --- Enhanced model -------------------------------------------------
        enhanced_model = CMF(method="als", k=k, lambda_=lambda_reg, verbose=False)
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
# Single-network evaluation
# ---------------------------------------------------------------------------


def evaluate_single_network(
    data: pd.DataFrame,
    model_name: str,
    network_index: int,
    k: int = 20,
    lambda_reg: float = 1.0,
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
) -> dict[str, list[float]]:
    """
    Evaluate a random sample of networks for all three diffusion models.

    For each sampled network the mean enhanced RMSE, paired baseline RMSE, and
    improvement percentage are saved back to ``inferred_edges_<short>.csv``.

    Args:
        data:                Ratings DataFrame to use for training (should be the
                             global train split so test ratings are never seen).
                             Obtained via :func:`recommender.data.load_and_split_dataset`.
        sample_networks:     Number of networks to randomly sample per model.
        transform:           Feature normalisation method.
        include_communities: Whether to include community features.
        n_splits:            Cross-validation splits per network.

    Returns:
        Dict mapping model name → list of mean enhanced RMSE values (one per
        sampled network).
    """
    all_results: dict[str, list[float]] = {m: [] for m in Models.ALL}

    # Hyperparameter search uses only train data — no leakage from held-out test.
    print("Searching best hyperparameters (baseline) …")
    search_result = search_best_params(data, n_iter=50, n_splits=3)
    if search_result is None:
        print("Hyperparameter search failed — using defaults k=20, lambda_reg=1.0")
        best_k, best_lambda = 20, 1.0
    else:
        best_k = search_result["best_params"]["k"]
        best_lambda = search_result["best_params"]["lambda_reg"]
        # m-3 fix: keep only the minimum (the first assignment was dead code).
        best_search_rmse = float(min(r["rmse"] for r in search_result["all_results"]))
        print(
            f"Best params: k={best_k}, lambda_reg={best_lambda:.4f} "
            f"(search RMSE={best_search_rmse:.4f})"
        )
        save_search_results(search_result)

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
