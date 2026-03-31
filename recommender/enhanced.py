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
from recommender.data import load_dataset, split_data_single, evaluate_single_split
from recommender.baseline import search_best_params, save_search_results


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_network_features(
    model_name: str,
    network_index: int,
    transform: str = "standard",
    include_communities: bool = True,
) -> pd.DataFrame | None:
    """
    Load centrality (and optionally community) features for one inferred network.

    The returned DataFrame is indexed by ``UserId`` and has one column per
    feature, all in float format, with the chosen normalisation applied.

    Args:
        model_name:          Diffusion model name (exponential / powerlaw / rayleigh).
        network_index:       Zero-based network index (selects the CSV by filename).
        transform:           Scaler to apply — ``"standard"``, ``"minmax"``, or
                             ``"normalizer"``.
        include_communities: If ``True``, merge LPH and ``num_communities`` from the
                             corresponding community CSV.

    Returns:
        DataFrame with features per user, or ``None`` if the file is missing.
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

    df = df.set_index("UserId").fillna(0.0)

    # Apply normalisation
    if transform == "standard":
        scaler = StandardScaler()
    elif transform == "minmax":
        scaler = MinMaxScaler()  # type: ignore[assignment]
    elif transform == "normalizer":
        scaler = Normalizer()  # type: ignore[assignment]
    else:
        raise ValueError(
            f"Unknown transform: {transform!r}. "
            "Use 'standard', 'minmax', or 'normalizer'."
        )

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df.values),
        index=df.index,
        columns=df.columns,
    )
    return df_scaled


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
) -> list[float]:
    """
    Evaluate enhanced CMF via repeated random train/test splits.

    The user-attribute matrix is passed as the ``U`` parameter to
    :class:`cmfrec.CMF`.  Only users present in *user_attributes* can receive
    predictions; rows from *data* whose ``UserId`` is absent are excluded from
    evaluation.

    Args:
        data:             Full ratings DataFrame.
        user_attributes:  Feature DataFrame indexed by ``UserId``.
        k:                Number of latent factors.
        lambda_reg:       L2 regularisation strength.
        n_splits:         Number of random splits.
        test_size:        Test fraction.

    Returns:
        List of per-split RMSE values.
    """
    # Keep only users for whom attributes are available
    valid_users = set(user_attributes.index)
    filtered = data[data["UserId"].isin(list(valid_users))]  # type: ignore[arg-type]

    if filtered.empty:
        print("  Warning: no overlap between rating users and network users.")
        return []

    rmse_scores: list[float] = []
    for split_idx in range(n_splits):
        train_df, test_df = split_data_single(
            filtered, test_size=test_size, random_state=split_idx  # type: ignore[arg-type]
        )

        # Build user-attribute DataFrame aligned to the training set.
        # cmfrec requires UserId as a column (not the index) in U.
        train_users = sorted(train_df["UserId"].unique())
        u_matrix = user_attributes.loc[
            [u for u in train_users if u in user_attributes.index]
        ].reset_index()  # brings UserId back as a column

        model = CMF(method="als", k=k, lambda_=lambda_reg, verbose=False)
        model.fit(X=train_df, U=u_matrix)

        result = evaluate_single_split(model, test_df)
        rmse_scores.append(result["rmse"])

    return rmse_scores


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
) -> list[float]:
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
        List of RMSE values (one per split), or empty list on failure.
    """
    features = load_network_features(
        model_name,
        network_index,
        transform=transform,
        include_communities=include_communities,
    )
    if features is None:
        print(f"  Skipping {model_name} #{network_index:03d}: features not found.")
        return []

    return evaluate_cmf_with_user_attributes(
        data, features, k=k, lambda_reg=lambda_reg, n_splits=n_splits
    )


def _save_rmses(
    model_name: str,
    network_index: int,
    rmse_scores: list[float],
    baseline_rmse: float | None = None,
) -> None:
    """
    Append mean RMSE, std, and improvement vs baseline to the results file.

    The columns ``rmse_mean``, ``rmse_std``, and ``improvement_pct`` are added
    (or overwritten) for the given network index row.

    Args:
        model_name:    Diffusion model name.
        network_index: Zero-based network index.
        rmse_scores:   List of per-split RMSE values.
        baseline_rmse: Baseline RMSE for improvement calculation.
    """
    model_short = Models.SHORT[model_name]
    results_file = Paths.NETWORKS / model_name / f"inferred_edges_{model_short}.csv"

    if not results_file.exists():
        return

    df = pd.read_csv(results_file, sep="|")
    for col in ("rmse_mean", "rmse_std", "improvement_pct"):
        if col not in df.columns:
            df[col] = np.nan

    if network_index < len(df):
        mean_rmse = float(np.mean(rmse_scores))
        std_rmse = float(np.std(rmse_scores))
        df.loc[network_index, "rmse_mean"] = mean_rmse
        df.loc[network_index, "rmse_std"] = std_rmse
        if baseline_rmse is not None and baseline_rmse > 0:
            df.loc[network_index, "improvement_pct"] = (
                (baseline_rmse - mean_rmse) / baseline_rmse
            ) * 100.0
        df.to_csv(results_file, sep="|", index=False)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def run_network_evaluation(
    sample_networks: int = 5,
    transform: str = "standard",
    include_communities: bool = True,
    n_splits: int = 5,
) -> dict[str, list[float]]:
    """
    Evaluate a random sample of networks for all three diffusion models.

    For each sampled network the mean RMSE is saved back to the
    ``inferred_edges_<short>.csv`` file so it can be used later for
    alpha-vs-RMSE analysis.

    Args:
        sample_networks:     Number of networks to randomly sample per model.
        transform:           Feature normalisation method.
        include_communities: Whether to include community features.
        n_splits:            Cross-validation splits per network.

    Returns:
        Dict mapping model name → list of mean RMSE values (one per sampled
        network).
    """
    data = load_dataset()
    all_results: dict[str, list[float]] = {m: [] for m in Models.ALL}

    print("Searching best hyperparameters (baseline) …")
    search_result = search_best_params(data, n_iter=50, n_splits=3)
    if search_result is None:
        print("Hyperparameter search failed — using defaults k=20, lambda_reg=1.0")
        best_k, best_lambda = 20, 1.0
        baseline_rmse: float | None = None
    else:
        best_k = search_result["best_params"]["k"]
        best_lambda = search_result["best_params"]["lambda_reg"]
        baseline_rmse = search_result["all_results"][0]["rmse"]  # best RMSE from search
        # use the minimum RMSE across all search iterations as baseline
        baseline_rmse = float(min(r["rmse"] for r in search_result["all_results"]))
        print(
            f"Best params: k={best_k}, lambda_reg={best_lambda:.4f}  (baseline RMSE={baseline_rmse:.4f})"
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
            rmses = evaluate_single_network(
                data,
                model_name,
                net_idx,
                k=best_k,
                lambda_reg=best_lambda,
                transform=transform,
                include_communities=include_communities,
                n_splits=n_splits,
            )
            if rmses:
                mean_rmse = float(np.mean(rmses))
                if baseline_rmse is not None and baseline_rmse > 0:
                    improvement = ((baseline_rmse - mean_rmse) / baseline_rmse) * 100.0
                    sign = "+" if improvement > 0 else ""
                    print(
                        f"  Mean RMSE = {mean_rmse:.4f}  "
                        f"(baseline={baseline_rmse:.4f}, "
                        f"improvement={sign}{improvement:.2f}%)"
                    )
                else:
                    print(f"  Mean RMSE = {mean_rmse:.4f}")
                _save_rmses(model_name, net_idx, rmses, baseline_rmse=baseline_rmse)
                all_results[model_name].append(mean_rmse)

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

    _eval_results = run_network_evaluation(
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
