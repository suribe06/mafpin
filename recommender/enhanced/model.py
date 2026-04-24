"""
CMF model training and evaluation with user-side attributes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from cmfrec import CMF  # type: ignore[import-untyped]

from config import Defaults
from recommender.data import evaluate_ranking, evaluate_single_split, split_data_single
from recommender.enhanced.features import _SCALERS


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
    compute_ranking: bool = False,
    ranking_k: int = 10,
    cmf_nthreads: int = -1,
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
        w_user:           Weight for the user side-information reconstruction loss.
        n_splits:         Number of random splits.
        test_size:        Test fraction.
        transform:        Scaler to apply per fold — ``"standard"``, ``"minmax"``,
                          or ``"normalizer"``.
        baseline_k:       Number of latent factors for the paired plain-CMF baseline.
        baseline_lambda:  L2 regularisation for the paired baseline.
        compute_ranking:  When ``True``, compute NDCG@K, Precision@K, Recall@K, and MRR.
        ranking_k:        Cut-off for rank-based metrics.  Defaults to 10.
        cmf_nthreads:     Number of BLAS threads for cmfrec.  Use ``1`` in
                          multi-process contexts to avoid oversubscription.

    Returns:
        List of per-split result dicts with keys ``rmse_enhanced``,
        ``rmse_baseline``, ``improvement`` (baseline − enhanced), and
        (when *compute_ranking* is ``True``) ``ndcg_at_k``, ``precision_at_k``,
        ``recall_at_k``, ``mrr``.
    """
    if transform not in _SCALERS:
        raise ValueError(
            f"Unknown transform: {transform!r}. Use one of {list(_SCALERS)}."
        )

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

        # M-2: fit scaler on training users only to prevent leakage.
        train_users = sorted(train_df["UserId"].unique())
        train_feats = user_attributes.loc[train_users]
        scaler = _SCALERS[transform]()
        scaler.fit(train_feats.values)
        scaled_all = pd.DataFrame(
            scaler.transform(user_attributes.values),
            index=user_attributes.index,
            columns=user_attributes.columns,
        )
        u_matrix = scaled_all.reset_index()

        # Enhanced model
        enhanced_model = CMF(
            method="als",
            k=k,
            lambda_=lambda_reg,
            w_main=w_main,
            w_user=w_user,
            nthreads=cmf_nthreads,
            verbose=False,
        )
        enhanced_model.fit(X=train_df, U=u_matrix)
        enhanced_rmse = evaluate_single_split(enhanced_model, test_df)["rmse"]

        # M-3: paired baseline on the same filtered subset.
        if baseline_k is not None and baseline_lambda is not None:
            baseline_model = CMF(
                method="als",
                k=baseline_k,
                lambda_=baseline_lambda,
                nthreads=cmf_nthreads,
                verbose=False,
            )
            baseline_model.fit(X=train_df)
            baseline_rmse = evaluate_single_split(baseline_model, test_df)["rmse"]
        else:
            baseline_rmse = float("nan")

        result: dict = {
            "rmse_enhanced": enhanced_rmse,
            "rmse_baseline": baseline_rmse,
            "improvement": baseline_rmse - enhanced_rmse,
        }

        if compute_ranking:
            ranking = evaluate_ranking(enhanced_model, train_df, test_df, k=ranking_k)
            result.update(ranking)

        results.append(result)

    return results
