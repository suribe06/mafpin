"""Boundary-guided social regularization for Phase 6.

This module is intentionally separate from the existing enhanced CMF path. It
uses the patched local ``cmfrec`` L-BFGS implementation and passes weighted
social edges as COO arrays through ``CMF(lambda_social=..., social_row=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import DatasetPaths, Defaults, Models
from networks.network_io import directed_to_undirected, load_as_networkx
from recommender._cmfrec import CMF  # type: ignore[attr-defined]
from recommender.data import evaluate_single_split
from recommender.enhanced.features import _SCALERS

SocialMode = Literal[
    "uniform",
    "community_jaccard",
    "boundary_downweight",
    "bridge_preserve",
]


@dataclass(frozen=True)
class SocialEdges:
    """COO representation consumed by patched cmfrec."""

    row: np.ndarray
    col: np.ndarray
    val: np.ndarray
    mode: str
    n_edges: int
    mean_weight: float
    min_weight: float
    max_weight: float


def _network_file(dataset: str, model_name: str, network_index: int) -> Path:
    short = Models.SHORT[model_name]
    return (
        DatasetPaths(dataset).NETWORKS
        / model_name
        / f"inferred-network-{short}-{network_index:03d}.txt"
    )


def _community_file(dataset: str, model_name: str, network_index: int) -> Path:
    return (
        DatasetPaths(dataset).COMMUNITIES
        / model_name
        / f"communities_{model_name}_{network_index:03d}.csv"
    )


def load_community_frame(
    dataset: str,
    model_name: str,
    network_index: int,
) -> pd.DataFrame:
    """Load the raw community CSV used to derive social edge weights."""
    path = _community_file(dataset, model_name, network_index)
    if not path.exists():
        raise FileNotFoundError(f"Community file not found: {path}")
    return pd.read_csv(path).set_index("UserId")


def _parse_community_sets(community_frame: pd.DataFrame) -> dict[int, set[int]]:
    if "community_ids" not in community_frame.columns:
        return {int(user): set() for user in community_frame.index}

    parsed: dict[int, set[int]] = {}
    for user, raw_value in community_frame["community_ids"].fillna("").items():
        ids = {
            int(part)
            for part in str(raw_value).split(";")
            if part not in {"", "nan", "None"}
        }
        parsed[int(str(user))] = ids
    return parsed


def _boundary_intensity(community_frame: pd.DataFrame) -> dict[int, float]:
    """Return a 0-1 boundary score where larger means more boundary-like."""
    if "lph_score" in community_frame.columns:
        raw = (-community_frame["lph_score"].astype(float)).clip(lower=0.0)
        max_raw = float(raw.max())
        if max_raw > 0:
            values = raw / max_raw
        elif "is_boundary" in community_frame.columns:
            values = community_frame["is_boundary"].astype(float)
        else:
            values = raw
    elif "is_boundary" in community_frame.columns:
        values = community_frame["is_boundary"].astype(float)
    elif "local_pluralistic_hom" in community_frame.columns:
        lph = community_frame["local_pluralistic_hom"].astype(float)
        values = 1.0 - ((lph - lph.min()) / (lph.max() - lph.min() + 1e-12))
    else:
        values = pd.Series(0.0, index=community_frame.index)

    return {
        int(str(user)): float(np.clip(value, 0.0, 1.0))
        for user, value in values.items()
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _edge_weight(
    source: int,
    target: int,
    communities: dict[int, set[int]],
    boundary: dict[int, float],
    mode: SocialMode,
    beta: float,
    gamma: float,
) -> float:
    if mode == "uniform":
        return 1.0

    shared = _jaccard(communities.get(source, set()), communities.get(target, set()))
    boundary_pair = max(boundary.get(source, 0.0), boundary.get(target, 0.0))

    if mode == "community_jaccard":
        return shared
    if mode == "boundary_downweight":
        return shared * max(0.0, 1.0 - beta * boundary_pair)
    if mode == "bridge_preserve":
        return float(1.0 / (1.0 + np.exp(-(gamma * shared - beta * boundary_pair))))

    raise ValueError(f"Unknown social mode: {mode!r}")


def build_social_edges(
    dataset: str,
    model_name: str,
    network_index: int,
    user_index: pd.Index | range | list[int],
    mode: SocialMode = "boundary_downweight",
    beta: float = 0.5,
    gamma: float = 1.0,
    symmetrization: str = "union",
    normalize: bool = True,
    dtype: np.dtype | type = np.float32,
) -> SocialEdges:
    """Build weighted upper-triangle social COO arrays for patched cmfrec."""
    if model_name not in Models.ALL:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Choose from {Models.ALL}."
        )

    network_path = _network_file(dataset, model_name, network_index)
    community_frame = load_community_frame(dataset, model_name, network_index)
    graph, _ = load_as_networkx(network_path)
    graph_u = directed_to_undirected(graph, method=symmetrization)

    users = {int(user) for user in user_index}
    communities = _parse_community_sets(community_frame)
    boundary = _boundary_intensity(community_frame)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for source, target in graph_u.edges():
        source_i = int(source)
        target_i = int(target)
        if source_i == target_i or source_i not in users or target_i not in users:
            continue
        row, col = sorted((source_i, target_i))
        weight = _edge_weight(row, col, communities, boundary, mode, beta, gamma)
        if weight <= 0.0 or not np.isfinite(weight):
            continue
        rows.append(row)
        cols.append(col)
        vals.append(weight)

    values = np.asarray(vals, dtype=dtype)
    if normalize and values.size:
        mean_weight = float(values.mean())
        if mean_weight > 0.0:
            values = values / mean_weight

    if values.size:
        mean_value = float(values.mean())
        min_value = float(values.min())
        max_value = float(values.max())
    else:
        mean_value = min_value = max_value = 0.0

    return SocialEdges(
        row=np.asarray(rows, dtype=np.int32),
        col=np.asarray(cols, dtype=np.int32),
        val=values,
        mode=mode,
        n_edges=int(values.size),
        mean_weight=mean_value,
        min_weight=min_value,
        max_weight=max_value,
    )


def _ratings_to_coo(
    ratings: pd.DataFrame,
    n_users: int,
    n_items: int,
    dtype: np.dtype | type,
) -> sp.coo_array:
    return sp.coo_array(
        (
            ratings["Rating"].to_numpy(dtype=dtype, copy=False),
            (
                ratings["UserId"].to_numpy(dtype=np.int32, copy=False),
                ratings["ItemId"].to_numpy(dtype=np.int32, copy=False),
            ),
        ),
        shape=(n_users, n_items),
        dtype=dtype,
    )


def _scaled_user_matrix(
    user_attributes: pd.DataFrame,
    train_users: np.ndarray,
    n_users: int,
    transform: str,
    dtype: np.dtype | type,
) -> np.ndarray:
    if transform not in _SCALERS:
        raise ValueError(
            f"Unknown transform: {transform!r}. Use one of {list(_SCALERS)}."
        )

    aligned = user_attributes.reindex(range(n_users)).fillna(0.0)
    train_index = [user for user in train_users if user in user_attributes.index]
    if not train_index:
        raise ValueError("No training users have side attributes.")

    scaler = _SCALERS[transform]()
    scaler.fit(user_attributes.loc[train_index].values)
    scaled = scaler.transform(aligned.values)
    return np.require(scaled, dtype=dtype, requirements=["ENSUREARRAY", "C_CONTIGUOUS"])


def fit_social_cmf_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_attributes: pd.DataFrame,
    social_edges: SocialEdges,
    k: int = Defaults.K,
    lambda_reg: float = Defaults.LAMBDA_REG,
    w_main: float = Defaults.W_MAIN,
    w_user: float = Defaults.W_USER,
    lambda_social: float = 0.01,
    transform: str = "standard",
    maxiter: int = 25,
    nthreads: int = 1,
    random_state: int = 42,
    use_float: bool = True,
    include_user_attributes: bool = True,
) -> tuple[CMF, dict[str, float]]:
    """Fit one social-regularized CMF split and evaluate rating accuracy."""
    dtype = np.float32 if use_float else np.float64
    max_social_user = 0
    if social_edges.row.size:
        max_social_user = max(max_social_user, int(social_edges.row.max()))
    if social_edges.col.size:
        max_social_user = max(max_social_user, int(social_edges.col.max()))
    n_users = int(
        max(
            train_df["UserId"].max(),
            test_df["UserId"].max(),
            int(np.max(user_attributes.index.to_numpy(dtype=np.int64))),
            max_social_user,
        )
        + 1
    )
    n_items = int(max(train_df["ItemId"].max(), test_df["ItemId"].max()) + 1)
    train_matrix = _ratings_to_coo(train_df, n_users, n_items, dtype=dtype)
    user_matrix = None
    if include_user_attributes:
        user_matrix = _scaled_user_matrix(
            user_attributes,
            np.asarray(train_df["UserId"].unique(), dtype=np.int64),
            n_users,
            transform,
            dtype,
        )
    social_values = np.require(
        social_edges.val,
        dtype=dtype,
        requirements=["ENSUREARRAY", "C_CONTIGUOUS"],
    )

    model = CMF(
        method="lbfgs",
        k=k,
        lambda_=lambda_reg,
        w_main=w_main,
        w_user=w_user,
        lambda_social=lambda_social,
        social_row=social_edges.row,
        social_col=social_edges.col,
        social_val=social_values,
        user_bias=True,
        item_bias=True,
        center=True,
        maxiter=maxiter,
        random_state=random_state,
        verbose=False,
        nthreads=nthreads,
        use_float=use_float,
    )
    if user_matrix is None:
        model.fit(X=train_matrix)
    else:
        model.fit(X=train_matrix, U=user_matrix)
    metrics = evaluate_single_split(model, test_df)
    return model, metrics
