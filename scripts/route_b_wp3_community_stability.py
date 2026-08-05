#!/usr/bin/env python3
"""Route B WP3 — community / LPH stability across α neighbours and detectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as script from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.artifacts import NetworkArtifacts  # noqa: E402
from networks.communities.detection import (  # noqa: E402
    compute_node_community_membership,
    detect_overlapping_communities,
)
from networks.communities.lph import compute_lph_paper  # noqa: E402
from networks.network_io import directed_to_undirected, load_as_networkx  # noqa: E402
from recommender.experiment.route_b.communities_freeze import (  # noqa: E402
    resolve_m3_network,
)
from recommender.experiment.route_b.paths import RouteBPaths  # noqa: E402


def _lph_series_from_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return pd.Series(df.set_index("UserId")["lph_score"], dtype=float)


def _b10_set(lph: pd.Series) -> set[int]:
    clean = pd.Series(lph, dtype=float).dropna()
    if clean.empty:
        return set()
    thr = float(np.nanpercentile(clean.to_numpy(dtype=float), 10))
    mask = clean <= thr
    return {int(i) for i in list(clean.loc[mask].index)}


def _spearman(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(joined) < 5:
        return float("nan")
    a = pd.Series(joined["a"], dtype=float)
    b = pd.Series(joined["b"], dtype=float)
    return float(a.corr(b, method="spearman"))


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return float("nan")
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else float("nan")


def _recompute_lph(network_path: Path, algorithm: str, seed: int) -> pd.Series:
    del seed  # detection API has no seed; keep signature for callers
    graph, _ = load_as_networkx(network_path)
    undirected = directed_to_undirected(graph, method="union")
    communities = detect_overlapping_communities(undirected, algorithm=algorithm)
    membership = compute_node_community_membership(
        list(map(int, undirected.nodes())), communities
    )
    lph_scores, _s, _d = compute_lph_paper(undirected, membership)
    return pd.Series(lph_scores, dtype=float)


def run_stability(
    dataset: str,
    *,
    neighbors: int = 2,
    detectors: list[str] | None = None,
    seed: int = 42,
) -> Path:
    detectors = detectors or ["demon"]
    paths = RouteBPaths(dataset)
    paths.ensure_dirs()
    net = resolve_m3_network(dataset)
    model = net["diffusion_model"]
    center = int(net["alpha_index"])
    arts = NetworkArtifacts(dataset)
    indices = arts.list_community_indices(model)
    if center not in indices:
        raise FileNotFoundError(
            f"Center α index {center} not in communities for {model}"
        )

    center_csv = arts.communities_csv(model, center)
    center_lph = _lph_series_from_csv(center_csv)
    center_b10 = _b10_set(center_lph)

    rows: list[dict] = []
    # E3.2 — α neighbours, same diffusion model, precomputed DEMON CSVs
    for idx in indices:
        if idx == center or abs(idx - center) > neighbors:
            continue
        other = _lph_series_from_csv(arts.communities_csv(model, idx))
        rows.append(
            {
                "dataset": dataset,
                "comparison": "alpha_neighbor",
                "model": model,
                "center_index": center,
                "other_index": idx,
                "detector": "demon",
                "spearman_lph": _spearman(center_lph, other),
                "jaccard_b10": _jaccard(center_b10, _b10_set(other)),
                "n_users": int(
                    pd.concat([center_lph.rename("a"), other.rename("b")], axis=1)
                    .dropna()
                    .shape[0]
                ),
            }
        )

    # E3.3 — alternate detectors on frozen network
    net_path = arts.network_txt(model, center)
    for det in detectors:
        if det == "demon":
            continue
        try:
            alt_lph = _recompute_lph(net_path, det, seed)
        except Exception as exc:  # ponytail: detector may be unavailable
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": "detector",
                    "model": model,
                    "center_index": center,
                    "other_index": center,
                    "detector": det,
                    "spearman_lph": float("nan"),
                    "jaccard_b10": float("nan"),
                    "n_users": 0,
                    "error": str(exc),
                }
            )
            continue
        rows.append(
            {
                "dataset": dataset,
                "comparison": "detector",
                "model": model,
                "center_index": center,
                "other_index": center,
                "detector": det,
                "spearman_lph": _spearman(center_lph, alt_lph),
                "jaccard_b10": _jaccard(center_b10, _b10_set(alt_lph)),
                "n_users": int(
                    pd.concat([center_lph.rename("a"), alt_lph.rename("b")], axis=1)
                    .dropna()
                    .shape[0]
                ),
            }
        )

    # Static quality for center
    com = pd.read_csv(center_csv)
    rows.append(
        {
            "dataset": dataset,
            "comparison": "quality_static",
            "model": model,
            "center_index": center,
            "other_index": center,
            "detector": "demon",
            "spearman_lph": 1.0,
            "jaccard_b10": 1.0,
            "n_users": int(len(com)),
            "mean_num_communities": float(com["num_communities"].mean())
            if "num_communities" in com.columns
            else float("nan"),
            "frac_boundary": float(com["is_boundary"].mean())
            if "is_boundary" in com.columns
            else float("nan"),
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(paths.COMMUNITY_STABILITY, index=False)
    print(f"Community stability → {paths.COMMUNITY_STABILITY}")
    # Quick console verdict
    neigh = pd.Series(out.loc[out["comparison"] == "alpha_neighbor", "spearman_lph"]).dropna()
    if len(neigh):
        rho = float(neigh.mean())
        verdict = (
            "interpretable (ρ≥0.7)"
            if rho >= 0.7
            else "grey zone" if rho >= 0.4 else "unstable (ρ<0.4)"
        )
        print(f"Mean Spearman(α neighbours) = {rho:.3f} → {verdict}")
    return paths.COMMUNITY_STABILITY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["movielens", "ciao", "epinions"])
    parser.add_argument("--neighbors", type=int, default=2)
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["demon", "aslpaw"],
        help="Community detectors (demon uses precomputed CSV; others recompute)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_stability(
        args.dataset,
        neighbors=args.neighbors,
        detectors=list(args.detectors),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
