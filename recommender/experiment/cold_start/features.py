"""Feature loading helpers for cold-start (NetInf + trust)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DatasetPaths
from recommender.enhanced.features import load_network_features
from recommender.experiment.cold_start.strata import coverage_from_feature_index


def load_variant_features(
    *,
    dataset: str,
    model_name: str,
    network_index: int,
    include_communities: bool,
    paths: DatasetPaths | None = None,
) -> pd.DataFrame | None:
    """Load NetInf side features from core or cold-start paths."""
    return load_network_features(
        model_name,
        network_index,
        include_communities=include_communities,
        include_cascade_stats=True,
        dataset=dataset,
        paths=paths,
    )


def feature_coverage_from_csvs(
    user_ids: list[int],
    *,
    dataset: str,
    model_name: str,
    network_index: int,
    paths: DatasetPaths | None = None,
    trust_users: set[int] | None = None,
) -> pd.DataFrame:
    """Infer coverage flags from centrality/community CSVs when present.

    ``appears_in_netinf_graph`` / ``has_centrality_features`` require ``degree > 0``
    (cascade headers declare every user, so CSV membership alone is not enough).
    """
    dp = paths or DatasetPaths(dataset)
    index_str = f"{network_index:03d}"
    cent_csv = (
        dp.CENTRALITY / model_name / f"centrality_metrics_{model_name}_{index_str}.csv"
    )
    com_csv = (
        dp.COMMUNITIES / model_name / f"communities_{model_name}_{index_str}.csv"
    )
    centrality_users: set[int] = set()
    lph_users: set[int] = set()
    if cent_csv.exists():
        cent = pd.read_csv(cent_csv)
        if "degree" in cent.columns:
            centrality_users = set(
                cent.loc[cent["degree"].fillna(0) > 0, "UserId"].astype(int)
            )
        else:
            centrality_users = set(cent["UserId"].astype(int))
    if com_csv.exists():
        com = pd.read_csv(com_csv)
        # LPH only meaningful for users that also have graph degree > 0.
        if "local_pluralistic_hom" in com.columns:
            candidates = set(
                com.loc[com["local_pluralistic_hom"].notna(), "UserId"].astype(int)
            )
        else:
            candidates = set(com["UserId"].astype(int))
        lph_users = candidates & centrality_users if centrality_users else candidates
    return coverage_from_feature_index(
        user_ids,
        centrality_users=centrality_users,
        lph_users=lph_users,
        trust_users=trust_users,
    )


def resolve_selected_network(
    manifest: dict,
    variant_id: str,
    *,
    fallback_path: Path | None = None,
) -> dict | None:
    """Pull selected_network for a variant from the experiment manifest."""
    variants = manifest.get("variants") or {}
    entry = variants.get(variant_id) or {}
    selected = entry.get("selected_network")
    if selected:
        return dict(selected)
    # Soft / trust clones reuse M3 network selection.
    if variant_id in {"M3_soft", "M2_trust", "M3_trust"}:
        alt = (variants.get("M3") or {}).get("selected_network")
        if alt:
            return dict(alt)
    # Fall back to M3 / first network-bearing variant.
    for key in ("M3", "M2", "M4c", "M4d"):
        alt = (variants.get(key) or {}).get("selected_network")
        if alt:
            return dict(alt)
    if fallback_path and fallback_path.exists():
        import json

        payload = json.loads(fallback_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            # network_selection_results.json shapes vary; try common keys
            for key in ("M3", "selected", "best"):
                if key in payload and isinstance(payload[key], dict):
                    cand = payload[key]
                    if "diffusion_model" in cand or "model" in cand:
                        return {
                            "diffusion_model": cand.get(
                                "diffusion_model", cand.get("model")
                            ),
                            "alpha_index": cand.get(
                                "alpha_index", cand.get("network_index", 0)
                            ),
                            "alpha_value": cand.get("alpha_value"),
                        }
    return None


def remap_selected_network_to_paths(
    selected: dict,
    paths: DatasetPaths,
) -> dict:
    """Map a core selected_network onto an existing network under *paths*.

    Prefers the exact ``alpha_index`` file when present. Otherwise picks the
    closest ``alpha`` in ``inferred_edges_*.csv`` (needed when cold-start
    smoke runs use ``--n-alphas`` smaller than the core grid).
    """
    from networks.artifacts import NetworkArtifacts

    if not selected:
        raise ValueError("selected_network is required")
    model_name = selected["diffusion_model"]
    net_idx = int(selected["alpha_index"])
    # ColdStartPaths.BASE.name is "cold_start"; prefer dataset from paths.BASE parent.
    dataset = getattr(paths, "BASE", paths.NETWORKS).parent.name
    if dataset == "cold_start":
        dataset = paths.BASE.parent.name
    arts = NetworkArtifacts(dataset, paths=paths)
    exact = arts.network_txt(model_name, net_idx)
    if exact.exists():
        return dict(selected)

    edges_csv = arts.inferred_edges_csv(model_name)
    if not edges_csv.exists():
        raise FileNotFoundError(
            f"Missing cold-start networks for {model_name} under {paths.NETWORKS / model_name}. "
            "Re-run controlled without --skip-rebuild."
        )
    frame = pd.read_csv(edges_csv, sep="|")
    if "alpha" not in frame.columns or frame.empty:
        raise FileNotFoundError(f"Empty alpha grid in {edges_csv}")
    from config import Models

    short = Models.SHORT[model_name]
    edge_col = f"inferred_edges_{short}"
    usable = frame
    if edge_col in frame.columns and (frame[edge_col] > 0).any():
        usable = frame[frame[edge_col] > 0]
    target = selected.get("alpha_value")
    if target is None or (isinstance(target, float) and pd.isna(target)):
        best_i = int(usable.index[0])
    else:
        alphas = usable["alpha"].to_numpy(dtype=float)
        best_i = int(usable.index[int((abs(alphas - float(target))).argmin())])
    remapped = dict(selected)
    remapped["alpha_index"] = int(best_i)
    remapped["alpha_value"] = float(usable.loc[best_i, "alpha"])
    remapped["selection_source"] = (
        f"remapped_from_core_idx_{net_idx}_closest_alpha"
    )
    target_file = arts.network_txt(model_name, best_i)
    if not target_file.exists():
        raise FileNotFoundError(
            f"Remapped network missing: {target_file} "
            f"(from core alpha_index={net_idx})"
        )
    print(
        f"[cold_start] remapped {model_name} network "
        f"idx {net_idx} → {best_i} (α={remapped['alpha_value']:.6g})"
    )
    return remapped
