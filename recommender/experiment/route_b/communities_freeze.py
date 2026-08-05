"""Load frozen M3 community partition for Route B metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from networks.artifacts import NetworkArtifacts


def resolve_m3_network(dataset: str) -> dict:
    """Return selected_network for M3 from manifest or network_selection JSON."""
    from config import DatasetPaths

    dp = DatasetPaths(dataset)
    manifest_path = dp.EXPERIMENT_MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = (manifest.get("variants") or {}).get("M3") or {}
        selected = entry.get("selected_network")
        if selected:
            return dict(selected)
    sel_path = dp.NETWORK_SELECTION
    if sel_path.exists():
        payload = json.loads(sel_path.read_text(encoding="utf-8"))
        # Prefer M3 row if present
        if isinstance(payload, dict):
            for key in ("M3", "best", "selected"):
                if key in payload and isinstance(payload[key], dict):
                    net = payload[key]
                    if "diffusion_model" in net or "model_name" in net:
                        return {
                            "diffusion_model": net.get("diffusion_model")
                            or net.get("model_name"),
                            "alpha_index": int(net["alpha_index"]),
                            "alpha_value": net.get("alpha_value"),
                        }
            # Flat selection file
            if "diffusion_model" in payload:
                return {
                    "diffusion_model": payload["diffusion_model"],
                    "alpha_index": int(payload["alpha_index"]),
                    "alpha_value": payload.get("alpha_value"),
                }
        if isinstance(payload, list):
            for row in payload:
                if row.get("variant_id") == "M3" or row.get("model_variant") == "M3":
                    return {
                        "diffusion_model": row.get("diffusion_model")
                        or row.get("model_name"),
                        "alpha_index": int(row["alpha_index"]),
                        "alpha_value": row.get("alpha_value"),
                    }
    raise FileNotFoundError(
        f"Could not resolve M3 selected network for {dataset}. "
        "Need experiment_manifest.json or network_selection_results.json."
    )


def parse_community_ids(raw: object) -> set[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    text = str(raw).strip()
    if not text:
        return set()
    out: set[int] = set()
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
            out.add(int(part))
    return out


def load_frozen_communities(
    dataset: str,
    *,
    network: dict | None = None,
    paths=None,
) -> pd.DataFrame:
    """Community CSV for the frozen M3 network, indexed by UserId."""
    net = network or resolve_m3_network(dataset)
    model = net["diffusion_model"]
    idx = int(net["alpha_index"])
    arts = NetworkArtifacts(dataset, paths=paths)
    csv_path = arts.communities_csv(model, idx)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing frozen communities: {csv_path}")
    frame = pd.read_csv(csv_path)
    if "UserId" not in frame.columns:
        raise ValueError(f"No UserId in {csv_path}")
    frame = frame.set_index("UserId")
    if "community_ids" not in frame.columns:
        frame["community_ids"] = ""
    frame["community_set"] = frame["community_ids"].map(parse_community_ids)
    return frame


def item_dominant_communities(
    train_df: pd.DataFrame,
    user_communities: pd.Series,
) -> dict[int, set[int]]:
    """Map item → moda of communities among train raters (multiset mode)."""
    from collections import Counter

    # user_communities: UserId → set[int]
    item_to_coms: dict[int, Counter] = {}
    for _, row in train_df.iterrows():
        uid = int(row["UserId"])
        iid = int(row["ItemId"])
        coms = user_communities.get(uid, set())
        if not coms:
            continue
        bucket = item_to_coms.setdefault(iid, Counter())
        for c in coms:
            bucket[c] += 1
    dominant: dict[int, set[int]] = {}
    for iid, counter in item_to_coms.items():
        if not counter:
            continue
        top = counter.most_common(1)[0][1]
        dominant[iid] = {c for c, n in counter.items() if n == top}
    return dominant
