#!/usr/bin/env python3
"""Short validation analyses from the 2026-10-25 audit (C3, seeds, LPH ablation).

Subcommands
-----------
temporal-overlap      C3: look-ahead of train cascades relative to test ratings.
prediction-bootstrap  Paired per-user RMSE / CCE deltas with bootstrap CI + Holm.
frozen-seeds          Refit frozen variants (manifest HP + selected network) per seed.
lph-ablation          Column ablation of LPH features with frozen M3 HP and network.

None of these re-tune hyperparameters or modify canonical artifacts: every
output goes under ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DatasetPaths, SideUserFeatures  # noqa: E402
from recommender.data import (  # noqa: E402
    evaluate_single_split,
    load_and_split_dataset,
    warm_test_slice,
)
from recommender.experiment.cold_start.strata import bootstrap_mean_ci  # noqa: E402
from recommender.experiment.route_b.beyond_accuracy_stats import (  # noqa: E402
    _holm_adjust,
    _wilcoxon_p,
)
from recommender.experiment.route_b.boundary_strata import (  # noqa: E402
    per_user_rmse_from_predictions,
)

DEFAULT_PAIRS = [
    ("M3", "M1"),
    ("M2", "M1"),
    ("M3", "M2"),
    ("M4c", "M3"),
    ("M4d", "M3"),
    ("M4c", "M1"),
    ("M4d", "M1"),
]
CENTRALITY = [
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
    "clustering",
    "eccentricity",
]


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False))
    print(f"\n→ {path}")


# ---------------------------------------------------------------------------
# C3 — temporal look-ahead diagnostic
# ---------------------------------------------------------------------------


def temporal_overlap(dataset: str, out: Path) -> None:
    _, train, test = load_and_split_dataset(dataset=dataset)
    last_train_by_item = train.groupby("ItemId")["timestamp"].max()
    first_test_by_item = test.groupby("ItemId")["timestamp"].min()

    item_last = test["ItemId"].map(last_train_by_item)
    item_first_test = train["ItemId"].map(first_test_by_item)
    row = {
        "dataset": dataset,
        "n_train": len(train),
        "n_test": len(test),
        "train_max_ts": int(train["timestamp"].max()),
        "test_min_ts": int(test["timestamp"].min()),
        # Global ordering: share of train rows newer than the earliest test row.
        "frac_train_after_first_test": float(
            (train["timestamp"] > test["timestamp"].min()).mean()
        ),
        # NetInf channel: the item cascade used for inference contains events
        # later than this test rating.
        "frac_test_with_later_train_in_item": float(
            (item_last > test["timestamp"]).fillna(False).mean()
        ),
        # Share of cascade events posterior to some test rating of the same item.
        "frac_train_after_item_first_test": float(
            (train["timestamp"] > item_first_test).fillna(False).mean()
        ),
    }
    _write(pd.DataFrame([row]), out)


# ---------------------------------------------------------------------------
# Paired bootstrap on saved predictions
# ---------------------------------------------------------------------------


def _load_predictions(pred_dir: Path, variants: list[str]) -> dict[str, pd.DataFrame]:
    preds: dict[str, pd.DataFrame] = {}
    for vid in variants:
        path = pred_dir / f"{vid}.parquet"
        if not path.exists() and path.with_suffix(".csv").exists():
            path = path.with_suffix(".csv")
        if not path.exists():
            print(f"  skip {vid}: missing {path}")
            continue
        preds[vid] = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        )
    return preds


def prediction_bootstrap(
    dataset: str,
    pred_dir: Path,
    variants: list[str],
    out: Path,
    *,
    per_user_ba: Path | None,
    n_samples: int,
    seed: int,
) -> None:
    preds = _load_predictions(pred_dir, variants)
    rmse = {v: per_user_rmse_from_predictions(p) for v, p in preds.items()}
    for vid, frame in preds.items():
        err = frame["Rating"] - frame["Prediction"]
        print(f"  {vid}: global RMSE {float(np.sqrt((err**2).mean())):.6f}")

    cce: dict[str, pd.Series] = {}
    if per_user_ba is not None and per_user_ba.exists():
        ba = (
            pd.read_parquet(per_user_ba)
            if per_user_ba.suffix == ".parquet"
            else pd.read_csv(per_user_ba)
        )
        for vid in preds:
            sub = ba.loc[ba["model_variant"] == vid]
            if not sub.empty:
                cce[vid] = pd.Series(
                    sub.set_index("UserId")["cce_at_k"], dtype=float
                )

    rows: list[dict] = []
    for metric, table in (("rmse", rmse), ("cce_at_10", cce)):
        for first, second in DEFAULT_PAIRS:
            if first not in table or second not in table:
                continue
            joined = pd.concat(
                [table[first], table[second]], axis=1, keys=["a", "b"]
            ).dropna()
            deltas = (joined["a"] - joined["b"]).to_numpy(dtype=float)
            stats = bootstrap_mean_ci(deltas, n_samples=n_samples, seed=seed)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "comparison": f"{first}-{second}",
                    "n_users": int(stats["n"]),
                    "mean_delta": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "ci_excludes_0": bool(
                        stats["ci_low"] > 0 or stats["ci_high"] < 0
                    ),
                    "wilcoxon_p": _wilcoxon_p(deltas),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for metric in frame["metric"].unique():
            mask = frame["metric"] == metric
            frame.loc[mask, "wilcoxon_p_holm"] = _holm_adjust(
                frame.loc[mask, "wilcoxon_p"].tolist()
            )
    print("RMSE: negative delta = first variant better. CCE: positive = more exposure.")
    _write(frame, out)


# ---------------------------------------------------------------------------
# Frozen multi-seed refit
# ---------------------------------------------------------------------------


def _frozen_inputs(dataset: str) -> tuple[dict, dict, dict]:
    from recommender.experiment.final_eval import load_canonical_baseline
    from recommender.experiment.manifest import load_manifest

    dp = DatasetPaths(dataset)
    manifest = load_manifest(dataset)
    baseline = load_canonical_baseline(dataset)["best_params"]
    selection = (
        json.loads(dp.NETWORK_SELECTION.read_text(encoding="utf-8"))
        if dp.NETWORK_SELECTION.exists()
        else {}
    )
    return manifest, baseline, selection


def frozen_seeds(
    dataset: str,
    variants: list[str],
    seeds: list[int],
    out_dir: Path,
) -> None:
    from recommender.experiment import final_eval as fe

    # shortcut: ranking is skipped (only RMSE is compared); re-enable for NDCG.
    fe.evaluate_ranking = lambda *a, **k: dict.fromkeys(  # type: ignore[assignment]
        ("ndcg_at_k", "precision_at_k", "recall_at_k", "mrr"), float("nan")
    )
    manifest, baseline, selection = _frozen_inputs(dataset)
    _, train, test = load_and_split_dataset(dataset=dataset)

    rows: list[dict] = []
    for seed in seeds:
        for vid in variants:
            entry = manifest.get("variants", {}).get(vid, {})
            hyper = dict(baseline) if vid == "M1" else entry.get("hyperparameters")
            if not hyper:
                print(f"  skip {vid}: no hyperparameters in manifest")
                continue
            selected = entry.get("selected_network") or selection.get(
                "variants", {}
            ).get(vid)
            print(f"--- {dataset} {vid} seed={seed}", flush=True)
            row = fe.evaluate_variant_global_test(
                vid,
                train,
                test,
                dataset=dataset,
                hyperparameters=hyper,
                baseline_params=baseline,
                selected_network=selected,
                nthreads=1,
                random_state=seed,
                save_predictions_path=out_dir / f"seed{seed}" / f"{vid}.parquet",
            )
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model_variant": vid,
                    "rmse": row["rmse"],
                    "rmse_warm": row["rmse_warm"],
                    "valid": row["valid_metric_row"],
                }
            )
            print(f"    RMSE {row['rmse']:.6f}  warm {row['rmse_warm']:.6f}")

    raw = pd.DataFrame(rows)
    _write(raw, out_dir / "frozen_seeds_raw.csv")

    wide = raw.pivot(index="seed", columns="model_variant", values="rmse")
    summary: list[dict] = []
    for first, second in DEFAULT_PAIRS:
        if first in wide and second in wide:
            delta = wide[first] - wide[second]
            summary.append(
                {
                    "comparison": f"{first}-{second}",
                    "mean_delta": float(delta.mean()),
                    "sd_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                    "seeds_first_better": int((delta < 0).sum()),
                    "n_seeds": int(delta.notna().sum()),
                }
            )
    stats = raw.groupby("model_variant")[["rmse", "rmse_warm"]].agg(["mean", "std"])
    print(stats.to_string())
    _write(pd.DataFrame(summary), out_dir / "frozen_seeds_deltas.csv")


# ---------------------------------------------------------------------------
# LPH ablation (frozen M3 HP + network; only columns change)
# ---------------------------------------------------------------------------


def _feature_config(enabled: list[str]) -> dict[str, bool]:
    return {key: key in enabled for key in SideUserFeatures.FEATURES}


def lph_ablation(dataset: str, seeds: list[int], out: Path) -> None:
    from recommender.enhanced.features import load_network_features
    from recommender.enhanced.model import fit_enhanced_cmf

    manifest, _, selection = _frozen_inputs(dataset)
    entry = manifest["variants"]["M3"]
    hyper = entry["hyperparameters"]
    net = entry.get("selected_network") or selection["variants"]["M3"]
    model_name, net_idx = net["diffusion_model"], int(net["alpha_index"])
    _, train, test = load_and_split_dataset(dataset=dataset)
    warm = warm_test_slice(train, test)

    default_on = [k for k, v in SideUserFeatures.FEATURES.items() if v]
    configs = {
        "centrality": _feature_config(CENTRALITY),
        "centrality+pagerank_lph (=M2)": _feature_config(CENTRALITY + ["pagerank_lph"]),
        "centrality+h_tilde": _feature_config(CENTRALITY + ["lph_score"]),
        "centrality+jaccard_lph": _feature_config(
            CENTRALITY + ["local_pluralistic_hom"]
        ),
        "centrality+s_v": _feature_config(CENTRALITY + ["s_v"]),
        "M3_full": _feature_config(default_on),
    }

    rows: list[dict] = []
    for name, cfg in configs.items():
        attrs = load_network_features(
            model_name,
            net_idx,
            include_communities=True,
            dataset=dataset,
            feature_config=cfg,
        )
        if attrs is None:
            raise FileNotFoundError(f"Missing features for {model_name} #{net_idx}")
        for seed in seeds:
            model = fit_enhanced_cmf(
                train,
                attrs,
                k=int(hyper["k"]),
                lambda_reg=float(hyper["lambda_reg"]),
                w_main=float(hyper["w_main"]),
                w_user=float(hyper["w_user"]),
                nthreads=1,
                random_state=seed,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "feature_set": name,
                    "n_columns": attrs.shape[1],
                    "columns": ";".join(attrs.columns),
                    "seed": seed,
                    "rmse": evaluate_single_split(model, test)["rmse"],
                    "rmse_warm": evaluate_single_split(model, warm)["rmse"],
                }
            )
            print(f"  {name:32s} seed={seed} RMSE {rows[-1]['rmse']:.6f}", flush=True)

    raw = pd.DataFrame(rows)
    ref = raw.loc[raw["feature_set"] == "centrality"].set_index("seed")["rmse"]
    raw["delta_vs_centrality"] = raw["rmse"] - raw["seed"].map(ref)
    _write(raw, out)
    print(
        raw.groupby("feature_set")[["rmse", "rmse_warm", "delta_vs_centrality"]]
        .agg(["mean", "std"])
        .to_string()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("temporal-overlap")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("prediction-bootstrap")
    p.add_argument("--dataset", required=True)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--per-user-ba", type=Path, default=None)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("frozen-seeds")
    p.add_argument("--dataset", required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p = sub.add_parser("lph-ablation")
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "temporal-overlap":
        temporal_overlap(args.dataset, args.out)
    elif args.cmd == "prediction-bootstrap":
        prediction_bootstrap(
            args.dataset,
            args.pred_dir,
            args.variants,
            args.out,
            per_user_ba=args.per_user_ba,
            n_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    elif args.cmd == "frozen-seeds":
        frozen_seeds(args.dataset, args.variants, args.seeds, args.out_dir)
    else:
        lph_ablation(args.dataset, args.seeds, args.out)


if __name__ == "__main__":
    main()
