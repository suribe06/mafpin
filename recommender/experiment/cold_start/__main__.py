"""CLI entrypoint: ``python -m recommender.experiment.cold_start``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from config import DatasetPaths, Datasets, Defaults
from recommender.experiment.cold_start import artifacts
from recommender.experiment.cold_start.deltas import (
    bootstrap_delta_table,
    build_user_deltas,
)
from recommender.experiment.cold_start.evaluate import evaluate_variants_by_stratum
from recommender.experiment.cold_start.features import (
    feature_coverage_from_csvs,
    remap_selected_network_to_paths,
    resolve_selected_network,
)
from recommender.experiment.cold_start.paths import ColdStartPaths
from recommender.experiment.cold_start.rebuild import run_feature_pipeline
from recommender.experiment.cold_start.report import write_success_summary
from recommender.experiment.cold_start.splits import (
    global_strata_split,
    load_encoded_ratings,
    per_user_chrono_split,
    per_user_leave_k_split,
    split_manifest_payload,
    zero_shot_trust_split,
)
from recommender.experiment.cold_start.strata import build_user_strata
from recommender.experiment.cold_start.trust_variants import (
    build_trust_attribute_tables,
    encoded_trust_user_ids,
)
from recommender.experiment.final_eval import load_canonical_baseline
from recommender.experiment.manifest import load_manifest
from recommender.experiment.variants import COLD_START_VARIANT_IDS, TRUST_VARIANT_IDS
from networks.social import load_trust_graph


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cold-start experiment runner (diagnostic / controlled / trust)."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=Datasets.ALL,
        help="Dataset name",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["diagnostic", "controlled", "zero_shot_trust", "report"],
        help="Experiment mode",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variant IDs (default depends on mode)",
    )
    parser.add_argument("--seed", type=int, default=Defaults.CMF_RANDOM_STATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override cold_start output root",
    )
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="controlled mode: skip cascade/NetInf rebuild",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--include-ranking",
        action="store_true",
        help="Also compute global ranking metrics (secondary)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Per-user late-holdout fraction for controlled split",
    )
    parser.add_argument(
        "--split",
        choices=["leave_last", "leave_k"],
        default="leave_last",
        help=(
            "controlled split protocol: leave_last (default) or leave_k "
            "(stratified caps 0/2/7/all to populate cold strata on dense data)"
        ),
    )
    parser.add_argument("--n-alphas", type=int, default=Defaults.N_ALPHAS)
    parser.add_argument("--max-iter", type=int, default=Defaults.MAX_ITER)
    parser.add_argument("--k-avg-degree", type=int, default=Defaults.K_AVG_DEGREE)
    parser.add_argument("--cmf-maxiter", type=int, default=Defaults.CMF_MAXITER)
    return parser.parse_args(argv)


def _paths(args: argparse.Namespace) -> ColdStartPaths:
    return ColdStartPaths(args.dataset, root=args.output_dir)


def _load_experiment_context(dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(dataset)
    baseline = load_canonical_baseline(dataset)
    best = baseline.get("best_params") or baseline
    baseline_params = {
        "k": int(best["k"]),
        "lambda_reg": float(best["lambda_reg"]),
    }
    return manifest, baseline_params


def _write_eval_outputs(
    *,
    paths: ColdStartPaths,
    dataset: str,
    mode: str,
    user_strata,
    result_rows: list[dict[str, Any]],
    rmse_by_variant: dict,
    bootstrap_samples: int,
    seed: int,
) -> None:
    artifacts.write_csv(paths.USER_STRATA, user_strata)
    artifacts.upsert_results(
        paths.RESULTS,
        result_rows,
        keys=["dataset", "mode", "model_variant", "stratum"],
    )
    deltas = build_user_deltas(
        dataset=dataset,
        mode=mode,
        user_strata=user_strata,
        rmse_by_variant=rmse_by_variant,
    )
    artifacts.upsert_frame(
        paths.USER_DELTAS,
        deltas,
        keys=["dataset", "mode", "UserId"],
    )
    boot = bootstrap_delta_table(
        deltas,
        dataset=dataset,
        mode=mode,
        n_samples=bootstrap_samples,
        seed=seed,
    )
    artifacts.upsert_frame(
        paths.BOOTSTRAP_CIS,
        boot,
        keys=["dataset", "mode", "stratum", "comparison"],
    )
    artifacts.write_readme(paths, mode=mode, dataset=dataset)


def run_diagnostic(args: argparse.Namespace) -> None:
    paths = _paths(args)
    paths.ensure_dirs()
    variants = args.variants or COLD_START_VARIANT_IDS
    full_df, train_df, test_df = global_strata_split(args.dataset)
    manifest, baseline_params = _load_experiment_context(args.dataset)
    selected = resolve_selected_network(
        manifest,
        "M3",
        fallback_path=DatasetPaths(args.dataset).NETWORK_SELECTION,
    )
    coverage = None
    if selected:
        coverage = feature_coverage_from_csvs(
            sorted(set(full_df["UserId"].unique())),
            dataset=args.dataset,
            model_name=selected["diffusion_model"],
            network_index=int(selected["alpha_index"]),
            paths=DatasetPaths(args.dataset),
        )
    user_strata = build_user_strata(train_df, test_df, feature_coverage=coverage)
    artifacts.write_json(
        paths.SPLIT_MANIFEST,
        split_manifest_payload(
            mode="diagnostic",
            dataset=args.dataset,
            train_df=train_df,
            test_df=test_df,
            extra={
                "note": (
                    "Uses core global split and core NetInf features; "
                    "network attributes may include leakage relative to "
                    "per-user cold-start labels (§6.1)."
                ),
                "selected_network": selected,
            },
        ),
    )
    # Audit copy of the global split (not the scientific controlled split).
    artifacts.write_split_tables(paths, train_df, test_df)

    result_rows, rmse_by_variant = evaluate_variants_by_stratum(
        dataset=args.dataset,
        mode="diagnostic",
        train_df=train_df,
        test_df=test_df,
        user_strata=user_strata,
        variant_ids=variants,
        manifest=manifest,
        baseline_params=baseline_params,
        selected_network=selected,
        paths=DatasetPaths(args.dataset),
        include_ranking=args.include_ranking,
        maxiter=args.cmf_maxiter,
        random_state=args.seed,
    )
    _write_eval_outputs(
        paths=paths,
        dataset=args.dataset,
        mode="diagnostic",
        user_strata=user_strata,
        result_rows=result_rows,
        rmse_by_variant=rmse_by_variant,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_success_summary(paths, dataset=args.dataset, mode="diagnostic")


def run_controlled(args: argparse.Namespace) -> None:
    paths = _paths(args)
    paths.ensure_dirs()
    variants = args.variants or COLD_START_VARIANT_IDS
    full_df, user_enc, _item_enc = load_encoded_ratings(args.dataset)
    del user_enc  # encoding already applied; retained API for symmetry
    split_meta: dict[str, Any] = {"split_protocol": args.split, "test_frac": args.test_frac}
    if args.split == "leave_k":
        train_df, test_df, leave_meta = per_user_leave_k_split(
            full_df, test_frac=args.test_frac, seed=args.seed
        )
        split_meta.update(leave_meta)
        mode_label = "controlled_leave_k"
    else:
        train_df, test_df = per_user_chrono_split(full_df, test_frac=args.test_frac)
        mode_label = "controlled"
    manifest, baseline_params = _load_experiment_context(args.dataset)
    selected = resolve_selected_network(
        manifest,
        "M3",
        fallback_path=DatasetPaths(args.dataset).NETWORK_SELECTION,
    )

    rebuild_info: dict[str, Any] | None = None
    if not args.skip_rebuild:
        rebuild_info = run_feature_pipeline(
            args.dataset,
            train_df,
            all_user_ids=list(map(int, full_df["UserId"].tolist())),
            paths=paths,
            n_alphas=args.n_alphas,
            max_iter=args.max_iter,
            k_avg_degree=args.k_avg_degree,
        )
    elif not paths.CASCADES.exists():
        raise FileNotFoundError(
            f"Missing {paths.CASCADES}; run without --skip-rebuild first"
        )

    coverage = None
    if selected:
        selected = remap_selected_network_to_paths(selected, paths)
        coverage = feature_coverage_from_csvs(
            sorted(set(full_df["UserId"].unique())),
            dataset=args.dataset,
            model_name=selected["diffusion_model"],
            network_index=int(selected["alpha_index"]),
            paths=paths,
        )
    user_strata = build_user_strata(train_df, test_df, feature_coverage=coverage)
    artifacts.write_json(
        paths.SPLIT_MANIFEST,
        split_manifest_payload(
            mode=mode_label,
            dataset=args.dataset,
            train_df=train_df,
            test_df=test_df,
            extra={
                **split_meta,
                "selected_network": selected,
                "rebuild": rebuild_info,
                "skip_rebuild": bool(args.skip_rebuild),
                "cascades_path": str(paths.CASCADES),
            },
        ),
    )
    artifacts.write_split_tables(paths, train_df, test_df)

    # Remap each variant's network onto the cold-start grid before eval.
    remapped_manifest = dict(manifest)
    remapped_variants = dict(manifest.get("variants") or {})
    for vid, entry in remapped_variants.items():
        entry = dict(entry)
        net = entry.get("selected_network") or selected
        if net:
            entry["selected_network"] = remap_selected_network_to_paths(net, paths)
        remapped_variants[vid] = entry
    remapped_manifest["variants"] = remapped_variants

    # Keep CSV mode tag as "controlled" so report/filters stay stable; protocol in manifest.
    result_rows, rmse_by_variant = evaluate_variants_by_stratum(
        dataset=args.dataset,
        mode="controlled",
        train_df=train_df,
        test_df=test_df,
        user_strata=user_strata,
        variant_ids=variants,
        manifest=remapped_manifest,
        baseline_params=baseline_params,
        selected_network=selected,
        paths=paths,
        include_ranking=args.include_ranking,
        maxiter=args.cmf_maxiter,
        random_state=args.seed,
    )
    _write_eval_outputs(
        paths=paths,
        dataset=args.dataset,
        mode="controlled",
        user_strata=user_strata,
        result_rows=result_rows,
        rmse_by_variant=rmse_by_variant,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_success_summary(paths, dataset=args.dataset, mode="controlled")


def run_zero_shot_trust(args: argparse.Namespace) -> None:
    if args.dataset not in {"ciao", "epinions"}:
        raise SystemExit(
            "zero_shot_trust requires ciao or epinions (explicit trust graph)"
        )
    paths = _paths(args)
    paths.ensure_dirs()
    out = paths.ZERO_SHOT
    out.mkdir(parents=True, exist_ok=True)

    variants = args.variants or TRUST_VARIANT_IDS
    full_df, user_enc, _item_enc = load_encoded_ratings(args.dataset)
    m2_attrs, m3_attrs, trust_users = build_trust_attribute_tables(
        args.dataset, user_enc
    )
    # Also accept users present in the trust graph even if centrality empty.
    G = load_trust_graph(args.dataset)
    trust_users |= encoded_trust_user_ids(user_enc, G)

    train_df, test_df, zero_users = zero_shot_trust_split(
        full_df, encoded_trust_users=trust_users
    )
    from recommender.experiment.cold_start.strata import coverage_from_feature_index

    coverage = coverage_from_feature_index(
        sorted(set(full_df["UserId"].unique())),
        trust_users=trust_users,
    )
    user_strata = build_user_strata(train_df, test_df, feature_coverage=coverage)
    # Force stratum 0 for held-out trust users.
    user_strata.loc[user_strata["user_id"].isin(zero_users), "stratum"] = "0"
    user_strata.loc[user_strata["user_id"].isin(zero_users), "n_train_ratings"] = 0

    manifest, baseline_params = _load_experiment_context(args.dataset)
    trust_attrs = {
        "M2_trust": m2_attrs,
        "M3_trust": m3_attrs,
    }

    # Write zero-shot-specific copies under zero_shot_trust/
    z_train = out / "train.csv"
    z_test = out / "test.csv"
    z_strata = out / "user_strata.csv"
    z_manifest = out / "split_manifest.json"
    z_results = out / "cold_start_results.csv"
    z_deltas = out / "cold_start_user_deltas.csv"
    z_boot = out / "bootstrap_confidence_intervals.csv"

    artifacts.write_csv(z_train, train_df)
    artifacts.write_csv(z_test, test_df)
    artifacts.write_csv(z_strata, user_strata)
    artifacts.write_json(
        z_manifest,
        split_manifest_payload(
            mode="zero_shot_trust",
            dataset=args.dataset,
            train_df=train_df,
            test_df=test_df,
            extra={
                "n_zero_shot_users": len(zero_users),
                "note": (
                    "Users overlapping the trust graph have all ratings in test; "
                    "M2_trust/M3_trust use trust-graph attributes only."
                ),
            },
        ),
    )

    result_rows, rmse_by_variant = evaluate_variants_by_stratum(
        dataset=args.dataset,
        mode="zero_shot_trust",
        train_df=train_df,
        test_df=test_df,
        user_strata=user_strata,
        variant_ids=variants,
        manifest=manifest,
        baseline_params=baseline_params,
        selected_network=None,
        trust_attributes=trust_attrs,
        include_ranking=args.include_ranking,
        maxiter=args.cmf_maxiter,
        random_state=args.seed,
    )
    artifacts.upsert_results(
        z_results,
        result_rows,
        keys=["dataset", "mode", "model_variant", "stratum"],
    )
    deltas = build_user_deltas(
        dataset=args.dataset,
        mode="zero_shot_trust",
        user_strata=user_strata,
        rmse_by_variant=rmse_by_variant,
    )
    artifacts.write_csv(z_deltas, deltas)
    boot = bootstrap_delta_table(
        deltas,
        dataset=args.dataset,
        mode="zero_shot_trust",
        n_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    artifacts.write_csv(z_boot, boot)

    # Also mirror into top-level cold_start results for the report mode.
    artifacts.upsert_results(
        paths.RESULTS,
        result_rows,
        keys=["dataset", "mode", "model_variant", "stratum"],
    )
    write_success_summary(
        paths,
        dataset=args.dataset,
        mode="zero_shot_trust",
        results_path=z_results,
        bootstrap_path=z_boot,
    )
    print(f"Zero-shot trust artifacts → {out}")


def run_report(args: argparse.Namespace) -> None:
    paths = _paths(args)
    mode = "diagnostic"
    if paths.RESULTS.exists():
        import pandas as pd

        modes = pd.read_csv(paths.RESULTS)["mode"].dropna().unique().tolist()
        if "controlled" in modes:
            mode = "controlled"
        elif "diagnostic" in modes:
            mode = "diagnostic"
        elif modes:
            mode = str(modes[0])
    write_success_summary(paths, dataset=args.dataset, mode=mode)
    z_results = paths.ZERO_SHOT / "cold_start_results.csv"
    if z_results.exists():
        # Write a separate summary beside zero-shot artifacts.
        text_path = paths.ZERO_SHOT / "success_summary.md"
        from recommender.experiment.cold_start.report import build_success_summary
        import pandas as pd

        results = pd.read_csv(z_results)
        boot_path = paths.ZERO_SHOT / "bootstrap_confidence_intervals.csv"
        boot = pd.read_csv(boot_path) if boot_path.exists() else None
        text_path.write_text(
            build_success_summary(
                results, boot, dataset=args.dataset, mode="zero_shot_trust"
            ),
            encoding="utf-8",
        )
        print(f"Zero-shot summary → {text_path}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.mode == "diagnostic":
        run_diagnostic(args)
    elif args.mode == "controlled":
        run_controlled(args)
    elif args.mode == "zero_shot_trust":
        run_zero_shot_trust(args)
    elif args.mode == "report":
        run_report(args)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
