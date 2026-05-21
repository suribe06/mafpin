"""Pre-registration step: stratified network sample plan."""

from __future__ import annotations

import argparse
import json as _json


def run_preregister(args: argparse.Namespace) -> None:
    """Write a stratified network sample plan across alpha quantiles × models.

    The plan covers three density strata (sparse / medium / dense) defined by
    tertiles of the alpha index range for each diffusion model.  Five networks
    are sampled from each stratum, giving 45 reference networks in total
    (3 models × 3 strata × 5).  Saving this plan *before* running evaluations
    constitutes a pre-registration commitment and helps reviewers verify that
    reported results are not cherry-picked.
    """
    from config import DatasetPaths, Models

    dp = DatasetPaths(args.dataset)

    plan: dict = {"dataset": args.dataset, "strata": {}, "models": {}}
    total_networks = 0

    for model_name in Models.ALL:
        model_dir = dp.CENTRALITY / model_name
        csvs = sorted(model_dir.glob(f"centrality_metrics_{model_name}_*.csv"))
        if not csvs:
            print(f"  No centrality CSVs for {model_name} — skipping.")
            continue

        indices = sorted(int(p.stem.rsplit("_", 1)[-1]) for p in csvs)
        n = len(indices)
        # Tertile boundaries
        q33 = indices[n // 3]
        q67 = indices[2 * n // 3]

        sparse = [i for i in indices if i < q33]
        medium = [i for i in indices if q33 <= i < q67]
        dense = [i for i in indices if i >= q67]

        rng = __import__("random").Random(args.seed)
        sample_per_stratum = 5
        sampled_sparse = (
            sorted(rng.sample(sparse, min(sample_per_stratum, len(sparse))))
            if sparse
            else []
        )
        sampled_medium = (
            sorted(rng.sample(medium, min(sample_per_stratum, len(medium))))
            if medium
            else []
        )
        sampled_dense = (
            sorted(rng.sample(dense, min(sample_per_stratum, len(dense))))
            if dense
            else []
        )

        plan["models"][model_name] = {
            "total_networks": n,
            "quantile_boundaries": {"q33": q33, "q67": q67},
            "sampled": {
                "sparse": sampled_sparse,
                "medium": sampled_medium,
                "dense": sampled_dense,
            },
            "all_sampled": sampled_sparse + sampled_medium + sampled_dense,
        }
        total_networks += len(sampled_sparse) + len(sampled_medium) + len(sampled_dense)
        print(
            f"  {model_name}: {n} networks → "
            f"sparse={sampled_sparse}, medium={sampled_medium}, "
            f"dense={sampled_dense}"
        )

    plan["total_networks_sampled"] = total_networks
    plan["seed"] = args.seed
    plan["rationale"] = (
        "Pre-registered stratified sample: 5 networks per stratum (sparse/medium/"
        "dense defined by alpha-index tertiles) per diffusion model.  Networks "
        "were selected before evaluation to avoid cherry-picking."
    )

    out_path = dp.BASE / "preregistered_network_sample.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(plan, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    print(f"\nPre-registered sample plan saved → {out_path}")
    print(f"Total networks to evaluate: {total_networks}")
