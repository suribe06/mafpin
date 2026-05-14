"""Step 3 network sweep for Phase 6 social regularization."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import DatasetPaths, Datasets, Defaults, Models
from recommender.enhanced.social_regularization import SocialMode
from recommender.enhanced.social_smoke_test import run_social_smoke_test


def _lambda_slug(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def available_network_indices(dataset: str, model_name: str) -> list[int]:
    """Return indices that have network, centrality, and community artifacts."""
    paths = DatasetPaths(dataset)
    short = Models.SHORT[model_name]
    patterns = {
        "networks": (
            paths.NETWORKS / model_name,
            re.compile(rf"^inferred-network-{short}-(\d{{3}})\.txt$"),
        ),
        "centrality": (
            paths.CENTRALITY / model_name,
            re.compile(rf"^centrality_metrics_{model_name}_(\d{{3}})\.csv$"),
        ),
        "communities": (
            paths.COMMUNITIES / model_name,
            re.compile(rf"^communities_{model_name}_(\d{{3}})\.csv$"),
        ),
    }

    available_sets: list[set[int]] = []
    for directory, pattern in patterns.values():
        if not directory.exists():
            return []
        indices = {
            int(match.group(1))
            for path in directory.iterdir()
            if (match := pattern.match(path.name))
        }
        available_sets.append(indices)

    return sorted(set.intersection(*available_sets)) if available_sets else []


def sample_network_indices(
    dataset: str,
    model_names: Iterable[str],
    n_networks: int,
    random_state: int,
) -> dict[str, list[int]]:
    """Sample valid network indices for each diffusion model."""
    rng = np.random.default_rng(random_state)
    selected: dict[str, list[int]] = {}
    for model_name in model_names:
        available = available_network_indices(dataset, model_name)
        if len(available) < n_networks:
            raise ValueError(
                f"Only {len(available)} complete networks are available for "
                f"{dataset}/{model_name}; need {n_networks}."
            )
        sampled = rng.choice(available, size=n_networks, replace=False)
        selected[model_name] = sorted(int(index) for index in sampled)
    return selected


def _result_row(
    path: Path, result: dict, status: str, error: str | None = None
) -> dict:
    social_metrics = result.get("lambda_social_on", {})
    baseline_metrics = result.get("lambda_social_0", {})
    diagnostics = result.get("diagnostics", {})
    social_edges = result.get("social_edges", {})
    return {
        "status": status,
        "dataset": result.get("dataset"),
        "model_name": result.get("model_name"),
        "network_index": result.get("network_index"),
        "social_mode": result.get("social_mode"),
        "lambda_social": result.get("lambda_social"),
        "rmse": social_metrics.get("rmse"),
        "mae": social_metrics.get("mae"),
        "r2": social_metrics.get("r2"),
        "baseline_rmse": baseline_metrics.get("rmse"),
        "baseline_mae": baseline_metrics.get("mae"),
        "baseline_r2": baseline_metrics.get("r2"),
        "rmse_delta": result.get("rmse_delta"),
        "edges": social_edges.get("n_edges"),
        "min_weight": social_edges.get("min_weight"),
        "max_weight": social_edges.get("max_weight"),
        "baseline_reasonable": diagnostics.get("lambda_social_0_reasonable_scale"),
        "social_regularized_reasonable": diagnostics.get(
            "lambda_social_on_reasonable_scale"
        ),
        "path": str(path),
        "error": error,
    }


def run_social_network_sweep(
    dataset: str = Datasets.DEFAULT,
    model_names: list[str] | None = None,
    n_networks: int = 10,
    social_mode: SocialMode = "boundary_downweight",
    lambda_social: float = 0.001,
    beta: float = 0.5,
    gamma: float = 1.0,
    max_ratings: int = 5000,
    test_size: float = 0.2,
    k: int = 8,
    lambda_reg: float = 10.0,
    w_main: float = Defaults.W_MAIN,
    w_user: float = Defaults.W_USER,
    transform: str = "standard",
    maxiter: int = 20,
    random_state: int = 42,
    nthreads: int = 1,
    include_user_attributes: bool = False,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run Step 3 across sampled networks and save JSON plus CSV outputs."""
    selected_models = model_names or Models.ALL
    for model_name in selected_models:
        if model_name not in Models.ALL:
            raise ValueError(f"Unknown model {model_name!r}. Choose from {Models.ALL}.")

    base_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else DatasetPaths(dataset).BASE / "social_smoke_results" / "network_sweep"
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    selected = sample_network_indices(
        dataset=dataset,
        model_names=selected_models,
        n_networks=n_networks,
        random_state=random_state,
    )
    (base_output_dir / "selected_network_indices.json").write_text(
        json.dumps(selected, indent=2),
        encoding="utf-8",
    )

    rows: list[dict] = []
    lambda_part = _lambda_slug(lambda_social)
    for model_name, indices in selected.items():
        for network_index in indices:
            output_path = (
                base_output_dir
                / f"{model_name}_{network_index:03d}_{social_mode}_lambda_{lambda_part}.json"
            )
            try:
                if output_path.exists() and not overwrite:
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                else:
                    result = run_social_smoke_test(
                        dataset=dataset,
                        model_name=model_name,
                        network_index=network_index,
                        social_mode=social_mode,
                        lambda_social=lambda_social,
                        beta=beta,
                        gamma=gamma,
                        max_ratings=max_ratings,
                        test_size=test_size,
                        k=k,
                        lambda_reg=lambda_reg,
                        w_main=w_main,
                        w_user=w_user,
                        transform=transform,
                        maxiter=maxiter,
                        random_state=random_state,
                        nthreads=nthreads,
                        include_user_attributes=include_user_attributes,
                        output_path=output_path,
                    )
                rows.append(_result_row(output_path, result, status="ok"))
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                rows.append(
                    _result_row(
                        output_path,
                        {
                            "dataset": dataset,
                            "model_name": model_name,
                            "network_index": network_index,
                            "social_mode": social_mode,
                            "lambda_social": lambda_social,
                        },
                        status="error",
                        error=str(exc),
                    )
                )

    summary = pd.DataFrame(rows).sort_values(["model_name", "network_index"])
    summary.to_csv(base_output_dir / "network_sweep_summary.csv", index=False)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 6 social regularization over sampled networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=Datasets.DEFAULT, choices=Datasets.ALL)
    parser.add_argument("--models", nargs="+", default=Models.ALL, choices=Models.ALL)
    parser.add_argument("--n-networks", type=int, default=10)
    parser.add_argument(
        "--social-mode",
        default="boundary_downweight",
        choices=[
            "uniform",
            "community_jaccard",
            "boundary_downweight",
            "bridge_preserve",
        ],
    )
    parser.add_argument("--lambda-social", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max-ratings", type=int, default=5000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambda-reg", type=float, default=10.0)
    parser.add_argument("--w-user", type=float, default=Defaults.W_USER)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-user-attributes",
        action="store_true",
        help="Also pass enhanced side-user features as U; off by default for smoke stability.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = run_social_network_sweep(
        dataset=args.dataset,
        model_names=args.models,
        n_networks=args.n_networks,
        social_mode=args.social_mode,
        lambda_social=args.lambda_social,
        beta=args.beta,
        gamma=args.gamma,
        max_ratings=args.max_ratings,
        k=args.k,
        lambda_reg=args.lambda_reg,
        w_user=args.w_user,
        maxiter=args.maxiter,
        random_state=args.random_state,
        nthreads=args.nthreads,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        include_user_attributes=args.include_user_attributes,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else DatasetPaths(args.dataset).BASE / "social_smoke_results" / "network_sweep"
    )
    ok_count = int((summary["status"] == "ok").sum())
    print(f"Completed {ok_count}/{len(summary)} Step 3 network-sweep runs")
    print(f"Saved selected indices to {output_dir / 'selected_network_indices.json'}")
    print(f"Saved summary to {output_dir / 'network_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
