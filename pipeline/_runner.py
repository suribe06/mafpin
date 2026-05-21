"""Step registry and main() entry point."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

from pipeline._cpu import _configure_cpu_limits
from pipeline._logging import _TeeStream, _open_pipeline_log
from pipeline.steps.cascade import run_cascade
from pipeline.steps.centrality import run_centrality
from pipeline.steps.communities import run_communities
from pipeline.steps.delta import run_delta
from pipeline.steps.hypertune import run_hypertune
from pipeline.steps.inference import run_inference
from pipeline.steps.preregister import run_preregister
from pipeline.steps.recommend import run_recommend
from pipeline.steps.shap import run_shap

STEPS: dict[str, tuple[str, Any]] = {
    "cascade": ("Generate diffusion cascades from ratings", run_cascade),
    "delta": ("Compute median inter-event delta", run_delta),
    "inference": ("Infer diffusion networks (NetInf)", run_inference),
    "communities": ("Detect overlapping communities + LPH", run_communities),
    "centrality": ("Compute SNAP centrality metrics", run_centrality),
    "recommend": ("Train and evaluate CMF recommender", run_recommend),
    "hypertune": ("Optuna search for enhanced CMF hyperparameters", run_hypertune),
    "shap": ("SHAP feature importance for enhanced CMF", run_shap),
}

ALL_STEPS: list[str] = list(STEPS.keys())


def main(argv: list[str] | None = None) -> None:
    """Entry point for the MAFPIN pipeline."""
    from config import Defaults
    from pipeline._cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(argv)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = _open_pipeline_log(args)
    if log_file is not None:
        sys.stdout = _TeeStream(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _TeeStream(original_stderr, log_file)  # type: ignore[assignment]

    # ALS does not support social regularization (no weighted-edge objective).
    if (
        getattr(args, "social_regularization", False)
        and getattr(args, "cmf_method", Defaults.CMF_METHOD) == "als"
    ):
        parser.error(
            "--social-regularization requires L-BFGS; ALS cannot incorporate "
            "weighted social edges. Use --cmf-method lbfgs (the default)."
        )

    try:
        print(
            f"\n=== Pipeline run started "
            f"{datetime.now().isoformat(timespec='seconds')} ===",
            flush=True,
        )
        print(f"Command: python pipeline.py {' '.join(sys.argv[1:])}", flush=True)
        _configure_cpu_limits(args)

        # --preregister-networks runs independently before any pipeline step.
        if getattr(args, "preregister_networks", False):
            print(
                "\n[PREREGISTER] Building pre-registered network sample …",
                flush=True,
            )
            run_preregister(args)
            print("[PREREGISTER] Done.", flush=True)

        # Support running --preregister-networks alone (without --steps/--all).
        if not getattr(args, "all", False) and not getattr(args, "steps", None):
            if not getattr(args, "preregister_networks", False):
                parser.error(
                    "one of --all, --steps, or --preregister-networks is required."
                )
            # Pre-registration-only run: nothing else to do.
            return

        steps = ALL_STEPS if args.all else args.steps

        print(f"Running steps: {', '.join(steps)}", flush=True)
        print("-" * 50, flush=True)

        for step in steps:
            description, runner = STEPS[step]
            print(f"\n[{step.upper()}] {description}", flush=True)
            print("=" * 50, flush=True)
            runner(args)
            print(f"[{step.upper()}] Done.", flush=True)

        print("\nPipeline finished.", flush=True)
    finally:
        print(
            f"=== Pipeline run ended "
            f"{datetime.now().isoformat(timespec='seconds')} ===\n",
            flush=True,
        )
        if log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()
