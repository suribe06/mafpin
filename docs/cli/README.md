# CLI Reference

MAFPIN exposes several command-line entry points. There are no setuptools
`console_scripts`; run everything from the repository root with the project
Python environment (typically the `mafpin` conda env).

```bash
cd /path/to/mafpin
conda run --no-capture-output -n mafpin python -m <module> --help
```

Use `--no-capture-output` with `conda run` so progress streams to the terminal
instead of buffering until exit.

---

## Documents in this folder

| Document | Covers |
| --- | --- |
| [pipeline.md](pipeline.md) | Unified pipeline (`python -m pipeline` / `pipeline.py`) |
| [networks.md](networks.md) | Cascades, delta, NetInf, communities, centrality |
| [recommender.md](recommender.md) | Enhanced CMF eval and Phase 6 social tooling |
| [experiments.md](experiments.md) | Core experiment batch, cold-start, Route B |
| [visualization.md](visualization.md) | Model, network, social, and SHAP plots |

Narrative walkthroughs (when to run what) live elsewhere:

- [Usage guide](../usage.md) — step-by-step pipeline tutorial
- [Core experiment commands](../experiments/core_experiment_commands.md)
- [Cold-start commands](../experiments/cold_start_commands.md)
- [Route B commands](../experiments/route_b_commands.md)
- [Visualization guide](../visualization.md)

---

## Quick map of entry points

### Preferred: unified pipeline

```bash
python -m pipeline --all
# equivalent:
python pipeline.py --all
```

Most day-to-day work goes through this CLI. See [pipeline.md](pipeline.md).

### Standalone modules (`python -m …`)

| Invocation | Role |
| --- | --- |
| `python -m networks.cascades` | Cascade generation |
| `python -m networks.delta` | Median Δ and α centres |
| `python -m networks.inference` | NetInf batch / single model |
| `python -m networks.communities` | Demon / ASLPAw + LPH |
| `python -m networks.centrality` | SNAP centrality metrics |
| `python -m recommender.enhanced` | Enhanced CMF network evaluation |
| `python -m recommender.baseline` | Baseline Optuna search (no flags) |
| `python -m recommender.enhanced.social_search` | Phase 6 Optuna search |
| `python -m recommender.enhanced.social_smoke_test` | Phase 6 smoke / grids |
| `python -m recommender.enhanced.social_network_sweep` | Phase 6 network sweep |
| `python -m recommender.enhanced.social_best_params_eval` | Best social params vs baseline |
| `python -m recommender.experiment.cold_start` | Cold-start experiment modes |
| `python -m recommender.experiment.route_b.boundary_strata` | Route B WP2 strata |
| `python -m visualization.model_plots` | CMF evaluation plots |
| `python -m visualization.model_plots.social_regularization` | Phase 6 plots |
| `python -m visualization.network_plots <cmd>` | Cascade / centrality plots |
| `python -m visualization.shap_plots` | SHAP plots (no flags) |

### Scripts

| Invocation | Role |
| --- | --- |
| `./scripts/run_core_experiment.sh` | Full core-experiment ladder |
| `python scripts/route_b_wp3_community_stability.py` | Route B WP3 stability |
| `python scripts/parse_experiment_logs.py` | Parse logs → JSON (no flags) |

---

## Shared conventions

### Datasets

| Value | Raw data |
| --- | --- |
| `movielens` (default in most CLIs) | `datasets/movielens/` |
| `ciao` | `datasets/ciao/` |
| `epinions` | `datasets/epinions/` |

Generated artefacts land under `data/<dataset>/`. Keep `--dataset` consistent
across steps in the same experiment so cascade IDs and recommender encodings
stay aligned.

### Diffusion models

| Value | Meaning |
| --- | --- |
| `exponential` | Exponential transmission model |
| `powerlaw` | Power-law transmission model |
| `rayleigh` | Rayleigh transmission model |

### Core experiment variants

Used by `--model-variant` / `--all-variants` on the pipeline and by experiment
CLIs:

| ID | Meaning (short) |
| --- | --- |
| `M1` | Baseline CMF (no network features) |
| `M2` | Enhanced CMF, no communities |
| `M3` | Enhanced CMF with communities / LPH |
| `M4a`–`M4d` | Social regularization modes (`uniform` … `bridge_preserve`) |
| `M4c_robustness` | M4c with `normalized_laplacian` |
| `M2_trust` / `M3_trust` | Trust-network features (Ciao / Epinions) |
| `M3_soft` | Soft community assignment |

### Threading note

L-BFGS in local `cmfrec` should run with **one BLAS thread per fit** (higher
values can segfault). Prefer `--n-jobs` to parallelize across networks, not a
large `--cmf-nthreads`.

### Discovering flags at runtime

Every argparse CLI supports `--help`. Defaults in this documentation match the
current parsers; if they diverge, trust `--help` and open an issue / PR against
these docs.
