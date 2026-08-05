# Pipeline CLI

Unified entry point for the MAFPIN research pipeline.

```bash
python -m pipeline [OPTIONS]
# equivalent:
python pipeline.py [OPTIONS]
```

Parser: `pipeline/_cli.py`. Step runners: `pipeline/steps/`.

For a narrative walkthrough (what each step writes and when to chain them), see
[usage.md](../usage.md). This page is the full flag reference.

---

## Invocation patterns

```bash
# Every registered step in order (includes Phase-2 experiment steps)
python -m pipeline --all

# Selected steps, in the given order
python -m pipeline --steps cascade inference centrality communities recommend

# Single diffusion model
python -m pipeline --steps inference recommend --model exponential

# Phase 6 social path
python -m pipeline --steps recommend shap --social-regularization --dataset movielens

# Core experiment Phase 2 only
python -m pipeline --steps import_manifest canonical_baseline network_selection final_eval \
  --dataset movielens --all-variants
```

`--all` and `--steps` are mutually exclusive. Either one of them, or
`--preregister-networks` alone, is required to actually run work.

Long runs tee stdout/stderr to `data/<dataset>/pipeline.log` unless
`--no-log` or a custom `--log-file` is set. Monitor with:

```bash
tail -f data/movielens/pipeline.log
```

---

## Steps

| Step | Description | Typical inputs |
| --- | --- | --- |
| `cascade` | Generate diffusion cascades from ratings | `datasets/<name>/` |
| `delta` | Compute median inter-event Δ | `data/<ds>/cascades.txt` |
| `inference` | Infer networks with NetInf | cascades + α grid |
| `communities` | Overlapping communities + LPH | inferred networks |
| `centrality` | SNAP centrality metrics | inferred networks |
| `recommend` | Train / evaluate CMF (baseline + enhanced / social) | networks + features |
| `hypertune` | Optuna search for enhanced (or social) CMF | features |
| `shap` | TreeSHAP on GBT surrogate of CMF outputs | tuned params + networks |
| `import_manifest` | Import HPs from recommend logs into experiment manifest | recommend logs |
| `canonical_baseline` | Freeze canonical M1 baseline for final test | manifest |
| `network_selection` | Freeze best diffusion model + α per variant | recommend results |
| `final_eval` | Evaluate variants on global held-out test | frozen networks / HPs |

`--all` runs **every** key in `STEPS` (including `import_manifest` …
`final_eval`). For exploratory network → recommend work without Phase 2, pass
an explicit `--steps` list instead.

---

## Step selection

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--all` | flag | off | Run every registered step in order (including Phase 2). Mutually exclusive with `--steps`. |
| `--steps STEP …` | one or more step names | — | Run the listed steps in the given order. |

---

## Dataset, model, and NetInf grid

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | `movielens` \| `ciao` \| `epinions` | `movielens` | Dataset; raw files under `datasets/<name>/`, artefacts under `data/<name>/`. |
| `--model` | `exponential` \| `powerlaw` \| `rayleigh` | all models | Restrict inference / recommendation to one diffusion model. |
| `--n-alphas` | int | `100` | Number of α values in the NetInf log-spaced grid. |
| `--max-iter` | int | `5000` | Fallback edge budget *k* when average-degree scaling is disabled. |
| `--k-avg-degree` | float | `2` | Sets `k = avg_degree × N` edges per network. Pass `0` to disable and use `--max-iter` only. Paper default is `2`. |

---

## Recommender / CMF

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--no-communities` | flag | communities **included** | Exclude community membership / LPH features from enhanced CMF (variant M2). |
| `--cmf-method` | `lbfgs` \| `als` | `lbfgs` | CMF optimizer. Social regularization requires `lbfgs`. |
| `--cmf-maxiter` | int | `25` | L-BFGS iteration budget per fit. |
| `--cpu-fraction` | float | `0.4` | Fraction of detected CPU cores for CMF/BLAS when `--cmf-nthreads` is `0`. |
| `--cmf-nthreads` | int | `0` | Explicit CMF/BLAS thread cap. `0` → derive from `--cpu-fraction`. Keep low for L-BFGS. |
| `--sample-networks` | int | `5` | Networks sampled per diffusion model in `recommend`. |
| `--n-jobs` | int | `1` | Parallel workers for recommend network evaluation. `1` = sequential; `-1` = CPU cap from `--cpu-fraction`. |
| `--seed` | int | `42` | Random seed (also used for SHAP network sampling). |

**Constraint:** `--social-regularization` with `--cmf-method als` is rejected by the parser.

---

## Logging

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--log-file` | path | `data/<dataset>/pipeline.log` | Tee stdout/stderr to this path. Use distinct files per experiment run. |
| `--no-log` | flag | off | Disable the tee log. |

---

## Phase 6 social regularization

Enable with `--social-regularization`. Affects `recommend`, `hypertune`, and `shap`.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--social-regularization` | flag | off | Use social-regularized CMF. |
| `--social-mode` | see below | `boundary_downweight` | Edge weighting mode. |
| `--lambda-social` | float | `0.001` | Fallback social strength when no searched params are loaded. |
| `--social-beta` | float | `0.5` | Boundary penalty for edge weighting. |
| `--social-gamma` | float | `1.0` | Shared-community gain (esp. `bridge_preserve`). |
| `--social-normalization` | see below | `mean_weight` | How social edge weights are scaled. |
| `--social-search-max-ratings` | int | `5000` | Rating cap during social Optuna search; `0` disables the cap. |
| `--social-n-trials` | int | `200` | Optuna trial budget for the social search space. |

### `--social-mode` values

| Mode | Intent |
| --- | --- |
| `uniform` | Equal weight on social edges (M4a) |
| `community_jaccard` | Weight by community overlap (M4b) |
| `boundary_downweight` | Down-weight boundary-spanning edges (M4c) |
| `bridge_preserve` | Preserve bridge edges via γ (M4d) |

### `--social-normalization` values

`none`, `mean`, `mean_weight`, `edges`, `n_edges`, `sum_weight`, `normalized_laplacian`.

Aliases: `mean` → `mean_weight`, `edges` → `n_edges` (handled in the social regularization code path).

See also [social_regularization.md](../social_regularization.md) and the standalone tools in [recommender.md](recommender.md).

---

## SHAP

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--k-networks` | int | `20` | Networks sampled per diffusion model for SHAP. |
| `--all-networks` | flag | off | Use every available network (overrides `--k-networks`). |
| `--seed` | int | `42` | Sampling seed. |

---

## Core experiment / Route B flags

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--preregister-networks` | flag | off | Write stratified network sample JSON under `data/<dataset>/preregistered_network_sample.json`. |
| `--run-id` | string | — | After `recommend`, archive artefacts under `data/<dataset>/runs/<run-id>/`. |
| `--model-variant` | variant ID | — | Single variant for `import_manifest` / `network_selection` / `final_eval`. |
| `--all-variants` | flag | off | Process all registered variants (`ALL_VARIANT_IDS`). |
| `--beyond-accuracy` | flag | off | Route B WP1: also compute CCE / ILD / novelty / coverage on `final_eval`. |
| `--save-predictions` | flag | off | Route B WP2: write per-rating predictions under `data/<ds>/route_b/predictions/`. |
| `--force` | flag | off | Re-run `canonical_baseline` even if it already exists. |

Variant IDs: `M1`, `M2`, `M3`, `M4a`, `M4b`, `M4c`, `M4d`, `M4c_robustness`, `M2_trust`, `M3_trust`, `M3_soft`.

Mapping of variants → CLI flags for recommend/hypertune is in
[core_experiment_commands.md](../experiments/core_experiment_commands.md).
Batch automation: [experiments.md](experiments.md#run_core_experimentsh).

---

## Examples by goal

**Networks only (MovieLens):**

```bash
python -m pipeline --steps cascade delta inference communities centrality --dataset movielens
```

**Recommend with fewer sampled networks, parallel workers:**

```bash
python -m pipeline --steps recommend \
  --dataset ciao \
  --sample-networks 10 \
  --n-jobs -1 \
  --cpu-fraction 0.4 \
  --log-file data/ciao/logs/m3_recommend.log
```

**M4c social recommend:**

```bash
python -m pipeline --steps recommend \
  --social-regularization \
  --social-mode boundary_downweight \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --cmf-method lbfgs \
  --dataset movielens
```

**Final eval with beyond-accuracy metrics:**

```bash
python -m pipeline --steps final_eval \
  --dataset movielens \
  --all-variants \
  --beyond-accuracy
```
