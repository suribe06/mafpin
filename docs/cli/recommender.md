# Recommender CLI

Standalone evaluators and Phase 6 social-regularization tools. Day-to-day
training/evaluation usually goes through [pipeline.md](pipeline.md)
(`recommend`, `hypertune`, `shap`). Use these modules for focused experiments
or debugging.

Background: [hyperparameter_tuning.md](../hyperparameter_tuning.md),
[social_regularization.md](../social_regularization.md).

---

## Enhanced CMF — `python -m recommender.enhanced`

Evaluate enhanced CMF with network side-information on sampled networks.

```bash
python -m recommender.enhanced --sample-networks 5
python -m recommender.enhanced --all --no-communities --transform standard
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--all` | flag | off | Evaluate every available network for all models (sets a large sample count). |
| `--sample-networks N` | int | `5` | Random networks per diffusion model (ignored when `--all`). |
| `--transform` | `standard` \| `minmax` \| `normalizer` | `standard` | Feature normalisation. |
| `--no-communities` | flag | off | Exclude LPH / `num_communities` features. |
| `--n-splits` | int | `5` | Cross-validation splits per network. |

Uses the default dataset load/split from `recommender.data` (MovieLens unless
configured otherwise). Pipeline `recommend` is the fuller multi-dataset path.

---

## Baseline — `python -m recommender.baseline`

Runs the complete baseline Optuna search + CV + final train
(`run_complete_example()`). **No CLI flags.**

```bash
python -m recommender.baseline
```

---

## Social search — `python -m recommender.enhanced.social_search`

Optuna search over Phase 6 social-regularized CMF hyperparameters (and optional
side-user attributes).

```bash
python -m recommender.enhanced.social_search \
  --dataset movielens \
  --model exponential \
  --network-index 0 \
  --n-trials 200 \
  --social-modes uniform community_jaccard boundary_downweight bridge_preserve
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | dataset choices | `movielens` | Dataset. |
| `--model` | diffusion model | `exponential` | Which model's network to load. |
| `--network-index` | int | `0` | Index into that model's inferred networks. |
| `--n-trials` | int | `200` | Optuna trials. |
| `--timeout` | int | `None` | Optional Optuna timeout (seconds). |
| `--max-ratings` | int | `5000` | Cap ratings used during search. |
| `--test-size` | float | `0.2` | Hold-out fraction for search evaluation. |
| `--maxiter` | int | `25` | L-BFGS iterations. |
| `--random-state` | int | `42` | RNG seed. |
| `--nthreads` | int | `1` | CMF/BLAS threads (keep at 1 for L-BFGS). |
| `--transform` | string | `standard` | Feature transform name. |
| `--social-normalization` | normalization choices | `mean_weight` | Social edge normalization. |
| `--social-modes` | one or more modes | all four modes | Modes to search over. |
| `--k-min` / `--k-max` | int | `5` / `50` | Latent factor search bounds. |
| `--lambda-reg-min` / `--lambda-reg-max` | float | `0.01` / `10.0` | Regularization bounds. |
| `--w-main-min` / `--w-main-max` | float | `0.1` / `1.0` | Main weight bounds. |
| `--w-user-min` / `--w-user-max` | float | `0.01` / `1.0` | User-attribute weight bounds. |
| `--lambda-social-min` / `--lambda-social-max` | float | `1e-4` / `1.0` | Social λ bounds. |
| `--beta-min` / `--beta-max` | float | `0.0` / `1.0` | β bounds. |
| `--gamma-min` / `--gamma-max` | float | `0.1` / `3.0` | γ bounds. |
| `--output-path` | path | auto | Where to write search results JSON. |
| `--no-user-attributes` | flag | off | Search without side-user matrix `U`. |

Normalization choices: `none`, `mean`, `mean_weight`, `edges`, `n_edges`,
`sum_weight`, `normalized_laplacian`.

Social modes: `uniform`, `community_jaccard`, `boundary_downweight`,
`bridge_preserve`.

Prints best params JSON to stdout when trials succeed.

---

## Social smoke test — `python -m recommender.enhanced.social_smoke_test`

Small fixed-hyperparameter smoke run (or Step-4 `λ_reg × w_user` grid).

```bash
python -m recommender.enhanced.social_smoke_test \
  --dataset movielens \
  --social-mode boundary_downweight \
  --maxiter 5

python -m recommender.enhanced.social_smoke_test \
  --user-attribute-grid \
  --lambda-reg-grid 1.0 3.0 10.0 \
  --w-user-grid 0.01 0.05 0.1
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | dataset choices | `movielens` | Dataset. |
| `--model` | diffusion model | `exponential` | Network model. |
| `--network-index` | int | `0` | Network index. |
| `--social-mode` | mode choices | `boundary_downweight` | Edge weighting mode. |
| `--social-normalization` | normalization choices | `mean_weight` | Edge normalization. |
| `--lambda-social` | float | `0.01` | Social strength. |
| `--beta` | float | `0.5` | Boundary penalty. |
| `--gamma` | float | `1.0` | Shared-community gain. |
| `--max-ratings` | int | `5000` | Rating subsample cap. |
| `--k` | int | `8` | Latent factors. |
| `--lambda-reg` | float | `1.0` | Regularization (single-run mode). |
| `--lambda-reg-grid` | float+ | `1.0 3.0 10.0` | Grid for `--user-attribute-grid`. |
| `--w-user` | float | config default | User-attribute weight (single-run). |
| `--w-user-grid` | float+ | `0.01 0.05 0.1` | Grid for `--user-attribute-grid`. |
| `--maxiter` | int | `5` | L-BFGS iterations (smoke-sized). |
| `--nthreads` | int | `1` | Threads. |
| `--output-path` | path | `data/<ds>/social_smoke_results.json` | JSON for single-run mode. |
| `--output-dir` | path | — | Directory for grid summaries / JSON. |
| `--overwrite` | flag | off | Overwrite existing outputs. |
| `--include-user-attributes` | flag | off | Pass enhanced `U` features (off by default for stability). |
| `--user-attribute-grid` | flag | off | Run Step 4 over `λ_reg × w_user` with side-user attributes on. |

Report: [reports/social_smoke_test.md](../reports/social_smoke_test.md).

---

## Social network sweep — `python -m recommender.enhanced.social_network_sweep`

Apply fixed social hyperparameters across sampled networks.

```bash
python -m recommender.enhanced.social_network_sweep \
  --dataset movielens \
  --models exponential powerlaw rayleigh \
  --n-networks 10 \
  --social-mode boundary_downweight
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | dataset choices | `movielens` | Dataset. |
| `--models` | one or more models | all three | Diffusion models to sweep. |
| `--n-networks` | int | `10` | Networks sampled per model. |
| `--social-mode` | mode choices | `boundary_downweight` | Edge weighting. |
| `--lambda-social` | float | `0.001` | Social strength. |
| `--beta` / `--gamma` | float | `0.5` / `1.0` | Social weighting params. |
| `--max-ratings` | int | `5000` | Rating cap. |
| `--k` | int | `8` | Latent factors. |
| `--lambda-reg` | float | `10.0` | Regularization. |
| `--w-user` | float | config default | User-attribute weight. |
| `--maxiter` | int | `20` | L-BFGS iterations. |
| `--random-state` | int | `42` | Sampling / split seed. |
| `--nthreads` | int | `1` | Threads. |
| `--output-dir` | path | `data/<ds>/social_smoke_results/network_sweep` | Output directory. |
| `--overwrite` | flag | off | Overwrite existing files. |
| `--include-user-attributes` | flag | off | Pass enhanced `U` features. |

Writes `selected_network_indices.json` and `network_sweep_summary.csv` under
the output directory.

---

## Best params eval — `python -m recommender.enhanced.social_best_params_eval`

Compare Optuna-best social CMF against plain baseline CMF.

```bash
python -m recommender.enhanced.social_best_params_eval \
  --search-result-path path/to/social_search.json \
  --output-path path/to/comparison.json \
  --report-path path/to/report.md
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--search-result-path` | path | module default | Optuna search result JSON. |
| `--output-path` | path | module default | Comparison JSON output. |
| `--report-path` | path | module default | Markdown report path. |
| `--maxiter` | int | from search / default | Override L-BFGS iterations. |
| `--nthreads` | int | from search / default | Override thread count. |
| `--social-retries` | int | `5` | Retries on social fit failures. |

Prints the comparison block as JSON to stdout.

---

## Related pipeline flags

For production-style runs prefer:

```bash
python -m pipeline --steps hypertune recommend shap --social-regularization …
```

See [pipeline.md](pipeline.md#phase-6-social-regularization) for the shared
social flags (`--social-mode`, `--social-normalization`, `--social-n-trials`, …).
