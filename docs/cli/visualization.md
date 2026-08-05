# Visualization CLI

Generate PNG figures from saved pipeline / experiment artefacts. Narrative plot
catalogue: [visualization.md](../visualization.md).

Default save locations are under `plots/<dataset>/` (via `DatasetPaths`).

---

## Model plots — `python -m visualization.model_plots`

CMF evaluation plots (α–RMSE, hyperparameter search, ranking, …).

```bash
python -m visualization.model_plots
python -m visualization.model_plots --plot alpha-rmse delta-rmse --models exponential
python -m visualization.model_plots --plot ranking --dataset ciao --no-save
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--plot` | one or more plot IDs | `all` | Which plots to generate. |
| `--models` | model names and/or `all` | `all` | Models for α–RMSE / Δ–RMSE style plots. |
| `--no-save` | flag | off | Show interactively without writing files. |
| `--dataset` | string | `Datasets.DEFAULT` (`movielens`) | Dataset whose result files to load. |

### `--plot` choices

| ID | Content |
| --- | --- |
| `alpha-rmse` | RMSE vs α |
| `delta-rmse` | Δ-RMSE vs α |
| `alpha-edges` | Edge count vs α |
| `hyperparam` | Hyperparameter search overview |
| `heatmap` | Parameter-space heatmap (RMSE) |
| `metrics` | Metrics distribution comparison |
| `convergence` | Optuna / search convergence |
| `ranking` | Ranking metrics per α |
| `ranking-comparison` | Ranking comparison across models |
| `all` | Everything above |

Hyperparam / heatmap / metrics / convergence need baseline search JSON
(`data/<ds>/…` baseline results). If missing, those plots warn and skip —
run `python -m recommender.baseline` or the pipeline recommend/hypertune path
first.

---

## Social regularization plots — `python -m visualization.model_plots.social_regularization`

Phase 6 λ-sweep and network-sweep figures.

```bash
python -m visualization.model_plots.social_regularization --plot-kind all --dataset movielens
python -m visualization.model_plots.social_regularization --plot-kind lambda --results-dir path/to/lambda_sweep
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | dataset choices | `movielens` | Dataset. |
| `--plot-kind` | `lambda` \| `network` \| `all` | `lambda` | Which Phase 6 plot family. |
| `--results-dir` | path | kind-specific under `data/<ds>/social_smoke_results/` | Results directory for the selected kind. |
| `--output-dir` | path | `plots/<ds>/models/social_regularization` | Where PNGs are written. |

Default results dirs:

- `lambda` → `data/<ds>/social_smoke_results/lambda_sweep`
- `network` → `data/<ds>/social_smoke_results/network_sweep`

Produce those artefacts with the tools in [recommender.md](recommender.md).

---

## Network plots — `python -m visualization.network_plots`

Subcommands are **required**.

### `cascades`

```bash
python -m visualization.network_plots cascades
python -m visualization.network_plots cascades --n 50 --no-save
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--n` | int | `30` | Number of cascades to show on the timeline. |
| `--no-save` | flag | off | Show without saving. |

### `centrality`

```bash
python -m visualization.network_plots centrality --model exponential --network 000
python -m visualization.network_plots centrality --model powerlaw --metric pagerank
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | diffusion model | **required** | Which model's centrality files to load. |
| `--network` | string | `"000"` | Network index string (matches inferred-network suffix). |
| `--metric` | metric name or `all` | `all` | Single metric distribution, or full grid. |
| `--no-save` | flag | off | Show without saving. |

Metric choices: `degree`, `betweenness`, `closeness`, `eigenvector`,
`pagerank`, `clustering`, `eccentricity`, `all`.

---

## SHAP plots — `python -m visualization.shap_plots`

Runs `plot_all_shap()` (importance + beeswarm for all models). **No CLI flags.**

```bash
python -m visualization.shap_plots
```

Requires SHAP artefacts from `python -m pipeline --steps shap` (or the analysis
API). See [usage.md](../usage.md) and the SHAP sections of the pipeline docs.
