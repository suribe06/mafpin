# Usage Guide

This guide walks through each step of the MAFPIN pipeline using `pipeline.py`.

For the full flag reference across **all** entry points (pipeline, networks,
recommender, experiments, visualization), see [cli/README.md](cli/README.md).
Pipeline-only options: [cli/pipeline.md](cli/pipeline.md).

---

## Quick Start — Full Pipeline

Run all steps in order using the default **MovieLens** dataset:

```bash
python pipeline.py --all
```

Run the full pipeline with Phase 6 social regularization enabled:

```bash
python pipeline.py --all --social-regularization
```

When running through conda, use `--no-capture-output` so progress is streamed
to the terminal instead of buffered until the process exits:

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
   --steps recommend shap \
   --dataset movielens \
   --social-regularization
```

Every pipeline run also appends stdout and stderr to
`data/<dataset>/pipeline.log` by default, so long runs can be monitored with
`tail -f data/movielens/pipeline.log`.

This uses the L-BFGS CMF solver by default so the baseline, enhanced model,
social-regularized model, and SHAP analysis are trained with the optimizer
required by the social graph penalty.

To use a different dataset:

```bash
python pipeline.py --all --dataset ciao
python pipeline.py --all --dataset epinions
```

Available datasets: `movielens` (default), `ciao`, `epinions`.  
Raw files are read from the `datasets/<name>/` directory.

---

## Step-by-step

### 1. Generate Cascades

Convert the ratings dataset into a cascades file for NetInf:

```bash
python pipeline.py --steps cascade
```

This reads from `datasets/movielens/` by default, applies the **global 80/20 split** (seed from `config.Split.RANDOM_STATE`), and writes `data/<dataset>/cascades.txt` built from training interactions only. Held-out test ratings are never seen by NetInf.

To use a different dataset:

```bash
python pipeline.py --steps cascade --dataset ciao
python pipeline.py --steps cascade --dataset epinions
```

> **Note:** Use the same `--dataset` flag consistently for all subsequent steps in the same pipeline run so that cascade IDs and recommender encodings stay aligned.

---

### 2. Inspect Delta (optional)

Print the median inter-event delta and suggested alpha centres:

```bash
python pipeline.py --steps delta
```

---

### 3. Infer Diffusion Networks

Run NetInf across the log-spaced alpha grid for all models:

```bash
python pipeline.py --steps inference
```

Options:

```bash
# Single model only
python pipeline.py --steps inference --model exponential

# Custom alpha grid
python pipeline.py --steps inference --n-alphas 50 --max-iter 1000
```

Output: `data/<dataset>/inferred_networks/<model>/inferred_edges_<short>_<alpha>.csv`

---

### 4. Compute Centrality Metrics

```bash
python pipeline.py --steps centrality
```

Output: `data/<dataset>/centrality_metrics/<model>/centrality_metrics_<short>_<id>.csv`

---

### 5. Detect Communities and Compute LPH

```bash
python pipeline.py --steps communities
```

Output: `data/<dataset>/communities/<model>/communities_<short>_<id>.csv`

---

### 6. Train and Evaluate Recommenders

Baseline CMF + enhanced CMF with all three models:

```bash
python pipeline.py --steps recommend
```

This step:

1. Loads the dataset and applies the **global split** (same seed as the cascade step).
2. Runs hyperparameter search and trains the **baseline CMF** on the training partition, then reports RMSE/MAE/R² on the global test set.
3. Evaluates the **enhanced CMF** (with network side-information) using repeated random sub-splits of the training partition, with a paired baseline per fold for fair comparison.

The pipeline default CMF solver is `lbfgs`. This keeps the plain baseline and
the enhanced model comparable to the social-regularized path, which requires
L-BFGS. You can still override it for non-social runs:

```bash
python pipeline.py --steps recommend --cmf-method als
```

Community features are included by default. To exclude them from the enhanced or
social CMF side-user matrix:

```bash
python pipeline.py --steps recommend --no-communities
```

Run the same recommendation flow with Phase 6 social regularization:

```bash
python pipeline.py --steps recommend --social-regularization
```

When `--social-regularization` is enabled, `recommend` performs a social Optuna
search over `k`, `lambda_reg`, `w_main`, `w_user`, `lambda_social`,
`social_mode`, `beta`, and `gamma`, while holding `social_normalization` fixed
to the selected `--social-normalization` value. It saves the result to
`data/<dataset>/social_hyperparam_search_results.json`, and then evaluates the
selected social CMF settings across the sampled inferred networks. The social
search uses 200 trials by default because it has a larger eight-parameter search
space; baseline and non-social enhanced searches still use 50 trials. The chosen
normalization is recorded in the search result and per-trial metadata.

Useful social options:

```bash
# Restrict the social run to one diffusion model
python pipeline.py --steps recommend --model exponential --social-regularization

# Use all available networks in the recommendation evaluation
python pipeline.py --steps recommend --all-networks --social-regularization

# Adjust social-search cost and L-BFGS iterations
python pipeline.py --steps recommend --social-regularization \
   --social-n-trials 200 \
   --social-search-max-ratings 10000 \
   --social-normalization mean_weight \
   --cmf-maxiter 50
```

---

### 7. Tune Enhanced or Social CMF Hyperparameters (standalone)

Run the Optuna search for the enhanced model in isolation, without triggering
the full network evaluation:

```bash
python pipeline.py --steps hypertune
```

This step performs one Optuna TPE search over the model hyperparameters using
the first available network as a representative sample. The non-social enhanced
search uses 50 trials over four parameters (`k`, `lambda_reg`, `w_main`,
`w_user`). Results are saved to
`data/<dataset>/enhanced_search_results.json` and consumed by the `shap` step.

To run the social-regularized search instead:

```bash
python pipeline.py --steps hypertune --social-regularization
```

This saves `data/<dataset>/social_hyperparam_search_results.json`. The SHAP
step will load that file when it is also run with `--social-regularization`.
The social search uses 200 trials by default over eight tuned parameters plus
the fixed/logged `social_normalization` setting.

Use `hypertune` instead of `recommend` when you only need the best params — for
example before running the SHAP analysis without re-evaluating all networks.

---

### 8. SHAP Feature Importance

Compute SHAP values to explain which network features drive the enhanced CMF
predictions:

```bash
python pipeline.py --steps shap
```

This step requires `data/<dataset>/enhanced_search_results.json` to exist (generated by
either `recommend` or `hypertune`).  For each of the three diffusion models it:

1. Samples 5 random networks (configurable with `--k-networks`).
2. Trains the enhanced CMF with the saved best hyperparameters.
3. Predicts per-user mean ratings on the test set.
4. Fits a GBT surrogate on `(network features → mean predicted rating)`.
5. Applies `TreeExplainer` (exact TreeSHAP) and averages |SHAP| values across
   the sampled networks.

Output: `data/<dataset>/shap_results.json` with per-model feature importance rankings.

To explain the social-regularized CMF instead of the regular enhanced CMF, pass
the same social flag:

```bash
python pipeline.py --steps shap --social-regularization
```

In this mode SHAP loads
`data/<dataset>/social_hyperparam_search_results.json`, trains the L-BFGS social
CMF for each sampled network, and fits the same GBT surrogate on the resulting
CMF predictions.

Options:

```bash
# Fewer/more networks per model
python pipeline.py --steps shap --k-networks 10

# Single diffusion model
python pipeline.py --steps shap --model exponential

# Reproducible sampling with a different seed
python pipeline.py --steps shap --seed 7
```

---

### Experiment Tracking (MLflow)

The `recommend`, `hypertune`, and `shap` steps automatically log parameters,
metrics, and artifacts to a local MLflow store in `mlruns/`.

Inspect results in the MLflow UI after any tracked step:

```bash
mlflow ui --backend-store-uri mlruns/
# → http://127.0.0.1:5000
```

See [mlflow.md](mlflow.md) for the full list of tracked metrics and how to
query results programmatically.

---

## Datasets

All three rating datasets are stored under `datasets/` and are ready to use without any preprocessing.

| Name | Folder | File | Format |
| --- | --- | --- | --- |
| `movielens` | `datasets/movielens/` | `ratings_small.csv` | CSV with header: `UserId,ItemId,Rating,timestamp` |
| `ciao` | `datasets/ciao/` | `rating_with_timestamp.txt` | Whitespace-separated, no header. Columns: `user, product, category, rating, helpfulness, timestamp` |
| `epinions` | `datasets/epinions/` | `rating_with_timestamp.txt` | Whitespace-separated, no header. Columns: `user, product, category, rating, helpfulness, timestamp` |

Column mapping and separators are configured in `config.Datasets.CONFIG`.  
Adding a new dataset only requires adding an entry to that dict and placing the raw file in `datasets/<name>/`.

---

## CLI Reference

The complete, up-to-date flag tables for `pipeline.py` (including Phase-2
experiment steps, `--k-avg-degree`, `--preregister-networks`, `--run-id`,
variants, and Route B flags) live in **[cli/pipeline.md](cli/pipeline.md)**.

Index of every CLI in the repo: **[cli/README.md](cli/README.md)**.

Quick check of the installed parser:

```bash
python -m pipeline --help
```

---

## Combining Steps

Steps can be chained in a single call:

```bash
python pipeline.py --steps cascade inference centrality communities recommend
```

Typical workflow when only SHAP analysis is needed after networks are ready:

```bash
python pipeline.py --steps hypertune shap
```

Equivalent workflow for the Phase 6 social-regularized model:

```bash
python pipeline.py --steps hypertune shap --social-regularization
```

---

## Python API

Each module can also be used directly:

```python
# Global split (same partition as the pipeline) — MovieLens default
from recommender.data import load_and_split_dataset
data, train_df, test_df = load_and_split_dataset()              # movielens
data, train_df, test_df = load_and_split_dataset(dataset="ciao")
data, train_df, test_df = load_and_split_dataset(dataset="epinions")

# Cascade generation from a pre-split DataFrame
from networks.cascades import generate_cascades_from_df
generate_cascades_from_df(train_df, all_user_ids=data["UserId"])

# Centrality
from networks.centrality import calculate_centrality_for_all_models
calculate_centrality_for_all_models()

# Baseline recommendation
from recommender.baseline import search_best_params, train_final_model
from recommender.data import evaluate_single_split
results = search_best_params(train_df, n_iter=50)
model = train_final_model(train_df, **results["best_params"])
metrics = evaluate_single_split(model, test_df)  # RMSE on held-out test

# Enhanced recommendation
from recommender.enhanced import run_network_evaluation
run_network_evaluation(data=train_df, sample_networks=5, include_communities=True)

# Social-regularized recommendation
run_network_evaluation(
   data=train_df,
   sample_networks=5,
   include_communities=True,
   use_social_regularization=True,
   social_mode="boundary_downweight",
   lambda_social=0.001,
)

# Enhanced hyperparameter search (standalone)
from recommender.enhanced import search_enhanced_params, save_enhanced_search_results, load_network_features
features = load_network_features("exponential", 0, include_communities=True)
search = search_enhanced_params(train_df, features, n_trials=50, n_splits=3)
save_enhanced_search_results(search)

# SHAP feature importance
from config import DatasetPaths
from analysis.shap_analysis import run_shap_analysis, save_shap_results
results = run_shap_analysis(k_networks=5, include_communities=True)                 # movielens
results = run_shap_analysis(k_networks=5, include_communities=True, dataset="ciao")
social_results = run_shap_analysis(
   k_networks=5,
   include_communities=True,
   social_regularization=True,
   params_path=DatasetPaths("movielens").SOCIAL_RESULTS,
)
save_shap_results(results)
```

---

## Visualisation

```python
# Community + LPH plots
from visualization.community_plots import plot_lph_distribution
plot_lph_distribution(save=True)

# Alpha vs RMSE
from visualization.model_plots import plot_alpha_rmse_analysis
plot_alpha_rmse_analysis("exponential", rmse_values, baseline_rmse=0.92)

# Centrality distributions
from visualization.network_plots import plot_all_centrality_distributions
plot_all_centrality_distributions("exponential", "001")

# Cascade timeline
from visualization.network_plots import plot_cascades_timeline
plot_cascades_timeline(n=30)
```

---

## Output Summary

| Step | Output location |
| --- | --- |
| Cascades | `data/<dataset>/cascades.txt` |
| Inferred networks | `data/<dataset>/inferred_networks/<model>/` |
| Centrality metrics | `data/<dataset>/centrality_metrics/<model>/` |
| Communities + LPH | `data/<dataset>/communities/<model>/` |
| Enhanced hyperparams | `data/<dataset>/enhanced_search_results.json` |
| Social hyperparams | `data/<dataset>/social_hyperparam_search_results.json` |
| SHAP results | `data/<dataset>/shap_results.json` |
| Plots | `plots/<dataset>/` |
