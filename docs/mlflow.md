# MLflow Integration

MAFPIN uses [MLflow](https://mlflow.org) for experiment tracking. Every run of the
`recommend`, `hypertune`, and `shap` pipeline steps is automatically logged to a local
MLflow tracking server — no extra configuration required.

---

## Quick start

```bash
# 1. Run any tracked step (runs are logged automatically)
python pipeline.py --steps recommend --all-networks

# 2. Open the MLflow UI to inspect results
mlflow ui --backend-store-uri mlruns/
# → http://127.0.0.1:5000
```

---

## Configuration

All MLflow settings live in `config.py` under the `MLflow` class:

```python
class MLflow:
    EXPERIMENT_NAME = "mafpin"          # experiment name shown in the UI
    TRACKING_URI    = str(ROOT / "mlruns")  # local file store, no server needed
```

To point to a remote MLflow server instead of the local file store, change
`TRACKING_URI` to a remote URI, e.g. `"http://my-server:5000"`.

---

## What gets tracked

### `recommend` step

Parent run name: **`recommend`**

| What | MLflow entity |
| --- | --- |
| `include_communities`, `sample_networks`, `all_networks`, `model`, `n_optuna_trials`, `n_cv_splits` | params |
| `k_baseline`, `lambda_baseline`, `k_enhanced`, `lambda_enhanced`, `w_main`, `w_user` | params (post-search) |
| `baseline_rmse`, `baseline_mae`, `baseline_r2` | metrics (global test set) |
| `<model>_rmse_enhanced`, `<model>_rmse_baseline` | metrics per network (step = network index) |
| `<model>_improvement_pct` | metrics per network (step = network index) |
| `<model>_mean_rmse_enhanced`, `<model>_n_networks_evaluated` | summary metrics |
| `baseline_search_results.json`, `enhanced_search_results.json` | artifacts |

Two **nested runs** are created inside the parent:

- **`baseline_search`** — logs `baseline_trial_rmse` per Optuna trial (step = trial number) and `baseline_best_k`, `baseline_best_lambda_reg`, `baseline_best_rmse`.
- **`enhanced_search`** — logs `enhanced_trial_rmse` per Optuna trial and `enhanced_best_k`, `enhanced_best_lambda_reg`, `enhanced_best_w_main`, `enhanced_best_w_user`, `enhanced_best_rmse`.

---

### `hypertune` step

Run name: **`hypertune`**

| What | MLflow entity |
| --- | --- |
| `include_communities`, `n_optuna_trials`, `n_cv_splits` | params |
| `enhanced_best_k`, `enhanced_best_lambda_reg`, `enhanced_best_w_main`, `enhanced_best_w_user` | params |
| `enhanced_best_rmse` | metric |
| `enhanced_trial_rmse` (per trial) | metric (step = trial number) |
| `enhanced_search_results.json` | artifact |

---

### `shap` step

Run name: **`shap`**

| What | MLflow entity |
| --- | --- |
| `k_networks`, `include_communities`, `seed`, `all_networks`, `model` | params |
| `<model>_n_networks` | metric (number of networks processed) |
| `shap_<model>_<feature_name>` (one per feature per model) | metrics (mean \|SHAP\|) |
| `shap_results.json` | artifact |

---

## Comparing runs in the UI

After running the pipeline multiple times (e.g. with and without communities, or with
different `--sample-networks` values), open the UI and use the **Compare** view to plot
metric curves side by side:

```bash
mlflow ui --backend-store-uri mlruns/
```

Useful comparisons:

- `baseline_rmse` vs `<model>_mean_rmse_enhanced` — is the network side-information helping?
- `enhanced_trial_rmse` over steps — Optuna convergence curve.
- `shap_<model>_<feature>` across runs — feature importance stability.

---

## Programmatic access

Read results from any run without opening the UI:

```python
import mlflow

mlflow.set_tracking_uri("mlruns/")
client = mlflow.MlflowClient()

experiment = client.get_experiment_by_name("mafpin")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'recommend'",
    order_by=["metrics.baseline_rmse ASC"],
)

for run in runs:
    print(run.info.run_id, run.data.metrics["baseline_rmse"])
```

---

## Storage layout

MLflow stores everything under `mlruns/` in the project root:

```text
mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── meta.yaml
        ├── params/
        ├── metrics/
        └── artifacts/
            ├── baseline_search_results.json
            └── enhanced_search_results.json
```

`mlruns/` is excluded from version control via `.gitignore`.  
To share results, export runs with:

```bash
mlflow experiments csv -x <experiment_id> -o results.csv
```
