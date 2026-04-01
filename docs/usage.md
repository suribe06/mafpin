# Usage Guide

This guide walks through each step of the MAFPIN pipeline using `pipeline.py`.

---

## Quick Start — Full Pipeline

Run all steps in order for all three diffusion models:

```bash
python pipeline.py --all
```

---

## Step-by-step

### 1. Generate Cascades

Convert the MovieLens ratings CSV into a cascades file for NetInf:

```bash
python pipeline.py --steps cascade
```

This reads `data/ratings_small.csv`, applies the **global 80/20 split** (seed from `config.Split.RANDOM_STATE`), and writes `data/cascades.txt` built from training interactions only. Held-out test ratings are never seen by NetInf.

To use a different dataset:

```bash
python pipeline.py --steps cascade --dataset my_ratings
```

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

Output: `data/inferred_networks/<model>/inferred_edges_<short>_<alpha>.csv`

---

### 4. Compute Centrality Metrics

```bash
python pipeline.py --steps centrality
```

Output: `data/centrality_metrics/<model>/centrality_metrics_<short>_<id>.csv`

---

### 5. Detect Communities and Compute LPH

```bash
python pipeline.py --steps communities
```

Output: `data/communities/<model>/communities_<short>_<id>.csv`

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

Include community features (LPH + community count):

```bash
python pipeline.py --steps recommend --include-communities
```

---

## Combining Steps

Steps can be chained in a single call:

```bash
python pipeline.py --steps cascade inference centrality communities recommend
```

---

## Python API

Each module can also be used directly:

```python
# Global split (same partition as the pipeline)
from recommender.data import load_and_split_dataset
data, train_df, test_df = load_and_split_dataset()  # uses config.Split

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
| Cascades | `data/cascades.txt` |
| Inferred networks | `data/inferred_networks/<model>/` |
| Centrality metrics | `data/centrality_metrics/<model>/` |
| Communities + LPH | `data/communities/<model>/` |
| Plots | `plots/` |
