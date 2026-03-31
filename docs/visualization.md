# Visualization Reference

All plots are generated via `visualization/model_plots.py` and saved to `plots/models/` by default.

---

## Quick reference

| Plot | Command | Output file |
| ---- | ------- | ----------- |
| Alpha vs RMSE (all models) | `python -m visualization.model_plots --plot alpha-rmse --models all` | `alpha_rmse_<model>.png` |
| Alpha vs ΔRMSE % (all models) | `python -m visualization.model_plots --plot delta-rmse --models all` | `alpha_delta_rmse_<model>.png` |
| Alpha vs Inferred Edges | `python -m visualization.model_plots --plot alpha-edges` | `alpha_edges.png` |
| Hyperparameter search | `python -m visualization.model_plots --plot hyperparam` | `hyperparam_search.png` |
| Parameter heatmap (k vs λ) | `python -m visualization.model_plots --plot heatmap` | `heatmap_rmse.png` |
| Metrics distribution | `python -m visualization.model_plots --plot metrics` | `metrics_comparison.png` |
| Convergence curve | `python -m visualization.model_plots --plot convergence` | `convergence.png` |
| All plots | `python -m visualization.model_plots --plot all` | (all of the above) |

---

## Main analysis plots

### Alpha vs RMSE

Shows how the enhanced model RMSE varies across the alpha grid, with a ±1σ shaded band and a dashed baseline reference line.

```bash
python -m visualization.model_plots --plot alpha-rmse --models all
```

Generate for a single model:

```bash
python -m visualization.model_plots --plot alpha-rmse --models exponential
```

---

### Alpha vs ΔRMSE %

Signed percentage difference `(enhanced − baseline) / baseline × 100` per alpha value.  
Points **above zero** (red) are worse than baseline; points **below zero** (blue) are better.  
This is the primary plot to assess whether the network side information helps.

```bash
python -m visualization.model_plots --plot delta-rmse --models all
```

---

### Alpha vs Inferred Edge Count

Three-subplot figure (one per diffusion model) showing how many edges NetInf inferred for each alpha value. Useful for understanding how the network density changes across the grid.

```bash
python -m visualization.model_plots --plot alpha-edges
```

---

## Hyperparameter plots

These require `data/baseline_search_results.json` to exist.  
Run `python pipeline.py --steps recommend` or `python -m recommender.baseline` first.

### Hyperparameter search overview

Six-panel plot: scatter of (k, λ) pairs coloured by RMSE, histograms, and error bars.

```bash
python -m visualization.model_plots --plot hyperparam
```

### Parameter space heatmap

Heatmap of RMSE across the (k, λ) grid.

```bash
python -m visualization.model_plots --plot heatmap
```

### Metrics distribution

Distribution of RMSE, MAE, and R² across all search trials.

```bash
python -m visualization.model_plots --plot metrics
```

### Convergence curve

Best RMSE improvement over hyperparameter search iterations.

```bash
python -m visualization.model_plots --plot convergence
```

---

## Common options

| Flag | Description |
| ---- | ----------- |
| `--plot PLOT [PLOT ...]` | One or more plot types (see table above) |
| `--models MODEL [MODEL ...]` | `exponential`, `powerlaw`, `rayleigh`, or `all` |
| `--no-save` | Show plots interactively without writing PNG files |

Example — two plots, two models, no save:

```bash
python -m visualization.model_plots --plot alpha-rmse delta-rmse --models exponential powerlaw --no-save
```

---

## Network / cascade plots

Separate module: `visualization/network_plots.py`.

```bash
# Cascade arrival-time distributions (default n=5)
python -m visualization.network_plots cascades --n 5

# Centrality metrics for a specific network
python -m visualization.network_plots centrality --model exponential --network 0
python -m visualization.network_plots centrality --model exponential --network 0 --metric degree
```
