# MAFPIN — Overview

**MAFPIN** (Matrix Factorization with Properties of Inferred Networks) is a research framework that combines **information diffusion modelling** with **collaborative filtering** to build network-aware recommender systems.

---

## Research Motivation

Standard collaborative filtering (CF) methods treat users as independent entities and ignore the social relationships that drive adoption behaviour.  
MAFPIN addresses this by:

1. **Inferring latent influence networks** from user interaction timestamps using NETINF (Gomez-Rodriguez et al., 2010).
2. **Characterising those networks** via seven centrality metrics and overlapping community detection.
3. **Incorporating network features** as user side-information in Collective Matrix Factorisation (CMF), improving personalised rating prediction.

---

## Pipeline at a Glance

```text
    │
    ▼
[global split]        ──→  train (80%) / test (20%)  [config.Split]
    │
    ▼
[cascade generation]  ──→  cascades.txt  (train interactions only)
    │
    ▼
[network inference]   ──→  inferred_edges_<model>_<alpha>.csv   (per α)
    │
    ▼
[centrality metrics]  ──→  centrality_metrics_<model>_<id>.csv
    │
    ▼
[community detection] ──→  communities_<model>_<id>.csv  (+LPH)
    │
    ▼
[hypertune]           ──→  enhanced_search_results.json  (Optuna TPE)
    │
    ▼
[recommendation]      ──→  RMSE / MAE / R²  (evaluated on global test set)
    │
    ▼
[shap analysis]       ──→  shap_results.json  (per-model feature importance)
```

Three diffusion models are supported: **exponential**, **power-law**, and **Rayleigh**.  
Each model is evaluated across a log-spaced grid of the alpha (transmission rate) parameter.

---

## Key Concepts

| Concept | Description |
| --- | --- |
| **Cascade** | An ordered sequence of (user, timestamp) pairs for a single item. |
| **Alpha (α)** | Transmission rate parameter of the chosen diffusion model. |
| **NetInf** | Maximum-likelihood network structure learner from cascades. |
| **LPH** | *Local Pluralistic Homophily* — per-node community overlap similarity (see [lph.md](lph.md)). |
| **CMF** | *Collective Matrix Factorisation* — jointly factorises ratings and side-information matrices. |

---

## Repository Structure

```text
mafpin/
├── config.py               # Centralised paths, default parameters, and global split settings (Split class)
├── pipeline.py             # Unified CLI entry point
├── networks/               # Cascade generation, inference, centrality, communities
├── recommender/            # Dataset utilities, baseline CMF, enhanced CMF
├── analysis/               # Post-hoc analyses (SHAP feature importance)
├── visualization/          # All plot functions (model, network, community)
├── data/                   # Raw data + generated artefacts
│   ├── cascades.txt
│   ├── ratings_small.csv
│   ├── inferred_networks/
│   ├── centrality_metrics/
│   ├── communities/
│   ├── enhanced_search_results.json
│   └── shap_results.json
├── plots/                  # Generated PNG figures
└── docs/                   # This documentation
```

---

## References

- Gomez-Rodriguez, M., Leskovec, J., & Krause, A. (2010). *Inferring networks of diffusion and influence*. KDD.
