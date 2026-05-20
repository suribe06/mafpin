# MAFPIN: Matrix Factorization with Properties of Inferred Networks

MAFPIN is a research framework that combines **information diffusion modelling** with **collaborative filtering** to build network-aware recommender systems.

The core hypothesis is that user centrality metrics and community structure in inferred social influence networks can improve traditional matrix factorisation accuracy.

## Documentation

| Document | Description |
| --- | --- |
| [docs/overview.md](docs/overview.md) | Project overview, pipeline diagram, key concepts |
| [docs/installation.md](docs/installation.md) | Dependencies, virtual environment setup, NetInf binary |
| [docs/methodology.md](docs/methodology.md) | Cascade generation, alpha grid, NetInf, CMF formulation |
| [docs/lph.md](docs/lph.md) | Local Pluralistic Homophily definition and Demon algorithm |
| [docs/centrality_metrics.md](docs/centrality_metrics.md) | All seven centrality metrics with formulas |
| [docs/usage.md](docs/usage.md) | Step-by-step CLI and Python API guide |
| [docs/hyperparameter_tuning.md](docs/hyperparameter_tuning.md) | Optuna TPE search, parameter ranges, two-baseline strategy |
| [docs/social_regularization.md](docs/social_regularization.md) | Phase 6 boundary-guided social regularization module reference |
| [docs/reports/social_smoke_test.md](docs/reports/social_smoke_test.md) | Phase 6 social regularization smoke-test report |
| [docs/visualization.md](docs/visualization.md) | Plot reference: alpha-RMSE, delta-RMSE, community and network plots |

## Quick Start

```bash
# 1. Create a Python 3.9 virtual environment
python3.9 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 2b. Build the local cmfrec Cython extensions
cd cmfrec-master && python setup.py build_ext --inplace && cd ..

# 3. Run the full pipeline
python pipeline.py --all

# Optional: run the full Phase 6 social-regularized pipeline
python pipeline.py --all --social-regularization
```

## Pipeline Steps

```text
cascade → delta → inference → centrality → communities → recommend
                                                              │
                                                    hypertune → shap
```

Run individual steps:

```bash
python pipeline.py --steps cascade inference centrality
python pipeline.py --steps recommend

# Tune enhanced CMF hyperparameters without running the full evaluation
python pipeline.py --steps hypertune

# SHAP feature importance (requires hypertune or recommend to have run first)
python pipeline.py --steps shap

# Social-regularized recommend + SHAP flow
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend shap \
  --dataset movielens \
  --social-regularization
```

Pipeline runs append progress to `data/<dataset>/pipeline.log` by default.
Use `tail -f data/movielens/pipeline.log` to watch long runs from another
terminal.

See [docs/usage.md](docs/usage.md) for the full reference.

## Project Structure

```text
mafpin/
├── config.py               # Centralised paths and default parameters
├── pipeline.py             # Unified CLI entry point
├── networks/               # Cascade generation, inference, centrality, communities
├── recommender/            # Dataset utilities, baseline CMF, enhanced CMF
├── cmfrec-master/          # Local cmfrec source compiled in place
├── analysis/               # Post-hoc analyses (SHAP feature importance)
├── visualization/          # Plot functions (model, network, community)
├── data/                   # Raw data + generated artefacts
├── plots/                  # Generated PNG figures
└── docs/                   # Full documentation
```

## Key Innovation

Traditional collaborative filtering relies solely on user-item rating patterns. MAFPIN extends this by:

1. **Inferring social influence networks** from temporal cascade data using NetInf.
2. **Computing seven per-node centrality metrics** (degree, betweenness, closeness, eigenvector, PageRank, clustering, eccentricity).
3. **Detecting overlapping communities** (Demon / ASLPAw) and computing **Local Pluralistic Homophily (LPH)**.
4. **Incorporating network features as user side-information** in Collective Matrix Factorisation (CMF).
5. **Evaluating RMSE improvement** across a log-spaced alpha (transmission rate) grid.
6. **Adding direct social regularization** — Phase 6 can use the inferred network as a weighted graph penalty over L-BFGS CMF user factors via `--social-regularization`.
7. **Explaining predictions via SHAP** — a GBT surrogate trained on CMF outputs is analysed with TreeSHAP to rank the contribution of each network feature per diffusion model, including the social-regularized path when enabled.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{doi:10.1142/S1793830923500520,
  author = {Uribe, Santiago and Ramirez, Carlos and Finke, Jorge},
  title  = {Recommender systems based on matrix factorization and the properties of inferred social networks},
  journal = {Discrete Mathematics, Algorithms and Applications},
  volume  = {16},
  number  = {05},
  pages   = {2350052},
  year    = {2024},
  doi     = {10.1142/S1793830923500520},
  url     = {https://doi.org/10.1142/S1793830923500520}
}
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
