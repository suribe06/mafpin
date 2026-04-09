# Methodology

This document describes the technical steps behind the MAFPIN pipeline.

---

## 1. Cascade Generation

A **cascade** encodes how a single item propagates through a user population.

Given the MovieLens ratings CSV (columns: `userId`, `movieId`, `rating`, `timestamp`):

1. The dataset is split into **train (80%) and test (20%)** using the global seed defined in `config.Split.RANDOM_STATE` (default: 42). Only training interactions are used to build cascades.
2. For each movie, collect all `(userId, timestamp)` pairs present in the **training split**.
3. Sort by timestamp **ascending** to form an ordered adoption sequence.
4. Write to `cascades.txt` in NetInf format: one line per cascade, values as comma-separated `userId,timestamp` pairs.

```text
userId_1,t1,userId_2,t2,...
```

The cascade **header** (self-loop lines that declare node existence) lists the full user-ID space — including users present only in the test set — so that NetInf compact node IDs remain aligned with the `LabelEncoder` mapping used by the CMF recommender.

Cascades with only one user are skipped; they carry no diffusion signal.

Source: `networks/cascades.py`, `config.Split`

---

## 2. Alpha Grid Generation

The algorithm automatically computes model-specific α ranges from cascade temporal statistics.

Source: `networks/delta.py`

### 2.1 Median Delta (Δ)

We first compute the **median of consecutive inter-event time differences** within each cascade:

$$\tilde{\Delta} = \text{median}\lbrace t_{i+1} - t_i \rbrace$$

Only adjacent pairs in time-sorted order are used. All-pairs computation would overestimate Δ by counting long-range differences that have no physical meaning under a Markov diffusion assumption.

In this implementation, Unix epoch seconds are divided by 86 400 at cascade-generation time so that all timestamps are stored in **days**. Consequently Δ is in days and α values are in days⁻¹ (exponential) or days⁻² (Rayleigh). This keeps the log-likelihood surface numerically well-conditioned for typical social-dataset inter-event times (hours to months). The unit chosen does not affect network topology — only the absolute scale of α; as long as cascades, Δ, and the α grid all use the **same unit**, the results are equivalent.

### 2.2 Model-Specific Alpha Centres

Each transmission model has a theoretical relationship between its **median** and **α**, which defines a natural centre point:

#### Exponential

$$m = \frac{\ln 2}{\alpha} \quad \Rightarrow \quad \alpha_{\text{center}} = \frac{\ln 2}{\tilde{\Delta}}$$

#### Rayleigh

$$m = \sqrt{\frac{2 \ln 2}{\alpha}} \quad \Rightarrow \quad \alpha_{\text{center}} = \frac{2 \ln 2}{\tilde{\Delta}^2}$$

#### Power-law

$$m = 2^{1/\alpha}\,\Delta_{\min} \quad \Rightarrow \quad \alpha = \frac{\ln 2}{\ln(\tilde{\Delta}/\Delta_{\min})}$$

However, for typical datasets this value falls below 1, while NetInf only supports $\alpha \geq 1$.
Therefore, in practice a fixed linear grid is used:

$$\alpha \in [1,\ 3] \quad \text{or} \quad [1,\ 5]$$

### 2.3 Log-Scale Grid (Exponential & Rayleigh)

For Exponential and Rayleigh models, α values are explored around the centre on a **logarithmic scale** to capture both slower and faster transmission rates:

- **Range factor** $r$ — typical values: 10–100
- **Grid size** $N$ — typical values: 20–100 (default: 100)

The grid formula is:

$$\alpha_i = \alpha_{\text{center}} \cdot r^{\left(\tfrac{2i}{N-1} - 1\right)}, \quad i = 0, 1, \dots, N-1$$

This ensures exactly half the values are smaller than the centre and half are larger, evenly spaced in log-space.

> **Important:** Cascades, Δ, and the α grid must all use the **same time unit**. This implementation uses **days** throughout.

---

## 3. Network Inference (NetInf)

For each α in the grid, NetInf is called as a subprocess:

```bash
./netinf -i:cascades.txt -m:<model> -a:<alpha> -n:<n_edges> -o:<output>
```

NetInf maximises the likelihood of observing the cascades under the chosen diffusion model using a greedy algorithm.  
The output is a pipe-separated edge list with columns `src`, `dst`, `alpha`.

Source: `networks/inference.py`

---

## 4. Centrality Metrics

Seven per-node metrics are computed with SNAP-py for each inferred network:

| Metric | Description |
| --- | --- |
| Degree | Fraction of connected neighbours (normalised) |
| Betweenness | Fraction of shortest paths passing through the node |
| Closeness | Inverse of mean shortest path to all other nodes |
| Eigenvector | Importance weighted by neighbour importance |
| PageRank | Random-walk stationary distribution |
| Clustering | Fraction of closed triangles among neighbours |
| Eccentricity | Maximum shortest path length from the node |

Results are saved to `data/centrality_metrics/<model>/centrality_metrics_<short>_<id>.csv`.

Source: `networks/centrality.py`

---

## 5. Community Detection and LPH

Overlapping communities are detected using two algorithms:

- **Demon** (Democratic Estimate of the Modular Organization of a Network) — ego-net-based hierarchical merging.  
- **ASLPAw** — Asymmetric Speaker-listener Label Propagation Algorithm (weighted variant).

For each node, **Local Pluralistic Homophily (LPH)** is computed as the mean Jaccard similarity between its community set and those of its neighbours.  
See [lph.md](lph.md) for a full definition.

Source: `networks/communities.py`

---

## 6. Recommendation

### Models

**Baseline CMF** — a standard Collective Matrix Factorisation model trained only on the rating matrix:

$$\min_{U, V}\ \|R - UV^\top\|_F^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)$$

**Enhanced CMF** — the baseline augmented with a user side-information matrix **S** built from:

- Seven centrality metrics computed over the inferred diffusion network.
- Optionally: community membership count and LPH score.

Features are standardised (or min–max / L2-normalised, configurable) and passed to `cmfrec.CMF` as `U=S`. Scaling is fitted on **training users only** within each fold, preventing leakage.

---

### Comparison Design

The goal of the evaluation is to answer a single question: **does adding diffusion-network side-information improve a CMF recommender, relative to the best possible plain CMF of the same class?**

To answer this fairly two separate hyperparameter searches are run:

| Search | Parameters | Purpose |
| --- | --- | --- |
| Baseline Optuna (k, λ) | latent factors, L2 regularisation | Find the best achievable RMSE for plain CMF |
| Enhanced Optuna (k, λ, w_main, w_user) | all above + loss weights | Find the best achievable RMSE with network side-info |

The searches are **independent**: optimal λ for plain CMF is systematically different from optimal λ for enhanced CMF (the side-information term changes the effective regularisation landscape), so a shared search would bias the result.

#### Two-level evaluation

**Level 1 — Global baseline (whole dataset)**  
The baseline model is trained with its optimally-tuned (k\*, λ*) on the global training split and evaluated on the global held-out test set. This gives the best-case RMSE achievable by plain CMF on the full user population.

**Level 2 — Per-network paired comparison**  
For each inferred network, only the users present in that network have features available. The evaluation is therefore restricted to that user subset, making a direct global comparison invalid. Instead, for every network and every cross-validation fold:

1. The **enhanced model** is trained with enhanced-optimal (k, λ, w_main, w_user) on the filtered training fold.
2. A **paired baseline** is trained with baseline-optimal (k\*, λ*) on the **same filtered fold** and evaluated on the **same test fold**.

Because both models see exactly the same users and the same data, the difference in RMSE is attributable solely to the network side-information.

> **Why not reuse the enhanced λ for the paired baseline?**  
> The enhanced model is jointly regularised by the rating loss and the side-info reconstruction loss. Its optimal λ is larger than what a plain CMF needs, so applying it to the baseline would over-regularise it and inflate its RMSE — making the improvement look larger than it truly is.

---

### Evaluation Protocol

- **Single global 80/20 split** (seed: `config.Split.RANDOM_STATE`). Cascade generation, feature computation, hyperparameter search, and model training all use the training partition only.
- **Feature scaling** is fitted on training users within each CV fold and applied to all users.
- **Final metrics** (RMSE, MAE, R²) on the held-out global test set.

Source: `recommender/baseline.py`, `recommender/enhanced.py`, `config.Split`
