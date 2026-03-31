# Methodology

This document describes the technical steps behind the MAFPIN pipeline.

---

## 1. Cascade Generation

A **cascade** encodes how a single item propagates through a user population.

Given the MovieLens ratings CSV (columns: `userId`, `movieId`, `rating`, `timestamp`):

1. For each movie, collect all `(userId, timestamp)` pairs where a rating exists.
2. Sort by timestamp to form an ordered adoption sequence.
3. Write to `cascades.txt` in NetInf format: one line per cascade, values as comma-separated `userId,timestamp` pairs.

```text
userId_1,t1,userId_2,t2,...
```

Source: `networks/cascades.py`

---

## 2. Alpha Grid Generation

The algorithm automatically computes model-specific α ranges from cascade temporal statistics.

Source: `networks/delta.py`

### 2.1 Median Delta (Δ)

We first compute the **median time difference** between consecutive user interactions within cascades:

$$\tilde{\Delta} = \text{median}\lbrace t_i - t_j \mid t_i > t_j \rbrace$$

Timestamps are assumed to be in **Unix epoch seconds**, but may be converted to days or years as long as the same unit is used consistently for Δ, α, and cascade timestamps.

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

> **Important:** Always use the same unit system for Δ, α, and cascade timestamps.

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

### Baseline CMF

A standard Collective Matrix Factorisation model trained only on the rating matrix:

$$\min_{U, V}\ \|R - UV^\top\|_F^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)$$

Fitted with ALS using `cmfrec.CMF`.

### Enhanced CMF

The baseline is augmented with a user side-information matrix **S** built from:

- Seven centrality metrics (per network).
- Optionally: community membership counts and LPH value.

Features are optionally standardised or min–max normalised before being passed to CMF as `U=S`.

Source: `recommender/baseline.py`, `recommender/enhanced.py`
