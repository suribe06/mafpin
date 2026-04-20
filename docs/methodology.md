# Methodology

This document describes the technical steps behind the MAFPIN pipeline.

---

## 1. Cascade Generation

A **cascade** encodes how a single item propagates through a user population.

Three rating datasets are supported and read from the `datasets/` directory.  
The active dataset is selected via the `--dataset` CLI flag (default: `movielens`).

| Dataset | Source file | Columns used |
| --- | --- | --- |
| `movielens` | `datasets/movielens/ratings_small.csv` | `UserId`, `ItemId`, `Rating`, `timestamp` |
| `ciao` | `datasets/ciao/rating_with_timestamp.txt` | user (col 0), product (col 1), rating (col 3), timestamp (col 5) |
| `epinions` | `datasets/epinions/rating_with_timestamp.txt` | user (col 0), product (col 1), rating (col 3), timestamp (col 5) |

Column mappings are defined in `config.Datasets.CONFIG`.

Given the selected dataset:

1. The dataset is split into **train (80%) and test (20%)** using the global seed defined in `config.Split.RANDOM_STATE` (default: 42). Only training interactions are used to build cascades.
2. For each item, collect all `(userId, timestamp)` pairs present in the **training split**.
3. Sort by timestamp **ascending** to form an ordered adoption sequence.
4. Write to `cascades.txt` in NetInf format: one line per cascade, values as comma-separated `userId,timestamp` pairs.

```text
userId_1,t1,userId_2,t2,...
```

The cascade **header** (self-loop lines that declare node existence) lists the full user-ID space — including users present only in the test set — so that NetInf compact node IDs remain aligned with the `LabelEncoder` mapping used by the CMF recommender.

Cascades with only one user are skipped; they carry no diffusion signal.

### 1.1 Cascade User Statistics

After the cascade file is written, a second pass computes three **per-user temporal influence statistics** from the cascade timeline.  These are saved once per dataset to `data/<dataset>/cascade_user_stats.csv` and later merged into the Enhanced CMF side-information matrix.

For each user $u$ appearing in cascade $c$, let $\text{pos}(u, c)$ be the **1-indexed temporal rank** of $u$ inside cascade $c$ (rank 1 = seed, i.e. earliest adopter in that item's adoption sequence).  Let $B_u$ be the set of cascades that contain $u$.

| Column | Definition | Interpretation |
| --- | --- | --- |
| `cascade_breadth` | $\|B_u\|$ — number of distinct cascades user $u$ participates in | Activity level; how broadly a user rates across items |
| `mean_cascade_position` | $\frac{1}{\|B_u\|} \sum_{c \in B_u} \text{pos}(u, c)$ | Average adoption rank; lower = earlier adopter on average |
| `min_cascade_position` | $\min_{c \in B_u} \text{pos}(u, c)$ | Best (earliest) rank ever observed; `1` means the user was the seed in at least one cascade |

**Reliability threshold**: `mean_cascade_position` and `min_cascade_position` are set to `NaN` for users who appear in fewer than 5 cascades, because positional estimates from very few observations are unreliable.  The `NaN` values are replaced with `0.0` when the feature matrix is assembled by `load_network_features()`.

**Alignment with centrality IDs**: The compact `UserId` keys used in `cascade_user_stats.csv` are derived by sorting all node IDs declared in the cascade header and assigning 0-based indices — the same logic as `_build_mapper` in `networks/network_io.py`.  This ensures a direct join with the centrality-metric CSVs without any re-mapping.

Source: `networks/cascades.py :: compute_cascade_user_stats`, `config.DatasetPaths.CASCADE_USER_STATS`

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

The power-law (Pareto) transmission density used by NetInf is:

$$f(\Delta t;\,\alpha) = (\alpha - 1)\cdot \Delta t^{-\alpha}, \quad \Delta t \geq 1, \quad \alpha > 1$$

Unlike the exponential and Rayleigh parameters, **α is a dimensionless shape exponent** — it is not a rate in any time unit, so it does not scale with whether timestamps are stored in seconds, hours, or days. For this reason no data-driven centre is derived from the median Δ; instead a fixed linear sweep is used:

$$\alpha \in [1.1,\ 5.0]$$

**Why the lower bound is 1.1, not 1.0:** At α = 1 the integrand becomes $t^{-1}$, whose integral $\int_1^{\infty} t^{-1}\,dt$ diverges — the distribution is non-normalizable and the likelihood is undefined. NetInf may accept α = 1 without raising an error, but the result is numerically meaningless. The bound 1.1 enforces a valid density with a finite mean (mean exists when α > 2; variance exists when α > 3).

**Why the upper bound is 5.0:** Real social-influence cascades rarely exhibit exponents above 3–4. The paper's synthetic-network experiments (Gomez-Rodriguez et al. 2011, Section 5.2) use α ∈ {1.5, 2.0, 2.5}. Sweeping to 5.0 gives comfortable margin while keeping runtime tractable.

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

Eleven per-node metrics are computed with SNAP-py for each inferred network:

| Metric | Column | Description |
| --- | --- | --- |
| Degree | `degree` | Fraction of connected neighbours (normalised total degree) |
| In-Degree | `in_degree` | Normalised in-degree — influence sinks / late adopters |
| Out-Degree | `out_degree` | Normalised out-degree — influence sources / taste-makers |
| Betweenness | `betweenness` | Fraction of shortest paths passing through the node |
| Closeness | `closeness` | Inverse of mean shortest path to all other nodes |
| Eigenvector | `eigenvector` | Importance weighted by neighbour importance |
| PageRank | `pagerank` | Random-walk stationary distribution |
| Clustering | `clustering` | Fraction of closed triangles among neighbours |
| Eccentricity | `eccentricity` | Maximum shortest path length from the node |
| Hub Score | `hub_score` | HITS hub: points to authoritative nodes (aggregators) |
| Authority Score | `auth_score` | HITS authority: pointed to by many hubs (canonical taste-makers) |

Results are saved to `data/centrality_metrics/<model>/centrality_metrics_<short>_<id>.csv`.

See [centrality_metrics.md](centrality_metrics.md) for full formulas and interpretations.

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

- Eleven centrality metrics computed over the inferred diffusion network (`degree`, `in_degree`, `out_degree`, `betweenness`, `closeness`, `eigenvector`, `pagerank`, `clustering`, `eccentricity`, `hub_score`, `auth_score`).
- Optionally: community membership count and LPH score.
- Optionally: cascade temporal statistics shared across all networks for the dataset:
  - `mean_cascade_position` — average 1-indexed temporal position of the user across cascades (lower = earlier adopter).
  - `min_cascade_position` — earliest position ever observed (1 = seed in at least one cascade).
  - `cascade_breadth` — number of distinct cascades the user appears in.
  - Users appearing in fewer than 5 cascades have NaN positional stats; these are filled with 0.0 downstream.

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
