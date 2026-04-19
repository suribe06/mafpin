# MAFPIN Pipeline Review Report

**Date**: April 19, 2026  
**Reviewed by**: AI Code Review Agent (GitHub Copilot — Claude Sonnet 4.6)  
**Pipeline version**: feat/multi-datasets branch  
**Files reviewed**: `pipeline.py`, `config.py`, `networks/cascades.py`, `networks/inference.py`, `networks/centrality.py`, `networks/communities.py`, `networks/delta.py`, `networks/network_io.py`, `recommender/data.py`, `recommender/baseline.py`, `recommender/enhanced.py`, `analysis/shap_analysis.py`

---

## Executive Summary

MAFPIN is a research pipeline that derives graph-structural user features from a NetInf-inferred influence network and passes them as side information to a Collective Matrix Factorization (CMF) recommender. The overall architecture is sound and the codebase shows evidence of iterative refinement (references to C-1, C-3, M-2, M-3 fixes indicate an active bug-fixing history). The core data-leakage controls are correctly placed: cascade generation uses only training interactions, scalers are fitted on training users only, and SHAP values are computed against test-set predictions rather than raw test ratings.

However, three issues warrant attention before this pipeline is used in a published evaluation. First, the train/test split is **random rather than temporal**: for cascade-based network inference this creates an implicit temporal inconsistency where test interactions may precede training interactions in the cascade timeline. Second, the **entire inferred network is treated as undirected** despite NetInf outputting directed edges; this degrades the semantic validity of every directional metric (PageRank, betweenness, in-degree). Third, **no ranking metrics** (NDCG@K, Precision@K) are computed: the pipeline is evaluated solely on RMSE/MAE/R², which do not measure recommendation quality.

The pipeline is viable for research, but the temporal split issue and directedness issue should be resolved before submitting results to a venue where these design choices will be scrutinized. The feature expansion section below identifies six additional features that would materially strengthen the contribution.

---

## Issue Registry

| # | Location | Severity | Category | Summary |
|---|----------|----------|----------|---------|
| 1 | `config.py :: Split` / all steps | **Critical** | Leakage / Correctness | Random split used instead of temporal split; test interactions may predate training interactions in the cascade timeline |
| 2 | `networks/network_io.py :: load_as_snap / load_as_networkx` | **Critical** | Correctness | NetInf-inferred directed graph is loaded as undirected; all centrality metrics lose directional semantics |
| 3 | `recommender/enhanced.py :: run_network_evaluation` | **Major** | Reproducibility | Network sampling uses `np.random.choice` without a fixed seed; results are not reproducible across runs |
| 4 | `networks/inference.py :: infer_networks` (powerlaw branch) | **Major** | Correctness | Power-law exponent grid `[1.0, 5.0]` is dataset-independent and hardcoded; may miss the relevant regime for any given dataset |
| 5 | `recommender/enhanced.py :: search_enhanced_params` | **Major** | Evaluation | Hyperparameters tuned on a single representative network and applied to all others; transfers poorly when network density varies |
| 6 | `analysis/shap_analysis.py :: compute_shap_for_network` | **Major** | Validity | GBT surrogate is fitted on the full user set with no held-out validation; SHAP values are unreliable when `n_users < ~50` |
| 7 | `recommender/data.py :: evaluate_single_split` | **Major** | Evaluation | Only RMSE, MAE, R² are reported; no ranking metrics (NDCG@K, Precision@K, MRR) despite the task being a recommendation problem |
| 8 | `recommender/enhanced.py :: evaluate_cmf_with_user_attributes` | **Minor** | Correctness | `Normalizer` in `_SCALERS` normalises rows (samples), not features; semantically incorrect when used as a feature scaler |
| 9 | `networks/inference.py :: infer_networks` | **Minor** | Robustness | No minimum-edge guard before centrality computation; all-zero feature files are silently generated for degenerate networks |
| 10 | `networks/centrality.py :: pagerank_custom_beta` | **Minor** | Correctness | Docstring and equation reference a directed formulation, but `G` is always an undirected NetworkX graph (from `load_as_networkx`); the "out-degree" variable is actually the undirected degree |
| 11 | `networks/communities.py :: save_community_results` | **Minor** | Feature Quality | `community_ids` (semicolon-separated membership list) is persisted to CSV but never included in the feature matrix; community identity is entirely lost |
| 12 | `analysis/shap_analysis.py` (module-level constants) | **Minor** | Correctness | `_ENHANCED_PARAMS_PATH`, `_SHAP_RESULTS_PATH`, `_SHAP_MATRICES_DIR` default to `Datasets.DEFAULT` (MovieLens); calling module-level functions without explicit `path=` overrides would silently use wrong dataset paths |
| 13 | `recommender/baseline.py :: search_best_params` | Minor | Correctness | Legacy `search_best_params` uses random search (`scipy.stats.randint`) with only 20 iterations by default; the newer `search_baseline_params` (Optuna, 50 trials) is used in the pipeline but the old function remains and could be called accidentally |
| 14 | `networks/cascades.py :: generate_cascades_from_df` | Suggestion | Correctness | `num_items` counts only training items; if a user ID in the cascade header coincidentally equals an item offset due to different train sizes, the node namespace assumption breaks on future dataset variants |
| 15 | `datasets/` (trust files unused) | Suggestion | Feature Quality | Ciao and Epinions include explicit trust graphs (`trust.txt`) that are never loaded; cross-network features between the inferred graph and the explicit trust graph are a significant missed signal |

---

## Detailed Findings

### Category 1: Data Leakage

#### Issue 1 — Random split used instead of temporal split
**Location**: `config.py :: Split`, `pipeline.py :: _run_cascade`, `recommender/data.py :: load_and_split_dataset`  
**Severity**: Critical  
**Description**: `train_test_split(df, test_size=0.2, random_state=42)` is a random shuffle over all ratings, ignoring the `timestamp` column. For a cascade-based pipeline this means:

1. A user's test interactions can have earlier timestamps than some of their training interactions.  
2. The cascade file for item `i` may include a user rated it at time `t1` (training), while another user rated the same item at `t0 < t1` but that rating is in the test set. NetInf never sees that earlier event, so it infers the wrong influence direction.  
3. The inferred network is not guaranteed to reflect only the causal past of the training period.

In practice, with a large enough dataset the bias averages out, but for sparse users (≤5 ratings) the effect is substantial. This also means the evaluation does not simulate a realistic deployment scenario where the model is tested on future ratings.

**Fix**: Replace the random split with a temporal split. Sort by timestamp, use the last `TEST_SIZE` fraction of each user's interactions (or a global time cutoff) as the test set. Update `config.py :: Split` to add a `STRATEGY = "temporal"` option and modify `split_data_single` accordingly:

```python
def split_data_temporal(
    data: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.sort_values("timestamp")
    cutoff = int(len(data) * (1 - test_size))
    return data.iloc[:cutoff].copy(), data.iloc[cutoff:].copy()
```

---

#### Issue 6 — SHAP surrogate fitted on full user set without validation
**Location**: `analysis/shap_analysis.py :: compute_shap_for_network`  
**Severity**: Major  
**Description**: The GBT surrogate is trained on `(X, y)` where `X` is scaled centrality features and `y` is the mean CMF-predicted rating for each test user, with no held-out split. When `n_users` is small (the function accepts as few as `min_users=10`), the surrogate overfits trivially. SHAP values derived from an overfitted surrogate reflect noise rather than the CMF's systematic feature sensitivities.  
**Fix**: Add a surrogate quality guard. Fit on 80% of the common_users, evaluate on the remaining 20%, and return `None` (skip this network) when surrogate R² on the held-out split is below a threshold (e.g., 0.1). Log the surrogate R² alongside the network index. Also raise `min_users` to at least 30:

```python
if len(common_users) < 30:
    return None
split = int(0.8 * len(common_users))
surrogate.fit(X[:split], y[:split])
surrogate_r2 = r2_score(y[split:], surrogate.predict(X[split:]))
if surrogate_r2 < 0.05:
    print(f"  surrogate R²={surrogate_r2:.3f} too low — skipping.")
    return None
# Refit on full data for SHAP
surrogate.fit(X, y)
```

---

### Category 2: Methodological Correctness

#### Issue 2 — Directed network loaded as undirected
**Location**: `networks/network_io.py :: load_as_snap`, `load_as_networkx`  
**Severity**: Critical  
**Description**: NetInf infers a **directed** influence graph (edge `u → v` means "u tends to influence v"). Both loaders create undirected graphs (`snap.TUNGraph`, `nx.Graph`). This discards all directional information and changes the semantics of every derived metric:

- **PageRank** on an undirected graph is symmetric and does not distinguish high-influence senders from high-influence receivers.  
- **Betweenness centrality** uses shortest undirected paths, not directed reachability.  
- **`pagerank_custom_beta`** references out-degrees explicitly in its formulation, but the adjacency matrix `A` it receives is symmetric (from an `nx.Graph`), so the "out-degree" is actually the undirected degree.  
- **Closeness centrality** on a disconnected directed graph differs substantially from the undirected version.

**Fix**: Change both loaders to use directed graph types:

```python
# network_io.py
G = snap.TNGraph.New()           # directed SNAP graph
G = nx.DiGraph()                 # directed NetworkX graph
```

Additionally update `pagerank_custom_beta` to use `A.sum(axis=1)` for out-degree and `A.sum(axis=0)` for in-degree (accessible once G is directed), and add an `in_degree` feature to the centrality CSV. Community detection algorithms in `cdlib` accept directed graphs when using compatible algorithms.

---

#### Issue 4 — Power-law exponent grid is dataset-independent
**Location**: `networks/inference.py :: infer_networks` (model == 1 branch)  
**Severity**: Major  
**Description**: The power-law model uses a fixed linear sweep `np.linspace(1.0, 5.0, n)`. Unlike the exponential and Rayleigh models, this does not adapt to the median inter-event delta of the dataset. A dataset with very different temporal dynamics (e.g., Epinions with coarser timestamps vs. MovieLens with second-resolution Unix times) might have its meaningful exponent range entirely outside [1, 5].  
**Fix**: Use the median delta to anchor the power-law grid similarly to exponential/Rayleigh. The power-law decay $f(t) = (1 + \alpha t)^{-\beta}$ has its median at approximately $(2^{1/\beta} - 1)/\alpha$. Alternatively, expose the range as a configurable parameter:

```python
# config.py :: Defaults
POWERLAW_ALPHA_RANGE = (1.0, 5.0)  # (min, max) exponent for power-law sweep
```

And make it dataset-discoverable via the delta computation.

---

#### Issue 10 — `pagerank_custom_beta` docstring/implementation mismatch
**Location**: `networks/centrality.py :: pagerank_custom_beta`, `compute_pagerank_lph`  
**Severity**: Minor  
**Description**: The docstring states "G: Directed NetworkX graph" and the formula references $k_j^{out}$ (out-degree of node j). However, `compute_pagerank_lph` calls `load_as_networkx()` which returns an `nx.Graph()` (undirected). The `A.sum(axis=1)` then gives the undirected degree, not the out-degree. The computation is internally consistent for an undirected graph but misrepresents the Newman Eq. 7.18 formulation it claims to implement.  
**Fix**: Once Issue 2 is resolved (directed graphs), this is automatically corrected. Until then, update the docstring to state "Undirected NetworkX graph (directed formulation pending)."

---

#### Issue 8 — `Normalizer` mis-semantics in `_SCALERS`
**Location**: `recommender/enhanced.py :: _SCALERS`, `analysis/shap_analysis.py :: _SCALERS`  
**Severity**: Minor  
**Description**: `sklearn.preprocessing.Normalizer` normalises each **sample** (row) to unit norm, not each **feature** (column). When used as a feature scaler, it scales differently depending on how many features a user has non-zero values in, not on the distribution of each feature. This is unlikely to be the intended behaviour when the user selects `--transform normalizer`.  
**Fix**: Either remove `Normalizer` from `_SCALERS` or add a clear docstring warning, and prefer `StandardScaler` or `MinMaxScaler` as recommended options.

---

### Category 3: Evaluation Validity

#### Issue 7 — No ranking metrics
**Location**: `recommender/data.py :: evaluate_single_split`, throughout the evaluation pipeline  
**Severity**: Major  
**Description**: The pipeline reports only RMSE, MAE, and R². These are rating-prediction metrics, not recommendation-quality metrics. A model that accurately predicts a 3-star rating for a good item that the user would have rated 5 stars may still recommend it correctly; conversely a model with low RMSE can produce poor rankings. For a recommender evaluation, the community expects at minimum:

- **NDCG@K** (normalised discounted cumulative gain)  
- **Precision@K** and **Recall@K**  
- **MRR** (mean reciprocal rank)  

Without these, the paper cannot make claims about recommendation quality — only about rating prediction accuracy.  
**Fix**: Add a ranking evaluation function to `recommender/data.py`:

```python
from sklearn.metrics import ndcg_score

def evaluate_ranking(model, test_data: pd.DataFrame, k: int = 10) -> dict[str, float]:
    """Compute NDCG@K, Precision@K and Recall@K over the test set."""
    ...
```

Call it alongside `evaluate_single_split` and log results to MLflow.

---

#### Issue 5 — Hyperparameters tuned on a single representative network
**Location**: `recommender/enhanced.py :: search_enhanced_params`, `pipeline.py :: _run_recommend`  
**Severity**: Major  
**Description**: `search_enhanced_params` is called once using `sample_features` from the **first** available network (`network_index=0`). The resulting `(k, lambda_reg, w_main, w_user)` are then applied to all 100+ networks across all three diffusion models. Network density varies significantly across the alpha grid (sparse at extreme alphas, dense near the centre). Hyperparameters optimal for a moderately dense network may be suboptimal for sparse or near-complete graphs.  
**Fix**: Either (a) tune on a small stratified sample of networks (sparse, medium, dense) and pick the best, or (b) document this as a deliberate design choice in the paper with a note on sensitivity. As a minimum, log which network was used for tuning in MLflow.

---

### Category 4: Implementation Bugs & Inconsistencies

#### Issue 3 — Unseeded network sampling in `run_network_evaluation`
**Location**: `recommender/enhanced.py :: run_network_evaluation` (line ~624)  
**Severity**: Major  
**Description**: Network sampling uses `np.random.choice(indices, sample_networks, replace=False)` without setting a seed. Different runs of the same command will evaluate different networks, making result comparisons meaningless across re-runs.  
**Fix**: Accept a `seed` parameter (consistent with the `--seed` CLI argument already used for SHAP) and pass it:

```python
rng = np.random.default_rng(seed)
sampled = sorted(rng.choice(indices, min(sample_networks, len(indices)), replace=False))
```

---

#### Issue 9 — No minimum-edge guard before centrality/community computation
**Location**: `networks/centrality.py :: calculate_centrality_for_network`, `networks/communities.py :: calculate_communities_for_network`  
**Severity**: Minor  
**Description**: When NetInf infers zero edges for an alpha value, the resulting network file contains only self-loop node declarations. Computing betweenness or eccentricity on an edgeless graph is valid but produces meaningless all-zero outputs. These are silently saved to CSV and later passed as features to the CMF, diluting the side-information signal.  
**Fix**: Add an early-exit guard:

```python
if len(edges) == 0:
    print(f"  Warning: no edges in {network_file.name}, skipping.")
    return False
```

Add this check inside `parse_network_file` or at the top of each compute function.

---

#### Issue 14 — Item-count offset in cascade may break on future dataset variants
**Location**: `networks/cascades.py :: generate_cascades_from_df`  
**Severity**: Suggestion  
**Description**: User IDs in the cascade header are offset by `num_items = interactions["ItemId"].nunique()` (training-set items only). If two datasets have the same number of training items by coincidence, or if a train split produces a different `num_items` than expected, user IDs in cascades from different runs will differ. The offset is only used to separate the item and user namespaces for NetInf; since NetInf does not use item nodes directly, this offset is unnecessary.  
**Fix**: Simplify by using a large fixed offset (e.g., `USER_ID_OFFSET = 1_000_000`) or eliminate the item namespace entirely (NetInf only needs cascade participants, not item IDs).

---

### Category 5: Graph Feature Quality

#### Issue 11 — Community identity not encoded in feature matrix
**Location**: `networks/communities.py :: save_community_results`, `recommender/enhanced.py :: load_network_features`  
**Severity**: Minor  
**Description**: The community CSV stores `community_ids` as a semicolon-separated string (e.g., `"0;3;7"`). This column is excluded from the feature merge in `load_network_features`; only `num_communities`, `local_pluralistic_hom`, and `lph_score` are included. Community identity carries important signal: two users in the same community are more likely to share tastes than two users with the same number of communities but different memberships.  
**Fix**: Encode community membership as a binary or frequency vector. For small community counts (≤50), one-hot encode. For larger counts, use a hash-trick or community2vec embedding. At minimum, include the top-k most popular community IDs as binary features.

---

#### Issue 15 — Explicit trust graphs in Ciao/Epinions are unused
**Location**: `datasets/ciao/trust.txt`, `datasets/epinions/trust.txt` (not read anywhere)  
**Severity**: Suggestion  
**Description**: Both Ciao and Epinions provide explicit user trust/follow graphs. These are never loaded or used. The inferred network captures implicit co-rating influence; the explicit trust network captures stated social trust. Their overlap (or lack thereof) is a strong signal about social influence vs. taste similarity.  
**Fix**: Load the trust graph in a new `networks/social.py` module and compute:

1. **Trust-graph centrality features** (degree in trust graph, pagerank in trust graph)  
2. **Graph overlap score**: Jaccard similarity between trust-graph neighbourhood and inferred-network neighbourhood  
3. **Alignment ratio**: fraction of inferred edges that also appear in the trust graph

---

### Category 6: Pipeline Robustness

#### Issue 12 — Module-level dataset path constants default to MovieLens
**Location**: `analysis/shap_analysis.py` lines 65–67  
**Severity**: Minor  
**Description**: The three module-level constants `_ENHANCED_PARAMS_PATH`, `_SHAP_RESULTS_PATH`, `_SHAP_MATRICES_DIR` are evaluated at import time and point to the default dataset (MovieLens). Any function that uses them without an explicit `path=` override (e.g., if called directly from a notebook or test) will silently read/write the wrong dataset's files.  
**Fix**: Remove module-level path constants and derive paths inside each function, or add a module-level `_DATASET` variable that callers can override:

```python
def save_shap_results(results: dict, path: Path | None = None, dataset: str | None = None) -> None:
    dest = path or DatasetPaths(dataset or Datasets.DEFAULT).SHAP_RESULTS
    ...
```

---

## Graph Feature Expansion Recommendations

The current feature set (degree, betweenness, closeness, eigenvector, PageRank, clustering coefficient, eccentricity, num_communities, LPH scores) is a solid foundation. The following additions are ranked by expected signal-to-cost ratio.

---

### 1. In-Degree and Out-Degree (Separate)
**What it captures**: NetInf's directed edges distinguish "influence sources" (high out-degree: users whose ratings tend to be followed) from "influence sinks" (high in-degree: users who adopt others' preferences). The current degree metric merges both.  
**Why it may improve CMF**: Users with high out-degree are likely taste-makers whose ratings should be weighted differently than passive followers. The CMF's user embeddings can absorb this signal.  
**How to compute**: Once directed loading is restored (Issue 2), trivially:
```python
in_deg = {n: G.in_degree(n) for n in G.nodes()}
out_deg = {n: G.out_degree(n) for n in G.nodes()}
```
**Caveats**: Depends on fixing Issue 2 (directed graph loading). Zero cost beyond that.

---

### 2. K-Core Decomposition (Coreness)
**What it captures**: A node's k-core number (max k such that the node is in the k-core subgraph) measures its structural embeddedness — how deeply it is embedded in the dense core of the network.  
**Why it may improve CMF**: Core users tend to be highly connected active raters whose preferences propagate widely; periphery users may have niche tastes. Coreness provides a single scalar that captures this in a way that degree alone does not.  
**How to compute**:
```python
import networkx as nx
coreness = nx.core_number(G)  # returns {node: k-core number}
```
**Caveats**: Defined for undirected graphs. For directed graphs, use `nx.algorithms.core.onion_layers` or the weak-component k-core. O(m + n) time.

---

### 3. Structural Holes (Burt's Constraint Index)
**What it captures**: Burt's constraint C(i) ∈ [0, 1] measures how much a node's neighbours are interconnected with each other. High constraint = few structural holes (node is embedded in a clique). Low constraint = many structural holes (node bridges otherwise disconnected groups).  
**Why it may improve CMF**: Boundary-spanners with low constraint are often early adopters who are exposed to diverse items; highly-constrained users tend to rate within tight taste communities. This is complementary to the existing LPH scores.  
**How to compute**:
```python
import networkx as nx
constraint = nx.constraint(G)  # returns {node: constraint value}
```
**Caveats**: O(n * d²) where d is mean degree. Can be expensive on dense networks. Use only for networks with mean degree ≤ 100.

---

### 4. Personalized PageRank Vectors (as dense side info)
**What it captures**: The personalised PageRank (PPR) vector for user u, `ppr_u[v]`, measures how likely a random walk starting at u visits v. It encodes the "taste neighbourhood" of each user in the graph.  
**Why it may improve CMF**: Two users with similar PPR vectors are structurally similar — they have influence pathways to the same set of users. Passing a compressed version (via SVD) of PPR as user features gives the CMF access to latent social position.  
**How to compute**:
```python
ppr_matrix = []
for u in sorted(G.nodes()):
    ppr_u = nx.pagerank(G, personalization={u: 1.0})
    ppr_matrix.append([ppr_u.get(v, 0.0) for v in sorted(G.nodes())])
# Compress with TruncatedSVD
from sklearn.decomposition import TruncatedSVD
ppr_svd = TruncatedSVD(n_components=10).fit_transform(ppr_matrix)
```
**Caveats**: O(n × iter × m) for full PPR; feasible only for networks with ≤ 10,000 nodes. Use approximate PPR (ANDERSEN et al. 2006) for larger networks. SVD dimension is a new hyperparameter.

---

### 5. Cascade Depth and Breadth per User
**What it captures**: For each user u, the maximum depth (longest chain in which u participates, measured as cascade position relative to the seed) and breadth (number of distinct items in which u appears in the cascade) capture temporal influence patterns.  
**Why it may improve CMF**: Users who consistently appear early in cascades (low cascade depth, close to the seed) are influence sources; users who appear late are followers. This is a direct temporal-diffusion signal that the static graph metrics do not capture.  
**How to compute**: Parse `cascades.txt` directly. For each cascade (item), find user u's position (1-indexed rank in sorted timestamp order). Aggregate across items: `mean_position`, `min_position`, `cascade_breadth`.  
**Caveats**: Requires access to the cascade file alongside the network file. Cannot be computed after aggregation (must be added as a cascade-step output). Only meaningful for users appearing in ≥5 cascades.

---

### 6. Temporal Activity Entropy
**What it captures**: Shannon entropy of a user's rating timestamp distribution, binned into weekly or monthly buckets. High entropy = ratings spread uniformly over time; low entropy = ratings clustered in a burst.  
**Why it may improve CMF**: Bursty users may be responding to trending items (social contagion) rather than stable personal preferences, making their embeddings harder to generalise. The CMF can down-weight their side information if entropy is correlated with embedding instability.  
**How to compute**:
```python
from scipy.stats import entropy
bins = pd.cut(user_df["timestamp"], bins=52)  # weekly bins over 1 year
counts = user_df.groupby(bins).size()
activity_entropy = entropy(counts + 1)  # +1 Laplace smoothing
```
**Caveats**: Requires the timestamp column to be preserved through to the feature-extraction step (currently discarded after cascade generation). Add a `--save-user-stats` option to the cascade step that writes a `user_temporal_stats.csv`.

---

### 7. Node2Vec Embeddings (Latent Graph Features)
**What it captures**: 16–64 dimensional dense embeddings learned from random walks on the inferred graph. They capture higher-order structural similarity that cannot be expressed as a single scalar metric.  
**Why it may improve CMF**: Node2vec embeddings directly parameterise node proximity in graph space. Passing them as user side info allows the CMF to learn which dimensions of social position are predictive of rating behaviour.  
**How to compute**:
```python
from node2vec import Node2Vec
n2v = Node2Vec(G, dimensions=32, walk_length=30, num_walks=200, workers=4)
model = n2v.fit(window=10, min_count=1, batch_words=4)
embeddings = {int(n): model.wv[str(n)] for n in G.nodes()}
```
**Caveats**: Node2vec embeddings are not deterministic without fixing the random seed. Adds `dimensions` (32–64) new features — the largest expansion in this list. Run PCA/UMAP to check that embeddings capture meaningful variation before including them. Computational cost is O(walks × walk_length × n).

---

## Dataset Suitability Assessment

### Ciao
**Suitability**: ✅ Best choice among the three supported datasets.  
- **Social graph**: Explicit trust graph (`trust.txt`) available. Currently unused (see Issue 15). This provides ground-truth social relationships against which the inferred network can be validated.  
- **Cascade density**: 282,619 ratings from 7,375 users and 105,114 items gives a mean of ~38 ratings/user — sufficient for meaningful cascades. Single-user cascades are already filtered.  
- **Temporal granularity**: Unix timestamps in seconds. However, the Ciao dataset is known to have many ratings recorded with the same day-level timestamp (time stored as midnight of the review day). After converting to days, many cascade entries within the same item will have identical timestamps, producing zero inter-event deltas. `compute_median_delta` already skips zero deltas (`if diff > 0`), but this means the effective cascade size may be much smaller than the raw count.  
- **Rating scale**: 1–5 integer scale. Compatible with CMF's ALS formulation.  
- **Recommendation**: Use Ciao as the primary dataset. Load the trust graph and compute cross-network overlap features (Issue 15).

### MovieLens (ratings_small.csv)
**Suitability**: ⚠️ Suboptimal for this specific pipeline.  
- **Social graph**: No explicit social graph available. The inferred network is the only social signal. Without ground-truth social edges, there is no way to validate whether NetInf's inference is sensible.  
- **Cascade density**: The "small" version contains ~100,000 ratings from 671 users over 9,066 movies, giving ~149 ratings/user — dense enough for cascades.  
- **Temporal granularity**: Unix timestamps in seconds from 1996–2016. Good temporal resolution for NetInf.  
- **Rating scale**: 0.5–5.0 half-star scale (10 distinct values). Compatible with CMF.  
- **Recommendation**: Suitable for demonstrating the pipeline mechanics but should not be the primary dataset for a paper claiming social-network effects. The absence of a social graph makes it impossible to distinguish "co-rating influence" from "shared taste" as explanations for the inferred edges. Use as a secondary dataset to show that the pipeline can be applied to rating-only data.

### Epinions
**Suitability**: ✅ Strong alternative to Ciao with a denser trust network.  
- **Social graph**: Explicit trust/distrust graph (`trust.txt`). Epinions has ~49,000 unique users and ~140,000 trust edges — substantially denser than Ciao's trust graph.  
- **Cascade density**: ~664,824 ratings from 49,290 users — approximately 13 ratings/user. This is sparse; many users have fewer than 5 ratings, which means most of their cascades will be single-user and filtered. **Flag**: run a pre-check: `(df.groupby("UserId").size() >= 2).mean()` to estimate what fraction of users survive the cascade filter.  
- **Temporal granularity**: Unix timestamps in seconds. Good.  
- **Rating scale**: 1–5 star scale. Compatible with CMF.  
- **Recommendation**: Include as a second primary dataset (alongside Ciao). The denser trust graph makes it a better stress-test for the cross-network alignment features.

### General Timestamp Warning
Both Ciao and Epinions are known to have review timestamps recorded at day resolution (time component = midnight) despite being stored as Unix seconds. This collapses many events within the same cascade to identical timestamps, producing zero deltas. The `compute_median_delta` function skips zero-delta pairs, but if a majority of cascade pairs are zero-delta, the median will be dominated by the smallest non-zero delta (which may be 1 day). Verify by running:

```python
delta = compute_median_delta(dp.CASCADES)
# Check what fraction of deltas are zero before filtering
```

If > 50% of deltas are zero, the cascade structure is degenerate for NetInf's decay model and the inferred network will be low-quality regardless of alpha tuning.

### Recommended Datasets Not Currently Supported

| Dataset | Reason to consider |
|---------|-------------------|
| **Last.fm (HetRec)** | Rich per-song play timestamps at minute resolution; explicit friend graph; ~2M plays × 1,892 users. Ideal temporal granularity for NetInf. |
| **Amazon Review (Fine Foods / Movies)** | Very large, rich timestamps (day level), verified purchases provide temporal ordering. No explicit social graph but review helpfulness votes can serve as a proxy. |
| **Yelp Social Network** | Explicit friend graph + fine-grained review timestamps. Ratings 1–5. ~6M reviews. Strong social signal. |

---

## Prioritized Action Plan

### Must-Fix Before Publication

1. **[Issue 2] Switch to directed graph loading** — Change `snap.TUNGraph.New()` → `snap.TNGraph.New()` and `nx.Graph()` → `nx.DiGraph()` in `network_io.py`. Update all downstream consumers to handle directed graphs. This is a single-file change but touches centrality, community detection, and the LPH-PageRank computation. Estimated effort: 2–3 hours.

2. **[Issue 1] Implement temporal train/test split** — Add `split_data_temporal()` to `recommender/data.py` and make it the default. Update `_run_cascade()` to also use a temporal split. This ensures cascade interactions are always "past" and test interactions are always "future." Estimated effort: 2–4 hours.

3. **[Issue 7] Add NDCG@K and Precision@K metrics** — Extend `evaluate_single_split` to call a new `evaluate_ranking()` helper. Report top-5 and top-10 metrics. Log to MLflow. Estimated effort: 1–2 hours.

### Should-Fix for Reproducibility

4. **[Issue 3] Fix unseeded network sampling** — Pass `seed` to `run_network_evaluation` and use `np.random.default_rng(seed)` for all sampling. 30-minute fix.

5. **[Issue 6] Add surrogate quality gate in SHAP** — Require surrogate R² ≥ 0.05 on a held-out split before accepting SHAP values. Raise `min_users` to 30. 1-hour fix.

### Improvement for Research Quality

6. **[Issue 15] Load and use explicit trust graphs** — Add `networks/social.py` to compute cross-network features for Ciao and Epinions. Document the inferred-vs-explicit distinction in the paper.

7. **[Graph expansion] Add in/out-degree, k-core, and cascade position features** — These have low implementation cost and directly address the directed-graph gap. Together they add ≈4 new features per user.

8. **[Issue 11] Encode community identity** — Include the top-20 community memberships as binary features (or use node2vec on the community graph).

9. **[Issue 5] Stratified hyperparameter tuning** — Tune on a sparse + medium + dense representative network trio to improve transfer to the full alpha grid.

### Low Priority

10. **[Issue 8]** Remove `Normalizer` from `_SCALERS` or document its row-normalisation semantics.  
11. **[Issue 9]** Add minimum-edge guard (≥5 edges) before centrality computation.  
12. **[Issue 4]** Make the power-law exponent range data-driven.  
13. **[Issue 12]** Remove module-level dataset path constants from `shap_analysis.py`.

---

## Research Contribution Notes

### Strengthening the Experimental Design

1. **Ablation study**: Report results for each feature group independently (centrality-only, community-only, LPH-only, all-features) rather than only a combined model. This isolates which graph-structural signal drives the improvement and significantly strengthens the contribution's interpretability.

2. **Statistical significance**: The current pipeline reports mean RMSE across random CV splits but does not perform a statistical test. Add a Wilcoxon signed-rank test comparing per-fold RMSE of baseline vs. enhanced CMF across all sampled networks. A p-value < 0.05 is required for publication.

3. **Network quality vs. performance correlation**: Plot the relationship between network density (edges/nodes), graph cohesion metrics, and RMSE improvement. This tests whether "better-inferred" networks produce more useful features — a direct contribution to the research question.

4. **Sensitivity to alpha**: The RMSE is already tracked per-alpha-index via `inferred_edges_<short>.csv`. Plot RMSE improvement as a function of the inferred edge count. If the improvement peaks at a specific network density, document this as a design recommendation.

5. **Cold-start scenario**: Users with few ratings are exactly those for whom the side information is most valuable (the CMF latent factor is unreliable for them). Add a cold-start sub-evaluation that restricts the test set to users with ≤5 training interactions and compare baseline vs. enhanced RMSE separately. This is where the paper's core claim is most defensible.

6. **Cross-dataset transferability**: If the optimal alpha (or the RMSE improvement pattern) is consistent across Ciao and Epinions, this strongly suggests the result generalises. Report this explicitly.

### Novelty Positioning

The MAFPIN pipeline's most defensible novelty claim is the integration of **overlapping community structure** (LPH, h̃v score from Barraza et al. 2025) with CMF via a principled user-side information matrix. To strengthen this:

- Clearly differentiate from prior work that uses community labels (non-overlapping) as side info (e.g., Ma et al. 2011, Yang et al. 2012).  
- Show that the LPH-weighted PageRank (`pagerank_lph`) outperforms standard PageRank in the SHAP importance ranking — this would validate the LPH contribution.  
- Include the trust graph (when available) as an additional baseline: "inferred network features vs. explicit trust network features vs. both." If the inferred features provide additional value beyond trust-graph features, that is the key finding.
