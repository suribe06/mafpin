# Local Pluralistic Homophily (LPH)

---

## Definition

**Local Pluralistic Homophily (LPH)** measures how similar a node's community memberships are to those of its immediate neighbours.  
It quantifies the degree to which nodes that are linked to each other also belong to the same overlapping communities.

Formally, for a node $v$ in graph $G = (V, E)$:

$$\text{LPH}(v) = \frac{1}{|N(v)|} \sum_{u \in N(v)} J\!\left(\mathcal{C}(v),\, \mathcal{C}(u)\right)$$

where:

- $N(v)$ is the set of immediate neighbours of $v$.
- $\mathcal{C}(v)$ is the set of communities that $v$ belongs to.
- $J(A, B)$ is the **Jaccard similarity** between sets $A$ and $B$.

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

$\text{LPH}(v) \in [0, 1]$ for all $v$.

---

## Interpretation

| LPH value | Interpretation |
| --- | --- |
| 1.0 | Node and all its neighbours belong to exactly the same communities. |
| 0.5 | Moderate overlap — node shares about half its communities with each neighbour on average. |
| 0.0 | Node shares no community membership with any neighbour. |

A **high LPH** indicates strong community cohesion around a node; nodes with high LPH act as community hubs.  
A **low LPH** indicates a node sits at the boundary between distinct communities (potential broker or bridge node).

---

## Community Detection: Demon Algorithm

LPH is computed on top of the overlapping communities found by **Demon** (Democratic Estimate of the Modular Organisation of a Network).

### How Demon Works

1. For each node $v$, extract the **ego-network** $G_v$ (the induced subgraph of $v$'s neighbours, without $v$).
2. Apply a label-propagation algorithm on $G_v$ to detect local communities.
3. Add $v$ back — each local community becomes a candidate community that includes $v$.
4. **Merge** candidate communities across all ego-networks using a Jaccard-based merging threshold $\epsilon$:
   - Two communities $C_a$ and $C_b$ are merged if $J(C_a, C_b) \geq \epsilon$.
5. Discard communities smaller than `min_community`.

### Key Parameters

| Parameter | Default | Effect |
| --- | --- | --- |
| `epsilon` | 0.25 | Merge threshold — lower ⇒ more aggressive merging, fewer communities |
| `min_community` | 3 | Minimum community size to retain |

Demon is implemented in `cdlib.algorithms.demon`.

### Why Demon for LPH?

Demon produces **overlapping** communities, meaning a node can belong to multiple communities simultaneously.  
This is essential for LPH: if every node belonged to exactly one community, the Jaccard similarity would reduce to a binary same/different-community indicator and LPH would lose most of its nuance.

---

## Alternative: ASLPAw

As a complement, **ASLPAw** (Asymmetric Speaker-listener Label Propagation Algorithm) is also supported.  
ASLPAw is faster and produces softer, probabilistic community assignments. It is available via `cdlib.algorithms.aslpaw`.

---

## Implementation

LPH is computed in `networks/communities.py`:

```python
from networks.communities import compute_local_pluralistic_homophily

G = ...            # networkx graph
membership = ...   # dict[node_id, frozenset[int]]  — community sets per node
lph_scores = compute_local_pluralistic_homophily(G, membership)
# lph_scores: dict[node_id, float]
```

Results are stored in the communities CSV alongside the community count per node.

---

## Normalized Local Pluralistic Homophily (h̃v) — Paper Metric

The Jaccard-based LPH above measures raw similarity in the neighborhood, but its
values can be dominated by degree or global variance effects, making direct node
comparison unreliable. Barraza et al. (2025) propose a normalized, neighborhood-centered
score **h̃v** that surfaces membership-based boundary positions orthogonally to topology.

### Key idea

A **boundary node** is not characterized by its deviation from a graph-wide average,
but by how its community profile **differs from its local neighborhood**. h̃v captures
this local misalignment. Large negative h̃v marks nodes whose membership profile
diverges from those of their neighbors → boundary-spanning candidates.

### Step-by-step computation (Algorithm 1)

#### Step 1 — Neighborhood alignment s(v)

For each node v, count how many of its own communities are represented at least
once in its neighborhood:

$$s(v) = \left|\bigcup_{i \in N(v)} \left(\mathcal{C}(v) \cap \mathcal{C}(i)\right)\right|$$

*Interpretation:* s(v) measures which portions of v's community identity are present
among its immediate neighbors. No weighting is applied.

#### Step 2 — Network-level pluralistic homophily h

Compute the Pearson-style edge assortativity using s(·) as node attribute
(equivalent form, Supplementary S1):

$$h = \frac{\sum_{(i,j)\in E}(s(i)-\mu_q)(s(j)-\mu_q)}{\sum_{(i,j)\in E}(s(i)-\mu_q)^2}$$

where $\mu_q = \tfrac{1}{2M}\sum_{(i,j)\in E}(s(i)+s(j))$ is the degree-weighted mean.

#### Step 3 — Local dissimilarity δv

$$\delta_v = \frac{1}{d_v}\sum_{i \in N(v)} |s(v) - s(i)|, \qquad \delta_v = 0 \text{ if } d_v = 0$$

This is the mean absolute difference between v's alignment score and those of its
neighbors.

#### Step 4 — Scaling factor λ

$$\lambda = \frac{h + \sum_{u \in V}\delta_u}{N}$$

#### Step 5 — Local score h̃v

$$\tilde{h}_v = \lambda - \delta_v$$

### Properties

| Property | Value |
| --- | --- |
| Range | $(-\infty, \lambda]$ — can be negative |
| Sum | $\sum_v \tilde{h}_v = h$ (global-local consistency) |
| Complexity | $O(M \cdot \bar{s})$ end-to-end, near-linear in practice |
| Boundary signal | Most negative $\tilde{h}_v$ → strongest boundary-spanning position |

### Boundary Signal Interpretation

| h̃v sign | Meaning |
| --- | --- |
| $\tilde{h}_v > 0$ | **Assortative** — v and neighbors share similar community membership patterns |
| $\tilde{h}_v \approx 0$ | Neutral neighborhood alignment |
| $\tilde{h}_v < 0$ | **Disassortative** — v's membership profile differ from its neighbors → boundary position |

Nodes ranked lowest by h̃v are prime candidates for boundary-spanning roles:
removing them in ascending h̃v order depletes inter-community coupling faster than
random removal while preserving global connectivity.

### Comparison with Jaccard-LPH

| Aspect | Jaccard LPH | h̃v |
| --- | --- | --- |
| Range | [0, 1] | $(-\infty, \lambda]$ |
| Global reference | None | Network-level assortativity h |
| Boundary signal | Low value | Strongly negative value |
| Degree bias | Present | Corrected via δv and λ |

---

## Implementation of h̃v

`compute_lph_paper(G, membership)` in `networks/communities.py` implements Algorithm 1
and returns a `dict[node_id, float]` of h̃v scores. The result is stored in the
`lph_score` column of the communities CSV and loaded into the user-side attribute
matrix by `recommender/enhanced.py`.

```python
from networks.communities import compute_lph_paper

G = ...            # networkx graph
membership = ...   # dict[node_id, set[int]]  — community-index sets per node
htilde = compute_lph_paper(G, membership)
# htilde: dict[node_id, float]  — boundary score; most negative = strongest boundary
```

---

## Visualisation

Community-level and LPH plots are available in `visualization/community_plots.py`:

- `plot_lph_distribution` — histogram of LPH values across nodes and models.
- `plot_alpha_vs_lph` — how mean LPH evolves across the alpha grid.
- `plot_lph_vs_centrality` — scatter of LPH against each centrality metric.
- `plot_community_correlation_heatmap` — correlation between community features and centrality.
