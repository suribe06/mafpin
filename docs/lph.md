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

## Visualisation

Community-level and LPH plots are available in `visualization/community_plots.py`:

- `plot_lph_distribution` — histogram of LPH values across nodes and models.
- `plot_alpha_vs_lph` — how mean LPH evolves across the alpha grid.
- `plot_lph_vs_centrality` — scatter of LPH against each centrality metric.
- `plot_community_correlation_heatmap` — correlation between community features and centrality.
