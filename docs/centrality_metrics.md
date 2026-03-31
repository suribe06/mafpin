# Centrality Metrics

Seven per-node centrality metrics are computed using **SNAP-py** for each inferred diffusion network.

---

## Metrics

### Degree Centrality

$$C_D(v) = \frac{\deg(v)}{|V| - 1}$$

Proportion of network nodes that $v$ is directly connected to.  
High values indicate **hubs** — highly connected nodes that may spread influence quickly.

---

### Betweenness Centrality

$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Fraction of all shortest paths in the graph that pass through $v$, where $\sigma_{st}$ is the total number of shortest paths from $s$ to $t$ and $\sigma_{st}(v)$ is the count that go through $v$.  
High betweenness indicates **brokers** — nodes that control information flow between communities.

---

### Closeness Centrality

$$C_C(v) = \frac{|V| - 1}{\sum_{u \neq v} d(v, u)}$$

Reciprocal of the average shortest path distance from $v$ to all other nodes.  
High closeness indicates nodes that can **reach the rest of the network quickly**.

---

### Eigenvector Centrality

$$C_E(v) = \frac{1}{\lambda} \sum_{u \in N(v)} C_E(u)$$

Importance of $v$ weighted by the importance of its neighbours (dominant eigenvector of the adjacency matrix).  
High eigenvector centrality indicates nodes connected to other **influential** nodes.

---

### PageRank

$$PR(v) = \frac{1 - d}{|V|} + d \sum_{u \in N^-(v)} \frac{PR(u)}{|N^+(u)|}$$

Random-walk stationary distribution where $d$ is the damping factor (default 0.85), $N^-(v)$ is the set of in-neighbours, and $|N^+(u)|$ is the out-degree of $u$.  
PageRank captures **global prestige** of a node — how frequently a random surfer would visit it.

---

### Clustering Coefficient

$$C_{cl}(v) = \frac{2\,|\{e_{uw} : u, w \in N(v), e_{uw} \in E\}|}{\deg(v)(\deg(v)-1)}$$

Fraction of closed triangles among $v$'s neighbours.  
High clustering indicates nodes embedded in **tightly knit local groups**.

---

### Eccentricity

$$C_{ec}(v) = \max_{u \in V} d(v, u)$$

Largest shortest-path distance from $v$ to any other node.  
Low eccentricity indicates nodes near the **centre of the graph**.

---

## Storage Format

For each network, results are saved as:

```text
data/centrality_metrics/<model>/centrality_metrics_<short>_<id>.csv
```

Columns: `user_id`, `degree`, `betweenness`, `closeness`, `eigenvector`, `pagerank`, `clustering`, `eccentricity`.

---

## Usage

```python
from networks.centrality import calculate_centrality_for_all_models
calculate_centrality_for_all_models()
```

Or for a single file:

```python
from networks.centrality import calculate_centrality_for_network
calculate_centrality_for_network("data/inferred_networks/exponential/inferred_edges_expo_001.csv")
```

---

## Visualisation

Per-metric distribution plots are available in `visualization/network_plots.py`:

```python
from visualization.network_plots import plot_all_centrality_distributions
plot_all_centrality_distributions("exponential", "001", save=True)
```
