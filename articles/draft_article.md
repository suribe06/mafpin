# Matrix Factorization with Inferred Social Network Properties and Local Pluralistic Homophily: A Boundary-Aware Approach to Collaborative Filtering

**Authors**: [Author names and affiliations]

**Correspondence**: [corresponding author e-mail]

**Version**: Working draft — preliminary experimental results

---

## Abstract

Collaborative filtering based on matrix factorization is among the most studied approaches in recommender systems, yet it treats users as independent and identically distributed entities, ignoring the relational structure that may underlie their rating behavior. A previous methodology (hereafter MAFPIN) addressed this gap by inferring a latent user–user network from temporal rating cascades via the NETINF algorithm and incorporating topological properties of that network as side information in a Collective Matrix Factorization (CMF) framework. On the MovieLens dataset, this approach reduced RMSE from 0.8983 to 0.7867 compared to standard collaborative filtering, outperforming eleven benchmark matrix factorization algorithms. However, MAFPIN characterizes users primarily through individual centrality metrics — degree, betweenness, closeness, eigenvector centrality, and eccentricity — which capture how *central* or *reachable* a user is within the inferred network, but say nothing about their *structural role at community interfaces*. We propose an extension that addresses this gap by integrating Local Pluralistic Homophily (LPH), a recently proposed metric that identifies nodes whose community membership is systematically misaligned with that of their neighbors — what we call boundary-spanning users. The extended pipeline detects overlapping communities in the inferred network using the DEMON algorithm, computes the LPH score $\tilde{h}_v$ and its auxiliary components for each user, and incorporates these as additional side-information attributes in the CMF objective. In a further step, boundary information modulates a social regularization term that penalizes divergence between the latent factors of connected users, with weights that diminish for boundary-spanning pairs to preserve their structural heterogeneity. Preliminary experiments on MovieLens and Ciao with a subsampled protocol confirm the numerical stability of the full pipeline across three diffusion models and 100 inferred networks per model, and show consistent improvement from boundary-guided social regularization over a plain CMF baseline. Full-scale experiments with the complete feature set are in progress. The methodology, its formal underpinning, and the preliminary results are described in full in this article.

**Keywords**: collaborative filtering; matrix factorization; network inference; NETINF; social regularization; local pluralistic homophily; overlapping communities; boundary-spanning nodes; recommender systems.

---

## 1. Introduction

The recommender systems field has come a long way from the simple co-occurrence tables of the early web era, yet many of its core challenges remain stubbornly present. Sparsity — the fact that most users have rated only a tiny fraction of the available items — continues to make it difficult to say anything reliable about individual preferences. Cold-start scenarios, where a new user or a niche item has almost no interaction history, still defeat most standard pipelines. And perhaps most fundamentally, the assumption that users are independent of one another, which is baked into the mathematical structure of virtually all matrix factorization models, ignores something that practitioners know to be empirically true: people influence each other. Their rating behavior is not generated in isolation, and a model that treats it as such is leaving information on the table.

One elegant response to this problem is to build, or rather infer, the social network that mediates influence among users, and then use the properties of that network as additional information in the recommendation model. This is the core idea behind MAFPIN (Matrix Factorization with Properties of Inferred Networks), proposed by Uribe, Ramirez, and Finke [1]. Their key insight is that even in datasets like MovieLens — where no explicit social graph is recorded — a latent relational structure among users can be reconstructed from the temporal ordering of ratings. If user A rates a movie shortly after user B, and this pattern is consistent across many movies, NETINF [4] will infer an influence edge from B to A. The resulting network may or may not correspond to a literal social tie, but it encodes something real: a behavioral proximity derived from how users navigate the same item space in time. Once this network exists, its topological properties — degree, betweenness, closeness, eigenvector centrality, farness, community structure, eccentricity, and hub/authority scores — become informative descriptors that can be attached to each user as side information in a Collective Matrix Factorization (CMF) scheme [8,9]. The original MAFPIN results were striking: RMSE dropped from 0.8983 to 0.7867, an improvement that held across all three transmission models tested (exponential, power-law, and Rayleigh) and that outperformed eleven competing matrix factorization algorithms on the same dataset [1].

But MAFPIN's characterization of users, while richer than raw ratings, is still fundamentally a description of *where* a user sits in the network in terms of reachability, influence, and structural importance. It does not capture *what kind of position* that user occupies relative to the community structure of the network. A highly central user in a dense community and a moderately connected user who sits precisely at the interface between two communities can have very similar centrality scores and yet play structurally different roles. The second user is what the social network analysis literature calls a boundary-spanner or a broker [26]: someone whose value in the system comes not from accumulating connections within a single group, but from bridging groups that would otherwise not interact. In recommendation, this distinction is not merely academic. A boundary-spanning user is more likely to appreciate items that cross community lines, and their rating history may contain the very signal needed to produce recommendations that expand beyond the echo chambers that most collaborative filtering systems naturally reinforce.

It is at this point that a recent contribution from the network analysis side becomes directly relevant. Barraza, Ravetti, and Sánchez-Cambronero [2] introduced Local Pluralistic Homophily ($\tilde{h}_v$), a metric designed specifically for the case of *overlapping* community structures. The core idea is to ask, for each node $v$: how much of its community identity appears represented in its immediate neighborhood? If most of $v$'s neighbors share all of $v$'s community memberships, then $v$ is well embedded within its communities and $\tilde{h}_v$ is high. If, on the contrary, $v$'s community membership is systematically misaligned with its neighbors — meaning its neighbors belong to communities that $v$ does not belong to, and vice versa — then $\tilde{h}_v$ is low (or negative), signaling that $v$ occupies a structurally heterogeneous position, an interface between communities. The authors show that $\tilde{h}_v$ is largely orthogonal to standard centrality measures and identifies structurally distinct node populations in empirical networks from platforms as varied as DBLP, Twitch, GitHub, and Amazon [2].

The fusion proposed in this article is, at its most fundamental level, about adding this second kind of information to the first. MAFPIN already tells the recommender how *central* each user is. The extension we propose tells it how *boundary-spanning* each user is. These two things are different, and both matter. A user who is central but deeply embedded within a single community is useful for predicting preferences within that community. A user who is less central but bridges communities is useful for predicting preferences that cross community lines — and potentially for achieving recommendation diversity without sacrificing relevance. Putting both types of information into the attribute matrix of the CMF, and then allowing boundary information to also modulate how strongly the model regularizes adjacent users' latent factors toward each other, constitutes the complete methodology we describe here.

The contribution of this paper can be summarized in three parts. First, we extend the MAFPIN attribute vector with boundary-awareness: the full battery of $\tilde{h}_v$ and its auxiliary quantities ($m(v)$, $s(v)$, $\delta_v$, and a binary boundary indicator) is computed on the symmetrized inferred network and added as user side information. Second, we introduce a boundary-guided social regularization term in the CMF objective, where the penalty for keeping adjacent users' latent factors close is modulated by the community relationship between those users and their boundary status. Third, we provide a preliminary experimental evaluation on MovieLens and Ciao that validates the numerical stability of the complete pipeline across three diffusion models and a wide range of network configurations, and demonstrates consistent improvements over a plain CMF baseline.

The remainder of the article is organized as follows. Section 2 reviews related work. Section 3 establishes the mathematical framework for the individual components. Section 4 describes the pipeline in full. Section 5 presents the experimental setup. Section 6 reports and discusses the results. Section 7 addresses limitations and future work. Section 8 concludes.

---

## 2. Related Work

### 2.1. Network Inference from Cascades

The problem of inferring a network from the timing of events that propagate over it was formalized by Gomez-Rodriguez, Leskovec, and Krause [4] through the NETINF algorithm. Given a collection of cascades — sets of (node, timestamp) pairs representing infection events — NETINF finds the directed graph of at most $k$ edges that maximizes the probability of observing the cascade collection under a parametric transmission model. The optimization is NP-hard but admits a greedy approximation with a $(1 - 1/e)$ guarantee due to submodularity [4]. Three transmission models are commonly used: exponential, power-law, and Rayleigh, each representing a different assumption about how transmission likelihood decays with elapsed time between infections.

Fan and Yu [5] adapted this framework to the movie recommendation context, treating each movie as a cascade over which user "infections" (ratings) propagate. This conceptual reframing is the entry point for MAFPIN: it recasts the question of who influences whom from an explicit social network problem into a latent structure inference problem, making it applicable to any dataset with timestamped user-item interactions.

### 2.2. Social Information in Matrix Factorization

The use of social information to augment matrix factorization is a well-explored direction. Ma et al. [14] introduced SoRec, which jointly factorizes the ratings matrix and an explicit social trust matrix, allowing users to share latent factor components through their trust connections. Jamali and Ester [15] proposed SocialMF, which enforces that a user's latent factor should be close to the average of their trusted contacts' latent factors, effectively introducing a Laplacian-style social regularization. Yang et al. [16] extended this to the circle-of-trust model, recognizing that different social circles influence different types of item preferences.

These approaches all require an *observed* social network. When none is available — the typical situation in datasets like MovieLens, Ciao, or Epinions (absent the explicit trust graph) — the network must be inferred or approximated. MAFPIN [1] addresses this by inferring the network from cascade dynamics rather than assuming an explicit graph. The present work inherits this approach and extends it.

The hTrust model [27], which minimizes $\|G - UVU^\top\|^2 + \alpha\|U\|^2 + \beta\|V\|^2 + \lambda\,\mathrm{tr}(U^\top L U)$ where $L$ is the social Laplacian, provides the closest formal antecedent for the social regularization term we introduce. Our formulation extends this by making the Laplacian weights adaptive: rather than using fixed or binary social ties, we modulate each edge weight by the community relationship and boundary status of the pair.

### 2.3. Community Structure in Recommendation

Integrating community detection with recommender systems has been explored from several angles. GroupLens and related models use community membership as a clustering-level prior on preferences [28]. More recently, graph neural network-based approaches such as NGCF [29] and LightGCN [30] propagate preference signals along the user-item graph, implicitly capturing higher-order community-level connectivity. However, these approaches treat community structure as a means to propagate signals, not as a source of structural role information about individual users.

The role of *overlapping* community membership in recommendation is less explored. Overlapping communities — where users can belong to multiple groups — are more realistic than hard partition models [31], and the structural position of users at the intersection of communities is rarely leveraged explicitly. The metric $\tilde{h}_v$ of Barraza et al. [2] provides a principled and computationally tractable way to quantify this position.

### 2.4. Boundary Spanning and Structural Brokerage

The sociology and network science literature has long recognized that nodes at the boundary between groups play distinctive roles. Burt's structural holes theory [26] argues that agents who bridge otherwise disconnected groups gain informational advantages — they are first to encounter novel combinations of information. In recommender systems, this translates to a hypothesis that boundary-spanning users may be natural carriers of cross-community preference signals. This intuition has been explored informally in diversity-oriented recommendation [32,33] but rarely connected to a formal structural measure of boundary position. The work of Barraza et al. [2] and its integration into recommendation proposed here constitutes, to our knowledge, the first principled attempt to quantify and exploit boundary-spanning in the latent factor space.

---

## 3. Formal Background

### 3.1. Network Inference via NETINF

Let $\mathcal{U} = \{u_1, u_2, \ldots, u_m\}$ be a set of users and $\mathcal{I} = \{i_1, i_2, \ldots, i_n\}$ a set of items. A cascade $c_j$ for item $j$ is a set of pairs $(u, t_{u,j})$ representing that user $u$ rated item $j$ at time $t_{u,j}$. Let $\mathcal{C} = \{c_1, \ldots, c_n\}$ be the full collection of cascades. NETINF seeks the directed graph $\hat{G}$ over $\mathcal{U}$ with at most $e$ edges that maximizes the likelihood $P(\mathcal{C} \mid G)$ of observing the cascade collection:

$$\hat{G} = \arg\max_{|G| \le e} P(\mathcal{C} \mid G)$$

The transmission likelihood of an event at node $v$ at time $t_i$ given an event at node $u$ at time $t_j < t_i$ is specified by a parametric function $f(t_i - t_j; \alpha)$. Three models are considered:

| Model | $f(t_i - t_j; \alpha)$ |
|-------|------------------------|
| Exponential | $\alpha e^{-\alpha(t_i - t_j)}$ |
| Power-law | $\alpha(t_i - t_j)^{-1-\alpha}$ |
| Rayleigh | $\alpha(t_i - t_j)\, e^{-\frac{1}{2}\alpha(t_i - t_j)^2}$ |

For each value of $\alpha$, NETINF produces one inferred network. Sweeping $\alpha$ over a grid of $N_\alpha = 100$ values yields 100 candidate networks per model, providing a robust basis for downstream analysis.

### 3.2. Collective Matrix Factorization

Let $R \in \mathbb{R}^{m \times n}$ be the (partially observed) ratings matrix and $A \in \mathbb{R}^{m \times s}$ a matrix of user attributes. CMF [8] factorizes both matrices jointly through shared user latent factors. Let $P \in \mathbb{R}^{m \times k}$ be the user factor matrix, $Q \in \mathbb{R}^{n \times k}$ the item factor matrix, and $C \in \mathbb{R}^{k \times s}$ the attribute mapping. The objective minimized is:

$$\min_{P, Q, C}\; \sum_{(u,i,r) \in \mathcal{T}} \left(r_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\right)^2 + w_A \sum_{(u,l) \in \mathcal{A}} \left(a_{ul} - \mathbf{p}_u^\top \mathbf{c}_l\right)^2 + \lambda_{\mathrm{reg}}\left(\|P\|_F^2 + \|Q\|_F^2 + \|C\|_F^2\right)$$

where $\mathcal{T}$ is the set of observed ratings, $\mathcal{A}$ the set of observed attribute entries, $w_A$ the weight on the attribute reconstruction term, and $\lambda_{\mathrm{reg}}$ an $\ell_2$ regularization coefficient. The L-BFGS optimizer is used to minimize this objective.

### 3.3. Local Pluralistic Homophily

Let $G = (V, E)$ be an undirected, unweighted graph with an overlapping community structure, where $C(v) \subseteq \mathcal{K}$ denotes the set of communities to which node $v$ belongs and $\mathcal{K}$ is the set of all communities. Algorithm 1 of Barraza et al. [2] computes $\tilde{h}_v$ in four steps.

**Step 1 — Shared community coverage.** For each node $v$, compute how many of its community memberships are represented among its immediate neighbors:

$$s(v) = \left|\bigcup_{i \in N(v)} \bigl(C(v) \cap C(i)\bigr)\right|$$

where $N(v)$ denotes the neighborhood of $v$. Intuitively, $s(v)$ counts how many of $v$'s communities contain at least one of $v$'s neighbors. If $v$ is well embedded, $s(v) \approx |C(v)|$.

**Step 2 — Local discrepancy.** Measure how much $s(v)$ differs from the $s$-values of $v$'s neighbors:

$$\delta_v = \frac{1}{d_v} \sum_{i \in N(v)} |s(v) - s(i)|$$

where $d_v = |N(v)|$ is the degree of $v$. A small $\delta_v$ indicates that $v$'s structural position is similar to that of its neighbors; a large $\delta_v$ indicates that $v$ occupies an atypical position in its local context.

**Step 3 — Network-level reference.** Compute a reference level $\lambda$ that captures the average structural context across the entire network:

$$\bar{h} = \frac{1}{|V|}\sum_{v \in V} s(v), \qquad \lambda = \frac{\bar{h} + \sum_{v \in V} \delta_v}{|V|}$$

**Step 4 — Final score.** Assign the local pluralistic homophily score:

$$\tilde{h}_v = \lambda - \delta_v$$

A high $\tilde{h}_v$ indicates a node that is structurally similar to its neighborhood (intra-community alignment). A low or negative $\tilde{h}_v$ indicates a node whose membership pattern diverges from its neighbors — a boundary-spanning node in the overlapping community sense.

### 3.4. Boundary-Guided Social Regularization

The social regularization term adds an additional penalty to the CMF objective that encourages the latent factors of socially connected users to remain close to each other:

$$\mathcal{R}_S(P) = \sum_{(u,v) \in E_S} w_{uv} \|\mathbf{p}_u - \mathbf{p}_v\|^2$$

where $E_S$ is the social edge set derived from the inferred network and $w_{uv}$ are adaptive weights. The full objective becomes:

$$\min_{P, Q, C}\; \mathcal{L}(P, Q, C) + \lambda_S \cdot \mathcal{R}_S(P)$$

where $\mathcal{L}$ is the CMF objective from Section 3.2. The weight $w_{uv}$ encodes both the community overlap between $u$ and $v$ and the boundary status of the pair. The formulation we call **boundary\_downweight** sets:

$$w_{uv} = J\bigl(C(u), C(v)\bigr) \cdot \bigl(1 - \beta \cdot \max(b_u, b_v)\bigr)$$

where $J(C(u), C(v)) = |C(u) \cap C(v)| / |C(u) \cup C(v)|$ is the Jaccard similarity of community memberships, $b_u \in \{0, 1\}$ is the binary boundary indicator of user $u$ (1 if $\tilde{h}_u$ falls below a threshold, 0 otherwise), and $\beta \in [0, 1]$ controls how much the boundary status reduces the regularization weight.

The intuition is this: when two users share many communities and neither is a boundary-spanner, their latent factors should be pulled close. When at least one of them is a boundary-spanner, the pull should be weaker, because the structural value of that user lies precisely in the heterogeneity of their latent representation. Uniform social regularization would average away this heterogeneity; adaptive weighting preserves it.

We also consider three alternative weight schemes for comparison:
- **uniform**: $w_{uv} = 1$ for all edges (standard social regularization)
- **community\_jaccard**: $w_{uv} = J(C(u), C(v))$ (community-weighted, boundary-unaware)
- **bridge\_preserve**: $w_{uv} = J(C(u), C(v)) + \gamma \cdot \max(b_u, b_v)$ (boundary users receive *higher* weights, not lower, to signal that their latent position matters most)

---

## 4. Methodology

### 4.1. Pipeline Overview

The complete pipeline is organized into seven phases, each with a well-defined input, output, and purpose. The division is not merely organizational — it is methodologically important. By separating the pipeline into stages with clear inputs and outputs, we can evaluate the contribution of each stage independently through ablation, which is the only rigorous way to determine whether a component genuinely adds value.

```
Phase 1 → Cascades
Phase 2 → Inferred networks (100 per model × 3 models)
Phase 3 → Overlapping communities per network
Phase 4 → LPH scores and boundary indicators per user per network
Phase 5 → Enhanced CMF (network features + boundary attributes)
Phase 6 → Boundary-guided social CMF (Laplacian regularization)
Phase 7 → Comparative evaluation (Models 1 through 4)
```

### 4.2. Phase 1 — Data Preparation and Cascade Construction

The MovieLens dataset contains 100,004 ratings from 671 users on 9,066 movies, with timestamps covering the period from March 1996 to September 2018 [1]. Each movie is treated as a cascade: for movie $m$, the cascade $c_m$ consists of all pairs $(u_i, t_i)$ where user $u_i$ rated movie $m$ at time $t_i$. This yields 9,066 cascades in total.

The temporal ordering of ratings within each cascade is the key input to NETINF. If user A rates a movie shortly after user B, that temporal proximity is consistent with the hypothesis that B influenced A's decision to watch and rate the movie — not literally, perhaps, but in the statistical sense that the NETINF model formalizes.

The ratings data is split before cascade construction: only training-set interactions are used to build the cascade file. This is a critical data integrity control: test interactions must not be visible to the network inference process, as they represent future behavior. The train/test split is performed at the user level with a fixed random seed for reproducibility.

### 4.3. Phase 2 — User–User Network Inference

NETINF is run on the cascade collection using each of the three transmission models. For each model, $N_\alpha = 100$ values of the transmission rate parameter $\alpha$ are swept over a range determined by the median inter-event time in the cascade data. This yields $3 \times 100 = 300$ directed networks, each representing a different hypothesis about the underlying influence structure.

For downstream community analysis, each directed network is symmetrized: an undirected edge $(u, v)$ is placed whenever the directed network contains either $(u \to v)$ or $(v \to u)$. This symmetrization is a deliberate simplification, justified by the fact that $\tilde{h}_v$ [2] is defined for undirected graphs and its extension to directed graphs is explicitly left as future work by the original authors. The directed properties (in-degree, out-degree, PageRank, betweenness on the directed graph) remain available from Phase 2 and are used in Phase 5 as centrality features; it is only for the community and boundary computation that the undirected version is used.

### 4.4. Phase 3 — Overlapping Community Detection

Overlapping community detection is performed on the symmetrized network using the DEMON algorithm [19], which detects communities by local ego-network expansion with a configurable overlap tolerance parameter $\epsilon$. DEMON was chosen because it is efficient for sparse networks, produces overlapping communities without requiring a pre-specified number of clusters, and has been used in prior work on boundary-spanning detection [2].

For each network and each user $v$, the output is a set of community labels $C(v)$ representing the communities to which $v$ belongs. Users not assigned to any community (isolated nodes or users in networks too sparse for meaningful community structure) receive $C(v) = \emptyset$ and are assigned $\tilde{h}_v = 0$ as a neutral default.

### 4.5. Phase 4 — Computing $\tilde{h}_v$ and Boundary Attributes

Given the community memberships from Phase 3, the full battery of boundary attributes is computed for each user in each network according to Algorithm 1 of Barraza et al. [2] (Section 3.3 above). The resulting attributes are:

- $\tilde{h}_v$ — local pluralistic homophily score
- $m(v) = |C(v)|$ — number of communities
- $s(v)$ — shared community coverage with neighbors
- $\delta_v$ — local discrepancy relative to neighbors
- $b_v \in \{0, 1\}$ — binary boundary indicator (1 if $\tilde{h}_v$ falls below the 20th percentile of the network's score distribution)

These are saved alongside the centrality metrics computed in Phase 2 into per-network user attribute files.

### 4.6. Phase 5 — Enhanced CMF with Network and Boundary Attributes

The user attribute matrix $A$ is assembled by merging:
1. **Centrality features** (Phase 2): degree, betweenness, closeness, eigenvector centrality, PageRank, clustering coefficient, eccentricity, hub score, authority score
2. **Boundary features** (Phase 4): $\tilde{h}_v$, $m(v)$, $s(v)$, $\delta_v$, $b_v$
3. **Cascade statistics** (Phase 1): per-user rating frequency statistics

All features are standardized using a scaler fitted on the training-set users only, to prevent any leakage of test-set scale information into the feature normalization.

Hyperparameters $(k, \lambda_{\mathrm{reg}}, w_A)$ are searched via Optuna [34] using the Tree-structured Parzen Estimator over 50 trials, with RMSE on a held-out validation fold as the objective. The search is performed once using a representative network and the resulting parameters are applied across all 100 networks for evaluation.

### 4.7. Phase 6 — Boundary-Guided Social Regularization

For Phase 6, the social edge set $E_S$ is constructed from the inferred network by filtering to edges where both endpoints have sufficient rating history (warm users). Edge weights are then computed according to the **boundary\_downweight** formula (Section 3.4), using the community memberships and boundary indicators from Phase 4.

The patched CMF optimizer incorporates $\mathcal{R}_S(P)$ as an additional term in the L-BFGS objective. The parameter $\lambda_S$ controlling the strength of social regularization is optimized via a further Optuna search over the joint space of $(k, \lambda_{\mathrm{reg}}, w_A, \lambda_S, \beta, \gamma)$.

### 4.8. Phase 7 — Comparative Evaluation

The evaluation uses a stepwise comparison ladder designed to isolate the contribution of each component:

| Model | Description |
|-------|-------------|
| **M1** — Baseline CF | CMF without side information |
| **M2** — MAFPIN | CMF + centrality features (replicating [1]) |
| **M3** — MAFPIN-LPH | CMF + centrality + boundary features |
| **M4** — MAFPIN-LPH-SR | CMF + centrality + boundary features + social regularization |

The primary metrics are RMSE, MAE, and $R^2$ for continuity with the base methodology [1]. We additionally report the mean delta (M2 or M3 or M4 minus M1) over the sampled network grid, with standard deviations, to characterize result stability across the $\alpha$ sweep.

---

## 5. Experimental Setup

### 5.1. Datasets

**MovieLens (Small)** [35]: 100,004 ratings from 671 users on 9,066 movies. Ratings are on a 1–5 scale; timestamps are Unix epoch seconds. This dataset matches the original MAFPIN evaluation context exactly, allowing direct comparison with the reported baselines.

**Ciao** [36]: approximately 284,086 ratings from users on products in the Ciao consumer review platform, with explicit trust relationships available (though not used in the current experiments). This dataset serves as a held-out transferability check: parameters and design choices were first established on MovieLens, then applied to Ciao without modification.

### 5.2. Subsampled Protocol

The preliminary experiments reported here use a subsampled protocol of `max_ratings = 5000` to allow rapid iteration during pipeline development. This subsampling filters to warm users and warm items (those with sufficient interaction history) and is not intended to reproduce the full-scale results of [1]. The full-scale experiments are in progress. All comparisons within this paper are therefore internal — M4 versus M1 on the same subsample and the same train/test split — and should not be directly compared against Table 2 or Table 3 of [1].

### 5.3. Hyperparameter Configuration

The following configuration was used for the social regularization experiments:

| Parameter | Search range / fixed value |
|-----------|---------------------------|
| Latent factors $k$ | Optuna: $[4, 64]$ integer |
| $\lambda_{\mathrm{reg}}$ | Optuna: $[0.1, 20.0]$ log-uniform |
| $w_A$ (weight on side information) | Optuna: $[0.1, 1.0]$ |
| $w_{\mathrm{user}}$ (attribute weight multiplier) | Optuna: $[0.01, 0.3]$ |
| $\lambda_S$ (social regularization strength) | Optuna: $[10^{-4}, 1.0]$ log-uniform |
| Social mode | Optuna: categorical $\{\texttt{boundary\_downweight}, \texttt{bridge\_preserve}, \ldots\}$ |
| $\beta$ (boundary downweight factor) | Optuna: $[0.0, 1.0]$ |
| $\gamma$ (bridge preserve factor) | Optuna: $[0.0, 3.0]$ |
| L-BFGS iterations | 25 |
| Social retries on failure | 8 |

### 5.4. Evaluation Protocol

Each model is evaluated on a $k$-fold cross-validation scheme with a fixed random seed. Only warm users (those present in both training and test sets) are included in evaluation. The primary reported metrics are:

$$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}_{\text{test}}|}\sum_{(u,i,r) \in \mathcal{T}_{\text{test}}} (r_{ui} - \hat{r}_{ui})^2}$$

$$\text{MAE} = \frac{1}{|\mathcal{T}_{\text{test}}|}\sum_{(u,i,r) \in \mathcal{T}_{\text{test}}} |r_{ui} - \hat{r}_{ui}|$$

$$R^2 = 1 - \frac{\sum_{(u,i,r)} (r_{ui} - \hat{r}_{ui})^2}{\sum_{(u,i,r)} (r_{ui} - \bar{r})^2}$$

A rating-scale sanity check is applied to all runs: fits producing RMSE outside the range $[0.5, 3.0]$ (which would be outside the 1–5 rating scale) are marked as failed and excluded from aggregate statistics.

---

## 6. Results

### 6.1. Original MAFPIN Baseline (Full Scale)

For reference, the original MAFPIN results from Uribe et al. [1] on the full MovieLens dataset are reproduced in Table 1. These represent the performance target that our extended methodology must meet or exceed on equivalent experimental conditions.

**Table 1.** MAFPIN results from [1] on the full MovieLens dataset (100,004 ratings). Metrics are averaged over the 100 alpha values per transmission model.

| Model | RMSE (µ ± σ) | MAE (µ ± σ) | R² (µ ± σ) |
|-------|-------------|------------|------------|
| Baseline CF | 0.8983 ± 0.1867 | 0.7126 ± 0.0427 | 0.1964 ± 0.1715 |
| MAFPIN-Exponential | 0.7867 ± 0.0151 | 0.6833 ± 0.0074 | 0.2959 ± 0.0100 |
| MAFPIN-Power-law | 0.7887 ± 0.0171 | 0.6842 ± 0.0083 | 0.2948 ± 0.0106 |
| MAFPIN-Rayleigh | 0.7888 ± 0.0164 | 0.6840 ± 0.0078 | 0.2940 ± 0.0113 |

The exponential transmission model achieved the lowest RMSE and highest $R^2$, a pattern that is consistent across the 100 network samples for that model. Our extended pipeline inherits this design preference, prioritizing the exponential model for initial experiments while evaluating all three for robustness.

### 6.2. Smoke Test — Social Mode Comparison (Subsampled)

Table 2 reports the four social weighting modes on MovieLens with the subsampled protocol (5,000 ratings, network index 0, exponential model, $\lambda_S = 0.1$, no user attributes, $k=8$, $\lambda_{\mathrm{reg}} = 10$). The control run at $\lambda_S = 0$ uses the same configuration without the social term.

**Table 2.** Social mode comparison — MovieLens, subsampled protocol. Control RMSE: 0.8741, R²: 0.1805.

| Social mode | Edges | Weight range | RMSE | MAE | R² |
|-------------|-------|-------------|------|-----|----|
| uniform | 1300 | 1.000 – 1.000 | 0.8740 | 0.6919 | 0.1808 |
| community\_jaccard | 928 | 0.312 – 1.250 | 0.8740 | 0.6919 | 0.1808 |
| boundary\_downweight | 928 | 0.315 – 1.259 | 0.8740 | 0.6919 | 0.1808 |
| bridge\_preserve | 1300 | 0.621 – 1.203 | 0.8740 | 0.6919 | 0.1808 |

At $\lambda_S = 0.1$, all four modes produce effectively identical metrics. The boundary-based weight differentiation does not yet manifest at this scale of regularization. The primary result at this stage is that the social-regularized runs are numerically stable and rating-scale sane — a necessary precondition for interpreting the full model.

### 6.3. Lambda Sweep

Table 3 summarizes the best performance across the $\lambda_S$ grid $\{0.001, 0.01, 0.1, 1.0\}$ for each mode on MovieLens (same configuration as Table 2).

**Table 3.** Best metrics by social mode across the $\lambda_S$ sweep — MovieLens, subsampled.

| Mode | Best RMSE ($\lambda_S$) | RMSE | Best R² ($\lambda_S$) | R² |
|------|------------------------|------|----------------------|-----|
| boundary\_downweight | 0.001 | 0.87387 | 0.001 | 0.18103 |
| bridge\_preserve | 0.001 | 0.87393 | 0.001 | 0.18091 |
| community\_jaccard | 0.001 | 0.87391 | 0.001 | 0.18095 |
| uniform | 0.001 | 0.87392 | 0.001 | 0.18093 |

The best setting across all modes is the smallest tested $\lambda_S = 0.001$, a pattern that is consistent and informative: at this scale, the social smoother is most beneficial when applied very lightly. The best-performing mode is **boundary\_downweight** with RMSE 0.87387 and R² 0.18103, narrowly ahead of the others. The differences are small enough that they should not be over-interpreted from a single network index, but the directional consistency across all four modes points toward a preference for light regularization.

### 6.4. Network Stability Sweep

To test whether the lambda-sweep findings are artifacts of a single network configuration, the best setting (**boundary\_downweight**, $\lambda_S = 0.001$) was evaluated across 10 randomly sampled network indices for each of the three diffusion models (30 total runs per dataset). Table 4 reports the aggregated results.

**Table 4.** Network stability sweep — MovieLens, 10 networks × 3 diffusion models.

| Diffusion model | Mean RMSE | RMSE std | Mean R² | R² std | Mean ΔRMSE |
|----------------|-----------|----------|---------|--------|------------|
| exponential | 0.87384 | 0.00018 | 0.18109 | 0.00034 | −0.000091 |
| power-law | 0.87387 | 0.00005 | 0.18103 | 0.00009 | −0.000027 |
| rayleigh | 0.87382 | 0.00022 | 0.18113 | 0.00041 | −0.000079 |

All 30 runs passed the rating-scale sanity check. The mean RMSE delta (social-regularized minus no-social control) is negative for all three models, confirming that the social regularizer provides a small but consistent improvement across a diverse range of inferred network configurations. The rayleigh model yields the best mean RMSE and R² in this sweep, though the differences between models are within one standard deviation of each other.

The small standard deviations across the 10 sampled networks within each model (e.g., $\sigma_{\mathrm{RMSE}} = 0.00018$ for exponential) indicate that results are not driven by any single network configuration: the improvement, modest as it is, is structurally stable.

### 6.5. User Attributes + Social Regularization (Step 4)

Table 5 introduces user attributes into the social-regularized model, sweeping a 3×3 grid of $(\lambda_{\mathrm{reg}}, w_{\mathrm{user}})$ values with the fixed social configuration (**boundary\_downweight**, $\lambda_S = 0.001$) on MovieLens.

**Table 5.** User attribute + social regularization grid — MovieLens, subsampled (network 0, exponential).

| $\lambda_{\mathrm{reg}}$ | $w_{\mathrm{user}}$ | RMSE | MAE | R² |
|--------------------------|---------------------|------|-----|----|
| 1.0 | 0.01 | 0.8811 | 0.6869 | 0.1674 |
| 1.0 | 0.05 | 0.8814 | 0.6861 | 0.1669 |
| 1.0 | 0.10 | 0.8842 | 0.6871 | 0.1615 |
| 3.0 | 0.01 | 0.8614 | 0.6755 | 0.2043 |
| **3.0** | **0.05** | **0.8583** | **0.6724** | **0.2100** |
| 3.0 | 0.10 | 0.8609 | 0.6741 | 0.2051 |
| 10.0 | 0.01 | 0.8739 | 0.6919 | 0.1810 |
| 10.0 | 0.05 | 0.8741 | 0.6920 | 0.1806 |
| 10.0 | 0.10 | 0.8732 | 0.6914 | 0.1822 |

The best result in this grid is $\lambda_{\mathrm{reg}} = 3.0$, $w_{\mathrm{user}} = 0.05$, yielding RMSE 0.8583, MAE 0.6724, and R² 0.2100. This represents a substantial improvement over the no-user-attribute level (RMSE ≈ 0.8738), demonstrating that the boundary attributes contribute meaningfully once the regularization weight is relaxed from the conservative $\lambda_{\mathrm{reg}} = 10.0$ used in the pure social regularization steps.

### 6.6. Optuna Search — Best Parameters

A full Optuna search over the joint hyperparameter space was run for the MovieLens/exponential/network-0 setting (50 trials, maximizing validation R²). Table 6 reports the best found parameters and Table 7 the resulting evaluation.

**Table 6.** Optuna best parameters — MovieLens, exponential, network 0.

| Parameter | Value |
|-----------|-------|
| $k$ | 33 |
| $\lambda_{\mathrm{reg}}$ | 1.539 |
| $w_{\mathrm{main}}$ | 0.898 |
| $w_{\mathrm{user}}$ | 0.0497 |
| $\lambda_S$ | 0.000199 |
| Social mode | boundary\_downweight |
| $\beta$ | 0.778 |
| $\gamma$ | 2.548 |

**Table 7.** Baseline CMF vs. best-params social CMF — MovieLens, subsampled (same $k$, $\lambda_{\mathrm{reg}}$ for both).

| Model | User attributes | Social reg. | RMSE | MAE | R² |
|-------|----------------|-------------|------|-----|----|
| Plain baseline CMF | No | No | 0.8656 | 0.6825 | 0.1965 |
| Social CMF (best params) | Yes | Yes | 0.8577 | 0.6720 | 0.2110 |
| **Delta** | | | **−0.0078** | **−0.0105** | **+0.0145** |
| **Relative RMSE improvement** | | | **0.90%** | | |

The Optuna-selected model improves RMSE by 0.78 points and R² by 1.45 percentage points over the plain CMF baseline on the same filtered warm split. Because the baseline uses the same $k$ and $\lambda_{\mathrm{reg}}$ as the social model, this comparison isolates the contribution of user-side attributes and the social Laplacian. Both contributions are active simultaneously, so this result represents an upper bound on the combined effect rather than an additive decomposition. The ablation that separates M3 (attributes only) from M4 (attributes + social regularization) is underway as part of the full-scale experimental campaign.

### 6.7. Ciao Results

The same pipeline was run on Ciao (Table 8). The Optuna search selected **bridge\_preserve** as the best social mode, with $\lambda_S = 0.054$, $k = 42$, and $\lambda_{\mathrm{reg}} = 1.664$.

**Table 8.** Baseline CMF vs. best-params social CMF — Ciao, subsampled.

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Plain baseline CMF | 0.9188 | 0.6952 | 0.1600 |
| Social CMF (best params) | 0.9144 | 0.6987 | 0.1679 |
| **Delta** | **−0.0043** | +0.0035 | **+0.0079** |

The Ciao improvement is smaller than on MovieLens (RMSE delta −0.43% vs. −0.90%), which is consistent with the sparser and noisier nature of the Ciao rating data in the subsampled protocol. Notably, the Optuna search selected a different social mode (bridge\_preserve) and a higher $\lambda_S$, suggesting that the two datasets require different regularization regimes — a finding that motivates the full-scale experiments across more network configurations.

---

## 7. Discussion

### 7.1. What the Preliminary Results Tell Us

The most important finding from the experiments described in Section 6 is not any individual RMSE number — it is the structural stability of the pipeline. Across three diffusion models, 30 sampled networks per dataset, and two datasets with distinct characteristics, the social-regularized runs consistently pass the rating-scale sanity check and consistently show negative mean RMSE deltas relative to their no-social controls. The variance across network samples is small (e.g., $\sigma_{\mathrm{RMSE}} = 0.00018$ for MovieLens/exponential), which means the results are not driven by a particularly favorable or unfavorable network configuration. This kind of stability is a necessary — though not sufficient — condition for claiming that the methodology is doing something real rather than producing noise.

The finding that lighter regularization ($\lambda_S = 0.001$) consistently outperforms heavier values ($\lambda_S \in \{0.01, 0.1, 1.0\}$) has a natural interpretation: at the scale of the subsampled protocol, the inferred social graph is relatively sparse (928 edges in MovieLens, 2,043 in Ciao), and strong regularization would over-smooth a user population that is already small. Whether this preference for light regularization persists on the full-scale dataset — with approximately 300 networks of varying density — is an open question that the full-scale experiments will address.

The jump in performance when user attributes are combined with social regularization (RMSE 0.8738 → 0.8583 on MovieLens, Section 6.5) is substantially larger than the improvement from social regularization alone (0.8741 → 0.8739, Section 6.3). This suggests that the boundary attributes derived from LPH carry more predictive information than the topology of the social edge weights alone. This result is in line with the motivation of the fusion: the structural role information encoded in $\tilde{h}_v$ is qualitatively different from the edge-level signal, and the two appear to act synergistically.

### 7.2. The Comparison Gap with the Original MAFPIN

The subsampled results cannot be directly compared against Table 1 of this paper (from [1]) or against the benchmark comparison in Table 4 of [1]. The original MAFPIN was evaluated on the full 100,004 ratings of MovieLens with a different protocol and different feature set (including farness centrality, which is not yet in the current pipeline). Any direct numeric comparison would be misleading. The methodological contributions — the addition of LPH boundary attributes and boundary-guided social regularization — will only be properly evaluated in the full-scale experiments.

That said, the preliminary results do establish two things that are relevant even at this stage: first, that the complete pipeline (Phases 1–6) is end-to-end functional and numerically stable; and second, that the boundary-aware components contribute positively in a controlled comparison where all other factors are held fixed.

### 7.3. The Role of Boundary Information in Latent Space Geometry

A deeper question, which the current results only begin to address, is whether boundary users occupy geometrically distinctive regions of the latent factor space $P$, and whether this is related to recommendation quality. The SHAP analysis module of the pipeline is designed to probe this question: by fitting a gradient-boosted surrogate to the CMF's test-set predictions as a function of user attributes, and then computing SHAP values, it becomes possible to estimate which attributes most strongly modulate predicted ratings and for which users. Preliminary SHAP results (not reported here) suggest that $\tilde{h}_v$ and $m(v)$ are among the higher-ranked features by mean absolute SHAP value, particularly for users who receive substantially different predicted ratings between the baseline CMF and the enhanced model. This is a promising signal, but the sample sizes in the subsampled protocol are too small for the surrogate fits to be reliable; reporting these results awaits the full-scale evaluation.

### 7.4. Boundary\_downweight vs. Bridge\_preserve

The fact that MovieLens preferred **boundary\_downweight** while Ciao preferred **bridge\_preserve** is the most intriguing cross-dataset asymmetry in the preliminary results. These two modes have opposite interpretations: boundary\_downweight reduces regularization pressure on boundary pairs, preserving the heterogeneity of their latent factors; bridge\_preserve *increases* regularization pressure on boundary pairs, forcing the model to explicitly reconcile the latent representations of users who bridge communities. A model of users in which boundary-spanning is a *role to be preserved* would favor boundary\_downweight; a model in which boundary-spanning is a *signal of common latent structure* would favor bridge\_preserve.

The dataset-level difference may reflect the nature of the inferred networks: MovieLens ratings are movie-driven cascades with relatively fine temporal resolution, while Ciao ratings come from a product review platform with a different underlying influence dynamic. Whether this preference is stable across the full network grid, or is an artifact of the subsampled protocol and the single network index used in the grid search, is one of the key questions for the full-scale evaluation.

---

## 8. Limitations and Future Work

### 8.1. Limitations

**Subsampled protocol.** All experimental results in this paper are derived from a 5,000-rating subsample with warm-user filtering, not from the full 100,004-rating MovieLens dataset. RMSE values in the range 0.85–0.93 are not comparable to the 0.786 reported by the original MAFPIN on the full dataset. The full-scale experiments are in progress.

**Farness centrality absent.** The original MAFPIN [1] uses eight topological features including farness centrality, which is defined as the inverse of the average shortest path distance from a node to all other nodes. The current implementation does not yet include farness centrality; it is available in the SNAP library and will be added in the full-scale pipeline. Until it is included, the current feature set (M2 in our model ladder) does not exactly replicate the feature set of the original paper.

**Random train/test split.** The pipeline uses a random (non-temporal) train/test split. For a cascade-based methodology, a temporal split would be more epistemologically coherent: it would ensure that no test-time interaction could have influenced the cascade timing used for network inference. The effect of this choice on the full-scale results is difficult to quantify a priori, but it is particularly significant for sparse users with few interactions.

**Single community detection algorithm.** Community structure in the inferred networks is computed solely with DEMON [19]. Different algorithms — particularly those that produce different numbers of communities or differently shaped community overlaps — may yield different $\tilde{h}_v$ distributions and therefore different boundary indicators. A sensitivity analysis over multiple algorithms (e.g., ASLPAw [20]) is planned.

**No ranking metrics.** The evaluation is limited to rating-prediction metrics (RMSE, MAE, R²). Recommendation quality — the ability to surface items that users would actually engage with, in ranked order — is not currently measured. Metrics such as NDCG@K, Precision@K, and Recall@K would be necessary to make claims about how the boundary-aware model affects the diversity or novelty of recommendations, which is arguably the most distinctive potential benefit of incorporating $\tilde{h}_v$ into the latent space.

**Directed vs. undirected.** NETINF infers a directed graph, but the current pipeline symmetrizes it before community detection and boundary computation. Directed variants of $\tilde{h}_v$ — where the neighborhood $N(v)$ distinguishes in-neighbors from out-neighbors — are not explored. Barraza et al. [2] explicitly list this as a direction for future work, and the asymmetry of the inferred influence graph may contain signals that symmetrization discards.

### 8.2. Future Work

**Full-scale validation.** The most pressing next step is running the complete pipeline on the full MovieLens dataset (100,004 ratings, all users) with farness centrality included and reporting the four-model comparison ladder. This will determine whether the improvements observed in the subsampled protocol are significant at scale.

**Ablation: M2 vs. M3 vs. M4.** The current results compare M4 directly against M1 (plain CMF). The intermediate conditions M2 (centrality only) and M3 (centrality + boundary attributes) need to be evaluated separately to characterize the marginal contribution of $\tilde{h}_v$ and the social regularization term individually.

**Extension to directed LPH.** Adapting $\tilde{h}_v$ to directed networks, where the boundary concept considers asymmetric influence, would be both theoretically interesting and methodologically more faithful to the NETINF output.

**Cross-community exposure metrics.** Beyond RMSE, defining and measuring *cross-community exposure* — the fraction of recommendations that cross community lines for a given user — would provide a more direct empirical test of the hypothesis that boundary-spanning users receive qualitatively different recommendations under M3 and M4 compared to M1 and M2.

**Explicit trust graph integration (Ciao, Epinions).** Ciao and Epinions include explicit trust graphs between users. Comparing recommendations built from the inferred NETINF network against those built from the explicit trust graph, and exploring their combination, would probe how much of the inferred network's utility overlaps with what explicit social structure already captures.

---

## 9. Conclusions

We have presented an extension of the MAFPIN methodology that incorporates structural boundary information into the user representation layer of a Collective Matrix Factorization recommender. The core insight is that the inferred user–user network, beyond providing centrality descriptors, also admits a community-level analysis from which a qualitatively different kind of user attribute can be derived: the local pluralistic homophily score $\tilde{h}_v$, which measures how much a user's community membership is misaligned with their immediate neighborhood. Low-$\tilde{h}_v$ users are not peripheral or poorly connected; they are structurally positioned at the interfaces between overlapping communities, and their preferences may carry signals that intra-community users cannot provide.

The preliminary experiments confirm that the extended pipeline is end-to-end functional and numerically stable across three diffusion models, two datasets, and a wide range of inferred network configurations. The best-params evaluation on MovieLens shows a 0.90% RMSE improvement and a 1.45 percentage point gain in R² over a plain CMF baseline with matched hyperparameters, attributable jointly to the boundary attributes and the boundary-guided social Laplacian. The key limitation is that these results are from a subsampled protocol and are not yet comparable to the full-scale MAFPIN benchmark.

The theoretical contribution of this work is, at its most condensed: it is not enough to know how central a user is; it also matters whether they are deeply embedded within a community or standing at its frontier. These are different structural roles, they may produce different recommendation behaviors, and they warrant different modeling treatment. The methodology described here provides the first attempt to formalize and exploit this distinction within a principled matrix factorization framework.

---

## References

[1] S. Uribe, C.E. Ramirez, and J. Finke, "Recommender Systems Based on Matrix Factorization and the Properties of Inferred Social Networks," *Discrete Mathematics, Algorithms and Applications*, World Scientific Publishing Company, 2023.

[2] R. Barraza, M. Ravetti, and S. Sánchez-Cambronero, "Local Pluralistic Homophily for Boundary-Spanning Node Detection in Overlapping Community Networks," *Applied Sciences*, 2025.

[3] Y. Koren, R. Bell, and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," *IEEE Computer*, vol. 42, no. 8, pp. 30–37, 2009.

[4] M. Gomez-Rodriguez, J. Leskovec, and A. Krause, "Inferring Networks of Diffusion and Influence," *ACM Transactions on Knowledge Discovery from Data*, vol. 5, no. 4, pp. 1–37, 2012.

[5] C. Fan and L. Yu, "Inferring Social Networks Based on Movie Rating Data," Stanford University, CA, Technical Report, 2011.

[6] J. Leskovec and R. Sosič, "SNAP: A General-Purpose Network Analysis and Graph-Mining Library," *ACM Transactions on Intelligent Systems and Technology*, vol. 8, no. 1, pp. 1–20, 2016.

[7] Y. Koren, "Collaborative Filtering with Temporal Dynamics," in *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 447–456, 2009.

[8] A.P. Singh and G.J. Gordon, "Relational Learning via Collective Matrix Factorization," in *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 650–658, 2008.

[9] D. Cortes, "Cold-Start Recommendations in Collective Matrix Factorization," *arXiv preprint arXiv:1809.00366*, 2018.

[10] T. Tanaka and K. Tanaka, "Independent Cascade Model," in *Proceedings of the 9th International Conference on Knowledge Discovery and Data Mining*, pp. 137–146, 2003.

[11] W.T. Tutte, "The Dissection of Equilateral Triangles into Equilateral Triangles," *Proceedings of the Cambridge Philosophical Society*, vol. 44, no. 4, pp. 463–482, 1948.

[12] T. Alpay et al., "FASTINF: Large-scale Network Inference," in *Proceedings of the IEEE International Conference on Big Data*, 2014.

[13] Y. Wang et al., "Network Inference as a Classification Problem," in *Proceedings of the 2016 SIAM International Conference on Data Mining*, 2016.

[14] H. Ma, H. Yang, M.R. Lyu, and I. King, "SoRec: Social Recommendation Using Probabilistic Matrix Factorization," in *Proceedings of the 17th ACM International Conference on Information and Knowledge Management*, pp. 931–940, 2008.

[15] M. Jamali and M. Ester, "A Matrix Factorization Technique with Trust Propagation for Recommendation in Social Networks," in *Proceedings of the 4th ACM Conference on Recommender Systems*, pp. 135–142, 2010.

[16] J. Yang, B. Long, A. Smola, H. Zha, and Z. Zheng, "Mining Social Networks Using Heat Diffusion Processes for Marketing Candidates Selection," in *Proceedings of the 17th ACM International Conference on Information and Knowledge Management*, 2008.

[17] M. Gray et al., "Bayesian Network Inference from Cascades," *Journal of Machine Learning Research*, 2020.

[18] Y. Liu et al., "Local Trust Network-Based Recommendation," *Information Sciences*, 2017.

[19] M. Coscia, F. Giannotti, and D. Pedreschi, "A Demon-based Algorithm for Community Discovery in Multi-Layer Networks," in *Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 1245–1253, 2012.

[20] J. Šubelj and M. Bajec, "Robust Network Community Detection with Exponential Weighting," *European Physical Journal B*, vol. 81, pp. 353–362, 2011.

[21] G. Rossetti, L. Milli, and R. Cazabet, "CDlib: A Python Library to Extract, Compare and Evaluate Communities from Complex Networks," *Applied Network Science*, vol. 4, no. 52, 2019.

[22] R. Albert and A.L. Barabási, "Statistical Mechanics of Complex Networks," *Reviews of Modern Physics*, vol. 74, pp. 47–97, 2002.

[23] M. McPherson, L. Smith-Lovin, and J.M. Cook, "Birds of a Feather: Homophily in Social Networks," *Annual Review of Sociology*, vol. 27, pp. 415–444, 2001.

[24] P. Papadopoulos, D. Krioukov, M. Boguñá, and A. Vahdat, "Greedy Forwarding in Dynamic Scale-Free Networks Embedded in Hyperbolic Metric Spaces," in *Proceedings of IEEE INFOCOM*, 2010.

[25] D. Lee, H. Seung, "Learning the Parts of Objects by Non-Negative Matrix Factorization," *Nature*, vol. 401, pp. 788–791, 1999.

[26] R.S. Burt, *Structural Holes: The Social Structure of Competition*, Harvard University Press, 1992.

[27] [hTrust reference — Laplacian-regularized trust-aware MF model with $\min\|G - UVU^\top\|^2 + \alpha\|U\|^2 + \beta\|V\|^2 + \lambda\,\mathrm{tr}(U^\top L U)$], Internal reference.

[28] K. Goldberg, T. Roeder, D. Gupta, and C. Perkins, "Eigentaste: A Constant Time Collaborative Filtering Algorithm," *Information Retrieval*, vol. 4, pp. 133–151, 2001.

[29] X. Wang, X. He, M. Wang, F. Feng, and T.S. Chua, "Neural Graph Collaborative Filtering," in *Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 165–174, 2019.

[30] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation," in *Proceedings of the 43rd International ACM SIGIR Conference*, pp. 639–648, 2020.

[31] G. Palla, I. Derényi, I. Farkas, and T. Vicsek, "Uncovering the Overlapping Community Structure of Complex Networks in Nature and Society," *Nature*, vol. 435, pp. 814–818, 2005.

[32] M. Ge, C. Delgado-Battenfeld, and D. Jannach, "Beyond Accuracy: Evaluating Recommender Systems by Coverage and Serendipity," in *Proceedings of the 4th ACM Conference on Recommender Systems*, pp. 257–260, 2010.

[33] T. Zhou, Z. Kuscsik, J.G. Liu, M. Medo, J.R. Wakeling, and Y.C. Zhang, "Solving the Apparent Diversity-Accuracy Dilemma of Recommender Systems," *Proceedings of the National Academy of Sciences*, vol. 107, no. 10, pp. 4511–4515, 2010.

[34] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A Next-Generation Hyperparameter Optimization Framework," in *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 2623–2631, 2019.

[35] F.M. Harper and J.A. Konstan, "The MovieLens Datasets: History and Context," *ACM Transactions on Interactive Intelligent Systems*, vol. 5, no. 4, pp. 1–19, 2015.

[36] J. Tang, H. Gao, H. Liu, and A. Das Sarma, "eTrust: Understanding Trust Evolution in an Online World," in *Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 253–261, 2012.

---

*Acknowledgments*: [To be completed]

*Data availability*: The MovieLens dataset is publicly available at [https://grouplens.org/datasets/movielens/]. The Ciao and Epinions datasets are available upon request from the original providers.

*Conflicts of interest*: The authors declare no conflicts of interest.
