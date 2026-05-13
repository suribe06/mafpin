# Work Guide: Merging ws-dmaa with Appl Sci

## 1. Starting Point: What to Take from Each Article and Why

The core idea is straightforward, though it has several methodological layers. From the **ws-dmaa** article, I want to preserve the heart of the pipeline: from user–item interactions, infer an implicit social network among users with NETINF, extract properties from that network, and use those properties as attributes within a **Collective Matrix Factorization (CMF)** scheme to improve rating prediction. That article works with MovieLens, treats each movie as a cascade, and then adds network attributes such as degree, betweenness, closeness, eigenvector, farness, community structure, eccentricity, and hubs/authorities. It also reports that this strategy improves over the classical collaborative filtering baseline, and that the exponential transmission model had the best overall performance.

From the **Appl. Sci.** article I want to take a different but highly complementary idea: it is not enough to know which users are "well connected"; it also matters to detect which ones are situated at the **borders between overlapping communities**. This is where the metric $\tilde{h}_v$ comes in — it does not seek classical structural hubs, but nodes whose community membership is misaligned with that of their neighborhood. The paper insists that $\tilde{h}_v$ is orthogonal to purely topological measures like BridgeCC, is interpretable, and is especially useful when communities capture semantically or behaviorally coherent groups.

In one sentence, the proposed fusion is:

> **Maintain the predictive logic of ws-dmaa, but enrich the user attribute layer with a new inter-community boundary signal derived from $\tilde{h}_v$, so that the recommender sees not only central users, but also boundary users.**

## 2. Conceptual Intuition of the Fusion

This fusion should not be sold as "adding two techniques just because." The better framing is: in recommendation, part of the signal comes from **proximity** and another part comes from **crossing between groups**.

ws-dmaa already exploits the first quite well. When you infer a user network from their temporal rating sequences and use properties of that network as model attributes, you are capturing a reasonable idea: there are users who, even without being explicitly connected on the platform, behave as if there were an underlying relational structure between them.

But something is still missing: that inferred network can have very dense zones, communities, interfaces, overlap regions. Not all users located in those zones play the same role. Some are highly representative of their community; others are right on the edge, connecting profiles, tastes, or trajectories that are not the same. That is where Appl. Sci. contributes something that fits almost naturally: **detecting boundary nodes based on membership misalignment**.

In more human terms: if ws-dmaa helps me know **who resembles whom** within an implicit network, Appl. Sci. helps me know **who connects different worlds** within that same network.

And in recommendation that can be very valuable, because often the interesting jump does not come from the most similar neighbor, but from the user who lives between two communities and who, precisely because of that position, pulls or anticipates less obvious combinations of preferences.

## 3. Formal Objective of the Fused Methodology

The objective of the methodology would not be solely to predict ratings with lower RMSE. That would be important, yes, but not sufficient to justify the fusion.

The formal formulation:

> **Build a recommendation model based on matrix factorization over an inferred social network, incorporating inter-community boundary signals in networks with overlapping communities, so that the user's latent space reflects not only structural proximity but also capacity for connection between groups.**

Stated more simply: the user is not described solely by what they rate and how "central" they are, but also by **what role they play between communities**.

## 4. General Pipeline Structure

The complete methodology is divided into seven phases — not to bureaucratize it, but to make it very clear where one idea ends and another begins.

### Phase 1. Data Preparation and Cascade Construction

Start from the user–item–time dataset. Following the ws-dmaa logic, the natural case is MovieLens: 100,004 ratings, 671 users, and 9,066 movies. Each movie is interpreted as a cascade; that is, for a movie $m_c$, take the set of pairs $(u_i, t_i)$ indicating which user rated that movie and at what instant. This yields a total of 9,066 cascades, one per movie.

Important clarification: we are not assuming that a user "literally infects" another to watch a movie. We are adopting the diffusion inference logic as a way to **reconstruct a latent network of influence or behavioral proximity**. That is precisely the point of ws-dmaa: use the NETINF apparatus to go from a temporal sequence of interactions to a relational structure among users.

**Output of Phase 1:** A set of temporal cascades ready for the network inference process.

### Phase 2. User–User Network Inference

Run **NETINF** on the cascades from Phase 1. ws-dmaa considers three parametric transmission models: exponential, power-law, and Rayleigh, following the classical formulation of the network inference problem from cascades. The problem being optimized is, in essence, finding a hidden graph that maximizes the probability of observing the set of cascades under the adopted transmission model.

The practical recommendation is to start with the **exponential model** — not arbitrarily, but because in ws-dmaa it yielded the best empirical performance: average RMSE 0.7867, MAE 0.6833, and $R^2$ of 0.2959, outperforming not only the collaborative filtering baseline but also the other two transmission models.

For the fusion with Appl. Sci., an important methodological decision arises here. NETINF originally infers a **directed** network, because diffusion logic is directional. In contrast, the Appl. Sci. article formulates $\tilde{h}_v$ on **undirected and unweighted networks**, and explicitly leaves the extension to directed and weighted networks as future work. Therefore, for a clean and defensible first version of the fusion, the proposed controlled simplification is: **convert the inferred network to an undirected version via symmetrization**, and work in that space first.

**Output of Phase 2:** A user–user inferred network, preferably in a refined undirected version for subsequent community analysis.

### Phase 3. Overlapping Community Detection

This is where the fusion with Appl. Sci. truly begins.

An **overlapping community detection algorithm** is run on the inferred network. This point is not decorative. Appl. Sci. makes it quite clear that the utility of $\tilde{h}_v$ depends on the quality of the underlying communities. If the communities are poor, artificial, or overly forced, the score's interpretive strength weakens. The article explicitly recommends using overlap-aware methods when communities are not given a priori.

The goal here is not "pretty" communities for visualization; the goal is communities that represent **behaviorally or semantically coherent groups**. In our context, that means communities should reflect plausible relational patterns among users derived from rating behavior.

**Practical recommendation:** Do not commit to a single algorithm from the start. It is reasonable to test two or three overlap-aware detectors and evaluate stability.

**Output of Phase 3:** For each user $v$, a set of memberships $C(v)$ — the overlapping communities to which they belong.

### Phase 4. Computing Boundary Attributes with $\tilde{h}_v$

Once community memberships are obtained, compute the **local pluralistic homophily** metric.

The idea, summarized without losing rigor: for each node $v$, evaluate how much of its community identity appears represented in its immediate neighborhood. This is summarized in a quantity $s(v)$. Then the local discrepancy $\delta_v$ is measured by comparing $s(v)$ with the values of its neighbors. Finally, this is normalized with respect to a global reference to obtain $\tilde{h}_v = \lambda - \delta_v$. **More negative values correspond to nodes whose membership is most misaligned with that of their neighbors — natural candidates for boundary-spanners.**

The key insight: we are not importing another centrality measure. We are importing a signal of a different nature. The article itself shows that $\tilde{h}_v$ tends to diverge from topological baselines and detects functionally distinct node populations. In plain terms: this metric does not chase the typical hub, but the user who acts as an **interface between communities**.

**Attributes to derive here:**

- $\tilde{h}_v$ — local pluralistic homophily score
- $m(v) = |C(v)|$ — number of memberships per user
- $s(v)$ — local community alignment
- $\delta_v$ — local neighborhood dissimilarity
- A binary or quantile indicator of "boundary user" based on low percentiles of $\tilde{h}_v$

(Not all of these need to go into the final model — having the full battery allows for ablation studies.)

**Output of Phase 4:** A vector of community and boundary attributes for each user.

### Phase 5. Integration with the CMF User Attribute Layer

Here the fusion becomes fully operational.

ws-dmaa uses a **Collective Matrix Factorization** scheme in which, in addition to approximating the ratings matrix $R$, a user attribute matrix $A$ is approximated. That is exactly the entry point. No need to break the model or invent an exotic architecture: **simply extend matrix $A$**.

In the original article, $A$ already includes network properties such as degree, betweenness, closeness, eigenvector, farness, community structure, eccentricity, and hubs/authorities. The proposal is to now expand that matrix with attributes from the overlapping community analysis: $\tilde{h}_v$, $m(v)$, $s(v)$, $\delta_v$, and any complementary descriptor that proves stable.

In practical terms, the attribute vector of user $u$ would be approximately:

$$a_u = [\text{degree, betweenness, closeness, eigenvector, farness, eccentricity, hub/authority, community structure, } \tilde{h}_u, m(u), s(u), \delta_u]$$

The interpretation: the user's latent factor no longer depends only on what they rate and how "central" they are — it also depends on **whether they are embedded within a single community or move between several**.

**Output of Phase 5:** An enriched attribute matrix, ready to feed the CMF model.

### Phase 6. Strong Variant: Boundary-Guided Social Regularization

Up to this point there is already a valid and publishable fusion. But for a more ambitious second version, $\tilde{h}_v$ can also be incorporated into the **social regularization of the latent space**.

The idea: if two users are connected in the inferred network, it is not always appropriate to force their latent vectors to be equally close. If both are highly intra-community users, strong regularization may make sense. But if one is a boundary user with a very negative $\tilde{h}_v$, over-smoothing may not be appropriate, because their value lies precisely in mixing signals from different groups.

A regularization term of the following form can be introduced:

$$\sum_{(u,v) \in E} w_{uv} \|p_u - p_v\|^2$$

where the weight $w_{uv}$ depends not only on the existence of the edge, but on the **community relationship between $u$ and $v$ and their boundary values**. This is a proposed extension — not something implemented as-is in either paper — but it is the most natural and interesting prolongation of the fusion. The advantage is that boundary information ceases to be merely descriptive and begins to **modify the geometry of the latent space**.

**Output of Phase 6:** A socially regularized CMF model, sensitive to inter-community interfaces.

### Phase 7. Experimental Evaluation and Comparison Logic

A disciplined ladder of comparison, not all at once:

| Model | Description |
| ----- | ----------- |
| **Model 1** | MF or base collaborative filtering |
| **Model 2** | ws-dmaa original: inferred network + classical topological properties in CMF |
| **Model 3** | ws-dmaa + boundary attributes: $\tilde{h}_v$, $m(v)$, $s(v)$, $\delta_v$ |
| **Model 4** | ws-dmaa + boundary attributes + adaptive social regularization |

Basic evaluation must maintain **RMSE, MAE, and $R^2$** for continuity with the base article. ws-dmaa reports exactly those metrics and shows clear improvement over the baseline.

To justify why the boundary layer contributes something beyond "another feature," complementary recommendation metrics should also be added — for example, coverage, diversity, or some notion of inter-community exposure. Concepts like **cross-community exposure** or **boundary-spanning exposure** connect better with the structural logic of Appl. Sci.

## 5. Working Hypotheses Supporting the Fusion

**Hypothesis 1**
The network inferred from rating cascades contains useful relational signal for recommendation, beyond the user–item matrix alone. This is the foundation of ws-dmaa.

**Hypothesis 2**
Within that inferred network, users located at interfaces between overlapping communities play a different role from hubs or densely embedded nodes. This is precisely the thesis of Appl. Sci.

**Hypothesis 3**
Incorporating boundary attributes into the CMF allows better modeling of hybrid, transitional, or structurally multi-group-exposed users, which should translate into prediction improvements and, above all, less myopic recommendations.

**Hypothesis 4**
Latent regularization guided by boundary structure can better preserve the inter-community richness of the system than uniform social regularization.

## 6. Problems This Fusion Actually Solves

**Sparsity.** ws-dmaa already shows that using properties of an inferred network improves performance over simpler baselines, especially when the user–item matrix is sparse.

**Hybrid users.** Some users' preferences do not fit cleanly into a single community. The classical approach tends to average them poorly or push them toward the center of a single latent region. The boundary signal helps recognize that such a user is not poorly defined — they are, rather, **poorly modeled** if their inter-community role is not accounted for.

**Cross-group exposure.** If the goal is to discuss recommendations that cross communities without losing relevance, this fusion provides a much more solid mathematical and structural foundation than a generic discourse on diversity.

## 7. Limitations to Declare from the Start

1. $\tilde{h}_v$ **depends on the quality of detected communities**. If the communities are poor, the score loses interpretive force. The article itself states this clearly.

2. We are **combining a network inferred from an originally directed process with a metric formulated for undirected and unweighted networks**. For a first experimental phase this can be resolved with symmetrization, but if the work grows, an extension of $\tilde{h}_v$ to directed/weighted networks would be an interesting contribution in itself. Appl. Sci. even mentions it as future work.

3. **Not every boundary user will automatically be useful for recommendation.** Being at the interface between communities does not by itself guarantee predictive capacity. Therefore, evaluation must use ablations, not just intuition.

## 8. Summary in One Idea

> **The methodology fuses a matrix factorization-based recommender over an inferred social network with a layer of overlapping community boundary analysis, so that user attributes describe not only centrality or proximity, but also structural misalignment and inter-group interface.**

Or, more concisely:

> **The idea is not to settle only for who resembles whom, but to put into the model who is between whom.**

## 9. Implementation Roadmap

1. **Replicate ws-dmaa as-is**: cascades, NETINF, classical attributes, CMF. This establishes the internal baseline and forces a deep understanding of the original pipeline.

2. **On the inferred network, detect overlapping communities and compute $\tilde{h}_v$** and its auxiliary variables. This is the first real injection of Appl. Sci.

3. **Add those signals to the user attribute matrix and repeat the evaluation.** If improvement appears here, the minimal fusion is justified.

4. **Only after that, introduce the stronger version**: boundary-guided adaptive social regularization.

This order matters greatly. If everything is done at once, it becomes impossible to know what actually contributed.
