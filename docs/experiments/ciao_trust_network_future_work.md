# Ciao Explicit Trust Graph — Suggestions & Future Work

**Status:** Proposal (post core experiment refresh, 2026-07-06)  
**Related:** [core_experiment_ciao_findings.md](core_experiment_ciao_findings.md), [core_experiment_movielens_findings.md](core_experiment_movielens_findings.md), `networks/social.py`, `datasets/ciao/trust.txt`

---

## Context

The CiaoDVD dataset ships with an **explicit user–user trust graph** (`trust.txt`: ~57k directed edges, ~2.3k users). MovieLens has no equivalent.

The Phase 6 core experiment ladder (M1–M4d) builds all network-derived features and social regularisation on **NetInf-inferred influence networks** from rating cascades — not on the declared trust graph.

Code to load and analyse the trust graph already exists in `networks/social.py` (`load_trust_graph`, `compute_trust_features`, `compute_neighbourhood_overlap`, `compute_alignment_ratio`). It is **not wired into** the `recommend` / `final_eval` pipeline today (see pipeline review Issue 15).

In the current Ciao split (2 248 rating users), **2 238** also appear in the trust graph (10 rating-only, 104 trust-only).

---

## Trust vs NetInf — different objects

| Source | What it represents | How it is obtained |
| --- | --- | --- |
| **Ciao trust** | Declared “user A trusts user B” | Original dataset (social link) |
| **NetInf** | Latent influence / diffusion among users who adopt items at similar times | Inferred from cascades on the **training** rating split |

Literature on trust-aware recommendation (e.g. mTrust / eTrust, Tang et al.) treats these as **complementary**: trust is explicit and relatively static; cascade-based influence is behavioural and model-dependent (exponential / power-law / Rayleigh, α).

**Do not assume** inferred edges ≈ trust edges. Quick check on the frozen M3 network (exponential, α index 83):

| Metric | Value |
| --- | ---: |
| Inferred edges (mapped to user IDs) | 4 496 |
| Global alignment \|E_inferred ∩ E_trust\| / \|E_inferred\| | **~1.5%** |
| Mean Jaccard overlap of out-neighbourhoods | **~0.001** |

NetInf on Ciao co-rating cascades is recovering a **different** structure than the declared trust graph. That does not invalidate NetInf; it supports using Ciao as a **validation dataset** where explicit social ground truth exists.

**Additional caveat:** Ciao timestamps are often day-granular (many zero inter-event deltas within an item). That makes cascades harder for NetInf than on MovieLens and may contribute to numerical instability and dataset-specific social-mode rankings.

---

## What the trust graph is useful for

### 1. Methodological validation (high value, low cost)

**Goal:** Quantify how inferred networks relate to declared trust.

- Alignment ratio and neighbourhood Jaccard per (diffusion model, α index).
- Correlate alignment with Stage B CV RMSE and `final_eval` test RMSE per variant.
- **Research question:** Do “better” inferred networks align more with trust, or is improvement orthogonal?

**Deliverable:** One analysis script + figures (heatmap alignment vs α; scatter alignment vs Δ RMSE). No re-run of the 31 h Stage B batch required.

### 2. Trust-derived user features (medium cost)

**Goal:** Augment CMF user attributes without replacing NetInf.

`load_social_features(dataset, G_inferred)` already returns:

- `trust_in_degree`, `trust_out_degree`, `trust_pagerank`
- `jaccard_overlap` (trust vs inferred neighbourhood)

**Possible ablation:** Merge into the M3 attribute matrix (or a thin `M3+trust` variant) and run a single `final_eval` row.

### 3. Trust-graph ladder branch — `M_trust` (medium cost, Ciao-only)

**Goal:** Same M2/M3/M4 machinery but **social topology = trust.txt** instead of a NetInf network.

| Variant | Idea |
| --- | --- |
| `M2_trust` | Centrality on trust graph |
| `M3_trust` | + communities / boundary on trust |
| `M4*_trust` | Social regularisation on trust edges |

**Why:** Answers “does behavioural inference beat declared social structure on Ciao?” MovieLens cannot run this branch.

**Scope:** One `final_eval` session per finalist design — not a full α × 3-model NetInf sweep.

### 4. Alternative graph methods (lower priority / discussion only)

Reasonable comparisons for a related-work or limitations subsection — **not** a full replacement of the NetInf-centric ladder without paper justification:

| Method | Role |
| --- | --- |
| Trust graph (explicit) | Strong social baseline on Ciao |
| NetInf (current) | Influence from temporal co-adoption |
| User–item bipartite projections | Taste similarity, not social declaration |
| Embeddings (node2vec, etc.) on trust or bipartite | Heavier; optional extension |

The project thesis is **cascade-based influence inference**; trust is best positioned as **ground truth for validation** and **competitive baseline**, not as the only network source.

---

## Recommended priority order

1. **Trust–NetInf alignment report** — script + plots; cite in Ciao findings §discussion.
2. **`M3_trust` or minimal trust baseline** — one row in `core_experiment_results.csv` for Ciao.
3. **Trust overlap features** — only if (1) shows user-level overlap predicts when M3/M4 help.
4. **SHAP on finalists** (M3, M4c, M4d) — already planned; independent of trust work.
5. **Epinions** — repeats trust-available validation (also has `trust.txt`).

**Explicitly not recommended now:** Re-running the full Stage B `recommend --all-networks` ladder (~31 h on Ciao with `--n-jobs 1`) solely to switch graph source.

---

## Concrete next steps

### Step A — Alignment analysis (1–2 h implementation)

```text
For each frozen network in experiment_manifest / CSV:
  - Load inferred edges (map NetInf node IDs: user_id = netinf_id - 1_000_000)
  - compute_alignment_ratio(G_trust, G_inferred)
  - compute_neighbourhood_overlap → distribution of jaccard_overlap
  - Join with per-network CV RMSE from archived runs/*/network_metrics
Output: data/ciao/trust_netinf_alignment.json + plots/
```

Existing helpers: `networks/social.py`.

### Step B — `M_trust` minimal eval (few hours)

```text
1. Build pipeline path: trust graph → centrality / communities (same steps as inference branch, different edge source)
2. Reuse M3 hyperparameters from manifest OR re-tune only k, λ, w_* on trust features
3. final_eval --model-variant M3_trust (Ciao only)
4. Add row to findings: M3 (NetInf) vs M3_trust (explicit)
```

### Step C — Paper narrative (no new compute)

- Ciao = **dual network** dataset: declared trust + inferred influence.
- Report low edge alignment (~1.5%) to justify that Phase 6 adds **behavioural** signal beyond copying the social graph.
- If `M3_trust` beats `M3`, state that honestly; if not, strengthens the NetInf storyline.

### Step D — Code hygiene (optional)

- Wire `network_selection` to **merge** JSON on single-variant runs (avoid overwriting `network_selection_results.json` with one variant).
- Extend `parse_experiment_logs.py` with `--dataset` (still MovieLens-hardcoded).

---

## Open questions

1. Does higher trust–inferred overlap predict larger RMSE gains for M3 vs M1 at user level?
2. Should social regularisation on **trust** (M4c_trust) beat M4c on NetInf when alignment is low?
3. Is day-level timestamp granularity hurting NetInf enough to justify trust-first features on Ciao only?
4. Epinions trust graph is denser (~140k edges) — does alignment with NetInf improve there?

---

## References

- Ciao dataset readme: `datasets/ciao/readme.txt` (trustnetwork.mat → `trust.txt`)
- Tang, Gao, Liu, Das Sarma — **eTrust: Understanding Trust Evolution in an Online World** (KDD 2012). [PDF](https://www.cse.msu.edu/~tangjili/publication/trustEvolution.pdf)
- Tang, Gao, Liu — **mTrust: Discerning Multi-faceted Trust in a Connected World** (WSDM 2012) — related multi-faceted trust on review platforms
- Gomez-Rodriguez et al., NetInf (KDD 2010; TKDD 2012) — infer diffusion networks from cascades
- KONECT CiaoDVD trust network statistics: [librec-ciaodvd-trust](http://www.konect.cc/networks/librec-ciaodvd-trust/)
