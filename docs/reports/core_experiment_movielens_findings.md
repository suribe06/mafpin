# Core Experiment Findings — MovieLens (Phase 6)

**Analysis date:** 2026-06-30 (updated after Phase 2 `final_eval`)  
**Protocol:** [core_experiment_plan.md](../core_experiment_plan.md)  
**Primary result file:** `data/movielens/core_experiment_results.csv`  
**Supporting sources:** Stage A/B logs (`data/movielens/logs/`), `experiment_manifest.json`, `network_selection_results.json`, `canonical_baseline.json`, MLflow (`mlruns/`, experiment `mafpin`)  
**Dataset:** MovieLens (`ratings_small`), global temporal 80/20 split (80 003 train / 20 001 test), seed 42  

**Scope note:** Only **MovieLens** has the full ladder (Stage A + B + Phase 2 final evaluation). Ciao has smoke-test logs only.

---

## Experiment context

This report compares a **ladder of recommender models** built on the same ratings split and the same inferred user–user networks (NetInf: exponential, power-law, Rayleigh diffusion). Each step adds one design choice so we can see *where* any improvement comes from.

| ID | Name | What it uses | What it tests |
| --- | --- | --- | --- |
| **M1** | Baseline CMF | User–item ratings only (matrix factorisation, L-BFGS) | Lower bound: no network topology, no side features. |
| **M2** | Enhanced CMF (centrality) | M1 + **centrality metrics** from the inferred network as user side information | Does network topology help when encoded as per-user features (ws-dmaa style)? |
| **M3** | Enhanced CMF (full attributes) | M2 + **community membership** and **boundary** features | Do overlapping-community and boundary signals add value beyond centrality alone? |
| **M4a** | Social CMF (uniform) | M3-style user attributes + **social graph regularisation** with **uniform** edge weights | Does the graph regulariser itself help, independent of boundary-aware weighting? |
| **M4b** | Social CMF (community Jaccard) | Same as M4a, but edge weights reflect **shared-community overlap** (Jaccard) | Community-aware smoothing without explicit boundary downweighting. |
| **M4c** | Social CMF (boundary downweight) | Same as M4a, but edges across community **boundaries are downweighted** (main Phase 6 design) | Boundary-guided adaptive social regularisation. |
| **M4d** | Social CMF (bridge preserve) | Same as M4a, but a **bridge-preserve** rule keeps some cross-boundary links | More flexible boundary handling; secondary variant unless it clearly wins. |

**How the ladder is read:**

```text
M1  →  plain collaborative filtering
M2  →  + network centrality as side information
M3  →  + community / boundary attributes
M4* →  + social regularisation on the inferred graph (mode a/b/c/d)
```

**Planned pairwise comparisons** (see [core_experiment_plan.md](../core_experiment_plan.md)):

- **M2 vs M1** — value of network-derived centrality features  
- **M3 vs M2** — value of community/boundary user attributes  
- **M4a vs M3** — value of any social regulariser  
- **M4c vs M4a / M4b** — value of boundary-aware edge weighting  
- **M4d vs M4c** — bridge preservation vs strict boundary downweighting  
- **M4c vs M1** — headline “full Phase 6 vs baseline” (in practice we also report **M4d** and **M3** as stronger test winners)

M1 uses its **own** hyperparameter search. M2–M4 each tune CMF factors, regularisation, and (for M4) social strength and mode-specific parameters. Network type and α are selected per variant on validation CV, then frozen for the global test evaluation reported below.

---

## Executive summary

### Headline (global held-out test — authoritative)

After freezing hyperparameters from Stage B logs, selecting one network per variant (`network_selection`), and evaluating once on the global test split (`final_eval`):

| Rank | Variant | Test RMSE | Δ vs M1 | Δ vs M3 |
| ---: | --- | ---: | ---: | ---: |
| 1 | **M4d** (`bridge_preserve`) | **1.0243** | **−2.5%** | **−0.18%** |
| 2 | **M3** (+ communities) | **1.0261** | **−2.4%** | — |
| 3 | M4a (`uniform`) | 1.0322 | −1.8% | −0.60% |
| 4 | M2 (centrality only) | 1.0385 | −1.2% | −1.23% |
| 5 | M4b (`community_jaccard`) | 1.0410 | −1.0% | −1.49% |
| 6 | M1 (baseline CMF) | 1.0510 | — | −2.49% |
| 7 | M4c (`boundary_downweight`) | 1.0421 | −0.85% | −1.56% |

**Key conclusions on test:**

1. **Enhanced CMF with network features beats plain CMF (M1).** M3 reduces RMSE by ~2.4% vs M1; M2 also wins (~1.2%).
2. **Social regularization does not uniformly beat M3 on test.** M4d is the only social variant that clearly beats M3 on RMSE; M4a is close; **M4b and M4c are worse than M3** despite strong CV numbers.
3. **M4c is not the best social mode** — on test it ranks last among M4 variants for RMSE. **M4d (`bridge_preserve`)** is the test winner.
4. **Community/boundary attributes (M3 vs M2) help on test** — opposite to validation CV, where M3 was slightly worse. The test gap is ~1.2% RMSE in favour of M3.
5. **CV and test rankings diverge** for social modes (especially M4b). Validation sweeps remain useful for tuning but **paper claims must use `core_experiment_results.csv`**.

### Validation CV (Stage B — diagnostic)

On training-split CV across all networks, social modes still show the pattern from the first analysis pass: M4b ≈ M4d < M4a < M4c, with ~0.5–1.8% RMSE gains over M3. See §4 for detail.

---

## 1. Methodology (current)

Phase 2 pipeline steps close the gaps identified in the first analysis pass:

| Step | Output | Role |
| --- | --- | --- |
| `import_manifest` | `experiment_manifest.json` | Hyperparameters + per-family CV winners from Stage B logs |
| `canonical_baseline` | `canonical_baseline.json` | Single M1 search (k=39, λ≈7.01); CV RMSE 0.895, test RMSE 1.041 |
| `network_selection` | `network_selection_results.json` | Frozen `(diffusion_model, alpha_index)` per variant |
| `final_eval` | `core_experiment_results.csv` | Train on full train, evaluate once on global test |

**Resolved in codebase (no longer affect new runs):**

- Missing global test for M2–M4 → `final_eval` step.
- Non-canonical M1 per `recommend` run → `canonical_baseline`.
- Unfrozen network choice → `network_selection`.
- Bogus `improvement_pct` when paired baseline RMSE is degenerate → sanity filter in `network_eval.py`.
- L-BFGS segfault at `nthreads > 1` → forced single-thread for L-BFGS.
- Non-finite Optuna trials crashing search → pruning in baseline/enhanced/social search.

**Remaining operational notes:**

- Shared `inferred_edges_*.csv` columns can still be overwritten across `recommend` runs; use `--run-id` to archive per-variant snapshots under `data/<dataset>/runs/<run_id>/`.
- `M4c_robustness` (`normalized_laplacian`) has Stage B CV logs only; **no `final_eval` row yet**.
- SHAP (`shap_results.json`, 2026-05-20) predates this campaign.

Commands to reproduce Phase 2: [core_experiment_commands.md](../core_experiment_commands.md) §10.

---

## 2. What was run

### Stage A — `hypertune` (representative network `exponential_000`)

| Variant | Log | CV RMSE (tuning) | Trials |
| --- | --- | ---: | --- |
| M2 | `m2_hypertune.log` | 0.8855 | 50 enhanced |
| M3 | `m3_hypertune.log` | 0.8856 | 50 enhanced |
| M4a | `m4a_hypertune.log` | 0.8728 | 200 social |
| M4b | `m4b_hypertune.log` | 0.8736 | 200 social |
| M4c | `m4c_hypertune.log` | **0.8726** | 200 social |
| M4d | `m4d_hypertune.log` | 0.8733 | 200 social |

Stage A on `exponential_000`: M4c ≈ M4a < M4d < M4b < M3 ≈ M2.

### Stage B — `recommend --all-networks`

| Variant | Log | Social mode / flags |
| --- | --- | --- |
| M2 | `m2_recommend.log` | `--no-communities` |
| M3 | `m3_recommend.log` | default (communities on) |
| M4a–M4d | `m4a`–`m4d_recommend.log` | respective `social_mode` |
| M4c robustness | `m4c_robustness_laplacian.log` | `boundary_downweight` + `normalized_laplacian` |

All runs: `--cmf-method lbfgs --cmf-maxiter 25 --n-jobs 1 --seed 42`.

### Phase 2 — frozen evaluation

Executed 2026-06-29 via `import_manifest` → `canonical_baseline` → `network_selection` → `final_eval --all-variants`.

---

## 3. Primary results — global held-out test

Source: `data/movielens/core_experiment_results.csv` (single `final_eval` session).

### 3.1 Accuracy metrics

| ID | Description | Network (frozen) | RMSE | MAE | R² | Δ RMSE vs M1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| M1 | Baseline CMF | — | 1.0510 | 0.8185 | 0.069 | — |
| M2 | Centrality only | rayleigh, α₀ | 1.0385 | 0.8001 | 0.091 | −0.0125 |
| M3 | + community/boundary attrs | rayleigh, α₆₃ | **1.0261** | **0.7955** | **0.113** | **−0.0249** |
| M4a | + uniform social | rayleigh, α₂₁ | 1.0322 | 0.7971 | 0.102 | −0.0188 |
| M4b | + community_jaccard | powerlaw, α₄₇ | 1.0410 | 0.8017 | 0.087 | −0.0100 |
| M4c | + boundary_downweight | rayleigh, α₂₁ | 1.0421 | 0.8023 | 0.085 | −0.0089 |
| **M4d** | + bridge_preserve | powerlaw, α₄₃ | **1.0243** | **0.7935** | **0.116** | **−0.0267** |

Canonical baseline from `canonical_baseline.json` (same search, earlier fit): test RMSE **1.0412**. M1 in `final_eval` is **1.0510** — small L-BFGS nondeterminism between runs. **Within-table comparisons use the same `final_eval` session** and are internally consistent.

### 3.2 Ranking metrics (test)

| ID | NDCG@10 | Precision@10 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| M1 | 0.375 | 0.330 | 0.046 | 0.656 |
| M2 | 0.373 | 0.343 | 0.048 | 0.653 |
| M3 | 0.260 | 0.218 | 0.030 | 0.606 |
| M4a | 0.136 | 0.085 | 0.012 | 0.525 |
| M4b | 0.070 | 0.052 | 0.009 | 0.263 |
| **M4c** | **0.401** | **0.365** | **0.049** | **0.657** |
| M4d | 0.173 | 0.113 | 0.018 | 0.573 |

**Interpretation:** RMSE and ranking metrics **disagree** for social models. M4c achieves the **best NDCG@10** despite mediocre RMSE; M4a/M4b collapse on ranking while M3 is mid-tier. If the paper targets rating accuracy, lead with RMSE (M4d/M3); if top-N recommendation quality, M4c warrants separate reporting — not as the overall “best” model without stating the metric.

### 3.3 Frozen network selection

From `network_selection_results.json` (best CV RMSE per family from Stage B logs):

| Variant | Diffusion model | α index | α value | CV RMSE (selected) |
| --- | --- | ---: | ---: | ---: |
| M2 | rayleigh | 0 | 5.74×10⁻⁶ | 0.8883 |
| M3 | rayleigh | 63 | 0.00201 | 0.8885 |
| M4a | rayleigh | 21 | 4.05×10⁻⁵ | 0.8838 |
| M4b | powerlaw | 47 | 2.95 | 0.8828 |
| M4c | rayleigh | 21 | 4.05×10⁻⁵ | 0.8854 |
| M4d | powerlaw | 43 | 2.79 | 0.8834 |

No single diffusion family dominates; α shifts with social mode — per-variant selection is necessary.

---

## 4. Validation CV results (Stage B — supplementary)

These metrics come from training-split CV across up to 100 α × 3 diffusion families. They informed hyperparameter and network selection but **do not match test ranking** for all variants.

### 4.1 Tuning network (`exponential_000`)

| ID | CV RMSE | Δ vs M2 | Δ vs M3 |
| --- | ---: | ---: | ---: |
| M2 | 0.8869 | — | — |
| M3 | 0.8872 | +0.0003 (worse) | — |
| M4a | 0.8714 | −0.0155 | −0.0158 |
| M4b | **0.8708** | −0.0161 | −0.0164 |
| M4c | 0.8721 | −0.0148 | −0.0151 |
| M4d | 0.8705 | −0.0164 | −0.0167 |

### 4.2 Network-wide sweep — best CV RMSE per family

| Variant | Exponential | Powerlaw | Rayleigh | **Min** |
| --- | ---: | ---: | ---: | ---: |
| M2 | 0.8883 | 0.8884 | 0.8883 | 0.8883 |
| M3 | 0.8890 | 0.8894 | 0.8885 | 0.8885 |
| M4a | 0.8840 | 0.8841 | 0.8838 | 0.8838 |
| **M4b** | 0.8829 | **0.8828** | 0.8828 | **0.8828** |
| M4c | 0.8855 | 0.8856 | 0.8854 | 0.8854 |
| M4d | 0.8837 | 0.8834 | 0.8835 | 0.8834 |

### 4.3 CV vs test — where they agree and diverge

| Comparison | CV winner | Test winner | Notes |
| --- | --- | --- | --- |
| M3 vs M2 | M2 (slightly) | **M3** | Community attrs help on test, not on CV |
| M4 vs M3 | All M4 | **Only M4d** (RMSE) | M4b best on CV, worst among M4 on test RMSE |
| M4c vs M4a/M4b | M4b | **M4a > M4d > M4c > M4b** | Boundary downweight never wins on either metric set for RMSE |
| M4c vs M1 | M4c (CV) | M4c (−0.9% RMSE) | Headline “beats baseline” holds weakly for M4c; **M4d/M3 beat it clearly** |

---

## 5. Ablation answers (plan interpretation rules)

Rules reference **global test** where available; CV noted when test is ambiguous.

### M2 vs M1 — centrality attributes

- **Test:** 1.0385 vs 1.0510 → **−1.2% RMSE**. Centrality side-information helps on held-out data.
- **Verdict:** **Supported** on MovieLens test.

### M3 vs M2 — community/boundary attributes

- **Test:** 1.0261 vs 1.0385 → **−1.2% RMSE** in favour of M3.
- **CV:** M3 slightly worse.
- **Verdict:** **Supported on test**; CV was misleading. Community/boundary features add value for rating accuracy on this split.

### M4a vs M3 — any social regularizer

- **Test:** 1.0322 vs 1.0261 → M4a **worse** by 0.6%.
- **CV:** M4a clearly better.
- **Verdict:** Social regularization **helps on CV but not guaranteed on test**; uniform mode does not beat M3 here.

### M4c vs M4a — boundary-aware vs uniform weighting

- **Test:** 1.0421 vs 1.0322 → M4c worse.
- **CV:** M4c worse.
- **Verdict:** **Boundary downweighting does not beat uniform** on RMSE.

### M4c vs M4b — boundary downweight vs community Jaccard

- **Test:** 1.0421 vs 1.0410 → both worse than M3; M4c slightly worse than M4b.
- **CV:** M4b clearly better.
- **Verdict:** **Community Jaccard preferred over boundary downweight** on CV; on test both underperform M3.

### M4d vs M4c — bridge preserve vs boundary downweight

- **Test:** 1.0243 vs 1.0421 → **M4d better by 1.7%**.
- **CV:** M4d better.
- **Verdict:** **Bridge preserve is the strongest social mode on test RMSE.**

### M4c vs M1 — headline Phase 6 claim

- **Test:** 1.0421 vs 1.0510 → **−0.85% RMSE** (weak).
- **Better headline:** **M4d vs M1: −2.5%**; **M3 vs M1: −2.4%**.
- **Verdict:** Phase 6 improves over baseline CMF, but **not primarily through M4c** — attribute-enhanced M3 and bridge-social M4d drive the gain.

### Plan rule check

> “The boundary claim needs M3 vs M2 and M4c vs M4a/M4b.”

- M3 vs M2: **passes on test** (fails on CV).
- M4c vs M4a/M4b: **fails on test and CV**.

**Overall boundary narrative:** community/boundary **user attributes** may be defensible (M3 > M2 on test); **boundary-guided social weighting (M4c)** is **not** supported as best design.

---

## 6. Hyperparameters and search quality

### 6.1 Frozen hyperparameters (from manifest, used in `final_eval`)

| Variant | k | λ_reg | λ_social | w_main | w_user | β | γ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 39 | 7.01 | — | — | — | — | — |
| M2 | 23 | 5.59 | — | 0.89 | 0.56 | — | — |
| M3 | 9 | 4.27 | — | 0.67 | 0.60 | — | — |
| M4a | 46 | 1.79 | 0.71 | 0.72 | 0.42 | 0 | 1.0 |
| M4b | 44 | 2.43 | 0.81 | 0.93 | 0.47 | 0 | 1.0 |
| M4c | 45 | 4.98 | 0.14 | 0.96 | 0.013 | 0.95 | 1.0 |
| M4d | 37 | 2.20 | 0.30 | 0.66 | 0.45 | 0.58 | 1.98 |

M4c relies on high `w_main`, very low `w_user`, and moderate `lambda_social` with high `beta` — a different balance than M4d.

### 6.2 Optuna trial completion (Stage B social search)

| Variant | COMPLETE | PRUNED | Prune rate |
| --- | ---: | ---: | ---: |
| M4a | 129 | 71 | 35% |
| M4b | 79 | 121 | **61%** |
| M4c | 115 | 85 | 43% |
| M4d | **182** | 18 | **9%** |

M4b has the best CV metrics but the highest prune rate; M4d has the most stable search and the best test RMSE — a useful robustness argument for M4d.

---

## 7. Robustness — `normalized_laplacian` vs `mean_weight` (M4c)

| Normalization | Tuning CV RMSE | Best rayleigh CV RMSE |
| --- | ---: | ---: |
| `mean_weight` (M4c) | 0.8721 | 0.8854 |
| `normalized_laplacian` | 0.8730 | 0.8887 |

**CV:** Laplacian normalization is uniformly worse (~0.3%).  
**Test:** `final_eval` for `M4c_robustness` not yet run — add with `--model-variant M4c_robustness` when needed.

---

## 8. Secondary findings

### 8.1 Stage A vs Stage B hyperparameter drift

M4c tuning-network CV: Stage A 0.8726 vs Stage B 0.8721 — small drift; acceptable.

### 8.2 SHAP

No SHAP for this campaign. Suggested finalists for interpretability: **M4d** (best test RMSE) and/or **M4c** (best NDCG@10), not M4c alone on accuracy grounds.

### 8.3 Ciao dataset

Not evaluated — cross-dataset claims remain open.

---

## 9. Synthesis — research questions

| # | Question | MovieLens answer |
| --- | --- | --- |
| 1 | Does enhanced CMF beat baseline CMF? | **Yes on test** — M3 −2.4%, M2 −1.2% vs M1. |
| 2 | Does social regularization beat enhanced-with-attributes? | **Mixed** — M4d wins (−0.18% vs M3); M4a/M4b/M4c lose on RMSE. |
| 3 | Which network and α? | **Per variant**; test winners use rayleigh (M2/M3/M4a/M4c) or powerlaw (M4b/M4d). |

---

## 10. Claims safe vs unsafe for writing

### Supported on global test

- Enhanced CMF with network-derived user features beats plain CMF (M2, M3 vs M1).
- Full attribute set (M3) beats centrality-only (M2) on RMSE.
- Bridge-preserve social regularization (M4d) gives the **best test RMSE** in the ladder.
- M4c improves weakly over M1 on RMSE but is **not** the best Phase 6 variant.
- Per-variant network selection is required (α and diffusion family differ).

### Not supported / contradicted

- “M4c is the best social mode” (last on test RMSE among M4).
- “M4b is the best overall” (best on CV, poor on test RMSE and ranking).
- “Boundary-downweight social weighting beats uniform/Jaccard” (M4c loses to M4a on test).
- “Social regularization always beats M3” (only M4d does on RMSE).

### Open

- Ciao replication.
- `final_eval` for M4c + `normalized_laplacian`.
- SHAP on M4d and/or M4c.
- Whether NDCG-first narrative (M4c) vs RMSE-first (M4d) should be dual-track in the paper.

---

## 11. Recommended next steps

1. **Paper tables:** Lead with `core_experiment_results.csv`; relegate CV sweep to appendix or hyperparameter-selection subsection.
2. **Ciao:** Repeat Phase 2 sequence (`import_manifest` → `canonical_baseline` → `network_selection` → `final_eval`).
3. **SHAP:** Run on **M4d** (accuracy) and optionally **M4c** (ranking).
4. **Robustness:** `final_eval --model-variant M4c_robustness` to close laplacian test comparison.
5. **Future `recommend` runs:** Always pass `--run-id` for archival.

---

## Appendix A — Artifact index

| File | Role |
| --- | --- |
| `data/movielens/core_experiment_results.csv` | **Primary** — global test metrics |
| `data/movielens/network_selection_results.json` | Frozen networks |
| `data/movielens/experiment_manifest.json` | Imported hyperparameters |
| `data/movielens/canonical_baseline.json` | Canonical M1 |
| `data/movielens/logs/m*_*.log` | Stage A/B logs |
| `data/movielens/logs/phase2_*.log` | Phase 2 pipeline logs |

## Appendix B — Reproducing Phase 2

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps import_manifest canonical_baseline network_selection final_eval \
  --dataset movielens --all-variants \
  --cmf-method lbfgs --cmf-maxiter 25 --seed 42
```

See [core_experiment_commands.md](../core_experiment_commands.md) §10 for full command sequence and per-variant options.
