# Core Experiment Findings — Ciao (Phase 6)

**Analysis date:** 2026-07-06 (Phase 2 refresh: `canonical_baseline --force` + `final_eval`, incl. `M4c_robustness`)  
**Protocol:** [core_experiment_plan.md](../../core_experiment_plan.md)  
**Primary result file:** `data/ciao/core_experiment_results.csv`  
**Supporting sources:** Stage A/B logs (`data/ciao/logs/`), `experiment_manifest.json`, `network_selection_results.json`, `canonical_baseline.json`  
**Dataset:** CiaoDVD, global temporal 80/20 split (28 852 train / 7 213 test), 2 248 users, 16 861 items, 36 065 ratings, seed 42  

**Scope note:** Ciao completes the same ladder as MovieLens (prerequisites → Stage A `hypertune` → Stage B `recommend --all-networks` → Phase 2 frozen evaluation). See [core_experiment_movielens_findings.md](core_experiment_movielens_findings.md) for the first dataset and [§12](#12-cross-dataset-comparison-movielens-vs-ciao) below for divergence.

---

## Experiment context

This report compares the **same model ladder** as MovieLens on CiaoDVD — a sparser trust/review network with a much larger item catalogue and weaker rating density per user.

| ID | Name | What it uses | What it tests |
| --- | --- | --- | --- |
| **M1** | Baseline CMF | User–item ratings only (matrix factorisation, L-BFGS) | Lower bound: no network topology, no side features. |
| **M2** | Enhanced CMF (centrality) | M1 + **centrality metrics** from the inferred network as user side information | Does network topology help when encoded as per-user features? |
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

**Planned pairwise comparisons** (see [core_experiment_plan.md](../../core_experiment_plan.md)):

- **M2 vs M1** — value of network-derived centrality features  
- **M3 vs M2** — value of community/boundary user attributes  
- **M4a vs M3** — value of any social regulariser  
- **M4c vs M4a / M4b** — value of boundary-aware edge weighting  
- **M4d vs M4c** — bridge preservation vs strict boundary downweighting  
- **M4c vs M1** — headline “full Phase 6 vs baseline” (also report **M3** as the test RMSE winner here)

M1 uses its **own** hyperparameter search. M2–M4 each tune CMF factors, regularisation, and (for M4) social strength and mode-specific parameters. Network type and α are selected per variant on validation CV, then frozen for the global test evaluation reported below.

---

## Executive summary

### Headline (global held-out test — authoritative)

After freezing hyperparameters from Stage B logs, selecting one network per variant (`network_selection`), and evaluating on the global test split (`final_eval`, **2026-07-06**):

| Rank | Variant | Test RMSE | Δ vs M1 | Δ vs M3 |
| ---: | --- | ---: | ---: | ---: |
| 1 | **M3** (+ communities) | **0.9217** | **−0.42%** | — |
| 2 | **M4c** (`boundary_downweight`) | **0.9238** | **−0.19%** | −0.23% |
| 3 | M4a (`uniform`) | 0.9242 | −0.15% | −0.27% |
| 4 | M4b (`community_jaccard`) | 0.9250 | −0.06% | −0.36% |
| 5 | M1 (baseline CMF) | 0.9256 | — | −0.42% |
| 6 | M2 (centrality only) | 0.9260 | −0.04% | −0.47% |
| 7 | M4d (`bridge_preserve`) | 0.9292 | −0.39% | −0.81% |
| 8 | M4c robustness (`normalized_laplacian`) | 0.9328 | −0.78% | −1.21% |

*Δ vs M1 from `rmse_delta_vs_baseline` in `core_experiment_results.csv` (same-session M1 = 0.9256). Percentages = \((\mathrm{RMSE}_\mathrm{M1} - \mathrm{RMSE}_\mathrm{variant}) / \mathrm{RMSE}_\mathrm{M1}\).*

**Key conclusions on test:**

1. **M3 is the test RMSE winner** — attribute-enhanced CMF without social regularisation beats all other variants, including every M4 mode.
2. **M4c is second overall** and the best social mode (`mean_weight`); unlike MovieLens, boundary downweight is the strongest M4 design on Ciao.
3. **No social variant beats M3 on test RMSE.** M4c comes closest (−0.23% vs M3); M4d is the worst among core M4 variants.
4. **M1 is mid-pack (5th of 8), not last.** With a proper canonical baseline search (k=50, λ≈5.25), plain CMF beats M2 and `M4c_robustness` but loses to M3/M4a/M4b/M4c.
5. **Centrality-only (M2) does not beat M1** on this refresh (0.9260 vs 0.9256) — community/boundary attributes (M3) carry the enhanced-CMF gain.
6. **`normalized_laplacian` robustness fails on test** — M4c_robustness (0.9328) is **worst overall**, worse than M1 and much worse than M4c (0.9238) despite similar CV (~0.904).
7. **Ranking metrics remain near zero** (NDCG@10 ≈ 0.0006–0.0012). Lead with RMSE; treat ranking as exploratory.

### Validation CV (Stage B — diagnostic)

On training-split CV across all networks, **M4c** has the lowest minimum RMSE (~0.8990), followed by M4a ≈ M4d < M4b < M3 < M2. Social modes show ~0.5–0.6% CV gains over M3. See §4 for detail.

---

## 1. Methodology (current)

Phase 2 pipeline steps match MovieLens:

| Step | Output | Role |
| --- | --- | --- |
| `import_manifest` | `experiment_manifest.json` | Hyperparameters + per-family CV winners from Stage B logs |
| `canonical_baseline` | `canonical_baseline.json` | Single M1 search (k=50, λ≈5.25); CV RMSE 0.920, test RMSE **0.9256** |
| `network_selection` | `network_selection_results.json` | Frozen `(diffusion_model, alpha_index)` per variant |
| `final_eval` | `core_experiment_results.csv` | Train on full train, evaluate once on global test |

**Ciao-specific operational notes (this run):**

- **Phase 2 refreshed 2026-07-06** after pipeline fix (`canonical_baseline --force` + `final_eval --all-variants`, then `M4c_robustness` network selection + final eval). Primary metrics below are from `data/ciao/core_experiment_results.csv`.
- **First Phase 2 pass (2026-07-05)** had a degenerate `canonical_baseline` global test (~10²⁸); superseded by the refresh. Code now sanity-checks and retries before save.
- **Stage B paired-baseline RMSE** sometimes blows up during network sweeps (`improvement=+100%` with huge RMSE). CV minima in §4 remain usable for network selection.
- **`network_selection_results.json`** currently lists only `M4c_robustness` (single-variant run overwrote the file). Frozen networks for M2–M4d remain in the CSV and `experiment_manifest.json`.

**Run configuration:** `--cmf-method lbfgs --cmf-maxiter 25 --n-jobs 1 --seed 42` throughout Stage B (sequential network evaluation). Batch via `./scripts/run_core_experiment.sh --dataset ciao`.

Commands to reproduce Phase 2: [core_experiment_commands.md](../../core_experiment_commands.md) §10.

---

## 2. What was run

### Prerequisites — `cascade` … `communities`

| Log | Started | Ended | Notes |
| --- | --- | --- | --- |
| `00_prerequisites.log` | 2026-07-04 14:48 | 15:12 | NetInf networks, centrality, communities |

### Stage A — `hypertune` (representative network `exponential_000`)

| Variant | Log | CV RMSE (tuning) | Trials |
| --- | --- | ---: | --- |
| M2 | `m2_hypertune.log` | 0.9028 | 50 enhanced |
| M3 | `m3_hypertune.log` | 0.9062 | 50 enhanced |
| M4a | `m4a_hypertune.log` | 0.8995 | 200 social |
| M4b | `m4b_hypertune.log` | **0.8993** | 200 social |
| M4c | `m4c_hypertune.log` | 0.8994 | 200 social |
| M4d | `m4d_hypertune.log` | 0.8996 | 200 social |

Stage A on `exponential_000`: M4b ≈ M4c ≈ M4a < M4d < M2 < M3.

### Stage B — `recommend --all-networks`

| Variant | Log | Social mode / flags |
| --- | --- | --- |
| M2 | `m2_recommend.log` | `--no-communities` |
| M3 | `m3_recommend.log` | default (communities on) |
| M4a–M4d | `m4a`–`m4d_recommend.log` | respective `social_mode` |
| M4c robustness | `m4c_robustness_laplacian.log` | `boundary_downweight` + `normalized_laplacian` |

All runs: `--cmf-method lbfgs --cmf-maxiter 25 --n-jobs 1 --seed 42`, with `--run-id m*_recommend`.

**Wall time:** prerequisites ~24 min; full ladder from `01_preregister` (2026-07-04 15:56) through Phase 2 `final_eval` (2026-07-05 22:34) ≈ **31 hours** (sequential `--n-jobs 1`).

### Phase 2 — frozen evaluation

| Session | Steps | Notes |
| --- | --- | --- |
| 2026-07-05 22:32–22:34 | First pass | Degenerate canonical baseline JSON (superseded) |
| **2026-07-06 01:01–01:03** | **`canonical_baseline --force` + `final_eval --all-variants`** | **Authoritative M1–M4d test rows** |
| 2026-07-06 01:03 | `network_selection` + `final_eval` for `M4c_robustness` | Laplacian robustness test row added |

---

## 3. Primary results — global held-out test

Source: `data/ciao/core_experiment_results.csv` (`final_eval` session **2026-07-06T01:01–01:03** for M1–M4d; `M4c_robustness` appended 01:03).

### 3.1 Accuracy metrics

| ID | Description | Network (frozen) | RMSE | MAE | R² | Δ RMSE vs M1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| **M3** | + community/boundary attrs | exponential, α₈₃ | **0.9217** | **0.7050** | **0.081** | **+0.0039** |
| M4c | + boundary_downweight | rayleigh, α₇₈ | 0.9238 | 0.7112 | 0.077 | +0.0018 |
| M4a | + uniform social | exponential, α₆₉ | 0.9242 | 0.7091 | 0.076 | +0.0014 |
| M4b | + community_jaccard | rayleigh, α₄₅ | 0.9250 | 0.7116 | 0.075 | +0.0006 |
| M1 | Baseline CMF | — | 0.9256 | 0.7093 | 0.074 | — |
| M2 | Centrality only | rayleigh, α₄₄ | 0.9260 | 0.7138 | 0.073 | −0.0004 |
| M4d | + bridge_preserve | powerlaw, α₈₈ | 0.9292 | 0.7159 | 0.067 | −0.0036 |
| M4c_robustness | + boundary_downweight, laplacian | powerlaw, α₆₄ | 0.9328 | 0.7190 | 0.059 | −0.0072 |

Canonical baseline (`canonical_baseline.json`, 2026-07-06): k=50, λ≈5.25, CV RMSE **0.9202**, global test RMSE **0.9256** (matches M1 row).

### 3.2 Ranking metrics (test)

| ID | NDCG@10 | Precision@10 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| M1 | 0.0018 | 0.0014 | 0.0014 | 0.0079 |
| M2 | 0.0010 | 0.0010 | 0.0011 | 0.0056 |
| M3 | 0.0006 | 0.0006 | 0.0002 | 0.0047 |
| M4a | 0.0007 | 0.0006 | 0.0010 | 0.0040 |
| M4b | 0.0007 | 0.0006 | 0.0002 | 0.0062 |
| **M4c** | **0.0011** | 0.0008 | 0.0006 | **0.0068** |
| M4d | 0.0011 | 0.0008 | 0.0009 | 0.0061 |
| M4c_robustness | 0.0012 | 0.0008 | 0.0009 | 0.0057 |

**Interpretation:** All ranking values are **extremely low** (sparse ratings, 16k items, 2k users). M4c/M4d show marginally higher NDCG@10 than M3, but the absolute scale (~0.1% NDCG) makes ranking claims fragile. **Lead with RMSE** for Ciao; treat ranking as exploratory.

### 3.3 Frozen network selection

From `experiment_manifest.json` / CSV (2026-07-05 network selection for M2–M4d; M4c_robustness added 2026-07-06):

| Variant | Diffusion model | α index | α value | CV RMSE (selected) |
| --- | --- | ---: | ---: | ---: |
| M2 | rayleigh | 44 | 1.75×10⁻⁴ | 0.9047 |
| M3 | exponential | 83 | 0.227 | 0.9039 |
| M4a | exponential | 69 | 0.0616 | 0.8997 |
| M4b | rayleigh | 45 | 1.92×10⁻⁴ | 0.9012 |
| **M4c** | rayleigh | 78 | 4.13×10⁻³ | **0.8990** |
| M4d | powerlaw | 88 | 4.57 | 0.8997 |
| M4c_robustness | powerlaw | 64 | 3.62 | 0.9038 |

M3’s winner is **exponential** with a relatively large α (0.23) — unlike MovieLens, where rayleigh dominated several enhanced variants.

---

## 4. Validation CV results (Stage B — supplementary)

These metrics come from training-split CV across pre-registered networks (45 networks × 3 diffusion families). They informed hyperparameter and network selection but **do not fully match test ranking** (especially M4d).

### 4.1 Social / enhanced search on tuning setup (Stage B `recommend`)

| ID | Enhanced / social CV RMSE | Notes |
| --- | ---: | --- |
| M2 | 0.9034 | enhanced search |
| M3 | 0.9017 | enhanced search |
| M4a | 0.8988 | social search |
| M4b | 0.8999 | social search |
| M4c | 0.8999 | social search |
| M4d | 0.8999 | social search |

### 4.2 Network-wide sweep — best CV RMSE per family

| Variant | Exponential | Powerlaw | Rayleigh | **Min** |
| --- | ---: | ---: | ---: | ---: |
| M2 | 0.9052 | 0.9050 | 0.9047 | 0.9047 |
| M3 | **0.9039** | 0.9040 | 0.9040 | **0.9039** |
| M4a | **0.8997** | 0.9003 | 0.8997 | **0.8997** |
| M4b | 0.9013 | 0.9013 | 0.9012 | 0.9012 |
| **M4c** | 0.8990 | 0.8994 | **0.8990** | **0.8990** |
| M4d | 0.8998 | **0.8997** | 0.8998 | **0.8997** |

### 4.3 CV vs test — where they agree and diverge

| Comparison | CV winner | Test winner | Notes |
| --- | --- | --- | --- |
| M3 vs M2 | **M3** | **M3** | Both metrics favour full attributes on Ciao |
| M4 vs M3 | All M4 (CV) | **M3** | Social regularisation does not transfer to test |
| M4c vs M4a/M4b | **M4c** | **M4c > M4a > M4b** | Boundary downweight best among M4 on both metrics |
| M4d vs M4c | M4c (slightly) | **M4c** | Bridge preserve hurts test RMSE on Ciao |
| M4c vs M1 | M4c (CV) | M4c (−0.19%) but **M3 better (−0.42%)** | Headline vs baseline weak for M4c; M3 is stronger |
| M4c vs M4c_robustness | M4c (CV) | **M4c** (0.9238 vs 0.9328) | Laplacian normalization hurts on test despite similar CV |

---

## 5. Ablation answers (plan interpretation rules)

Rules reference **global test** where available; CV noted when test is ambiguous.

### M2 vs M1 — centrality attributes

- **Test:** 0.9260 vs 0.9256 → M2 **worse** by 0.04%.
- **Verdict:** **Not supported** on this refresh — centrality alone does not beat a well-tuned baseline CMF on Ciao.

### M3 vs M2 — community/boundary attributes

- **Test:** 0.9217 vs 0.9260 → **−0.46% RMSE** in favour of M3.
- **CV:** M3 min 0.9039 vs M2 0.9047 — M3 also better.
- **Verdict:** **Supported on test and CV.** Full attribute set beats centrality-only; the enhanced-CMF gain is in community/boundary features, not centrality alone.

### M4a vs M3 — any social regularizer

- **Test:** 0.9242 vs 0.9217 → M4a **worse** by 0.27%.
- **CV:** M4a clearly better (~0.8988 vs ~0.9017).
- **Verdict:** Social regularization **helps on CV but not on test**; uniform mode does not beat M3 here.

### M4c vs M4a — boundary-aware vs uniform weighting

- **Test:** 0.9238 vs 0.9242 → M4c **slightly better** (−0.04% RMSE).
- **CV:** M4c better (0.8990 vs 0.8997 min).
- **Verdict:** **Weak support** for boundary downweight over uniform on Ciao — opposite direction to MovieLens, but effect size is tiny.

### M4c vs M4b — boundary downweight vs community Jaccard

- **Test:** 0.9238 vs 0.9250 → M4c better.
- **CV:** M4c better (0.8990 vs 0.9012).
- **Verdict:** **Boundary downweight preferred over Jaccard** on Ciao (both still lose to M3 on test).

### M4d vs M4c — bridge preserve vs boundary downweight

- **Test:** 0.9292 vs 0.9238 → **M4c better by 0.58%**.
- **CV:** Rough tie (0.8997 vs 0.8990).
- **Verdict:** **Bridge preserve is the weakest social mode on Ciao test** — opposite to MovieLens, where M4d won.

### M4c vs M1 — headline Phase 6 claim

- **Test:** 0.9238 vs 0.9256 → **−0.19% RMSE** (weak).
- **Better headline:** **M3 vs M1: −0.42%**; attribute-enhanced CMF without social reg is the clear winner.
- **Verdict:** Phase 6 beats baseline CMF modestly through M4c; **primarily through M3**.

### M4c (`mean_weight`) vs M4c_robustness (`normalized_laplacian`)

- **Test:** 0.9238 vs 0.9328 → **M4c better by 0.97%**; robustness variant is **worst overall**.
- **CV:** M4c min 0.8990 vs M4c_robustness 0.9038 — M4c also better.
- **Verdict:** **Do not use laplacian normalization on Ciao** for this design; `mean_weight` is clearly preferred on test.

### Plan rule check

> “The boundary claim needs M3 vs M2 and M4c vs M4a/M4b.”

- M3 vs M2: **passes on test and CV**.
- M4c vs M4a/M4b: **passes weakly on test and CV** (small margins).

**Overall boundary narrative on Ciao:** community/boundary **user attributes** (M3) are the strongest signal; **boundary-guided social weighting (M4c)** is the best social mode but still **does not beat M3** on RMSE.

---

## 6. Hyperparameters and search quality

### 6.1 Frozen hyperparameters (from manifest, used in `final_eval`)

| Variant | k | λ_reg | λ_social | w_main | w_user | β | γ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 50 | 5.25 | — | — | — | — | — |
| M2 | 33 | 1.81 | — | 0.46 | 0.043 | — | — |
| M3 | 41 | 0.90 | — | 0.23 | 0.016 | — | — |
| M4a | 49 | 1.71 | 0.18 | 0.51 | 0.050 | 0 | 1.0 |
| M4b | 12 | 3.54 | 0.21 | 0.89 | 0.033 | 0 | 1.0 |
| M4c | 50 | 0.33 | 0.043 | 0.10 | 0.68 | 0.044 | 1.0 |
| M4d | 27 | 2.29 | 0.31 | 0.71 | 0.071 | 0.85 | 1.27 |
| M4c_robustness | 14 | 1.36 | 0.89 | 0.55 | 0.017 | 0.43 | 1.0 |

M4c uses **low** `w_main`, **high** `w_user`, and **low** `lambda_social` with small `beta` — a different balance than MovieLens M4c (which favoured high `w_main`).

### 6.2 Optuna trial completion (Stage B social search)

| Variant | COMPLETE | PRUNED | Prune rate |
| --- | ---: | ---: | ---: |
| M4a | 132 | 68 | 34% |
| M4b | 125 | 75 | 38% |
| M4c | 92 | 108 | **54%** |
| M4d | 131 | 69 | 35% |

M4c has the highest prune rate and the best test performance among M4 variants — search instability does not imply poor generalisation here.

---

## 7. Robustness — `normalized_laplacian` vs `mean_weight` (M4c)

| Normalization | Stage B min CV RMSE | Test RMSE (`final_eval`) | Network (frozen) |
| --- | ---: | ---: | --- |
| `mean_weight` (M4c) | 0.8990 | **0.9238** | rayleigh, α₇₈ |
| `normalized_laplacian` (M4c_robustness) | 0.9038 | **0.9328** | powerlaw, α₆₄ |

**Verdict:** Laplacian normalization is **worse on CV and much worse on test** (−0.97% vs M4c, +0.78% vs M1). Keep `mean_weight` as the primary M4c configuration on Ciao.

---

## 8. Secondary findings

### 8.1 Numerical instability

The first Phase 2 pass had a degenerate `canonical_baseline` fit; the **2026-07-06 refresh** resolved it (test RMSE 0.9256). Occasional Stage B paired-baseline RMSE spikes remain in logs but do not affect frozen test evaluation.

### 8.2 SHAP

No SHAP for this campaign. Suggested finalists for interpretability: **M3** (best test RMSE) and **M4c** (best social mode / NDCG).

### 8.3 Runtime

`--n-jobs 1` made Stage B network sweeps dominate wall time (~31 h). The batch script now defaults to parallel network evaluation for future datasets.

---

## 9. Synthesis — research questions

| # | Question | Ciao answer |
| --- | --- | --- |
| 1 | Does enhanced CMF beat baseline CMF? | **Partially** — M3 −0.42% vs M1; M2 does **not** beat M1; M4c/M4a/M4b beat M1 weakly. |
| 2 | Does social regularization beat enhanced-with-attributes? | **No** — M3 beats all M4; M4c closest (−0.23% vs M3). |
| 3 | Which network and α? | **Per variant**; see §3.3. |
| 4 | Laplacian vs mean_weight (M4c)? | **`mean_weight` wins** on test (0.9238 vs 0.9328). |

---

## 10. Claims safe vs unsafe for writing

### Supported on global test

- **M3 gives the best test RMSE** on Ciao (0.9217).
- Full attribute set (M3) beats centrality-only (M2) on RMSE.
- M4c is the **best social mode** (second overall); M4a/M4b also beat M1 weakly.
- **`mean_weight` beats `normalized_laplacian`** for boundary-downweight social CMF (M4c vs M4c_robustness).
- Per-variant network selection is required (α and diffusion family differ).

### Not supported / contradicted

- “Social regularization beats M3 on RMSE” (none do on Ciao).
- “M4d is the best social mode” (worst among core M4 on test).
- “Centrality features alone beat baseline CMF” (M2 slightly worse than M1 on refresh).
- “Laplacian normalization is a viable M4c alternative” (worst test RMSE overall).
- “M1 is always worst on Ciao” (M1 is 5th of 8 after canonical refresh).

### Open

- SHAP on M3 and M4c.
- Epinions third dataset (if planned).
- Restore full `network_selection_results.json` (re-run `network_selection --all-variants`) if a single JSON file for all variants is needed.

---

## 11. Recommended next steps

1. **Paper tables:** Use `core_experiment_results.csv` (2026-07-06 refresh); report MovieLens and Ciao side by side (§12).
2. **SHAP:** Run on **M3** (accuracy winner) and **M4c** (best social mode).
3. **Epinions:** Repeat full ladder with parallel `--n-jobs` where stable.
4. **Optional housekeeping:** Re-run `network_selection --all-variants` to repopulate `network_selection_results.json` for all variants.

---

## 12. Cross-dataset comparison (MovieLens vs Ciao)

| Topic | MovieLens | Ciao |
| --- | --- | --- |
| **Test RMSE winner** | M4d (1.0243) | **M3 (0.9217)** |
| **M1 rank** | Mid (6th of 7) | **Mid (5th of 8)** |
| **M3 vs M1** | −2.4% | −0.42% |
| **M2 vs M1** | M2 wins | **M2 loses** (centrality alone insufficient) |
| **Social beats M3?** | Only M4d (−0.18%) | **No** (best M4c −0.23% vs M3) |
| **Best M4 mode** | M4d (`bridge_preserve`) | **M4c** (`boundary_downweight`) |
| **Laplacian robustness** | CV slightly worse | **Test much worse** (0.9328 vs M4c 0.9238) |
| **NDCG@10 scale** | 0.07–0.40 (usable) | ~0.001 (fragile) |

**Narrative implication:** The Phase 6 ladder is **not monotonic across datasets**. Attribute-enhanced CMF (M3) is a robust improvement over M1 on both sets; **social regularisation is not a universal win over M3** — it helps on MovieLens only through M4d, while on Ciao **no social variant beats M3**, and M4c is the preferred social design when social terms are discussed. Papers should present **both datasets** before claiming a single “best” social mode.

---

## Appendix A — Artifact index

| File | Role |
| --- | --- |
| `data/ciao/core_experiment_results.csv` | **Primary** — global test metrics |
| `data/ciao/network_selection_results.json` | Frozen networks (currently M4c_robustness only; see §1) |
| `data/ciao/experiment_manifest.json` | Hyperparameters + selected networks (all variants) |
| `data/ciao/canonical_baseline.json` | Canonical M1 (k=50, test RMSE 0.9256) |
| `data/ciao/pipeline.log` | Phase 2 refresh logs (2026-07-06) |
| `data/ciao/logs/m*_*.log` | Stage A/B logs |
| `data/ciao/runs/m*_recommend/` | Archived per-variant network metrics |

## Appendix B — Phase 2 commands (reference)

Phase 2 for Ciao was **completed 2026-07-06**. To reproduce:

```bash
# Core ladder test evaluation
$HOME/anaconda3/envs/mafpin/bin/python pipeline.py \
  --steps canonical_baseline final_eval \
  --dataset ciao --all-variants --force \
  --cmf-method lbfgs --cmf-maxiter 25 --seed 42

# M4c laplacian robustness (append row to CSV)
$HOME/anaconda3/envs/mafpin/bin/python pipeline.py \
  --steps network_selection final_eval \
  --dataset ciao --model-variant M4c_robustness \
  --cmf-method lbfgs --cmf-maxiter 25 --seed 42
```

See [core_experiment_commands.md](../../core_experiment_commands.md) §10 for the full command sequence.
