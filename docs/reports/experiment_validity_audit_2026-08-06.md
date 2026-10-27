# Experiment Validity Audit — Hypothesis Discard Safety

**Date:** 2026-08-06  
**Scope:** All experiments under `docs/experiments/` (core, cold-start, Route B) vs current implementation  
**Question:** Were discarded hypotheses discarded because the science failed, or because of experimental bugs?

---

## Executive verdict

**No classic train→test rating leakage** was found in the paths that produced the published findings (cascades from train, Optuna/network selection on train CV, scalers fit on train users, Phase-2 freeze-then-test). Historical Phase-6 leakage issues appear fixed in the current pipeline path.

**There are protocol / construct anomalies that can falsely weaken or discard some claims** — especially social-regularization / M4\* conclusions on the **core global-temporal test**, and any residual Route B “beyond-accuracy” positives. Cold-start **discards** of H1-stronger (and Ciao H2) look experimentally safe.

| Track | Classic leakage? | Safe to treat discards as settled? | Main risk |
|---|---|---|---|
| Core | No | **Partial — M4\* discards not settled** | Warm CV vs cold-dominated global test |
| Cold-start | No (controlled path) | **Yes** for H1-stronger / Ciao H2 | False *accepts* (H4, ML H1-gain) more than false rejects |
| Route B | No | **Yes** for strong-GO / M4c titular / soft NO-GO *if framed as unsupported* | Over-claiming weak positives; B2 “refutation” unsafe |

---

## Anomaly registry

Severity: **Critical** = can flip a discarded hypothesis; **Important** = changes interpretation / power; **Minor** = hygiene / docs.

### A1 — Critical — Core: global temporal test is mostly cold-start; CV is warm

| | |
|---|---|
| **Docs claim** | Global temporal 80/20 for final eval; Stage B / Optuna on train CV |
| **Code** | `recommender/data.py::split_data_temporal` (global cutoff); `recommender/enhanced/model.py::iter_warm_splits` + `filter_to_feature_users` for CV |
| **Measured** | MovieLens test: **82.7%** cold-user ratings, **9.3%** warm (user∧item). Ciao: **36.8%** cold-user, **76.7%** cold-item, **12.7%** warm |
| **Why it matters** | Social edges / NetInf features cannot touch MovieLens cold isolates. CV selects networks where social helps *warm* users; aggregate test RMSE is dominated by users where M4 does nothing |
| **Impact on discards** | **High flip risk** for “social / M4c fails vs M3” and “boundary social weighting discarded.” Findings already note CV↔test divergence; they understate that the mismatch is largely **regime mismatch**, not only overfitting |
| **Fix / re-eval** | Report warm-user (or per-user leave-last) test RMSE alongside aggregate; do not discard M4\* from aggregate alone |

### A2 — Important — Config docstring promises per-user temporal; code is global

| | |
|---|---|
| **Where** | `config.py` Split docstring (“last TEST_SIZE fraction of *each user’s* interactions”) vs `split_data_temporal` |
| **Actual** | Global chronological cutoff (locked by `tests/test_data_splits.py`); findings correctly say global |
| **Impact** | Docs/code contract bug. Per-user leave-last would keep users warm — the regime where social reg can matter — without reintroducing cascade leakage if cascades stay train-only |

### A3 — Important — Core: selection vs evaluation regime mismatch (compounding A1)

| Stage | Protocol |
|---|---|
| Enhanced Optuna | Warm random folds on train |
| Social Optuna | Single random warm split |
| Network sweep | Warm folds, feature users only |
| Final test | Full global temporal test, no warm filter |

Also: ranking columns in Stage B `_save_rmses` are shared across enhanced/social runs (last writer wins) — contaminates Stage B ranking diagnostics, **not** `core_experiment_results.csv`.

### A4 — Important — Core ranking protocol is weak / misleading

`evaluate_ranking` treats every test rating as relevant (`rating_threshold=None`), full catalog candidates, no cold filter. MovieLens: M4c best NDCG@10 while worst among M4 on RMSE. Ciao NDCG@10 ≈ 0.001. Softens “M4c failed” for ranking-first narratives; does not rescue RMSE discard without a declared primary metric.

### A5 — Important — Cold-start: discards look sound; some accepts are fragile

**Safe discards:** H1-stronger (both datasets), Ciao H2 — controlled rebuild is train-only; strata from `n_train_ratings`; no critical feature leakage found.

**Caveats (bias toward false accept or underpowered reject, not toward false H1-stronger reject):**

1. Ciao zero-shot trust: **33 train / 2215 test users** (514 vs 35 480 ratings) — H4 PASS is protocol-extreme  
2. Zero-shot trust scaler fit on all-zero train trust attrs → degenerate scaling  
3. MovieLens leave-k `1-3` has incomplete side-info coverage (~60%); H1-gain auto-PASS averages strata and can pass with a failing cold bin (CI FAIL — findings treat cautiously)  
4. Leave-k drops unused early ratings (anti-leakage, but artificial scarcity)

### A6 — Critical (for positives) / discard-safe — Route B CCE on N=16 / N=32

Artifacts: `n_users_with_cce` = **16.0** (MovieLens), **32.0** (Ciao) in `beyond_accuracy_results.csv`. Mean CCE deltas are not a dataset-level claim. Does **not** reverse discarding strong B1; makes early CCE “wins” unsafe as evidence.

### A7 — Important — Route B WP1 missing preregistered BA bootstrap / Holm

Protocol promised `beyond_accuracy_bootstrap.csv` + Wilcoxon/Holm family tests. Path exists in `RouteBPaths`; **no writer**; no BA bootstrap file on disk. WP1 judged from point estimates. Discarding “GO fuerte” is **conservative (OK)**; any “clean CI win” language was protocol-invalid.

### A8 — Important — B2 strata construct failure (mostly zero-community periphery)

`assign_lph_strata` uses low \(\tilde{h}_v\) percentiles; findings note most “boundary” users have `num_communities=0`. B10/B25 empty or N≪30. **Safe as “unsupported / no power”; unsafe as “H3/B2 refuted.”**

### A9 — Important — CCE denominator ≠ prereg \(1/K\)

`cce_at_k` uses `hits/counted`, skipping items without train dominant community (~22% ML / ~66% Ciao items lack \(D(i)\)). Secondary to A6.

### A10 — Medium — Post-hoc M4c multi-seed dual gate (HARKing risk)

Joint NDCG∧coverage ≥8/10 gate written after WP1 already showed both up. Titular FAIL still appropriate (coverage unstable); do not narrate as “no ranking signal” (NDCG often ↑).

### A11 — Medium — Trust zero-shot extreme split (same as A5.1)

Frente A trust PASS is real *under this protocol* but poor external validity; keep off the LPH/fusion thesis (consolidation docs already do this).

### A12 — Minor

| Item | Note |
|---|---|
| `networks/cascades/__main__.py` still random-splits | Footgun; pipeline path uses configured split |
| Ciao cutoff timestamp ties | Few shared-second rows; negligible |
| Ciao `network_selection_results.json` overwrite | `final_eval` prefers manifest — test rows OK |
| Protocol Δ sign vs Appendix A | Code/findings internally consistent (positive = named variant better) |
| Soft track bootstrap incomplete | Soft NO-GO still OK on near-zero magnitude |
| M4c mechanism script uses RMSE vs communities | Weak mechanism check; does not rescue or reverse titular FAIL |

---

## Hypothesis discard safety matrix

| Discarded / weakened claim | Safe? | Reason |
|---|---|---|
| Core: M4c boundary social best / beats M4a–b (MovieLens) | **Unsafe as settled** | A1 — test mostly cold isolates |
| Core: social never beats M3 (Ciao) / only M4d (ML) | **Unsafe as settled** | A1 + A3 |
| Core: M3 > M2 / enhanced > M1 | Mostly OK | Side info can still affect cold users; weaker threat |
| Cold-start: H1-stronger FAIL | **Safe** | Controlled anti-leakage path |
| Cold-start: Ciao H2 FAIL | **Safe** | Solid CIs / magnitude |
| Cold-start: MovieLens H1-gain PASS | Fragile accept | Not a discard issue |
| Cold-start: H4 trust PASS | Fragile accept | Extreme split + scaling |
| Route B: strong B1 GO discarded | **Safe** | Conservative; guards + A6/A7 |
| Route B: M4c titular multi-seed FAIL | **Safe** | Coverage unstable; don’t deny NDCG tendency |
| Route B: B2 / frontier concentration | Safe if “unsupported” | **Unsafe if “refuted”** (A8) |
| Route B: soft NO-GO | **Safe** | ~0 effect |
| Route B: WP4 skipped | **Safe** | Gate correctly blocked |

---

## What looks solid

1. Cascades / NetInf from **train only** (`pipeline/steps/cascade.py`); artifact manifests match temporal 80/20.  
2. Social Optuna receives `train_df`; scaler fit on train users.  
3. Phase 2: freeze HP/network → one global test (`final_eval`).  
4. Route B communities frozen from M3 for all variants; item \(D(i)\) from train raters.  
5. Controlled cold-start rebuild + stratum definition anti-leakage.  
6. Findings docs already flag CV↔test divergence and empty B2 cells in several places — the gap is mainly **not updating the discard language** after A1/A8.

---

## Recommended follow-ups (minimal)

1. **Re-evaluate core M3/M4\* on warm-user (and/or per-user temporal) test** before treating social discards as final.  
2. Align `config.Split` docstring with global temporal (or implement per-user).  
3. Do not cite MovieLens CCE as evidence without N≪30 disclosure; drop or re-power BA claims.  
4. Keep B2 language as “untested / invalid operationalization,” not refutation.  
5. Treat Ciao trust zero-shot as a separate extreme protocol, not ordinary cold-start confirmation.

---

## Code fixes applied (same day)

Implemented in-repo (requires **artifact rebuild** under the new default split):

| Anomaly | Fix |
|---|---|
| A1/A2 | `split_data_temporal` = per-user leave-last; `temporal_global` kept as legacy; config docs match |
| A1 eval | `final_eval` writes `rmse_warm`, `n_warm_test`, `warm_test_frac`, … |
| Cascade footgun | `networks/cascades/__main__.py` uses `load_and_split_dataset` |
| A5/A9 CCE | `cce_at_k` uses \(1/K\) denominator |
| A7 BA bootstrap | `beyond_accuracy_stats.write_beyond_accuracy_bootstrap` (+ Wilcoxon/Holm) from `final_eval` |
| A8 B2 strata | `assign_lph_strata` excludes `n_communities==0` as `ISO` |
| A3 ranking overwrite | Stage B ranking cols prefixed by `run_mode` |

**Not changed in code (report-only):** trust zero-shot protocol extremity; post-hoc M4c dual gate (HARKing) — process, not a bug.

---

## Key files reviewed

**Docs:** `docs/experiments/core_experiment_*.md`, `cold_start_*.md`, `findings/**`, `route_b/*`, `ciao_trust_network_future_work.md`  

**Code:** `recommender/data.py`, `config.py`, `recommender/enhanced/{model,search,social_search,social_regularization,network_eval}.py`, `recommender/experiment/{final_eval,network_selection,variants}.py`, `recommender/experiment/cold_start/*`, `recommender/experiment/route_b/*`, `pipeline/steps/cascade.py`, `scripts/route_b_m4c_mechanism.py`, related tests  

**Artifacts spot-checked:** `data/*/artifact_manifest.json`, `core_experiment_results.csv`, `route_b/beyond_accuracy_results.csv` (`n_users_with_cce`), Ciao zero-shot `split_manifest.json`, temporal cold-rate recompute on raw ratings
