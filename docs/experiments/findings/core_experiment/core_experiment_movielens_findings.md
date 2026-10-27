# Core Experiment Findings — MovieLens (Phase 6)

**Analysis date:** 2026-08-07 (per-user temporal re-run after validity audit)  
**Protocol:** [core_experiment_plan.md](../../core_experiment_plan.md)  
**Validity context:** [experiment_validity_audit_2026-08-06.md](../../../reports/experiment_validity_audit_2026-08-06.md)  
**Primary result file:** `data/movielens/core_experiment_results.csv`  
**Supporting sources:** Stage B logs (`*_per_user.log`), `experiment_manifest.json`, `network_selection_results.json`, `canonical_baseline.json`, `artifact_manifest.json`  
**Dataset:** MovieLens (`ratings_small`), **per-user chronological leave-last** 80/20 (`Split.STRATEGY=temporal`), 79 748 train / 20 256 test, seed 42  

**Supersedes:** the 2026-06-30 MovieLens write-up under **global** temporal cutoff (80 003 / 20 001). That split left ~83% of test ratings on cold users, so social regularisation was largely invisible on the held-out set. Absolute RMSE levels are **not** comparable across the two protocols.

**Archive of old numbers:** `data/movielens/core_experiment_results.pre_b1_backup.csv`

---

## Experiment context

Same model ladder as before (M1 → M2 → M3 → M4a–d). Cascades / NetInf / centrality / communities were **rebuilt** from the per-user train slice before Stage B and Phase 2.

| ID | Name | What it adds |
| --- | --- | --- |
| **M1** | Baseline CMF | Ratings only |
| **M2** | Enhanced (centrality) | + network centrality side info |
| **M3** | Enhanced (full attrs) | + community / boundary features |
| **M4a** | Social (uniform) | + social regulariser, uniform edges |
| **M4b** | Social (community Jaccard) | + Jaccard community edge weights |
| **M4c** | Social (boundary downweight) | + downweight cross-boundary edges |
| **M4d** | Social (bridge preserve) | + keep some cross-boundary bridges |

Pairwise questions (plan): M2 vs M1, M3 vs M2, M4a vs M3, M4c vs M4a/M4b, M4d vs M4c, M4\* vs M1.

---

## Executive summary

### Headline (held-out test — authoritative)

Warm coverage: **18 733 / 20 256** test ratings (92.5%); **0** cold-user test ratings. `final_eval` also reports `rmse_warm` on the warm user∧item slice.

| Rank | Variant | Test RMSE | Warm RMSE | Δ vs M1 (test) | Δ vs M3 (test) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **M4d** (`bridge_preserve`) | **0.9030** | 0.8950 | **−2.36%** | **−0.63%** |
| 2 | **M4a** (`uniform`) | **0.9034** | 0.8953 | −2.31% | −0.59% |
| 3 | **M4c** (`boundary_downweight`) | **0.9039** | **0.8947** | −2.26% | −0.54% |
| 4 | M3 (+ communities) | 0.9087 | 0.9009 | −1.73% | — |
| 5 | M2 (centrality) | 0.9088 | 0.9015 | −1.73% | +0.01% |
| 6 | M4b (`community_jaccard`) | 0.9172 | 0.9035 | −0.82% | +0.93% |
| 7 | M1 (baseline) | 0.9248 | 0.9007 | — | +1.76% |

**Key conclusions (this protocol):**

1. **Social regularisation beats M3 on test.** M4d, M4a, and M4c all improve ~0.5–0.7% RMSE vs M3. The June claim that “only M4d barely helps / M4c fails” does **not** hold under per-user leave-last.
2. **M4c is competitive, not discarded.** On aggregate RMSE it ties M4a/M4d within noise; on **`rmse_warm` it is the best M4** (0.8947). Boundary downweight is a viable design again.
3. **M4d remains a strong accuracy pick** (best aggregate RMSE), but the gap to M4c/M4a is tiny (~0.001).
4. **M3 ≈ M2.** Community/boundary *attributes* add essentially nothing over centrality alone on this split (−0.00006 RMSE). The old “M3 clearly beats M2 on test” finding does not replicate here.
5. **M3 vs M1 on the warm slice is ~flat** (0.9009 vs 0.9007). The ~1.7% M3-over-M1 gain on full test comes mostly from the ~7.5% non-warm (mostly cold-*item*) ratings. **Social** gains *do* show up on warm (~0.895 vs ~0.901).
6. **M4b remains the weak social mode** (worse than M3).

### What changed vs the global-cutoff campaign

| Claim (June / global cutoff) | Per-user re-run |
| --- | --- |
| Social rarely beats M3; M4c last among M4 | **Reversed** for M4a/c/d |
| M4c not supported as boundary social design | **Reopened** — competitive / best on warm |
| M3 ≫ M2 on test | **Not supported** (tie) |
| M4d test winner | Still among winners; no longer alone |
| Test dominated by cold users | Fixed (0 cold-user ratings) |

---

## 1. Methodology

| Step | Output | Role |
| --- | --- | --- |
| Prerequisites | cascades, NetInf×3, centrality, communities | Rebuilt after split change (`00_prereq_per_user.log`) |
| Stage B `recommend` | per-variant logs + network CSVs | M3, M4c, M4d re-run with `--n-jobs -1 --cpu-fraction 0.6` (M2/M4a/M4b present in Phase 2 via manifest/selection from this campaign’s artifacts) |
| `import_manifest` | `experiment_manifest.json` | Freeze HPs + per-family CV picks |
| `canonical_baseline` | `canonical_baseline.json` | M1: k=39, λ≈7.01 |
| `network_selection` | `network_selection_results.json` | One `(diffusion_model, α)` per variant |
| `final_eval` | `core_experiment_results.csv` | Full train → one test; includes `rmse_warm` |

**Split contract:** `config.Split.STRATEGY="temporal"` = per-user leave-last. Legacy global cutoff is `temporal_global` (do not mix artifacts across strategies).

---

## 2. Selected networks (frozen)

| Variant | Diffusion | α index | α value | Selection CV RMSE |
| --- | --- | ---: | ---: | ---: |
| M2 | rayleigh | 5 | 5.94×10⁻⁶ | 0.8883 |
| M3 | rayleigh | 68 | 2.09×10⁻³ | 0.8885 |
| M4a | rayleigh | 26 | 4.19×10⁻⁵ | 0.8838 |
| M4b | powerlaw | 47 | 2.95 | 0.8828 |
| M4c | rayleigh | 26 | 4.19×10⁻⁵ | 0.8854 |
| M4d | powerlaw | 43 | 2.79 | 0.8834 |

M4a and M4c share the same rayleigh α₂₆ network under this selection.

---

## 3. Pairwise verdicts (test RMSE)

Positive Δ% = first named variant better when written “A vs B → A better by …”.

### M2 vs M1 — centrality side info
- Test: 0.9088 vs 0.9248 → **M2 −1.73%**
- Warm: 0.9015 vs 0.9007 → ~flat / slightly worse
- **Verdict:** Helps on full test (cold-item mass); little warm-user gain.

### M3 vs M2 — community / boundary attributes
- Test: 0.9087 vs 0.9088 → **tie**
- **Verdict:** **Not supported** on this split. Do not claim attribute stacking as the main MovieLens win.

### M4a vs M3 — any social regulariser
- Test: 0.9034 vs 0.9087 → **M4a −0.59%**
- Warm: similar gap
- **Verdict:** **Supported.** Graph regularisation helps beyond attributes.

### M4c vs M4a — boundary downweight vs uniform
- Test: 0.9039 vs 0.9034 → M4a slightly better (~0.05%)
- Warm: **M4c better** (0.8947 vs 0.8953)
- **Verdict:** **Inconclusive / practical tie.** Boundary mode is not worse; warm prefers M4c.

### M4c vs M4b — boundary vs Jaccard
- Test: 0.9039 vs 0.9172 → **M4c clearly better**
- **Verdict:** Boundary downweight preferred over community Jaccard.

### M4d vs M4c — bridge preserve vs boundary downweight
- Test: 0.9030 vs 0.9039 → M4d edge (~0.1%)
- Warm: M4c edge (0.8947 vs 0.8950)
- **Verdict:** **Tie for practical purposes.** Either is a defensible social headline; report both.

### M4c vs M1 — headline Phase 6
- Test: 0.9039 vs 0.9248 → **−2.26%**
- Warm: 0.8947 vs 0.9007 → **−0.67%**
- **Verdict:** **Supported** on accuracy. Gain vs M1 is real on warm, not only cold-item artifact.

---

## 4. Ranking metrics (exploratory)

NDCG@10 on this re-run is low in absolute terms (~0.02–0.05) and **disagrees** with RMSE ordering:

| Variant | NDCG@10 | NDCG@10 warm |
| --- | ---: | ---: |
| M3 | 0.0496 | 0.0496 |
| M1 | 0.0495 | 0.0495 |
| M2 | 0.0490 | 0.0490 |
| **M4c** | **0.0475** | **0.0475** |
| M4d | 0.0299 | 0.0299 |
| M4a | 0.0218 | 0.0218 |
| M4b | 0.0218 | 0.0218 |

Lead with **RMSE / `rmse_warm`** for paper claims. Ranking needs a separate, declared protocol if used (relevance threshold, candidate set).

---

## 5. Hyperparameters (final_eval rows)

| Variant | k | λ | w_main | w_user | λ_social | β | γ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 39 | 7.01 | — | — | — | — | — |
| M2 | 23 | 5.59 | 0.89 | 0.56 | — | — | — |
| M3 | 9 | 4.27 | 0.67 | 0.60 | — | — | — |
| M4a | 46 | 1.79 | 0.72 | 0.42 | 0.71 | 0 | 1.00 |
| M4b | 44 | 2.43 | 0.93 | 0.47 | 0.81 | 0 | 1.00 |
| M4c | 45 | 4.98 | 0.96 | 0.013 | 0.14 | 0.95 | 1.00 |
| M4d | 37 | 2.20 | 0.66 | 0.45 | 0.30 | 0.58 | 1.98 |

M4c still uses high `w_main`, near-zero `w_user`, moderate `λ_social`, high `β` (strong boundary downweight).

---

## 6. Claims to use / avoid

**Safe to claim (MovieLens, per-user leave-last):**
- Social CMF (M4a/c/d) improves test RMSE over enhanced M3 and over M1.
- M4c is a competitive boundary-aware social mode (best warm RMSE among M4).
- M4b (Jaccard) underperforms.
- Side-info-only M3 does not clearly beat M2; social graph coupling is where the accuracy move is.

**Do not claim from this file alone:**
- Numbers from the June global-cutoff table (different split).
- “M4c failed / social does not beat M3” (falsified here).
- Strong ranking superiority for any M4 (NDCG weak / conflicting).
- Cross-dataset generality — needs Ciao under the **same** per-user protocol.

---

## 7. Next steps

1. **Ciao replication (same split + ladder)** — see recommendation below in chat / follow-up: at least M3, M4c, M4d (+ M1 via `canonical_baseline` / Phase 2).
2. Update Route B language that assumed M4c was dead on MovieLens accuracy.
3. Optional: M4a full Stage B if you want a cleaner ablation than manifest-only; current Phase 2 already includes M4a test row.
4. Keep `rmse_warm` in all future `final_eval` tables.

---

## 8. Paper table sketch (MovieLens)

Lead with per-user protocol; footnote the superseded global-cutoff campaign as a cold-heavy stress test where social could not transfer.

| Model | RMSE | RMSE (warm) | Δ vs M1 |
| --- | ---: | ---: | ---: |
| M1 | 0.925 | 0.901 | — |
| M2 | 0.909 | 0.901 | −1.7% |
| M3 | 0.909 | 0.901 | −1.7% |
| M4a | 0.903 | 0.895 | −2.3% |
| M4b | 0.917 | 0.903 | −0.8% |
| **M4c** | **0.904** | **0.895** | **−2.3%** |
| **M4d** | **0.903** | **0.895** | **−2.4%** |
