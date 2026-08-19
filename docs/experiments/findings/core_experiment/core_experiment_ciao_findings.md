# Core Experiment Findings — Ciao (Phase 6)

**Analysis date:** 2026-08-19 (per-user temporal re-run after validity audit)  
**Protocol:** [core_experiment_plan.md](../../core_experiment_plan.md)  
**Validity context:** [experiment_validity_audit_2026-08-06.md](../../../reports/experiment_validity_audit_2026-08-06.md)  
**Primary result file:** `data/ciao/core_experiment_results.csv`  
**Supporting sources:** `*_per_user.log`, `experiment_manifest.json`, `network_selection_results.json`, `canonical_baseline.json`, `artifact_manifest.json`  
**Dataset:** CiaoDVD, **per-user chronological leave-last** 80/20 (`Split.STRATEGY=temporal`), 27 898 train / 8 167 test, seed 42  

**Supersedes:** the 2026-07-06 Ciao write-up under **global** temporal cutoff (28 852 / 7 213). Absolute RMSE levels are **not** comparable across protocols.

**Scope of this re-run:** Phase 2 frozen eval for **M1, M3, M4c, M4d** only (minimal replication ladder after MovieLens revival). Rows for M2/M4a/M4b/`M4c_robustness` in the CSV are **stale** (no `rmse_warm`) — ignore them for claims.

---

## Experiment context

Same ladder as MovieLens. Cascades / NetInf / centrality / communities rebuilt from the per-user train slice. Stage B recommend logs: `m3/m4c/m4d_recommend_per_user.log` (copied to canonical `*_recommend.log` for `import_manifest`).

Warm coverage: **4 620 / 8 167** test ratings (56.6%); **155** cold-user test ratings (~1.9%). More residual cold-*item* mass than MovieLens (~92% warm).

---

## Executive summary

### Headline (held-out test — authoritative)

| Rank | Variant | Test RMSE | Warm RMSE | Δ vs M1 (test) | Δ vs M3 (test) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **M4d** (`bridge_preserve`) | **0.9156** | **0.8899** | **−0.36%** | **−0.44%** |
| 2 | **M4c** (`boundary_downweight`) | **0.9163** | **0.8905** | −0.29% | −0.36% |
| 3 | M1 (baseline) | 0.9190 | 0.8913 | — | −0.07% |
| 4 | M3 (+ communities) | 0.9196 | 0.8948 | +0.07% | — |

**Key conclusions (this protocol):**

1. **Social beats M3 on Ciao.** Both M4d and M4c improve test and warm RMSE vs M3 (~0.4–0.5%). The July claim “no social variant beats M3” does **not** hold under per-user leave-last.
2. **M4d is the accuracy winner** (best test + warm). Under global cutoff, M4d was the *worst* core M4 — that ranking **flips**.
3. **M4c is competitive** (2nd); boundary downweight remains a viable social mode, not discarded.
4. **M3 does not beat M1** (+0.07% test; **+0.40% warm** — attributes hurt on warm). Community/boundary *side info alone* is not the Ciao win under this split.
5. Gains vs M1 are **small** (~0.3%) but consistent on the warm slice (not only cold-item artifact).
6. NDCG@10 remains ~0.001–0.004 — lead with RMSE / `rmse_warm`.

### What changed vs the global-cutoff campaign

| Claim (July / global cutoff) | Per-user re-run |
| --- | --- |
| M3 is test RMSE winner; no social beats M3 | **Reversed** — M4d/M4c beat M3 and M1 |
| M4c best social, M4d worst among M4 | **M4d best**; M4c close 2nd |
| M3 carries enhanced-CMF gain vs M1 | **Not supported** (M3 ≳ M1) |
| Social “helps on CV but not on test” | Social **does** transfer to test when users are warm |

### Cross-check vs MovieLens per-user

| | MovieLens | Ciao |
|---|---|---|
| Social > M3? | Yes (M4a/c/d) | **Yes (M4c/d)** |
| M4c competitive? | Yes (best warm among M4) | Yes (2nd overall) |
| M3 ≫ M1 on attributes? | Weak / warm≈flat | **No** (M3 worse on warm) |
| Best social mode | M4d ≈ M4a ≈ M4c | **M4d ≥ M4c** |
| Effect size vs M1 | ~2.3% | ~0.3% (sparser / more cold items) |

**Replication verdict:** the MovieLens **revival of social/M4\*** **replicates in direction** on Ciao; magnitude is smaller. The old Ciao story “attributes (M3) win, social does not” is superseded.

---

## 1. Methodology

| Step | Notes |
| --- | --- |
| Prerequisites | `00_prereq_per_user.log` — cascade→communities on per-user train |
| Stage B | `m3/m4c/m4d_recommend_per_user.log` with `--all-networks` |
| Manifest | Import from canonical log names after copying per_user → `m*_recommend.log`; **import M3+M4c+M4d in one call** (per-variant import overwrites) |
| Network selection | `selection_source=manifest_network_best` (July `runs/*` archives set aside) |
| `final_eval` | M1/M3/M4c/M4d with `rmse_warm` (`phase2_04_fe_*_per_user.log`) |

---

## 2. Selected networks (frozen)

| Variant | Diffusion | α index | α value | Selection CV RMSE |
| --- | --- | ---: | ---: | ---: |
| M3 | exponential | 39 | 3.07×10⁻³ | 0.9096 |
| M4c | exponential | 82 | 0.168 | 0.9057 |
| M4d | rayleigh | 33 | 4.13×10⁻⁵ | 0.9058 |

Canonical M1: k=50, λ≈5.25.

---

## 3. Pairwise verdicts (test RMSE)

### M3 vs M1 — full attributes
- Test: 0.9196 vs 0.9190 → **M1 slightly better**
- Warm: 0.8948 vs 0.8913 → **M1 better (~0.4%)**
- **Verdict:** **Not supported.** Do not claim Ciao enhanced-CMF win from communities alone under per-user leave-last.

### M4c vs M3 — boundary social
- Test: 0.9163 vs 0.9196 → **M4c −0.36%**
- Warm: 0.8905 vs 0.8948 → **M4c −0.49%**
- **Verdict:** **Supported.** Graph regularisation helps beyond attributes.

### M4d vs M3 — bridge preserve
- Test: 0.9156 vs 0.9196 → **M4d −0.44%**
- Warm: 0.8899 vs 0.8948 → **M4d −0.55%**
- **Verdict:** **Supported.** Best accuracy pick on this re-run.

### M4d vs M4c
- Test: 0.9156 vs 0.9163 → M4d edge (~0.08%)
- Warm: similar tiny edge
- **Verdict:** **Practical tie / slight M4d preference.**

### M4c / M4d vs M1 — headline
- M4d −0.36% / M4c −0.29% on test; both also better on warm
- **Verdict:** **Supported**, modest effect size.

---

## 4. Ranking metrics (exploratory)

| Variant | NDCG@10 |
| --- | ---: |
| M1 | 0.0036 |
| M3 | 0.0022 |
| M4d | 0.0013 |
| M4c | 0.0012 |

Near-zero absolute scale; **do not** lead with ranking on Ciao.

---

## 5. Hyperparameters (fresh final_eval rows)

| Variant | k | λ | w_main | w_user | λ_social | β | γ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 50 | 5.25 | — | — | — | — | — |
| M3 | 48 | 2.87 | 0.93 | 0.037 | — | — | — |
| M4c | 50 | 1.84 | 0.54 | 0.021 | 0.276 | 0.79 | 1.00 |
| M4d | 34 | 1.96 | 0.57 | 0.010 | 0.175 | 0.64 | 1.59 |

---

## 6. Claims to use / avoid

**Safe to claim (Ciao, per-user leave-last, M1/M3/M4c/M4d ladder):**
- Social CMF (M4c/M4d) improves test and warm RMSE over M3 and over M1.
- M4d is the strongest of the three enhanced/social variants tested; M4c is close.
- M3 alone does not improve on M1 here.
- Directionally replicates the MovieLens per-user social revival (smaller deltas).

**Do not claim from this file alone:**
- July global-cutoff rank table (M3 winner / M4d last).
- Full M2/M4a/M4b/`M4c_robustness` comparisons (not re-frozen on this split).
- Large practical gains (effects are sub-percent).
- Ranking superiority.

---

## 7. Operational lesson (Phase 2)

`import_manifest` reads **fixed** log names (`m3_recommend.log`, …). Per-variant imports **overwrite** the whole manifest. For multi-variant Phase 2: copy/alias logs to canonical names, then import **M3+M4c+M4d in one call**, and run network selection once for that set.

---

## 8. Paper table sketch (Ciao, minimal ladder)

| Model | RMSE | RMSE (warm) | Δ vs M1 |
| --- | ---: | ---: | ---: |
| M1 | 0.919 | 0.891 | — |
| M3 | 0.920 | 0.895 | +0.1% |
| **M4c** | **0.916** | **0.891** | **−0.3%** |
| **M4d** | **0.916** | **0.890** | **−0.4%** |
