# Cold-start experiment findings

**Analysis date:** 2026-08-04  
**Protocol:** [cold_start_experiment_proposal.md](../../cold_start_experiment_proposal.md)  
**Commands:** [cold_start_commands.md](../../cold_start_commands.md)
**Auto summaries:** `data/<ds>/cold_start/success_summary.md` (and `zero_shot_trust/` for Ciao)

This note consolidates the controlled runs after the leave-k / report fixes. Machine-readable tables live under `data/`; this file is the interpretive summary for the paper.

---

## Hypotheses (what H1–H4 mean)

Evaluated on cold strata `1-3` and `4-10` unless noted. Δ = improvement in RMSE (positive ⇒ better). Auto-checks also require N≥10 users in a stratum.

| ID | Claim | Pass when |
| --- | --- | --- |
| **H1-gain** | Side info helps cold users | mean Δ (M1−M3) over cold strata > 0 |
| **H1-stronger** | That help is *larger* in cold than warm | H1-gain and mean Δ cold > Δ on `>10` |
| **H2** | Community/boundary (M3) beats centrality alone (M2) in cold | M3 RMSE < M2 RMSE on both `1-3` and `4-10` |
| **H3** | Social M4c/M4d add value beyond M3 | only report if they beat M3 by stratum with non-trivial evidence (not a binary gate in the auto summary) |
| **CI cold M3>M1** | Statistical support for H1-gain | bootstrap 95% CI of *per-user* (M1−M3) entirely > 0 on cold strata |
| **H4** | Pure 0-rating users need external/trust attributes | `--mode zero_shot_trust`: M2_trust and/or M3_trust beat M1 on stratum `0` (with CIs) |

**Why zero-shot shows N/A for H1/H2:** that track holds out *all* ratings for trust-graph users, so every evaluated user has `n_train=0` (only stratum `0`). There are no `1-3` / `4-10` / `>10` bins to compare, so H1-gain, H1-stronger, and H2 are **not applicable by design** — not a missing run. The track **did** finish: M1 / M2_trust / M3_trust trained and scored; criteria are **H4** only (`data/ciao/cold_start/zero_shot_trust/`).

---

## Executive summary

| Track | Protocol | H1-gain | H1-stronger | H2 | CI / H4 | Takeaway |
| --- | --- | --- | --- | --- | --- | --- |
| **MovieLens controlled** | leave-k (caps 0/2/7/all) | PASS (tiny) | FAIL | PASS (tiny) | CI FAIL | Side info helps a bit in places, but **not** “more in cold”; per-user CIs cross 0 |
| **Ciao controlled** | leave-last (natural depth) | PASS | FAIL | FAIL | CI PASS | Clear M3≪M1 gains everywhere; **larger on warm** than cold |
| **Ciao zero-shot trust** | hold out trust-graph users | N/A† | N/A† | N/A† | **H4 PASS** (esp. M2_trust) | **M2_trust** beats M1 strongly; M3_trust barely beats M1 and loses to M2_trust |

† N/A = hypothesis not defined on this track (only stratum `0`); run is complete.

**Headline claim for the article:** network/community side information improves cold users on Ciao (and weakly on artificial MovieLens leave-k), but the stronger hypothesis — *larger* gains in cold than warm — is **not** supported. On Ciao the warm gain is larger; on MovieLens leave-k the cold pooled Δ is ≈0 and bootstrap CIs include zero. Treat cold-start as a **general side-info benefit**, not a cold-specific amplification.

---

## 1. MovieLens — controlled leave-k

### Setup

- Split: `--mode controlled --split leave_k` (seed 42, `test_frac=0.2`)
- After chrono leave-last, each user gets a train cap in `{0, 2, 7, all}` (round-robin after shuffle)
- Unused early ratings **dropped** (59 613 rows) — neither train nor test
- NetInf rebuild: 100 α × {exponential, powerlaw, rayleigh} under `data/movielens/cold_start/`
- Selected network (from core M3): Rayleigh α≈0.002 (index 63)
- Variants: M1, M2, M3, M4c, M4d

Manifest: `data/movielens/cold_start/split_manifest.json` (`mode: controlled_leave_k`).

### Stratum balance (success of leave-k)

| Estrato | N users | N test ratings | Cap |
| --- | ---: | ---: | --- |
| 0 | 168 | 6037 | 0 |
| 1-3 | 168 | 4604 | 2 |
| 4-10 | 168 | 4882 | 7 |
| >10 | 167 | 4733 | all |

Leave-k **fixed** the empty cold strata of leave-last (previously 671/671 in `>10`).

### RMSE (pooled)

| Estrato | M1 | M2 | M3 | M4c | M4d | Δ M3−M1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.1157 | 1.1129 | 1.1097 | 1.1077 | **1.1010** | 0.0060 |
| 1-3 | **0.9366** | 0.9391 | 0.9373 | 0.9398 | 0.9525 | −0.0007 |
| 4-10 | 1.0024 | 1.0021 | 0.9991 | **0.9974** | 1.0073 | 0.0033 |
| >10 | 0.9763 | **0.9495** | 0.9514 | 0.9518 | 0.9521 | 0.0250 |

### Criteria

- **H1-gain:** PASS — mean cold Δ (1-3 + 4-10) = 0.0013 > 0, but **practically negligible**
- **H1-stronger:** FAIL — warm Δ (0.0250) ≫ cold
- **H2:** PASS on point estimates (M3 < M2 in 1-3 and 4-10) with tiny margins
- **Bootstrap CI** (per-user M3 vs M1 on cold): **FAIL** — all cold CIs cross 0 (e.g. 1-3: [−0.0043, 0.0021])

### Interpretation

1. Leave-k is methodologically sound for *populating* cold bins on dense MovieLens, but it is an **artificial** scarcity protocol: NetInf and CMF never see the dropped early ratings.
2. The only clear side-info win is on **warm** users (M2/M3 ≈ −0.025 vs M1). In cold bins, M3 is mixed (slightly worse than M1 on 1-3 pooled RMSE).
3. Social variants: M4d best on pure cold (0); M4c slightly best on 4-10; neither dominates consistently.
4. Do **not** claim statistically significant cold M3>M1 on MovieLens leave-k.

---

## 2. Ciao — controlled leave-last

Natural rating depth produces real cold strata without leave-k.

| Estrato | N users | M1 | M2 | M3 | Δ M3−M1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 155 | 0.9491 | 0.8818 | 0.8826 | 0.0665 |
| 1-3 | 438 | 1.0035 | 0.9267 | 0.9271 | 0.0763 |
| 4-10 | 868 | 1.0055 | 0.9391 | 0.9369 | 0.0687 |
| >10 | 787 | 0.9973 | 0.9097 | 0.9095 | **0.0878** |

- **H1-gain:** PASS (mean cold Δ ≈ 0.073)
- **H1-stronger:** FAIL (warm 0.088 > cold)
- **H2:** FAIL (M3 ≈ M2; community layer does not clearly beat centrality alone)
- **CI check:** PASS (cold per-user M3 vs M1 CIs entirely > 0)

**Read:** side information (mostly centrality / M2) helps cold *and* warm users on Ciao; the lift is real and large (~7–9% RMSE), but **not cold-specific**.

---

## 3. Ciao — zero-shot trust (H4 only)

Hold out all ratings for users present in the explicit trust graph (`n_train=0`). Features from trust centrality / simple community signals only (no NetInf).

**Nothing left to run for H1/H2 here.** Those need few-shot strata (`1-3`, `4-10`) and a warm baseline; this protocol intentionally puts everyone in stratum `0`. The executed command was `--mode zero_shot_trust` with variants `M1 M2_trust M3_trust` (artifacts under `data/ciao/cold_start/zero_shot_trust/`).

| | M1 | M2_trust | M3_trust |
| --- | ---: | ---: | ---: |
| RMSE (stratum 0, N=2215 users / 35 480 ratings) | 1.1914 | **1.0669** | 1.1740 |
| Δ vs M1 | — | **+0.1246** | +0.0175 |

- **H4-M2_trust:** PASS (large gain; per-user CI for M2 vs M1 ≈ [0.140, 0.150])
- **H4-M3_trust:** PASS vs M1 on point estimate + CI, but **worse than M2_trust** (M3 vs M2 mean Δ ≈ −0.096)
- **H1 / H2:** N/A (no cold/warm stratum contrast)

**Read:** for pure 0-rating users, **trust centrality (M2_trust)** is the useful signal. Extra trust-community features (M3_trust) do not help beyond M2_trust and can hurt.

---

## 4. Cross-dataset conclusions

1. **Reject H1-stronger** on both datasets under the current protocols.
2. **Accept a weaker claim:** structured user attributes improve RMSE for low-history users when those attributes exist (Ciao NetInf features; Ciao trust graph; weak/noisy on MovieLens leave-k).
3. **H2 (M3 > M2 in cold)** is not a robust story: MovieLens leave-k passes on tiny point estimates without CI support; Ciao fails (M2 ≈ M3).
4. **H4:** explicit trust enables true 0-shot; prefer reporting **M2_trust vs M1**, with M3_trust as a negative/neutral control.
5. MovieLens leave-last remains useful only as a **pipeline sanity check** (all warm); paper cold-start numbers for MovieLens should cite **leave-k**.

---

## 5. Caveats

- Leave-k drops history that a real system might still use for graph inference; results are conditional on that protocol.
- Pooled rating RMSE and per-user mean Δ can disagree in sign (MovieLens cold); prefer bootstrap tables for claims.
- Hyperparameters and network choice are frozen from the **core** experiment, then remapped onto the cold-start NetInf grid — not re-tuned per stratum.
- Diagnostic (global temporal) strata remain unsuitable for H1/H2 on MovieLens.

---

## 6. Artifact index

| Path | Role |
| --- | --- |
| `data/movielens/cold_start/success_summary.md` | Auto §10 checks (leave-k run) |
| `data/movielens/cold_start/split_manifest.json` | leave-k caps, rebuild counts |
| `data/movielens/cold_start/cold_start_results.csv` | RMSE by variant × stratum |
| `data/movielens/cold_start/bootstrap_confidence_intervals.csv` | Per-user Δ CIs |
| `data/ciao/cold_start/success_summary.md` | Ciao controlled |
| `data/ciao/cold_start/zero_shot_trust/` | H4 artifacts + summary |
