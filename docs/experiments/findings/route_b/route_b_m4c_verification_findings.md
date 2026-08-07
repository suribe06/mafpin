# WP / Frente B — Verificación M4c (hallazgos)

**Fecha:** 2026-08-06  
**Scope:** **B1 + B2 completados** (MovieLens). B3 pendiente (Frente C).  
**Fuentes:** `data/movielens/route_b/multiseed/` · `data/movielens/route_b/m4c_mechanism/` · `scripts/route_b_m4c_mechanism.py`  
**Criterio B1:** NDCG@10(M4c) > NDCG@10(M1) **y** cobertura(M4c) > cobertura(M1) en ≥ 8/10 semillas, RMSE en guardia.  
**Criterio B2 (plan):** si la mejora viene de usuarios *sin* info comunitaria / aristas sin comunidad → no es “boundary-aware”.

## B1 — Multi-semilla (corrida con BA upsert)

| Chequeo | Resultado |
|---|---|
| NDCG M4c > M1 | **9/10** (falla 101) |
| cov M4c > M1 same-seed | **7/10** (falla 101, 999, 31337) |
| RMSE guardia | **10/10** |
| Conjunto ≥ 8/10 | **7/10 → FAIL** |

Δ NDCG +0.085 ± 0.070; Δ cov +0.00031 ± 0.00082. Detalle por seed en `b1_summary.csv`.

**Veredicto B1: NO-GO** del umbral pre-registrado. Ranking frecuente + RMSE bueno; cobertura no estable.

## B2 — Mecanismo `boundary_downweight`

Red M4c congelada: **rayleigh #21**, β = 0.945. Predicciones M1/M4c coinciden con seed 42. Artefactos: `m4c_mechanism/mechanism_summary.json`.

### Edge stats (antes = Jaccard comunidad; después = boundary_downweight)

| Métrica | Valor |
|---|---|
| Aristas (usuarios del split) | 1279 |
| Frac. ≥ 1 extremo con comunidad | **0.994** |
| Frac. ambos extremos con comunidad | **0.776** |
| Frac. Jaccard > 0 | 0.775 (991 aristas) |
| Frac. des-ponderadas (entre Jaccard > 0) | **0.245** |
| Frac. eliminadas por boundary | 0.0 |
| w Jaccard mean / p50 | 0.796 / 1.0 |
| w BDW mean / p50 / p10–p90 | 0.773 / 0.931 / 0.41–1.0 |

A nivel de grafo el mecanismo **sí opera**: casi todas las aristas tocan comunidades y ~25 % de las aristas con overlap comunitario se des-ponderan (ninguna se apaga del todo con este β).

### Per-user: Δ RMSE (M1−M4c; + = M4c mejor) vs `n_communities`

Usuarios rankeados en beyond-accuracy seed 42: **N = 147**.

| Estrato | N | mean Δ RMSE |
|---|---:|---:|
| `n_communities = 0` | 131 (89 %) | +0.023 |
| `n_communities ≥ 1` | 16 | +0.042 |
| `n_communities ≥ 2` | 15 | +0.041 |

- Spearman(n_communities, Δ RMSE) = **0.073** (p = 0.38) — sin correlación detectable.
- La ganancia media es **mayor** en usuarios con comunidad que en los de 0 → **no** se cumple la falsificación “mejora concentrada en periferia sin comunidad”.
- Potencia baja: solo 16 usuarios rankeados con comunidad.

**Veredicto B2: mecanismo edge-level PLAUSIBLE; evidencia per-user DÉBIL pero no falsificada.** No se obliga a renombrar a “regularización selectiva incidental”. Tampoco demuestra que el beyond-accuracy de B1 sea causalmente boundary-aware (N pequeño en el estrato con comunidades).

## Veredicto conjunto Frente B

| Pieza | Resultado |
|---|---|
| B1 multi-seed beyond-accuracy | **FAIL** (7/10) |
| B2 mecanismo | **plausible / no falsificado** |
| B3 réplica ranking | pendiente de **Frente C** |

**GO de M4c como claim titular de beyond-accuracy: NO.**  
M4c queda como señal interesante (ranking a menudo ↑, RMSE ↓, regularizador que sí toca fronteras en el grafo), no como resultado blindado. El paper de fusión no debería titular con M4c hasta C (densificar) o un criterio revisado pre-registrado.

## B3

Espera densificación MovieLens (C1–C3). Ciao no aporta NDCG.
