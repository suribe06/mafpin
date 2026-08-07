# WP1 — Beyond-accuracy (hallazgos)

**Fecha:** 2026-08-07 (re-run per-user temporal, MovieLens)  
**Branch:** `feat/route-b-experiments`  
**Hipótesis:** B1 — Variantes con señal de frontera (M3, M4c, M4d) suben CCE@10, ILD@10 y/o cobertura vs M1/M2 **sin** degradar RMSE > 0.5 % relativo ni colapsar NDCG@10 (≥ 0.8 × NDCG@10 de M1).  
**Fuente:** `data/movielens/route_b/beyond_accuracy_results.csv`, `beyond_accuracy_bootstrap.csv`, `core_experiment_results.csv`  
**Split:** per-user leave-last (`artifact_manifest` train 79 748 / test 20 256). Backup del run global-cutoff: `data/movielens/route_b_backup_global_cutoff_20260807/`.

> **Ciao:** tablas previas (2026-08-05, cutoff global) **no** se regeneraron en este pase. No mezclar con MovieLens per-user.

## MovieLens (per-user)

| Var | RMSE | ΔRMSE vs M1 | NDCG@10 | Guardia NDCG (≥0.8×M1=0.0396) | CCE@10 | ILD lat | Novelty | Coverage | Gini | N CCE |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.9248 | — | 0.0495 | — | 0.680 | 1.040 | 3.01 | 0.0147 | 0.9980 | 281 |
| M2 | 0.9088 | −1.73 % | 0.0490 | ✅ | 0.684 | 1.023 | 3.07 | 0.0129 | 0.9981 | 281 |
| **M3** | 0.9087 | −1.73 % | **0.0496** | ✅ | **0.712** | 1.027 | 3.03 | 0.0106 | 0.9980 | 281 |
| M4a | 0.9034 | −2.31 % | 0.0218 | ❌ | 0.651 | 0.914 | 6.24 | 0.0071 | 0.9984 | 281 |
| M4b | 0.9172 | −0.82 % | 0.0218 | ❌ | 0.557 | 0.715 | 7.28 | 0.0092 | 0.9981 | 281 |
| **M4c** | 0.9039 | −2.26 % | **0.0475** | ✅ | 0.676 | 0.894 | 3.38 | **0.0149** | 0.9979 | 281 |
| M4d | 0.9030 | −2.36 % | 0.0299 | ❌ | 0.685 | 0.926 | 4.21 | 0.0051 | 0.9985 | 281 |

Bootstrap CCE (pareado, N=281, Holm):

| Comparación | mean Δ | CI 95 % | p Holm |
|---|---:|---|---:|
| M3−M2 | +0.028 | [0.018, 0.039] | \< 10⁻⁵ |
| M3−M1 | +0.032 | [0.021, 0.044] | \< 10⁻⁸ |
| M4c−M3 | −0.037 | [−0.050, −0.024] | \< 10⁻⁶ |
| M4d−M3 | −0.027 | [−0.040, −0.014] | \< 10⁻³ |

(Δ positivo ⇒ primera variante con más CCE.)

## Lectura (MovieLens)

1. **N CCE = 281** (antes ~16 bajo cutoff global). El beyond-accuracy ya no está sub-muestreado a un puñado de usuarios.
2. **Guardia RMSE:** todas mejoran vs M1.
3. **Guardia NDCG:** pasan M2, **M3**, **M4c**. Fallan M4a/M4b/M4d (ranking cae).
4. **CCE:** **M3 es el claro ganador** vs M1/M2 (CI lejos de 0). M4c/M4d **bajan** CCE vs M3.
5. **Cobertura:** máximo en **M4c** (0.0149), por encima de M1.
6. **Novelty:** M4a/b/d (y algo M4c) suben novelty; ILD latente cae en M4 vs M3.
7. Vs run global-cutoff: el “único caso limpio M4c” se **reformula**. Ahora el caso limpio de CCE+guardias es **M3**; M4c es limpio en **cobertura + RMSE + NDCG**, no en CCE.

## Veredicto B1 (MovieLens per-user)

**PARCIAL → más sólido que antes, pero no GO fuerte cross-métrica.**

- **GO débil / titular M3:** CCE↑ significativo, NDCG y RMSE OK.
- **GO débil / titular M4c:** cobertura↑, RMSE↓, NDCG OK; **no** CCE↑ vs M3.
- No hay variante que suba CCE **y** cobertura **y** NDCG a la vez vs M1/M3.

**WP4:** no disparar aún (sigue sin patrón multi-métrica limpio + Ciao pendiente).

## Ciao

Sin re-run en este pase. Hallazgos 2026-08-05 siguen como referencia provisional bajo cutoff global.
