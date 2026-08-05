# WP1 — Beyond-accuracy (hallazgos)

**Fecha:** 2026-08-05  
**Branch:** `feat/route-b-experiments`  
**Hipótesis:** B1 — Variantes con señal de frontera (M3, M4c, M4d) suben CCE@10, ILD@10 y/o cobertura vs M1/M2 **sin** degradar RMSE > 0.5 % relativo ni colapsar NDCG@10 (≥ 0.8 × NDCG@10 de M1).  
**Fuente:** `data/<ds>/route_b/beyond_accuracy_results.csv` + `data/<ds>/core_experiment_results.csv`

> **Nota de reproducibilidad:** los resultados de Ciao se regeneraron tras corregir el bug de sesgos sin inicializar en el L-BFGS del `cmfrec` vendorizado (arranque no reproducible). Todas las variantes de Ciao tienen ahora `valid_metric_row = True`.

## MovieLens

| Var | RMSE | ΔRMSE rel vs M1 | NDCG@10 | Guardia NDCG (≥0.300) | CCE@10 | ILD | Novelty | Coverage | Gini |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 1.0510 | — | 0.375 | — | 0.550 | 0.983 | 2.22 | 0.0079 | 0.9985 |
| M2 | 1.0385 | −1.2 % | 0.373 | ✅ | 0.550 | 1.059 | 2.13 | 0.0081 | 0.9985 |
| **M3** | 1.0287 | −2.1 % | **0.257** | ❌ | 0.575 | 1.046 | 3.15 | 0.0062 | 0.9986 |
| M4a | 1.0322 | −1.8 % | 0.136 | ❌ | 0.505 | 0.869 | 4.39 | 0.0034 | 0.9988 |
| M4b | 1.0410 | −0.9 % | 0.070 | ❌ | 0.332 | 0.924 | 7.10 | 0.0022 | 0.9988 |
| **M4c** | 1.0421 | −0.8 % | **0.401** | ✅ | 0.575 | 0.825 | 2.16 | **0.0090** | 0.9983 |
| M4d | 1.0243 | −2.5 % | 0.173 | ❌ | 0.556 | 0.809 | 3.70 | 0.0044 | 0.9988 |

## Ciao

| Var | RMSE | ΔRMSE rel vs M1 | NDCG@10 | CCE@10 | ILD | Novelty | Coverage | Gini |
|---|---|---|---|---|---|---|---|---|
| M1 | 0.9256 | — | 0.0018 | 0.736 | 0.955 | 6.28 | 0.0019 | 0.9994 |
| M2 | 0.9260 | +0.04 % | 0.0010 | 0.716 | 0.977 | 6.96 | 0.0024 | 0.9994 |
| **M3** | 0.9217 | −0.4 % | 0.0006 | **0.764** | 0.974 | 7.09 | 0.0018 | 0.9994 |
| M4a | 0.9242 | −0.2 % | 0.0007 | 0.731 | 0.408 | 6.99 | 0.0010 | 0.9994 |
| M4b | 0.9250 | −0.1 % | 0.0007 | 0.731 | 0.623 | 6.84 | 0.0012 | 0.9994 |
| M4c | 0.9238 | −0.2 % | 0.0011 | 0.725 | 0.801 | 6.96 | 0.0017 | 0.9994 |
| M4d | 0.9292 | +0.4 % | 0.0011 | 0.725 | 0.212 | 7.07 | 0.0009 | 0.9994 |

## Lectura

1. **Guardia RMSE:** se cumple en todas las variantes de ambos datasets (ninguna degrada > 0.5 % vs M1; casi todas mejoran).
2. **Tensión RMSE ↔ ranking (MovieLens):** las variantes que más bajan RMSE (M3, M4a, M4b, M4d) **colapsan el NDCG@10**. Solo **M2** y **M4c** conservan el ranking; M4c incluso lo mejora (0.401 > 0.375) y además tiene la **mayor cobertura** (0.0090) y CCE al alza (0.575). Ese es el único caso limpio de "beyond-accuracy sin costo de accuracy" en MovieLens.
3. **CCE:** M3 sube CCE en ambos datasets (ML 0.575 vs 0.550; Ciao 0.764 vs 0.736) — señal consistente de que las comunidades empujan recomendaciones cross-community. Pero en ML el precio es el NDCG.
4. **Ciao tiene ranking degenerado:** NDCG@10 ≈ 0.001 para todas las variantes. La guardia NDCG no es informativa en Ciao (los deltas son ruido cerca de cero); B1 en Ciao se juzga solo por RMSE + CCE/cobertura.
5. **ILD/novelty:** el proxy `ild_latent` no reemplaza la ILD por géneros primaria (no cableada; falta `movies.csv`). Tratar como diagnóstico.

## Veredicto

**B1 PARCIAL / DÉBIL.**

- Hay **una** ganancia beyond-accuracy limpia: **M4c en MovieLens** (cobertura + CCE arriba, NDCG y RMSE dentro de guardia).
- **M3 sube CCE en ambos datasets** pero, en MovieLens, a costa de un colapso de NDCG@10 → falla la guardia. En Ciao el NDCG es degenerado, así que no hay una historia de ranking que sostener.
- No hay un patrón consistente cross-dataset de "M3/M4d ganan en beyond-accuracy sin costo". No es un GO fuerte.
