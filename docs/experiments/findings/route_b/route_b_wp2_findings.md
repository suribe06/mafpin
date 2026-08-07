# WP2 — Estratos frontera + cross-community (hallazgos)

**Fecha:** 2026-08-07 (re-run per-user temporal, MovieLens)  
**Branch:** `feat/route-b-experiments`  
**Hipótesis:** B2 — La ganancia de M3 sobre M2 (y de M4 sobre M3) se concentra en usuarios frontera (B10/B25 de $\tilde{h}_v$) y/o en ratings cross-community.  
**Umbrales:** celda válida N ≥ 30; bootstrap 1000, CI 95 %.  
**Fuente:** `boundary_strata_*.csv`, `cross_community_items_results.csv`  
**Cambio de constructo:** `assign_lph_strata` excluye usuarios sin comunidad (`ISO`); percentiles solo sobre elegibles.

## Poblamiento de estratos (MovieLens)

| Estrato | N usuarios (en predicciones) |
|---|---:|
| B10 | **0** (merge→B25; elegibles insuficientes en cola baja) |
| B25 | **78** ✅ |
| MID | **132** ✅ |
| E75 | **71** ✅ |

Mejor que el run global-cutoff (B10/B25 vacíos / N≪30), pero **B10 sigue vacío** tras merge. Frontera analizable = **B25**.

## Bootstrap por estrato (mean Δ [CI 95 %])

Δ positivo ⇒ variante nombrada mejor en RMSE (convención código: M3−M2 etc. como en pipeline).

| Estrato | M3−M2 | M3−M1 | M4c−M3 | M4d−M3 |
|---|---|---|---|---|
| B25 (78) | −0.001 [−0.005, 0.003] | +0.003 [−0.007, 0.014] | **+0.009 [0.005, 0.014]** ✅ | **+0.011 [0.004, 0.018]** ✅ |
| MID (132) | +0.001 [−0.002, 0.003] | **+0.007 [0.003, 0.011]** ✅ | **+0.006 [0.004, 0.009]** ✅ | +0.003 [−0.001, 0.007] |
| E75 (71) | −0.003 [−0.006, 0.001] | −0.001 [−0.012, 0.010] | +0.004 [−0.000, 0.009] | +0.003 [−0.004, 0.009] |

## Cross-community items (RMSE, sin CI)

| Variant | n ratings | RMSE |
|---|---:|---:|
| M1 | 7117 | 0.890 |
| M2 | 7117 | 0.892 |
| M3 | 7117 | 0.892 |
| **M4c** | 7117 | **0.885** |
| M4d | 7117 | 0.886 |

## Lectura

1. **M3−M2 no se concentra en frontera:** CI cruza 0 en B25/MID/E75 (atributos comunidad ≈ null, coherente con core ML).
2. **M4c/M4d vs M3 sí ayudan en B25** (CI > 0) y M4c también en MID. No es exclusivo de frontera (MID también), pero **hay señal en B25** — mejora vs el WP2 viejo “sin potencia”.
3. Cross-community: M4c/M4d bajan RMSE vs M3 (~0.007); guiño alineado con social, sin CI formal.

## Veredicto B2 (MovieLens per-user)

**PARCIAL / DÉBIL — ya no “sin potencia”.**

- No soporta “M3−M2 es efecto frontera”.
- Soporta débilmente “M4 social ayuda en B25 (y MID para M4c)”.
- **No GO** para WP4 solo por B2.

## Soft assignment (anexo Route B)

Controlled leave-k, `--skip-rebuild`, variantes M1/M2/M3/M3_soft:

| Estrato | M3 | M3_soft | Δ (M3−soft, + = soft mejor) |
|---|---:|---:|---:|
| 0 | 1.112 | 1.107 | +0.005 |
| 1–3 | 0.936 | 0.939 | −0.003 |
| 4–10 | 1.002 | 0.997 | +0.004 |
| >10 | 0.957 | 0.955 | +0.003 |

**Soft NO-GO** para claim fuerte: deltas ~0–0.005 RMSE, sin patrón cold-específico claro.

## Ciao

Sin re-run en este pase.
