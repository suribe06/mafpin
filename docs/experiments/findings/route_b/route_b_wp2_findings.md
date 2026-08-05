# WP2 — Estratos frontera + cross-community (hallazgos)

**Fecha:** 2026-08-05  
**Branch:** `feat/route-b-experiments`  
**Hipótesis:** B2 — La ganancia de M3 sobre M2 (y de M4 sobre M3) se concentra en usuarios frontera (B10/B25 de $\tilde{h}_v$) y/o en ratings cross-community.  
**Umbrales (pre-registro):** celda válida con N ≥ 30 usuarios (fusionar B10→B25 si hace falta); bootstrap 1000 remuestras, CI 95 % percentil.  
**Fuente:** `data/<ds>/route_b/boundary_strata_results.csv`, `boundary_strata_bootstrap.csv`, `cross_community_items_results.csv`

## Poblamiento de estratos

| Dataset | B10 | B25 | MID | E75 |
|---|---|---|---|---|
| MovieLens | **0** | **0** | 17 | 130 |
| Ciao | 5 | 11 | 92 | 390 |

**Ningún estrato frontera alcanza N ≥ 30 en ningún dataset.** En MovieLens B10/B25 están vacíos (bajo la partición congelada casi no hay usuarios multi-comunidad en test). En Ciao B10=5, B25=11; incluso fusionados (16) siguen por debajo de 30. Esto es consecuencia directa del hallazgo de WP3 (Ciao: nº comunidades medio 0.77/usuario).

## Bootstrap por estrato (mean_delta [CI 95 %])

**MovieLens** (solo MID/E75 son evaluables, y MID con N=17 < 30):

| Estrato | M3−M2 | M3−M1 | M4c−M3 | M4d−M3 |
|---|---|---|---|---|
| MID (17) | +0.005 [−0.010, 0.021] | +0.056 [0.007, 0.109] ✅ | −0.014 [−0.030, 0.004] | +0.000 [−0.011, 0.014] |
| E75 (130) | +0.002 [−0.004, 0.006] | +0.029 [0.023, 0.035] ✅ | −0.006 [−0.012, 0.001] | −0.002 [−0.006, 0.001] |

**Ciao:**

| Estrato | M3−M2 | M3−M1 | M4c−M3 | M4d−M3 |
|---|---|---|---|---|
| B10 (5) | +0.004 [−0.005, 0.019] | +0.007 [−0.002, 0.016] | −0.017 [−0.062, 0.016] | −0.014 [−0.049, 0.018] |
| B25 (11) | −0.002 [−0.008, 0.004] | +0.004 [−0.021, 0.028] | +0.011 [−0.019, 0.048] | +0.008 [−0.002, 0.019] |
| MID (92) | +0.000 [−0.004, 0.005] | +0.004 [−0.004, 0.012] | +0.003 [−0.008, 0.016] | −0.001 [−0.007, 0.006] |
| E75 (390) | −0.000 [−0.002, 0.001] | **−0.005 [−0.008, −0.001]** ⚠️ | −0.000 [−0.005, 0.004] | −0.001 [−0.003, 0.001] |

(Delta positivo = M3 mejor. `M3−M2` mide el aporte de las comunidades; `M3−M1` el aporte total de side info.)

## Cross-community items (RMSE, sin CI)

| Dataset | n | M1 | M2 | M3 | M4c | M4d |
|---|---|---|---|---|---|---|
| MovieLens | 373 | 0.815 | 0.814 | **0.805** | 0.825 | 0.807 |
| Ciao | 66 | 0.826 | 0.820 | 0.819 | **0.807** | 0.817 |

## Lectura

1. **B2 no es testeable con potencia.** Todas las celdas frontera (B10/B25) tienen N < 30; en MovieLens están vacías. La hipótesis de "ganancia concentrada en frontera" no se puede sostener ni refutar con estos datos.
2. **Donde sí hay potencia (E75, MID), la ganancia no es frontera-específica.** En MovieLens M3−M1 es positivo y con CI que excluye 0 tanto en MID como en E75 (usuarios core) — es una mejora **amplia**, no de frontera. Y **M3−M2 es nulo en todos lados**: las comunidades (M3) no aportan sobre las features de centralidad (M2).
3. **Señal contraria en Ciao E75:** en el estrato mejor poblado (390 usuarios core), M3 es **ligeramente peor** que M1 (CI [−0.008, −0.001], excluye 0). Lo contrario de "M3 ayuda".
4. **Cross-community:** único indicio a favor. M3 tiene el menor RMSE en ratings cross-community de MovieLens (0.805) y M4c el menor en Ciao (0.807), consistente con la subida de CCE en WP1. Pero son n pequeños (373 / 66) y sin CI, así que es indicativo, no concluyente.

## Veredicto

**B2 NO SOPORTADA (principalmente por falta de potencia).** Los estratos frontera no se pueden poblar a N ≥ 30 con la partición actual; donde hay potencia, la ganancia de M3 es amplia (no de frontera) y desaparece frente a M2. El único guiño a favor es el RMSE cross-community, que habría que confirmar con más datos y CI.

---

## Anexo — Soft assignment (`M3_soft`), track aparte

No cuenta como B1–B5. GO informal (pre-registro): `M3_soft` mejora M3 (y/o M1) en estratos `1-3`/`4-10` con CI no trivial.  
**Fuente:** `data/<ds>/cold_start/cold_start_results.csv`, `bootstrap_confidence_intervals.csv`

RMSE por estrato — M3 vs M3_soft (modo controlled):

| Dataset (split) | Estrato | M3 | M3_soft | Δ (M3−soft) |
|---|---|---|---|---|
| MovieLens (leave_k) | 0 | 1.1094 | 1.1113 | −0.0019 |
| | 1-3 | 0.9370 | 0.9372 | −0.0002 |
| | 4-10 | 0.9990 | 1.0031 | −0.0040 |
| | >10 | 0.9481 | 0.9514 | −0.0033 |
| Ciao (leave_last) | 0 | 0.8848 | 0.8837 | +0.0011 |
| | 1-3 | 0.9279 | 0.9280 | −0.0000 |
| | 4-10 | 0.9403 | 0.9393 | +0.0010 |
| | >10 | 0.9114 | 0.9115 | −0.0000 |

Contexto de los CIs cold-start (M3 vs M1 / M2; el bootstrap **no** incluye comparaciones con `M3_soft`):

- **MovieLens:** todos los `M3_vs_M1` y `M3_vs_M2` tienen CI que incluye 0 en los 4 estratos → sin efecto cold-start detectable (consistente con hallazgos previos de leave-k débil en ML).
- **Ciao:** `M3_vs_M1` fuertemente positivo en todos los estratos (p.ej. estrato 0: +0.098 [0.049, 0.146]; 1-3: +0.097 [0.067, 0.128]), pero `M3_vs_M2` **nulo** → toda la ganancia es M2 sobre M1 (las features de centralidad); las comunidades no aportan.

**Veredicto soft:** **NO-GO.** En MovieLens `M3_soft` es marginalmente **peor** que M3 en los cuatro estratos; en Ciao es mejor por ~0.001 en dos estratos (magnitud trivial, sin CI para la comparación). No se cumple el GO informal. La membresía blanda por overlap de ítems, como está, no mueve la aguja.
