# Trust consolidation — Frente A (hallazgos)

**Fecha:** 2026-08-06  
**Scope:** A1 multi-semilla Ciao zero-shot trust. A2 (Epinions / réplica) aparcado. A3 (ablación) omitido — paper principal es core+LPH.  
**Fuente:** `data/ciao/route_b/trust_multiseed/seed_*/zero_shot_trust/` · resumen `data/ciao/route_b/trust_multiseed/a1_summary.csv`  
**Criterio pre-registrado:** signo de Δ(M2_trust−M1) en ≥ 8/10 semillas y CI bootstrap per-user excluyendo 0 en cada una.

## A1 — Multi-semilla (10)

Semillas: `{42, 7, 123, 2024, 31337, 101, 999, 12345, 271828, 314159}`. Solo cambia la init del CMF; el split zero-shot es determinista.

| seed | RMSE M1 | RMSE M2_trust | RMSE M3_trust | Δ pooled (M1−M2) | boot M2_vs_M1 [CI 95 %] |
|---:|---:|---:|---:|---:|---|
| 42 | 1.1914 | 1.0663 | 1.3820 | 0.1251 | 0.146 [0.141, 0.151] |
| 7 | 1.1877 | 1.0665 | 1.2955 | 0.1211 | 0.143 [0.137, 0.148] |
| 123 | 1.1913 | 1.0660 | 1.3170 | 0.1254 | 0.146 [0.141, 0.151] |
| 2024 | 1.1883 | 1.0678 | 1.2876 | 0.1205 | 0.142 [0.137, 0.147] |
| 31337 | 1.1879 | 1.0669 | 1.4471 | 0.1210 | 0.142 [0.137, 0.147] |
| 101 | 1.1894 | 1.0661 | 1.2512 | 0.1232 | 0.144 [0.139, 0.149] |
| 999 | 1.1876 | 1.0678 | 1.2371 | 0.1198 | 0.141 [0.136, 0.146] |
| 12345 | 1.1890 | 1.0672 | 1.3085 | 0.1218 | 0.143 [0.138, 0.148] |
| 271828 | 1.1881 | 1.0667 | 1.2920 | 0.1214 | 0.141 [0.136, 0.146] |
| 314159 | 1.1871 | 1.0670 | 1.3478 | 0.1201 | 0.141 [0.136, 0.146] |

**Resumen M2_trust vs M1**

| Métrica | Valor |
|---|---|
| Signo Δ pooled > 0 | **10/10** |
| CI bootstrap excluye 0 | **10/10** |
| Δ pooled mean ± SD | **0.1219 ± 0.0020** |
| Δ pooled peor / mejor | 0.1198 / 0.1254 |
| boot mean_delta mean ± SD | 0.1428 ± 0.0020 |

**Criterio ≥ 8/10: PASS.**

## Lectura

1. **El positivo trust es robusto a la semilla del CMF.** La ganancia de M2_trust (~10 % RMSE, ~0.12 pooled / ~0.14 per-user bootstrap) casi no se mueve entre semillas (SD ≈ 0.002).
2. **M3_trust no solo no ayuda: empeora en las 10 semillas** (Δ pooled M1−M3 = −0.128 ± 0.062; RMSE M3 entre 1.24 y 1.45). Confirma que la señal es centralidad del grafo trust, no features comunitarias sobre ese grafo. No valida LPH.
3. **A2 aparcado** (Epinions fuera de alcance). Sin réplica cross-dataset en esta fase.
4. **A3 omitido** (ablación leave-one-out); solo haría falta para un paper corto trust.

## Veredicto Frente A

**A1 PASS.** El claim "centralidad de confianza explícita ayuda en zero-shot (Ciao)" queda blindado multi-semilla. Listo como resultado auxiliar del paper de fusión o como núcleo de un manuscript trust aparte. Siguiente: Frente B (M4c) y C (compuerta de cobertura) — hilo core+LPH.
