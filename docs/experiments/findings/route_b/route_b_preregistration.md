# Route B pre-registration

**Date:** 2026-08-04  
**Branch:** `feat/route-b-experiments`  
**Protocol:** [route_b_protocol.md](route_b_protocol.md)  
**Commit:** `eca0171c9de9cccfd00e32e41b1f085390a068bf`

## Hypotheses (copied from protocol §3)

| ID | Enunciado | WP |
|---|---|---|
| **B1** | Variantes con señal de frontera (M3, M4c, M4d) aumentan CCE@10, ILD@10 y/o cobertura vs M1/M2 **sin** degradar RMSE > 0.5 % relativo ni colapsar NDCG@10. | WP1 |
| **B2** | Ganancia M3 over M2 (y M4 over M3) se concentra en usuarios frontera (B10/B25 de $\tilde{h}_v$) y/o ratings cross-community. | WP2 |
| **B3** | $\tilde{h}_v$ y frontera son estables vs α vecinos y detector. | WP3 |
| **B4** | Efectos se replican en ≥2/3 datasets, multi-semilla, competitivos vs SoRec/SocialMF/TrustSVD. | WP4 (solo si GO) |
| **B5** | Extensión dirigida/ponderada de $\tilde{h}_v$ recupera señal (condicional). | WP5 |

## Umbrales (no modificar post-hoc)

- Guardia accuracy: ΔRMSE relativo ≤ +0.5 %; NDCG@10 ≥ 0.8 × NDCG@10(M1)
- WP3 interpretable: Spearman ρ ≥ 0.7 (α vecinos) y Jaccard(B10) ≥ 0.5
- WP3 ruido: ρ < 0.4
- Celda WP2: N ≥ 30 usuarios (fusionar B10→B25 si hace falta)
- Multi-semilla: {42, 7, 123, 2024, 31337}
- Bootstrap: 1000 remuestras, CI 95 % percentil
- Holm sobre familia: {M3−M2, M4x−M3, M3−M1} × {global, B10, B25} × datasets

## Congelamiento

- [x] `experiment_manifest.json` / `canonical_baseline.json` / `network_selection_results.json` sin regenerar
- [ ] Epinions subsample window (si aplica): ________________

## Soft assignment (extensión, no B1–B5)

Track aparte: variante `M3_soft` en cold-start (membresía blanda por overlap de ítems → comunidades).  
GO informal: M3_soft mejora M3 (y/o M1) en estratos `1-3`/`4-10` con CI no trivial.  
No cuenta como B1/B2 del protocolo congelado.

## Compromiso

Todos los resultados (GO / nulos / NO-GO) se publican en `docs/experiments/findings/route_b/route_b_wp<k>_findings.md`.

## Veredicto del gate (2026-08-05, post WP1+WP2+WP3)

| Hipótesis | WP | Resultado | Detalle |
|---|---|---|---|
| B1 | WP1 | **Parcial / débil** | Único win limpio: M4c en MovieLens (cobertura+CCE sin costo de NDCG/RMSE). M3 sube CCE en ambos datasets pero colapsa NDCG@10 en ML; Ciao con ranking degenerado. Ver `route_b_wp1_findings.md`. |
| B2 | WP2 | **No soportada (sin potencia)** | Estratos frontera sin poblar (ML B10/B25 vacíos; Ciao N<30 aun fusionando). Donde hay potencia, la ganancia es amplia (no frontera) y nula vs M2; en Ciao E75 M3 es levemente peor que M1. Ver `route_b_wp2_findings.md`. |
| B3 | WP3 | **Soportada (eje α)** | ρ ≥ 0.91 (ML) / 0.98 (Ciao), Jaccard B10 ≥ 0.5 en media. Eje detector pendiente (ASLPAw sin deps). Ver `route_b_wp3_findings.md`. |
| soft | — | **NO-GO** | `M3_soft` no mejora M3 en `1-3`/`4-10`; en ML es marginalmente peor. |

**Decisión: NO-GO para WP4 tal cual.** La estructura es estable (B3) pero no hay una ganancia beyond-accuracy consistente cross-dataset (B1) ni evidencia de concentración en frontera (B2, sin potencia). No lanzar baselines Cornac/Epinions/multi-semilla todavía.

**Disparador de WP5 activo** (protocolo §8: "GO débil en WP1/WP2"). Antes de escalar, atacar: (1) potencia de estratos frontera — umbral de $\tilde{h}_v$ o dataset más denso; (2) colapso de NDCG en las variantes de comunidades en ML; (3) eje detector de WP3 (deps ASLPAw). El RMSE cross-community (único guiño a favor) debe confirmarse con CI.
