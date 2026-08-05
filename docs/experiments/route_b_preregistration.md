# Route B pre-registration

**Date:** 2026-08-04  
**Branch:** `feat/route-b-experiments`  
**Protocol:** [route_b_protocol.md](route_b_protocol.md)  
**Commit:** fill with `git rev-parse HEAD` before first WP1/WP3 run.

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

- [ ] `experiment_manifest.json` / `canonical_baseline.json` / `network_selection_results.json` sin regenerar
- [ ] Epinions subsample window (si aplica): ________________

## Soft assignment (extensión, no B1–B5)

Track aparte: variante `M3_soft` en cold-start (membresía blanda por overlap de ítems → comunidades).  
GO informal: M3_soft mejora M3 (y/o M1) en estratos `1-3`/`4-10` con CI no trivial.  
No cuenta como B1/B2 del protocolo congelado.

## Compromiso

Todos los resultados (GO / nulos / NO-GO) se publican en `docs/experiments/route_b_wp<k>_findings.md`.
