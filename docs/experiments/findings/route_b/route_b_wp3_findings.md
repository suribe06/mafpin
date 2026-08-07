# WP3 — Estabilidad de comunidades / LPH (hallazgos)

**Fecha:** 2026-08-07 (re-run per-user temporal, MovieLens)  
**Branch:** `feat/route-b-experiments`  
**Hipótesis:** B3 — $\tilde{h}_v$ y la frontera son estables frente a α vecinos y detector.  
**Umbrales:** interpretable si ρ ≥ 0.7 y Jaccard(B10) ≥ 0.5; ruido si ρ < 0.4.  
**Fuente:** `data/movielens/route_b/community_stability.csv`  
**Red centro:** M3 congelada = rayleigh α₆₈ (per-user campaign).

## MovieLens (per-user)

| eje | ρ Spearman (media) | Jaccard B10 (media) | Veredicto |
|---|---:|---:|---|
| α vecinos (68 vs 66/67/69/70), DEMON | **0.865** | **0.676** | interpretable (ρ≥0.7) |

Detalle ρ = {0.857, 0.873, 0.884, 0.847}; Jaccard B10 = {0.539, 0.714, 0.771, 0.681}.

Estático (misma partición): mean comunidades/usuario ≈ **1.04**; frac. frontera ≈ **0.231**.

**Detector ASLPAw:** no ejecutado (deps `gmpy2` / `ASLPAw` ausentes). B3 en eje detector sigue pendiente.

## Ciao

Sin re-run. Referencia 2026-08-05 (cutoff global): ρ media ≈ 0.977 — no actualizar hasta prereqs per-user.

## Veredicto

**B3 SOPORTADA en eje α-vecinos (MovieLens per-user).**  
Un poco menos estable que el run global (0.865 vs ~0.91) pero claramente sobre el umbral. Detector-swap no testeado.
