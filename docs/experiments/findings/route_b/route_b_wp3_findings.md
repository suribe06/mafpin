# WP3 — Estabilidad de comunidades / LPH (hallazgos)

**Fecha:** 2026-08-05  
**Branch:** `feat/route-b-experiments`  
**Hipótesis:** B3 — $\tilde{h}_v$ y la frontera son estables frente a α vecinos y detector.  
**Umbrales (pre-registro):** interpretable si ρ ≥ 0.7 y Jaccard(B10) ≥ 0.5; ruido si ρ < 0.4.  
**Fuente:** `data/<ds>/route_b/community_stability.csv`

## Resultado

| Dataset | eje | ρ Spearman (media) | Jaccard B10 (media) | Veredicto |
|---|---|---|---|---|
| MovieLens | α vecinos (63 vs 61/62/64/65) | **0.914** | 0.635 | interpretable |
| Ciao | α vecinos (83 vs 81/82/84/85) | **0.977** | 0.671 | interpretable |

Detalle por vecino:

- **MovieLens** ρ = {0.913, 0.949, 0.891, 0.905}; Jaccard = {0.692, 0.731, 0.485, 0.632}.
- **Ciao** ρ = {0.968, 0.985, 0.983, 0.974}; Jaccard = {0.708, 0.672, 0.657, 0.647}.

Contexto estructural (partición congelada, comparación estática):

| Dataset | nº comunidades (media/usuario) | frac. frontera |
|---|---|---|
| MovieLens | 1.20 | 0.277 |
| Ciao | 0.77 | 0.200 |

## Veredicto

**B3 SOPORTADA en el eje α-vecinos.** El ranking de LPH es muy estable (ρ ≥ 0.89 en todos los vecinos, muy por encima de 0.7) y el conjunto B10 se solapa por encima del umbral en promedio (un solo vecino de MovieLens, índice 64, queda en 0.485, marginalmente bajo 0.5; el resto pasa).

## Limitaciones

- **Eje detector sin probar:** ASLPAw no corrió por dependencias opcionales (`gmpy2`, `ASLPAw`). El único detector evaluado es DEMON. El componente "robustez al detector" de B3 queda **pendiente**. Instalar deps y re-correr, o declararlo fuera de alcance.
- **Ciao tiene nº de comunidades medio < 1** (0.77): muchos usuarios caen en 0 comunidades bajo DEMON, lo que restringe severamente la población de estratos frontera (ver WP2).
