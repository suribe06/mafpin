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

## Por qué Ciao tiene menos comunidades por usuario que MovieLens

`mean_num_communities` cuenta **membresías por usuario**, no comunidades de la red. Ambas redes tienen el mismo número de comunidades (13); lo que cambia es la cobertura:

| | MovieLens (rayleigh #63) | Ciao (exponential #83) |
|---|---|---|
| usuarios / aristas no dirigidas | 671 / 1252 | 2248 / 4394 |
| grado medio | 3.73 | 3.91 |
| nodos aislados (grado 0) | 164 (24 %) | 744 (33 %) |
| transitividad | 0.218 | **0.068** |
| comunidades DEMON | 13 | 13 |
| usuarios con ≥ 1 comunidad | 268 (40 %) | 444 (**20 %**) |
| membresías por usuario **cubierto** | 3.0 | **3.9** |

El promedio de Ciao está dominado por ceros: 1804 de 2248 usuarios (80 %) no pertenecen a ninguna comunidad. Condicionado a estar cubiertos, los usuarios de Ciao pertenecen a *más* comunidades que los de MovieLens. La causa es la red inferida, no el detector: NetInf reparte un presupuesto de aristas proporcional a N (`K_AVG_DEGREE = 2`), así que el grado medio es casi igual en ambos datasets, pero en Ciao las aristas forman tres veces menos triángulos. DEMON opera sobre ego-networks con `min_com_size = 3`, de modo que baja transitividad se traduce en baja cobertura. El corte por grado es nítido: todo usuario con grado ≤ 1 tiene 0 comunidades, y la media solo supera 1 a partir de grado ≈ 6.

## Limitaciones

- **Eje detector sin probar:** ASLPAw no corrió por dependencias opcionales (`gmpy2`, `ASLPAw`). El único detector evaluado es DEMON. El componente "robustez al detector" de B3 queda **pendiente**. Instalar deps y re-correr, o declararlo fuera de alcance.
- **Los artefactos congelados se generaron con DEMON sin semilla.** `cdlib.algorithms.demon` no expone semilla y usa el RNG global de `random`, así que era no determinista. Tres recálculos sobre la *misma* red congelada de Ciao dieron media de 0.725 / 0.753 / 0.754 (el CSV congelado dice 0.774) y ~10 % de usuarios (224/2248) cambiaron su conteo de comunidades entre corridas; en MovieLens, 141/671 (21 %). El ranking de LPH sí aguanta: ρ Spearman entre corridas = 0.992–0.993. Es decir, **el techo de estabilidad alcanzable es ρ ≈ 0.99**, y los α-vecinos quedan por debajo (0.89–0.985), así que lo medido no es solo ruido del detector — pero parte del gap sí lo es. Corregido a partir de `Defaults.COMMUNITY_SEED = 42` (`detect_overlapping_communities(..., seed=...)`): la detección ahora es reproducible bit a bit. Los artefactos **no** se regeneraron para no romper el pre-registro; regenerarlos movería la media de Ciao de 0.774 a 0.7576.
- **Ciao tiene nº de comunidades medio < 1** (0.77): el 80 % de los usuarios cae en 0 comunidades bajo DEMON, lo que restringe severamente la población de estratos frontera (ver WP2).
