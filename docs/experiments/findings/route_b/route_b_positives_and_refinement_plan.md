# Los dos positivos del proyecto y el plan para afinarlos

**Fecha:** 2026-08-05
**Rama:** `feat/route-b-experiments`
**Contexto:** campaña core ([MovieLens](../core_experiment_movielens_findings.md), [Ciao](../core_experiment_ciao_findings.md)), [cold start](../cold_start_findings.md) y Ruta B ([WP1](route_b_wp1_findings.md), [WP2](route_b_wp2_findings.md), [WP3](route_b_wp3_findings.md)).

---

Después de tres campañas de experimentos, la foto honesta es esta: la mayoría de las hipótesis que motivaron la fusión MAFPIN+LPH dieron efectos nulos o dentro del ruido, pero quedaron en pie **dos resultados positivos** que valen la pena. Este documento los describe con calma y deja el plan concreto para afinarlos — porque un positivo sin verificación de robustez es una anécdota, no un resultado.

## Positivo 1 — Centralidad de confianza en zero-shot (Ciao)

Este es, sin discusión, el resultado más fuerte que tenemos.

**Qué hicimos.** Tomamos los 2 215 usuarios de Ciao que aparecen en el grafo de confianza explícito y les ocultamos **todas** sus calificaciones (35 480 ratings al estrato de test). Es el cold start absoluto: el modelo nunca vio un solo rating de estos usuarios. La pregunta era si se puede decir algo útil sobre ellos usando únicamente su posición en el grafo de confianza.

**Qué encontramos.** Sí se puede, y no por poco:

| Variante | RMSE (estrato 0) | Δ vs M1 |
|---|---|---|
| M1 (CMF solo ratings) | 1.1914 | — |
| **M2_trust** (+ centralidades del grafo trust) | **1.0669** | **+0.1246 (~10.5 %)** |
| M3_trust (+ features comunitarias) | 1.1740 | +0.0175 |

El CI bootstrap per-user de M2_trust vs M1 es [0.140, 0.150] — limpio, lejos de cero, con N grande. Para un usuario sin historia, M1 solo puede tirar de medias globales; las centralidades del grafo de confianza le dan al CMF un ancla en el espacio latente, y esa ancla vale un 10 % de RMSE.

**La parte incómoda que hay que decir siempre:** las features comunitarias no solo no ayudaron — M3_trust pierde contra M2_trust por ~0.096. La señal está en la centralidad, no en las comunidades. Este positivo **no** valida la fusión LPH; valida "grafo social explícito + centralidad" para zero-shot.

**Debilidades actuales:** un dataset, una semilla, hiperparámetros congelados de la campaña core. Nada de eso invalida el efecto (la magnitud y el CI son demasiado claros), pero para publicarlo como claim central hay que cerrar esos flancos.

## Positivo 2 — M4c en MovieLens: beyond-accuracy sin pagar peaje

Este es el único brote verde de la fusión propiamente dicha, y hay que tratarlo con cariño y con escepticismo a partes iguales.

**Qué hicimos.** En WP1 evaluamos las variantes congeladas de la campaña core con métricas beyond-accuracy (CCE@10, ILD, novelty, cobertura de catálogo, Gini), manteniendo dos guardias pre-registradas: no degradar RMSE más de 0.5 % relativo y no colapsar NDCG@10 (≥ 0.8 × el de M1).

**Qué encontramos.** M4c (`boundary_downweight`: regularización social que des-pondera las aristas que cruzan fronteras comunitarias) fue la **única** variante que mejoró beyond-accuracy sin costo:

| Métrica (MovieLens) | M1 | M4c | Lectura |
|---|---|---|---|
| NDCG@10 | 0.375 | **0.401** | mejor ranking del ladder |
| Cobertura de catálogo | 0.0079 | **0.0090** | la más alta de todas las variantes |
| CCE@10 | 0.550 | 0.575 | más exposición cross-community |
| RMSE | 1.0510 | 1.0421 (−0.8 %) | dentro de guardia |

Para contraste: M3 y M4d bajan más el RMSE pero colapsan el NDCG (0.257 y 0.173); M4c es el único punto del ladder donde ranking, cobertura y exposición suben a la vez. Y hay un guiño coherente en WP2: en los ratings de ítems *cross-community*, M3 tiene el mejor RMSE en MovieLens (0.805) y M4c el mejor en Ciao (0.807).

**Por qué todavía no me lo creo del todo (y nadie debería):**

1. **Un dataset y una semilla.** En Ciao el ranking es degenerado (NDCG@10 ≈ 0.001 para todo el mundo), así que no puede corroborar ni refutar. Y ya sabemos que L-BFGS tiene no-determinismo apreciable entre corridas.
2. **El mecanismo está en duda.** WP3 mostró que solo el 40 % de los usuarios de MovieLens tiene al menos una comunidad (20 % en Ciao). Si la mayoría de aristas no tiene información comunitaria en ninguno de sus extremos, ¿cuántas aristas está des-ponderando realmente M4c? No lo hemos medido. El efecto podría venir de donde creemos — o de un sesgo colateral de la normalización.
3. **El guiño cross-community es indicativo, no concluyente:** n pequeños (373 / 66 ratings) y sin CI.

## El plan, en tres frentes

La lógica es: **A** y **B** son baratos y verifican lo que ya tenemos; **C** es el experimento que decide si el positivo de la fusión se puede obtener *de forma segura* — es decir, con la precondición estructural corregida y criterios fijados antes de mirar los resultados. Reglas de higiene: las mismas del [protocolo](route_b_protocol.md) §2 (pre-registro, artefactos congelados intactos, CIs bootstrap 1 000 remuestreos, Wilcoxon+Holm, todo se reporta).

### Frente A — Consolidar el positivo trust (bajo costo, alta prioridad)

**A1. Multi-semilla.** Repetir el track zero-shot con semillas {42, 7, 123, 2024, 31337}, variando solo la inicialización del CMF:

```bash
for s in 42 7 123 2024 31337; do
  conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
    --dataset ciao --mode zero_shot_trust --variants M1 M2_trust M3_trust \
    --seed $s --output-dir data/ciao/cold_start/zero_shot_trust_seed_$s
done
```

Criterio: el signo de Δ(M2_trust−M1) estable en ≥ 4/5 semillas y CI per-user excluyendo 0 en cada una. Con la magnitud actual (+0.125) esto debería pasar sobrado; si no pasa, tenemos un problema mayor y mejor saberlo ya.

**A2. Replicación en Epinions.** Ya está cableado en `config.py` y tiene trust explícito. Mismo protocolo, mismas variantes, semilla 42 primero y multi-semilla si replica:

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset epinions --mode zero_shot_trust --variants M1 M2_trust M3_trust --seed 42
```

Criterio de replicación: Δ(M2_trust−M1) > 0 con CI limpio. No exigimos la misma magnitud, solo el mismo fenómeno.

**A3. Ablación de centralidades (opcional pero barato).** Dentro de M2_trust, apagar familias de features (grado / betweenness / eigenvector / closeness) una a la vez para saber cuál carga el efecto. Esto convierte "las centralidades ayudan" en una afirmación mecánicamente interpretable, que es lo que un revisor va a pedir.

**Salida A:** `docs/experiments/findings/route_b/trust_consolidation_findings.md` con las tablas multi-semilla, la replicación y veredicto. Si A1+A2 pasan, este resultado queda listo como claim central o como paper corto independiente.

### Frente B — Afinar el positivo M4c (verificar antes de celebrar)

**B1. Multi-semilla en MovieLens.** Re-correr `final_eval` con beyond-accuracy para M1, M2, M3, M4c con las cinco semillas (flags exactos en [route_b_commands.md](route_b_commands.md); redes e HP congelados, solo cambia la semilla del CMF). Criterio: NDCG@10(M4c) > NDCG@10(M1) y cobertura(M4c) > cobertura(M1) en ≥ 4/5 semillas, con RMSE siempre dentro de guardia.

**B2. Chequeo de mecanismo.** Instrumentar `recommender/enhanced/social_regularization.py` para loguear, por corrida: (i) fracción de aristas efectivamente des-ponderadas, (ii) distribución de los pesos w_uv, (iii) fracción de aristas donde al menos un extremo tiene comunidad. Después, correlacionar el delta per-user de NDCG (M4c−M1) con la cobertura comunitaria del usuario. Si la mejora viene de usuarios *sin* información comunitaria, el mecanismo no es el que contamos y toca renombrar la historia (regularización selectiva, no "boundary-aware").

**B3. Segundo terreno con ranking sano.** Ciao no sirve como réplica de ranking. Dos opciones, en orden de preferencia: (a) esperar a las redes densificadas del Frente C y re-evaluar ahí mismo; (b) si C se retrasa, evaluar en Epinions cuando el ladder core exista. No inventar un protocolo de candidatos nuevo solo para esto: cambiaría la métrica y no sería comparable.

**Salida B:** `route_b_m4c_verification_findings.md`. GO si B1 y B2 confirman (efecto estable + mecanismo real); en ese caso M4c pasa de anécdota a resultado defendible.

### Frente C — La compuerta: ¿el positivo de la fusión se puede obtener de forma segura?

WP3 identificó la causa raíz de casi todos los nulos: con presupuesto de aristas k = 2N, las redes NetInf tienen transitividad bajísima y DEMON deja sin comunidad al 60–80 % de los usuarios. La señal LPH nunca tuvo sustrato. Antes de escribir "la fusión no funciona", hay que darle una prueba justa — y antes de dársela, fijar la vara.

**C1. Densificación.** Regenerar redes con `--k-avg-degree 4` y `--k-avg-degree 6` en ambos datasets. **Operativo, importante:** no pisar los artefactos congelados — antes de nada, `cp -r data/<ds>/inferred_networks data/<ds>/inferred_networks_k2_frozen` (y lo mismo con `communities/`), o dirigir la salida a un árbol aparte. La campaña core debe seguir siendo reproducible tal cual.

```bash
# snapshot primero; luego, por dataset y densidad:
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps inference centrality communities \
  --dataset movielens --k-avg-degree 4 --seed 42 \
  --log-file data/movielens/logs/route_b/c1_density_k4.log
```

**C2. Corrección del constructo frontera** (lo que WP2 dejó mandado):
- $\tilde{h}_v$ = *missing* (no λ) para usuarios sin comunidad — se acabó la masa degenerada del 53 %.
- Estrato frontera exige `num_communities ≥ 2`. Un usuario sin comunidad no puede ser puente de nada.
- DEMON siempre con `seed=42` (ya cableado en `Defaults.COMMUNITY_SEED`).

**C3. La compuerta (pre-registrar antes de correr C1, con umbrales inamovibles):**

| Condición | Umbral |
|---|---|
| Cobertura comunitaria | ≥ 60 % de usuarios con ≥ 1 comunidad, en ≥ 1 dataset |
| Población frontera | estrato B25 (con `num_communities ≥ 2`) con N ≥ 30 en ese dataset |
| Estabilidad (WP3 rápido sobre la densidad elegida) | ρ Spearman α-vecinos ≥ 0.7 y Jaccard(B10) ≥ 0.5 |

- **Si la compuerta NO pasa** con k=4 ni k=6: se cierra la vía. El artículo es el diagnóstico ("las señales de frontera comunitaria no son obtenibles de forma segura sobre redes de difusión inferidas dispersas, y este es el mecanismo"), que con WP1–WP3 ya está al 80 %.
- **Si la compuerta pasa:** re-tune **limitado y pre-registrado** (la densidad nueva invalida los HP congelados: 50 trials enhanced / 200 social, solo variantes M2, M3, M4c, mismo presupuesto que recibió la campaña core), y repetir WP1 + WP2 sobre las redes densas con el constructo corregido. Criterios idénticos a los del protocolo (§4.5 y §5.5). Si aparecen los efectos → paper de método, y solo entonces WP4 completo (baselines externos, Epinions, multi-semilla global).

**Salida C:** `route_b_gate_findings.md` con el veredicto de compuerta, y `route_b_dense_wp1_wp2_findings.md` si se cruza.

## Orden y punto de decisión

1. **A1, A2 y B1, B2 primero** (baratos, paralelizables). El positivo trust queda blindado o herido; M4c queda confirmado o degradado a nota al pie.
2. **C1–C3 después** (o en paralelo si hay cómputo), con el pre-registro de la compuerta commiteado **antes** del primer run.
3. **Sesión de decisión al cierre:** con A, B y el veredicto de compuerta sobre la mesa, se elige el artículo — método (fusión bajo precondición de cobertura) o diagnóstico (por qué la fusión no es obtenible en redes dispersas + qué sí funciona). En ambos, el positivo trust entra como resultado central o coprotagonista, y ninguna cifra titular se publica sin su multi-semilla.

La regla de oro no cambia: los umbrales se fijan antes de mirar, todo se reporta, y los artefactos congelados de la campaña core no se tocan.
