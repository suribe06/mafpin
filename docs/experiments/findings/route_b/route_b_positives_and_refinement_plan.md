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

**A1. Multi-semilla.** Repetir el track zero-shot con semillas {42, 7, 123, 2024, 31337}. El split es determinista (todos los usuarios del grafo trust van a test), así que `--seed` solo cambia la inicialización del CMF — exactamente lo que queremos aislar. `--output-dir` reemplaza el **root** de `ColdStartPaths`: cada corrida escribe bajo `<output-dir>/zero_shot_trust/`, sin tocar el run congelado de `data/ciao/cold_start/zero_shot_trust/`:

```bash
for s in 42 7 123 2024 31337; do
  conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
    --dataset ciao --mode zero_shot_trust --variants M1 M2_trust M3_trust \
    --seed $s --output-dir data/ciao/route_b/trust_multiseed/seed_$s
done
```

Criterio: el signo de Δ(M2_trust−M1) estable en ≥ 4/5 semillas y CI per-user excluyendo 0 en cada una. Con la magnitud actual (+0.125) esto debería pasar sobrado; si no pasa, tenemos un problema mayor y mejor saberlo ya.

**A2. Replicación en Epinions.** Verificado en el repo: `datasets/epinions/trust.txt` existe, `load_trust_graph` soporta `epinions`, y el track zero-shot no usa NetInf — no requiere ningún prerrequisito de la campaña core sobre Epinions. Mismo protocolo, mismas variantes, semilla 42 primero y multi-semilla si replica:

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset epinions --mode zero_shot_trust --variants M1 M2_trust M3_trust --seed 42
```

Criterio de replicación: Δ(M2_trust−M1) > 0 con CI limpio. No exigimos la misma magnitud, solo el mismo fenómeno.

**A3. Ablación de las features trust (opcional pero barato).** *(Corregido: la versión anterior hablaba de betweenness/eigenvector/closeness, que son features del core sobre la red inferida y no existen en este track.)* M2_trust usa exactamente **tres** features (`networks/social.py::compute_trust_features`): `trust_in_degree`, `trust_out_degree` y `trust_pagerank`; M3_trust añade `trust_community_size` y `trust_boundary_frac` (modularidad greedy sobre el grafo trust no dirigido — no DEMON ni LPH). Con solo tres features no hace falta SHAP: la ablación directa es más simple y más fuerte:

- **Leave-one-out:** re-entrenar M2_trust quitando una feature a la vez (3 fits) y **single-feature:** cada feature sola (3 fits), más el M2_trust completo. Siete fits por semilla — barato.
- Comparar RMSE en estrato 0 con delta pareado per-user y CI bootstrap vs el M2_trust completo.
- Reportar la correlación de Spearman entre las tres features sobre los usuarios del grafo trust: si `trust_pagerank` ≈ `trust_in_degree` (lo esperable), la historia interpretable es "número de seguidores", y eso se dice explícitamente.

Implementación: hoy **no hay CLI para subsets de features**. Camino mínimo: script `scripts/route_b_trust_ablation.py` que reutilice el split zero-shot de `recommender/experiment/cold_start/splits.py` y `build_trust_attribute_tables` de `recommender/experiment/cold_start/trust_variants.py`, filtre columnas de la tabla de atributos de M2_trust y entrene el mismo CMF por subset. Nada más que eso.

**Dos aclaraciones de alcance que conviene dejar escritas:**

1. **El SHAP existente no aplica aquí.** `shap_results.json` (2026-05-20) se calculó sobre el modelo enhanced del core — red inferida, otras features, otro modelo. No responde nada sobre la interpretabilidad del positivo 1; la interpretabilidad de M2_trust sale de la ablación A3, no de ese SHAP.
2. **A3 no reemplaza el trabajo sobre la red inferida.** El análisis de features del core (SHAP incluido, si se refresca) sigue viviendo en los frentes B y C. Son dos modelos distintos con dos preguntas distintas: A3 pregunta *qué ancla a un usuario sin ratings vía grafo explícito*; B/C preguntan *si la señal comunitaria de la red inferida aporta algo*.

**Salida A:** `docs/experiments/findings/route_b/trust_consolidation_findings.md` con las tablas multi-semilla, la replicación, la ablación y veredicto. Si A1+A2 pasan, este resultado queda listo como claim central o como paper corto independiente.

### Frente B — Afinar el positivo M4c (verificar antes de celebrar)

**B1. Multi-semilla en MovieLens.** Re-correr `final_eval --beyond-accuracy` para M1, M2, M3 y M4c con las cinco semillas (redes e HP congelados del manifest; solo cambia `--seed`). Dos comportamientos verificados en `recommender/experiment/final_eval.py` que condicionan el loop:

- `append_core_results` dedupea `core_experiment_results.csv` por `(dataset, model_variant)` quedándose con la **última** fila → cada semilla pisa la anterior, y además pisaría las filas canónicas de la campaña core. Lo mismo aplica a `data/<ds>/route_b/beyond_accuracy_results.csv` y al parquet per-user. Obligatorio: respaldar el CSV core antes del loop y archivar los artefactos por semilla.
- `apply_final_eval_deltas` calcula `rmse_delta_vs_baseline` contra la fila M1 **de la misma sesión** → M1 debe correrse en cada semilla, no solo M4c.

```bash
cp data/movielens/core_experiment_results.csv data/movielens/core_experiment_results.pre_b1_backup.csv
for s in 42 7 123 2024 31337; do
  for v in M1 M2 M3 M4c; do
    conda run --no-capture-output -n mafpin python pipeline.py \
      --steps final_eval --model-variant $v --beyond-accuracy \
      --dataset movielens --seed $s \
      --log-file data/movielens/logs/route_b/b1_${v}_seed_${s}.log
  done
  mkdir -p data/movielens/route_b/multiseed/seed_$s
  cp data/movielens/core_experiment_results.csv \
     data/movielens/route_b/beyond_accuracy_results.csv \
     data/movielens/route_b/beyond_accuracy_per_user.parquet \
     data/movielens/route_b/multiseed/seed_$s/
done
# restaurar la sesión canónica del core en el path estándar:
cp data/movielens/core_experiment_results.pre_b1_backup.csv data/movielens/core_experiment_results.csv
```

Criterio: NDCG@10(M4c) > NDCG@10(M1) y cobertura(M4c) > cobertura(M1) en ≥ 4/5 semillas, con RMSE siempre dentro de guardia.

**B2. Chequeo de mecanismo.** Dos piezas; ambas requieren cambios nuevos (hoy no existe ninguna de las dos):

1. **Instrumentar el regularizador** (`recommender/enhanced/social_regularization.py`): loguear por corrida (i) fracción de aristas con al menos un extremo con comunidad, (ii) fracción de aristas efectivamente des-ponderadas por `boundary_downweight`, (iii) distribución de `w_uv` antes/después. Con 40 % de cobertura comunitaria en MovieLens (WP3), si (i) es baja, el mecanismo "boundary-aware" opera sobre una minoría de aristas y eso cambia la historia.
2. **Correlación per-user con lo que ya se exporta.** Hoy **no hay NDCG per-user**: `compute_ranking_metrics` (`recommender/data.py`) calcula los scores por usuario internamente pero devuelve solo promedios. Opciones en orden: (a) usar lo disponible — RMSE per-user desde `data/<ds>/route_b/predictions/<variant>.parquet` (ya lo consume `recommender/experiment/route_b/boundary_strata.py`) y `cce_at_k`/`n_communities` per-user del parquet beyond-accuracy — y correlacionar Δ(M4c−M1) per-user con `n_communities`; (b) si se quiere NDCG per-user, exponer el vector desde `compute_ranking_metrics` (ya se calcula en el loop; es devolverlo, no recalcular).

Si la mejora de M4c viene de usuarios *sin* información comunitaria, el mecanismo no es el que contamos y toca renombrar la historia (regularización selectiva, no "boundary-aware").

**B3. Segundo terreno con ranking sano.** Ciao no sirve como réplica de ranking. Dos opciones, en orden de preferencia: (a) esperar a las redes densificadas del Frente C y re-evaluar ahí mismo; (b) si C se retrasa, evaluar en Epinions cuando el ladder core exista. No inventar un protocolo de candidatos nuevo solo para esto: cambiaría la métrica y no sería comparable.

**Salida B:** `route_b_m4c_verification_findings.md`. GO si B1 y B2 confirman (efecto estable + mecanismo real); en ese caso M4c pasa de anécdota a resultado defendible.

### Frente C — La compuerta: ¿el positivo de la fusión se puede obtener de forma segura?

WP3 identificó la causa raíz de casi todos los nulos: con presupuesto de aristas k = 2N, las redes NetInf tienen transitividad bajísima y DEMON deja sin comunidad al 60–80 % de los usuarios. La señal LPH nunca tuvo sustrato. Antes de escribir "la fusión no funciona", hay que darle una prueba justa — y antes de dársela, fijar la vara.

**C1. Densificación.** Regenerar redes con `--k-avg-degree 4` y `--k-avg-degree 6` en ambos datasets. `cascades.txt` y los deltas ya existen y no dependen de k, así que basta `--steps inference centrality communities`. **Operativo, importante:** esos steps escriben en `data/<ds>/inferred_networks/`, `data/<ds>/centrality_metrics/` y `data/<ds>/communities/` — exactamente los paths congelados del core (`DatasetPaths` no tiene override de salida). Snapshot antes de tocar nada; restaurar para volver a reproducir el core:

```bash
for d in inferred_networks centrality_metrics communities; do
  cp -r data/movielens/$d data/movielens/${d}_k2_frozen
done

conda run --no-capture-output -n mafpin python pipeline.py \
  --steps inference centrality communities \
  --dataset movielens --k-avg-degree 4 \
  --log-file data/movielens/logs/route_b/c1_density_k4.log
```

Regenerar in-place (con snapshot) tiene un beneficio concreto: los comandos existentes de WP1/WP2 ([route_b_commands.md](route_b_commands.md)) corren sin modificación sobre las redes densas.

**C2. Corrección del constructo frontera** (lo que WP2 dejó mandado):
- Estrato frontera exige `num_communities ≥ 2` y los usuarios con 0 comunidades quedan **fuera** de los percentiles de $\tilde{h}_v$ (elimina la masa degenerada en λ: 53 % de Ciao, 27 % de MovieLens). Punto de aplicación mínimo: la estratificación en `recommender/experiment/route_b/boundary_strata.py` — no hace falta tocar `networks/communities/lph.py`; el CSV puede seguir trayendo λ y la exclusión se hace al estratificar, dejándolo declarado en el findings.
- DEMON siempre con `seed=42` (ya cableado: `Defaults.COMMUNITY_SEED`, `detect_overlapping_communities(..., seed=...)`).

**C3. La compuerta (pre-registrar antes de correr C1, con umbrales inamovibles):**

| Condición | Umbral |
|---|---|
| Cobertura comunitaria | ≥ 60 % de usuarios con ≥ 1 comunidad, en ≥ 1 dataset |
| Población frontera | estrato B25 (con `num_communities ≥ 2`) con N ≥ 30 en ese dataset |
| Estabilidad (WP3 rápido sobre la densidad elegida) | ρ Spearman α-vecinos ≥ 0.7 y Jaccard(B10) ≥ 0.5 |

- **Si la compuerta NO pasa** con k=4 ni k=6: se cierra la vía. El artículo es el diagnóstico ("las señales de frontera comunitaria no son obtenibles de forma segura sobre redes de difusión inferidas dispersas, y este es el mecanismo"), que con WP1–WP3 ya está al 80 %.
- **Si la compuerta pasa:** los artefactos congelados del core dejan de aplicar sobre las redes densas — los índices α de `experiment_manifest.json` y `network_selection_results.json` pertenecen al grid k=2. El flujo mínimo válido es la escalera core reducida a M2, M3 y M4c: Etapa A `hypertune` (50 trials enhanced / 200 social — mismo presupuesto que recibió el core), Etapa B `recommend --all-networks`, `network_selection`, y entonces WP1 + WP2 con el constructo corregido. `canonical_baseline.json` (M1) no depende de la red y se reutiliza. Criterios idénticos a los del protocolo (§4.5 y §5.5). Si aparecen los efectos → paper de método, y solo entonces WP4 completo (baselines externos, Epinions, multi-semilla global).

**Salida C:** `route_b_gate_findings.md` con el veredicto de compuerta, y `route_b_dense_wp1_wp2_findings.md` si se cruza.

## Orden y punto de decisión

1. **A1, A2 y B1, B2 primero** (baratos, paralelizables). El positivo trust queda blindado o herido; M4c queda confirmado o degradado a nota al pie.
2. **C1–C3 después** (o en paralelo si hay cómputo), con el pre-registro de la compuerta commiteado **antes** del primer run.
3. **Sesión de decisión al cierre:** con A, B y el veredicto de compuerta sobre la mesa, se elige el artículo — método (fusión bajo precondición de cobertura) o diagnóstico (por qué la fusión no es obtenible en redes dispersas + qué sí funciona). En ambos, el positivo trust entra como resultado central o coprotagonista, y ninguna cifra titular se publica sin su multi-semilla.

La regla de oro no cambia: los umbrales se fijan antes de mirar, todo se reporta, y los artefactos congelados de la campaña core no se tocan.
