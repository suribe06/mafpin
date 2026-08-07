# Los dos positivos del proyecto y el plan para afinarlos

**Fecha:** 2026-08-06
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

**Debilidades actuales:** un dataset, una semilla, hiperparámetros congelados de la campaña core. Nada de eso invalida el efecto (la magnitud y el CI son demasiado claros), pero para publicarlo como claim hay que cerrar esos flancos.

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

Mapa de claims (importante: no son el mismo paper):

| Frente | Qué verifica | ¿Habla de LPH / fusión? |
|---|---|---|
| **A** (trust zero-shot) | Centralidad del grafo **explícito** ayuda sin ratings | **No.** Resultado lateral fuerte; no valida MAFPIN+LPH |
| **B** (M4c) | Beyond-accuracy de la regularización boundary-aware | **Sí**, único brote de la fusión hoy |
| **C** (densificar + constructo) | Si LPH puede operar con cobertura comunitaria decente | **Sí**, compuerta del paper de método |

**B** y **C** son el hilo del paper (core experiment + LPH). **A** se consolida para no desperdiciarlo — sección auxiliar o paper corto aparte — **no** como el pegamento narrativo de la fusión. Reglas de higiene: las del [protocolo](route_b_protocol.md) §2 (pre-registro, artefactos congelados intactos, CIs bootstrap 1 000 remuestreos, Wilcoxon+Holm, todo se reporta).

Semillas multi-seed (A1 y B1): `42 7 123 2024 31337 101 999 12345 271828 314159` (10; criterio ≥ 8/10). Cinco era poco para un claim publicable.

### Frente A — Consolidar el positivo trust (lateral al paper LPH)

**Qué NO es A.** No une el core experiment con LPH. El SHAP NetInf del core explica M2/M3 sobre la red **inferida**. M2_trust vive sobre el grafo **trust** y tres columnas ajenas a LPH. Mezclarlos en una sola historia es un error de framing.

**A1. Multi-semilla (10).** El split es determinista (todos los usuarios del grafo trust van a test); `--seed` solo cambia la inicialización del CMF. `--output-dir` reemplaza el **root** de `ColdStartPaths`: cada corrida escribe bajo `<output-dir>/zero_shot_trust/`, sin tocar el run congelado de `data/ciao/cold_start/zero_shot_trust/`:

```bash
SEEDS="42 7 123 2024 31337 101 999 12345 271828 314159"
for s in $SEEDS; do
  conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
    --dataset ciao --mode zero_shot_trust --variants M1 M2_trust M3_trust \
    --seed $s --output-dir data/ciao/route_b/trust_multiseed/seed_$s
done
```

Criterio: signo de Δ(M2_trust−M1) estable en **≥ 8/10** semillas y CI per-user excluyendo 0 en cada una. Reportar media ± SD del Δ entre semillas y el peor caso.

**A2. Segundo dataset con trust — aparcado.** Epinions está cableado en el repo (`datasets/epinions/`, ~356 k aristas trust / ~922 k ratings) pero **queda fuera de alcance por ahora**: es un orden de magnitud más grande que Ciao (~57 k / ~36 k) y el cómputo de baseline + zero-shot + multi-semilla no vale el costo en esta fase. Foco actual: **Ciao + MovieLens**. Si más adelante hace falta réplica cross-dataset del positivo trust, buscar un dataset con grafo explícito de tamaño comparable a Ciao (o un poco mayor), no a escala Epinions. Hasta entonces A2 no bloquea nada.

**A3. Ablación de features trust (opcional; solo si A se publica aparte).** M2_trust usa exactamente **tres** features (`networks/social.py::compute_trust_features`): `trust_in_degree`, `trust_out_degree`, `trust_pagerank`. M3_trust añade `trust_community_size` y `trust_boundary_frac` (modularidad greedy sobre el trust no dirigido — no DEMON ni LPH). Con tres features la ablación directa es más simple y más fuerte que SHAP:

- **Leave-one-out** (3 fits) + **single-feature** (3 fits) + M2_trust completo → 7 fits por semilla.
- RMSE estrato 0 con delta pareado per-user y CI bootstrap vs M2_trust completo.
- Spearman entre las tres features: si `trust_pagerank` ≈ `trust_in_degree`, la historia es "número de seguidores" y se dice explícitamente.

Implementación: script `scripts/route_b_trust_ablation.py` que reutilice el split zero-shot y `build_trust_attribute_tables`, filtre columnas y entrene el mismo CMF por subset. **Si el paper es core+LPH, A3 se omite** — no aporta al argumento de la fusión.

**Alcance (no negociable):**

1. El SHAP de `shap_results.json` es del enhanced NetInf del core; **no** interpreta M2_trust.
2. A3 no reemplaza el trabajo sobre la red inferida (B/C).

**Salida A:** `trust_consolidation_findings.md` (A1; A3 solo si paper corto trust; A2 aparcado). En el paper de fusión, trust entra como *auxiliar* o manuscript separado.

### Frente B — Afinar el positivo M4c (verificar antes de celebrar)

**B1. Multi-semilla en MovieLens (10).** Re-correr `final_eval --beyond-accuracy` para M1, M2, M3 y M4c (redes e HP congelados; solo cambia `--seed`). Comportamientos en `recommender/experiment/final_eval.py` que condicionan el loop:

- `append_core_results` dedupea `core_experiment_results.csv` por `(dataset, model_variant)` quedándose con la **última** fila → cada semilla pisa la anterior y las filas canónicas del core. Obligatorio: backup del CSV core y archivar por semilla.
- Beyond-accuracy ahora hace **upsert** (`upsert_beyond_accuracy_results` / per-user) por `(dataset, model_variant)` — antes un `to_csv` overwrite dejaba solo la última variante (bug B1). Con el fix, el loop por variante acumula M1–M4c en el CSV antes de archivar.
- `apply_final_eval_deltas` calcula `rmse_delta_vs_baseline` contra M1 **de la misma sesión** → M1 debe correrse en cada semilla.

```bash
SEEDS="42 7 123 2024 31337 101 999 12345 271828 314159"
cp data/movielens/core_experiment_results.csv data/movielens/core_experiment_results.pre_b1_backup.csv
for s in $SEEDS; do
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
cp data/movielens/core_experiment_results.pre_b1_backup.csv data/movielens/core_experiment_results.csv
```

Criterio: NDCG@10(M4c) > NDCG@10(M1) y cobertura(M4c) > cobertura(M1) en **≥ 8/10** semillas, con RMSE siempre dentro de guardia.

**B2. Chequeo de mecanismo.** Dos piezas nuevas:

1. **Instrumentar el regularizador** (`recommender/enhanced/social_regularization.py`): loguear por corrida (i) fracción de aristas con ≥ 1 extremo con comunidad, (ii) fracción efectivamente des-ponderadas por `boundary_downweight`, (iii) distribución de `w_uv` antes/después. Con 40 % de cobertura en MovieLens (WP3), si (i) es baja, "boundary-aware" opera sobre una minoría de aristas.
2. **Correlación per-user.** Hoy no hay NDCG per-user exportado. En orden: (a) Δ RMSE per-user desde `route_b/predictions/<variant>.parquet` y `cce_at_k`/`n_communities` del parquet beyond-accuracy → correlacionar Δ(M4c−M1) con `n_communities`; (b) si se quiere NDCG per-user, exponer el vector que `compute_ranking_metrics` ya calcula en el loop.

Si la mejora viene de usuarios *sin* información comunitaria → renombrar a regularización selectiva, no "boundary-aware".

**B3. Segundo terreno con ranking sano.** Ciao no sirve (NDCG degenerado). Por ahora la réplica es **MovieLens densificado (Frente C)** — mismo dataset, mejor sustrato comunitario. Un tercer dataset (trust/ranking) solo si aparece uno de tamaño Ciao-like; Epinions queda aparcado. No inventar protocolo de candidatos nuevo.

**Salida B:** `route_b_m4c_verification_findings.md`. GO si B1 y B2 confirman (efecto estable + mecanismo real).

### Frente C — La compuerta: ¿el positivo de la fusión se puede obtener de forma segura?

WP3: con k = 2N, transitividad baja y DEMON deja sin comunidad al 60–80 %. La señal LPH nunca tuvo sustrato. Antes de escribir "la fusión no funciona", prueba justa — y antes, vara fija.

**C1. Densificación.** Regenerar con `--k-avg-degree 4` y `6` en ambos datasets. `cascades.txt`/deltas no dependen de k → basta `--steps inference centrality communities`. Esos steps escriben en paths congelados del core (`DatasetPaths` sin override). Snapshot antes; restaurar para reproducir el core:

```bash
for d in inferred_networks centrality_metrics communities; do
  cp -r data/movielens/$d data/movielens/${d}_k2_frozen
done

conda run --no-capture-output -n mafpin python pipeline.py \
  --steps inference centrality communities \
  --dataset movielens --k-avg-degree 4 \
  --log-file data/movielens/logs/route_b/c1_density_k4.log
```

In-place + snapshot: los comandos de WP1/WP2 ([route_b_commands.md](route_b_commands.md)) corren sin cambio sobre redes densas.

**C2. Corrección del constructo frontera:**

- Estrato frontera exige `num_communities ≥ 2`; usuarios con 0 comunidades **fuera** de los percentiles de $\tilde{h}_v$ (elimina la masa en λ: 53 % Ciao, 27 % MovieLens). Aplicar en `boundary_strata.py` al estratificar; el CSV puede seguir trayendo λ.
- DEMON con `seed=42` (ya cableado: `Defaults.COMMUNITY_SEED`).

**C3. Compuerta (pre-registrar antes de C1):**

| Condición | Umbral |
|---|---|
| Cobertura comunitaria | ≥ 60 % usuarios con ≥ 1 comunidad, en ≥ 1 dataset |
| Población frontera | B25 con `num_communities ≥ 2` y N ≥ 30 |
| Estabilidad (WP3 rápido) | ρ Spearman α-vecinos ≥ 0.7 y Jaccard(B10) ≥ 0.5 |

- **Gate NO pasa** en k=4 ni k=6 → artículo diagnóstico (señales de frontera no obtenibles de forma segura sobre redes NetInf dispersas).
- **Gate pasa** → HP/índices α del core (k=2) no aplican. Escalera reducida M2/M3/M4c: hypertune (50 enhanced / 200 social) → `recommend --all-networks` → `network_selection` → WP1+WP2 con constructo corregido. M1 (`canonical_baseline.json`) se reutiliza. Criterios del protocolo §4.5 / §5.5. Efectos → paper de método. WP4 (baselines externos, dataset grande tipo Epinions) queda **después** y fuera de esta fase.

**Salida C:** `route_b_gate_findings.md`; si se cruza, `route_b_dense_wp1_wp2_findings.md`.

## Orden y punto de decisión

1. **Si el paper es core+LPH: B1/B2 y C1–C3 primero** (MovieLens + Ciao). Ahí está el hilo de la fusión. A1 (10 semillas Ciao) en paralelo como seguro barato del resultado auxiliar; A3 se omite; A2 aparcado.
2. **Si además se quiere paper corto trust:** A1 (+ A3); réplica cross-dataset solo con un corpus Ciao-like, no Epinions.
3. **Sesión de decisión al cierre:** con B, el veredicto de compuerta C, y (opcional) A1 sobre la mesa → paper de método o de diagnóstico. El positivo trust **no** carga la tesis LPH: auxiliar o manuscript separado. Ninguna cifra titular sin su multi-semilla (10). Scope de esta fase: **Ciao + MovieLens**.

La regla de oro no cambia: los umbrales se fijan antes de mirar, todo se reporta, y los artefactos congelados de la campaña core no se tocan.
