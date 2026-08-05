# Protocolo Experimental — Ruta B

**Fusión MAFPIN + Homofilia Pluralista Local (LPH): plan para fortalecer el caso positivo**

- **Fecha:** 4 de agosto de 2026 — versión 1.0
- **Copia PDF entregada:** `protocolo_ruta_B_mafpin.pdf` (generada desde este documento)
- **Findings relacionados:** [core_experiment_movielens_findings.md](core_experiment_movielens_findings.md) · [core_experiment_ciao_findings.md](core_experiment_ciao_findings.md) · [cold_start_findings.md](cold_start_findings.md)
- **Comandos de ejecución:** [route_b_commands.md](route_b_commands.md)
- **Pre-registro:** [route_b_preregistration.md](route_b_preregistration.md)
- **Implementación:** branch `feat/route-b-experiments` (`recommender.experiment.route_b`, flags `--beyond-accuracy` / `--save-predictions`, variante cold-start `M3_soft`)

---

## 1. Propósito y alcance

Este documento define el **protocolo completo de experimentos de la Ruta B**: la vía para darle a la fusión *red inferida + CMF* (MAFPIN / ws-dmaa) con *homofilia pluralista local* ($\tilde{h}_v$, Appl. Sci.) su mejor oportunidad de producir un resultado positivo publicable en revista indexada, **sin re-tunear ni re-abrir la campaña core ya cerrada**.

El protocolo está escrito para quien mantiene el repositorio: cada experimento indica **qué implementar, dónde, cómo ejecutarlo, qué artefactos produce y qué debería observarse en los resultados para que la ruta sea viable** (criterios GO / NO-GO pre-registrados).

**Punto de partida (evidencia ya consolidada):**

| Hallazgo | Fuente |
|---|---|
| M3 gana en test RMSE: −2.4% (MovieLens), −0.42% (Ciao) vs M1 | `core_experiment_*_findings.md` |
| M3 vs M2 (valor de la capa LPH/comunidades): débil e inconsistente; H2 FAIL en Ciao | ídem |
| M4c (regularización guiada por frontera) nunca supera a M3 en test RMSE | ídem |
| H1-stronger (ganancia mayor en cold que en warm): FAIL en ambos datasets | `cold_start_findings.md` |
| CIs bootstrap cold M3 vs M1 en MovieLens cruzan 0 | ídem |
| Zero-shot trust: M2_trust +0.125 RMSE (CI limpio); M3_trust **peor** que M2_trust | ídem |
| M4c tiene el **mejor NDCG@10** en MovieLens pese a mal RMSE | `core_experiment_movielens_findings.md` §3.2 |

**Diagnóstico que motiva la Ruta B:** las hipótesis se evaluaron casi exclusivamente en *error de rating global* (RMSE/MAE). La teoría de la fusión predice efectos en (a) **exposición intercomunitaria y diversidad** del ranking, y (b) **subpoblaciones de usuarios frontera**, no necesariamente en el promedio global. Además, nunca se verificó la **precondición** de la que depende $\tilde{h}_v$ según el propio paper de Appl. Sci.: la calidad y estabilidad de las comunidades detectadas sobre redes NetInf.

La Ruta B se organiza en **5 paquetes de trabajo (WP1–WP5)** más un árbol de decisión final (WP6).

## 2. Reglas de higiene experimental (obligatorias en todos los WP)

1. **Pre-registro.** Antes de ejecutar cualquier corrida, copiar las hipótesis B1–B5 y los umbrales de la sección 3 a un archivo `docs/experiments/route_b_preregistration.md`, con fecha y hash del commit. Los umbrales **no se modifican después de ver resultados**.
2. **Congelamiento.** No se re-tunean hiperparámetros ni se re-seleccionan redes. Se reutilizan `experiment_manifest.json`, `canonical_baseline.json` y `network_selection_results.json` de la campaña core de cada dataset. Las métricas nuevas se calculan sobre **los mismos modelos congelados**.
3. **Trazabilidad.** Cada corrida usa `--log-file` propio bajo `data/<dataset>/logs/route_b/` y `--run-id` propio. MLflow sigue activo; etiquetar corridas con `route_b` y `wp<k>`.
4. **Estadística.** Toda afirmación comparativa se soporta con: (i) *deltas pareados per-user* con **CI bootstrap percentil al 95 % (1 000 remuestreos)** — reutilizar la implementación de `recommender/experiment/cold_start/evaluate.py` —, y (ii) **Wilcoxon de rangos con signo pareado** con corrección de **Holm** sobre la familia de comparaciones pre-registradas.
5. **Sin selección de resultados.** Todos los resultados (positivos, nulos, negativos) se registran en `docs/experiments/route_b_wp<k>_findings.md`, con veredicto explícito contra los criterios pre-registrados.
6. **Una trampa conocida a vigilar.** En MovieLens, M4a/M4b colapsan en ranking manteniendo RMSE decente. Una "mejora de diversidad" acompañada de colapso de accuracy **no cuenta como éxito** (ver umbral de guardia en WP1).

## 3. Hipótesis pre-registradas de la Ruta B

| ID | Enunciado | WP que la evalúa |
|---|---|---|
| **B1** | Las variantes con señal de frontera (M3, M4c, M4d) aumentan la exposición intercomunitaria (CCE@10), la diversidad (ILD@10) y/o la cobertura de catálogo vs M1/M2, **sin degradar RMSE más de 0.5 % relativo** ni colapsar NDCG@10. | WP1 |
| **B2** | La ganancia de M3 sobre M2 (y de M4 sobre M3) se **concentra en usuarios frontera** (percentiles bajos de $\tilde{h}_v$) y/o en ratings de ítems fuera de la comunidad del usuario, aunque el promedio global sea nulo. | WP2 |
| **B3** | $\tilde{h}_v$ y el indicador de frontera son **estables** frente a la elección de red (índices $\alpha$ vecinos, modelo de difusión) y al detector de comunidades. *Precondición de interpretabilidad de B1/B2.* | WP3 |
| **B4** | Los efectos B1/B2 **se replican** en ≥ 2 de 3 datasets, sobreviven a multi-semilla, y las variantes MAFPIN+LPH son competitivas frente a baselines sociales externos (SoRec, SocialMF, TrustSVD). | WP4 |
| **B5** *(condicional)* | Una extensión **dirigida/ponderada** de $\tilde{h}_v$ recupera señal que la simetrización de la red NetInf destruye. | WP5 |

## 4. WP1 — Métricas *beyond-accuracy* sobre los modelos congelados

**Racional.** Es el experimento más barato y prometedor: la hipótesis 3 del documento de fusión promete "recomendación menos miope", no menor RMSE global. El NDCG@10 de M4c en MovieLens sugiere que la señal de frontera actúa sobre el *ranking*. El plan original proponía *cross-community exposure* y nunca se midió.

### 4.1. Qué implementar

Nueva función `compute_beyond_accuracy_metrics(...)` en `recommender/data.py`, junto a la función de ranking existente (misma convención de candidatos y top-K, K=10), que devuelva **valores per-user** (no solo promedios) para poder hacer bootstrap. Definiciones exactas (Apéndice A):

- **ItemCoverage@K** — fracción del catálogo candidato que aparece en algún top-K: $\mathrm{Cov@}K = |\bigcup_u \mathrm{top}K(u)| / |\mathcal{I}|$ (métrica global, sin per-user).
- **Gini@K** — concentración de la distribución de frecuencias con que cada ítem es recomendado (global).
- **ILD@K** (*intra-list diversity*, per-user) — con **dos vectorizaciones**:
  1. *Primaria (comparable entre modelos):* vectores de contenido fijos — géneros en MovieLens, categorías en Ciao/Epinions.
  2. *Diagnóstica:* factores latentes $\mathbf{q}_i$ del propio modelo (no comparable entre modelos; solo para inspección).
- **Novelty@K** (per-user) — $-\log_2$ de la popularidad relativa en train de los ítems recomendados.
- **CCE@K** (*cross-community exposure*, per-user) — **la métrica central del WP**, definida en §4.2.

### 4.2. CCE@K: definición formal propuesta

Para que la métrica sea **idéntica y comparable para todas las variantes**, la estructura comunitaria se toma de **una única fuente congelada**: la red seleccionada para **M3** en `network_selection_results.json` del dataset (comunidades DEMON ya calculadas por `networks/communities/batch.py`). La partición **no depende del modelo evaluado**.

1. $C(u)$ = comunidades del usuario $u$ en esa red (usuarios sin comunidad se excluyen del promedio y se reporta su $N$).
2. **Audiencia del ítem** $i$: multiconjunto de comunidades de los usuarios que calificaron $i$ en **train**. Comunidad dominante $D(i)$ = comunidad(es) moda de esa audiencia.
3. $$\mathrm{CCE@}K(u) = \frac{1}{K}\sum_{i \in \mathrm{top}K(u)} \mathbf{1}\left[D(i) \cap C(u) = \emptyset\right]$$
4. Reportar la media global y, cuando WP2 esté disponible, por estrato de $\tilde{h}_v$.

**Puntos de integración en el código:**

| Cambio | Archivo |
|---|---|
| Cálculo de las métricas (per-user + globales) | `recommender/data.py` |
| Columnas nuevas en el CSV de resultados | `recommender/experiment/final_eval.py` (donde hoy se escriben `ndcg_at_10`, etc.) |
| Flag opcional `--beyond-accuracy` | `pipeline/steps/final_eval.py` + `pipeline/_cli.py` |
| Test unitario con fixture pequeña y valores calculados a mano | `tests/test_beyond_accuracy.py` (nuevo, usando `tests/fixtures.py`) |

### 4.3. Ejecución

Con hiperparámetros y redes **ya congelados** (no correr `import_manifest`/`network_selection` con `--force`):

```bash
# MovieLens — re-evaluación final con métricas beyond-accuracy
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps final_eval --all-variants --beyond-accuracy \
  --dataset movielens --seed 42 \
  --log-file data/movielens/logs/route_b/wp1_final_eval.log

# Ciao — ídem
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps final_eval --all-variants --beyond-accuracy \
  --dataset ciao --seed 42 \
  --log-file data/ciao/logs/route_b/wp1_final_eval.log
```

*(Si se prefiere no tocar la firma del step, alternativa válida: script dedicado `scripts/route_b_wp1_beyond_accuracy.py` que cargue los modelos con los parámetros del manifest y evalúe. Lo importante es que **la fuente de hiperparámetros/red sea el manifest congelado**.)*

### 4.4. Artefactos esperados

| Artefacto | Contenido |
|---|---|
| `data/<ds>/route_b/beyond_accuracy_results.csv` | 1 fila por variante × métrica (globales + media per-user) |
| `data/<ds>/route_b/beyond_accuracy_per_user.parquet` | valores per-user (insumo del bootstrap y de WP2) |
| `data/<ds>/route_b/beyond_accuracy_bootstrap.csv` | CIs 95 % de deltas pareados per-user: M3−M2, M4c−M3, M4d−M3, M3−M1 |
| `docs/experiments/route_b_wp1_findings.md` | tablas + veredicto contra criterios |

### 4.5. Qué debería verse para que la ruta sea viable (criterios WP1)

- **GO fuerte:** M3, M4c o M4d mejoran **CCE@10** y al menos una de {ILD@10 primaria, Coverage@10} vs **M2** con CI per-user que **no cruza 0** en ≥ 1 dataset, cumpliendo la **condición de guardia**: ΔRMSE relativo ≤ +0.5 % y NDCG@10 ≥ 0.8 × NDCG@10(M1).
- **GO débil:** mejoras direccionalmente consistentes en **ambos** datasets sin CI limpio → continuar, la potencia estadística se busca en WP4.
- **NO-GO WP1:** sin patrón direccional consistente (o mejoras solo con colapso de accuracy) → la narrativa *beyond-accuracy* se abandona; la fusión se juega su viabilidad en WP2.

## 5. WP2 — Efectos heterogéneos: estratificación por frontera

**Racional.** La hipótesis central de la fusión (H3 del documento original) habla de *usuarios híbridos/frontera*. Nunca se evaluó si **esos usuarios en particular** mejoran. Un efecto real del 3–5 % en el 10–20 % de usuarios frontera es invisible en el promedio global — y sería un resultado publicable con narrativa limpia.

### 5.1. Diseño

- **Variable de estratificación:** $\tilde{h}_v$ del CSV de comunidades de la **red congelada de M3** (misma fuente que CCE@K). $\tilde{h}_v$ se computa de la red inferida sobre **train**, por lo que es *pre-tratamiento*: no hay fuga de test y es idéntica para todas las variantes.
- **Estratos pre-registrados** (percentiles entre usuarios con features): **B10** ($\tilde{h}_v \leq P10$, frontera dura), **B25** ($\leq P25$), **MID** (P25–P75), **E75** ($\geq P75$, incrustados). Adicional: el `boundary_flag` binario existente (`networks/communities/boundary.py::compute_boundary_indicator`).
- **Métricas:** RMSE per-user en test; deltas pareados per-user Δ(M3−M2), Δ(M4c−M3), Δ(M4d−M3), Δ(M3−M1) por estrato, con CI bootstrap 95 % y Wilcoxon+Holm.
- **Análisis complementario por ítems:** RMSE restringido a ratings de test cuyos ítems son *cross-community* para el usuario ($D(i) \cap C(u) = \emptyset$, definición de §4.2). Pregunta: ¿la fusión predice mejor el consumo **fuera** de la comunidad propia?

### 5.2. Qué implementar

1. **Persistencia de predicciones** — el cambio de código más importante de la Ruta B: flag `--save-predictions` en `final_eval` que escriba las predicciones per-rating de cada variante en `data/<ds>/route_b/predictions/<variant>.parquet` (columnas: `UserId, ItemId, Rating, Prediction, variant`). Sin esto, cada análisis exige re-entrenar.
2. **Módulo de análisis** `recommender/experiment/boundary_strata.py` (nuevo): carga predicciones + CSV de comunidades/LPH, construye estratos, calcula deltas pareados, bootstrap y Wilcoxon. Se ejecuta como `python -m recommender.experiment.boundary_strata --dataset <ds>`.
3. Test en `tests/` con fixture: estratos con N conocido y deltas verificados a mano.

### 5.3. Ejecución

```bash
# 1) Regenerar final_eval guardando predicciones (redes/HP congelados)
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps final_eval --all-variants --save-predictions \
  --dataset movielens --seed 42 \
  --log-file data/movielens/logs/route_b/wp2_final_eval_preds.log
# (repetir con --dataset ciao)

# 2) Análisis de estratos
conda run --no-capture-output -n mafpin python -m recommender.experiment.boundary_strata \
  --dataset movielens --bootstrap-samples 1000 --seed 42
# (repetir con --dataset ciao)
```

### 5.4. Artefactos esperados

| Artefacto | Contenido |
|---|---|
| `data/<ds>/route_b/predictions/<variant>.parquet` | predicciones per-rating congeladas |
| `data/<ds>/route_b/boundary_strata_results.csv` | RMSE por variante × estrato (N usuarios y N ratings por celda) |
| `data/<ds>/route_b/boundary_strata_bootstrap.csv` | CIs de deltas pareados por estrato |
| `data/<ds>/route_b/cross_community_items_results.csv` | RMSE en subconjunto cross-community |
| `docs/experiments/route_b_wp2_findings.md` | tablas + veredicto |

### 5.5. Qué debería verse para que la ruta sea viable (criterios WP2)

- **GO:** Δ(M3−M2) > 0 per-user con CI limpio en **B10 o B25** en ≥ 1 dataset, **con gradiente monótono** (efecto B10 ≥ B25 ≥ MID). Lo mismo aplicado a Δ(M4c/d−M3) cuenta como GO para la regularización guiada.
- **GO parcial:** efecto con CI limpio solo en el subconjunto de **ítems cross-community** (aunque los estratos de usuario sean nulos) → narrativa "la fusión ayuda a predecir consumo intercomunitario".
- **Requisito de reporte:** cada celda debe tener N ≥ 30 usuarios; si B10 queda por debajo, fusionar con B25 y declararlo.
- **NO-GO WP2:** sin gradiente por estrato ni efecto en ítems cross-community en ningún dataset → **H3 queda refutada en estos datos**; la única vía restante para la fusión es WP5 (¿la simetrización destruyó la señal?) — o pivotar a Ruta A.

## 6. WP3 — Diagnóstico de calidad y estabilidad de comunidades

**Racional.** El paper de Appl. Sci. advierte que $\tilde{h}_v$ **depende de la calidad de las comunidades**. Nadie verificó que las comunidades DEMON sobre redes NetInf sean estables. Si no lo son, los nulos de la campaña core quedan *explicados* y cualquier resultado de WP1/WP2 sería ruido. Es barato y debe correr **en paralelo** con WP1.

### 6.1. Experimentos

- **E3.1 — Calidad estática.** Para la red seleccionada de cada variante y sus vecinas de α (±2 índices, mismo modelo de difusión): número de comunidades, distribución de tamaños, fracción de nodos cubiertos, membresía media m(v), conductancia media y modularidad con traslape.
- **E3.2 — Estabilidad de $\tilde{h}_v$.** Correlación de **Spearman** de $\tilde{h}_v$ entre la red seleccionada y cada red vecina (α adyacente, mismo modelo), y entre modelos de difusión con α equivalente. Estabilidad del conjunto frontera: **Jaccard de los conjuntos B10** entre configuraciones.
- **E3.3 — Sensibilidad al detector.** Recalcular comunidades y $\tilde{h}_v$ con 2 detectores overlap-aware adicionales disponibles en CDlib (recomendados: **SLPA** y **Ego-Splitting**) sobre la red congelada de M3; Spearman de $\tilde{h}_v$ y Jaccard de B10 entre detectores.

### 6.2. Qué implementar

Script autónomo `scripts/route_b_wp3_community_stability.py` que reutilice `networks/communities/detection.py`, `lph.py` y `boundary.py`; no toca el pipeline. Dependencia nueva probable: `cdlib` (añadir a `requirements.txt` con versión fijada).

```bash
conda run --no-capture-output -n mafpin python scripts/route_b_wp3_community_stability.py \
  --dataset movielens --neighbors 2 --detectors demon slpa egosplit --seed 42
# (repetir con --dataset ciao)
```

### 6.3. Artefactos esperados

`data/<ds>/route_b/community_stability.csv` (una fila por par de configuraciones comparadas: ρ Spearman, Jaccard B10, métricas de calidad) y `docs/experiments/route_b_wp3_findings.md`.

### 6.4. Qué debería verse (criterios WP3 — precondición, no GO/NO-GO de publicación)

| Resultado | Lectura | Consecuencia |
|---|---|---|
| ρ ≥ 0.7 intra-modelo (α vecino) **y** Jaccard(B10) ≥ 0.5 | señal interpretable | B1/B2 se leen con confianza |
| 0.4 ≤ ρ < 0.7 | zona gris | B1/B2 se reportan con la inestabilidad como limitación declarada |
| ρ < 0.4 | $\tilde{h}_v$ sobre redes NetInf es esencialmente ruido | **explica los nulos de la campaña core**; STOP a la narrativa actual; activa WP5 (¿la direccionalidad estabiliza?) o pivote: paper crítico *"community-boundary signals on inferred networks are unstable"* (publicable en Applied Network Science / PLOS ONE) |

## 7. WP4 — Validez externa: baselines, tercer dataset, multi-semilla

**Racional.** Ninguna revista indexada de recomendación aceptará el paper sin baselines sociales externos, y los efectos pequeños de WP1/WP2 necesitan potencia estadística (más datasets, más semillas). **Ejecutar solo si WP1 o WP2 dieron GO (fuerte o débil).** Es el WP más costoso.

### 7.1. E4.1 — Baselines sociales externos

- **Modelos mínimos:** SoRec, SocialMF, TrustSVD (usan el grafo trust explícito de Ciao/Epinions) + el MF puro de la librería como *sanity check* contra M1.
- **Librería recomendada:** Cornac (o LibRecommender). Fijar versión en `requirements.txt`.
- **Regla de oro:** deben evaluar **exactamente el mismo split temporal 80/20 y semilla** que la campaña core. Implementar utilidad de exportación del split a CSV (`data/<ds>/route_b/split_export/{train,test}.csv`) si no existe, y alimentar las librerías desde esos CSV.
- Búsqueda de hiperparámetros de los baselines: presupuesto Optuna comparable al de M2/M3 (50 trials) sobre CV del train — **la misma cortesía que recibieron nuestras variantes, ni más ni menos**.
- Comparar en: RMSE global, RMSE por estratos cold (reutilizar `strata.py`), y estratos de frontera de WP2.

### 7.2. E4.2 — Tercer dataset: Epinions

Ya está cableado en `config.py` (`Datasets.ALL`) y tiene trust explícito. Ejecutar la misma escalera que Ciao:

```bash
# Prerrequisitos
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps cascade delta inference centrality communities \
  --dataset epinions --log-file data/epinions/logs/00_prerequisites.log

# Escalera completa (Etapa A + B + Fase 2), reutilizar el runner batch existente
./scripts/run_core_experiment.sh --dataset epinions

# Cold start controlado + zero-shot trust
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset epinions --mode controlled --seed 42
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset epinions --mode zero_shot_trust --variants M1 M2_trust M3_trust --seed 42
```

**Advertencia de cómputo (decisión pre-registrada, no post-hoc):** la escalera de Ciao tomó ≈ 31 h secuencial; Epinions es sustancialmente mayor. Si el presupuesto lo exige, submuestrear **temporalmente** (ventana contigua de ratings) hasta un tamaño comparable a Ciao, documentando la ventana elegida en el pre-registro **antes** de ver resultados.

### 7.3. E4.3 — Multi-semilla y tests de significancia

- **Semillas:** {42, 7, 123, 2024, 31337}, para M1, M2, M3 y el mejor M4 de cada dataset.
- **Qué varía:** solo la inicialización de CMF. **El split y las redes permanecen fijos** (aislar la varianza del optimizador; el no-determinismo de L-BFGS ya observado — M1 1.0412 vs 1.0510 — lo exige).
- **Análisis:** media ± desviación por variante; Wilcoxon pareado per-user por semilla; conclusión solo si el signo del efecto es estable en ≥ 4/5 semillas. Corrección de Holm sobre la familia pre-registrada: {M3−M2, M4x−M3, M3−M1} × {global, B10, B25} × {datasets}.

### 7.4. Artefactos esperados

`data/<ds>/route_b/external_baselines_results.csv`, `data/epinions/core_experiment_results.csv`, `data/<ds>/route_b/multiseed_results.csv`, y `docs/experiments/route_b_wp4_findings.md`.

### 7.5. Qué debería verse para que la ruta sea viable (criterios WP4)

- **GO:** (i) los efectos GO de WP1/WP2 conservan signo y CI en ≥ 2 de 3 datasets bajo multi-semilla; (ii) M3/M4 son **competitivos** (dentro del 1 % relativo de RMSE) frente a SocialMF/TrustSVD en global; y (iii) **ganan con CI limpio en el nicho pre-registrado** (estratos frontera y/o CCE@10). No es necesario ganar en todo: el claim del paper es el nicho.
- **NO-GO:** los baselines externos dominan también en los nichos → el paper de método no es defendible; pivotar a Ruta A (paper empírico honesto) o Ruta C (separar el resultado trust).

## 8. WP5 — (Condicional) Extensión dirigida/ponderada de $\tilde{h}_v$

**Disparadores:** (a) WP3 muestra inestabilidad severa o (b) WP1/WP2 dan GO débil y se necesita más señal, o (c) todo lo demás es nulo y se decide que la contribución sea metodológica. La sospecha es concreta: NetInf infiere una red **dirigida y ponderada** (likelihood por arista) y la simetrización actual puede estar destruyendo exactamente la señal direccional de frontera.

### 8.1. Diseño

1. **Formulación.** Extender el Algoritmo 1 de Appl. Sci.: calcular s(v) y δ_v separadamente sobre vecindario de entrada y de salida ($\tilde{h}_v^{in}$, $\tilde{h}_v^{out}$), y una variante ponderada donde la contribución de cada vecino se pesa por el peso NetInf de la arista. Documentar la formulación en `docs/lph.md`.
2. **Validación en sintético primero (obligatoria).** Antes de tocar MAFPIN: benchmarks LFR dirigidos con comunidades plantadas y *boundary nodes* conocidos. La extensión solo es contribución si demuestra recuperar fronteras plantadas mejor que la versión simetrizada. Script: `scripts/route_b_wp5_lfr_validation.py`.
3. **Aplicación.** Añadir $\tilde{h}_v^{in/out}$ y variantes ponderadas como features de una variante nueva **M3-dir** (extensión de `recommender/enhanced/features.py`), y repetir WP1 + WP2 comparando M3-dir vs M3.

### 8.2. Qué debería verse (criterios WP5)

- **GO método+aplicación:** la versión dirigida recupera fronteras plantadas en LFR (AUC de detección claramente superior a la simetrizada) **y** M3-dir mejora a M3 en los nichos de WP1/WP2.
- **GO solo-método:** la validación sintética es sólida pero MAFPIN no mejora → publicar la extensión como paper de análisis de redes (venue tipo Applied Network Science, Journal of Complex Networks), con MAFPIN como caso de aplicación con resultados mixtos honestos.
- **NO-GO:** la versión dirigida no supera a la simetrizada ni en sintético → cerrar la línea; la simetrización no era el problema.

## 9. WP6 — Árbol de decisión global y mapeo a publicación

| Combinación de veredictos | Paper resultante | Venue orientativo |
|---|---|---|
| B1 GO + B2 GO (+ B3 interpretable + B4 sobrevive) | Método: *boundary-aware CMF sobre redes inferidas*, claims en beyond-accuracy + segmentos frontera | ESWA, Information Processing & Management, ACM TORS, UMUAI |
| Solo B2 GO (+ B4) | Efectos heterogéneos de side-info social: *a quién ayuda la señal de frontera* | ACM TORS, RecSys (LBR primero como sonda) |
| Solo B1 GO (+ B4) | Exposición intercomunitaria/diversidad vía señales de frontera | RecSys, IP&M, Applied Sciences |
| B3 NO-GO (ρ < 0.4) | Estudio crítico: inestabilidad de señales comunitarias sobre redes inferidas + explicación de nulos | Applied Network Science, PLOS ONE |
| B5 GO solo-método | Extensión dirigida/ponderada de LPH validada en sintético | Applied Network Science, J. Complex Networks |
| Todo NO-GO | **Ruta A**: paper empírico honesto (escalera completa, nulos documentados, resultado fuerte de trust zero-shot como protagonista) | PLOS ONE, IEEE Access, Applied Sciences, track de reproducibilidad de RecSys |

**En cualquier rama**, el resultado zero-shot trust (M2_trust +0.125 RMSE, CI [0.140, 0.150]) es publicable y no debe desperdiciarse: entra como sección del paper principal o como paper corto separado (Ruta C).

## 10. Orden de ejecución y dependencias

```
        ┌────────────┐
        │ Pre-registro│  docs/experiments/route_b_preregistration.md
        └──────┬─────┘
     ┌─────────┴──────────┐
     ▼                    ▼
  WP3 (barato,         WP1 implementación
  paralelo)            └─> WP1 ejecución ──> veredicto B1
     │                    │
     │                    ▼
     │                 WP2 (--save-predictions primero) ──> veredicto B2
     │                    │
     ▼                    ▼
  veredicto B3 ────> ¿algún GO en {B1,B2}? ──sí──> WP4 (baselines, Epinions, semillas)
                          │                            │
                          no                           ▼
                          │                        veredicto B4 ──> WP6: paper de método
                          ▼
                     WP5 condicional / Ruta A
```

**Costo relativo:** WP3 bajo; WP1 bajo-medio (re-evaluación con modelos congelados); WP2 medio (requiere `--save-predictions` + análisis); WP4 alto (Epinions ≫ Ciao ≈ 31 h de escalera; baselines externos + 5 semillas); WP5 medio.

**Punto de decisión intermedio obligatorio:** tras WP1+WP2+WP3, sesión de revisión contra este protocolo antes de comprometer el cómputo de WP4.

## 11. Registro de resultados

1. Cada WP produce `docs/experiments/route_b_wp<k>_findings.md` con: fecha, commit, comandos exactos, tablas, y **veredicto explícito** (GO fuerte / GO débil / GO parcial / NO-GO) citando los umbrales de este protocolo.
2. Artefactos tabulares bajo `data/<dataset>/route_b/` (nunca sobrescribir artefactos de la campaña core).
3. MLflow: tags `route_b=true`, `wp=<k>`.
4. Al cierre: `docs/experiments/route_b_decision.md` con la rama del árbol de WP6 elegida y su justificación.

## Apéndice A — Definiciones matemáticas exactas

**Notación.** $\mathcal{U}$ usuarios de test, $\mathcal{I}$ ítems candidatos, $\mathrm{top}K(u)$ los K=10 ítems mejor rankeados para u (misma construcción de candidatos que la función de ranking existente en `recommender/data.py`), $\mathrm{pop}(i)$ = fracción de usuarios de train que calificaron i.

**ItemCoverage@K.**

$$\mathrm{Cov@}K = \frac{\left|\bigcup_{u \in \mathcal{U}} \mathrm{top}K(u)\right|}{|\mathcal{I}|}$$

**Gini@K.** Sea $f_i$ la frecuencia con que i aparece en los top-K, ordenadas ascendentemente ($f_{(1)} \leq \dots \leq f_{(n)}$, $n = |\mathcal{I}|$):

$$\mathrm{Gini@}K = \frac{\sum_{j=1}^{n} (2j - n - 1) f_{(j)}}{n \sum_j f_{(j)}}$$

**ILD@K** (per-user, con vectores $\mathbf{x}_i$ de contenido — primaria — o latentes — diagnóstica):

$$\mathrm{ILD@}K(u) = \frac{2}{K(K-1)} \sum_{\substack{i,j \in \mathrm{top}K(u) \\ i < j}} \left(1 - \frac{\mathbf{x}_i^\top \mathbf{x}_j}{\|\mathbf{x}_i\|\,\|\mathbf{x}_j\|}\right)$$

**Novelty@K** (per-user):

$$\mathrm{Nov@}K(u) = -\frac{1}{K} \sum_{i \in \mathrm{top}K(u)} \log_2 \mathrm{pop}(i)$$

**CCE@K** (per-user): con C(u) y D(i) definidos en §4.2,

$$\mathrm{CCE@}K(u) = \frac{1}{K}\sum_{i \in \mathrm{top}K(u)} \mathbf{1}\left[D(i) \cap C(u) = \emptyset\right]$$

**Delta pareado per-user y CI bootstrap.** Para variantes a, b y usuario u: $\Delta_u = \mathrm{RMSE}_u(a) - \mathrm{RMSE}_u(b)$ (positivo ⇒ b mejor). CI: bootstrap percentil 95 % sobre 1 000 remuestreos de usuarios (misma rutina que `cold_start/evaluate.py`).

**Estratos de frontera.** Percentiles de $\tilde{h}_v$ calculados **solo entre usuarios presentes en la red congelada de M3**: B10 $= \{u : \tilde{h}_u \leq P_{10}\}$, B25, MID $= (P_{25}, P_{75})$, E75 $= \{u : \tilde{h}_u \geq P_{75}\}$.

## Apéndice B — Checklist de pre-registro (copiar a `route_b_preregistration.md`)

- [ ] Fecha, autor y hash del commit del código con el que se ejecutará.
- [ ] Hipótesis B1–B5 copiadas literalmente con sus umbrales.
- [ ] Confirmación: manifests y redes congeladas de la campaña core sin regenerar (`experiment_manifest.json`, `network_selection_results.json`, `canonical_baseline.json` — fechas verificadas).
- [ ] Familia de comparaciones para Holm enumerada.
- [ ] Semillas fijadas: 42 (principal); multi-semilla {42, 7, 123, 2024, 31337}.
- [ ] Decisión de submuestreo de Epinions (si aplica) con ventana temporal declarada.
- [ ] Umbral de guardia de accuracy (RMSE ≤ +0.5 % rel.; NDCG@10 ≥ 0.8× M1) confirmado.
- [ ] Compromiso: todos los resultados se publican en los findings docs, incluidos nulos.

## Apéndice C — Resumen de cambios de código requeridos

| # | Cambio | Archivos | WP |
|---|---|---|---|
| 1 | `compute_beyond_accuracy_metrics` (per-user) + CCE@K | `recommender/data.py` | WP1 |
| 2 | Columnas beyond-accuracy en resultados + flag CLI | `recommender/experiment/final_eval.py`, `pipeline/steps/final_eval.py`, `pipeline/_cli.py` | WP1 |
| 3 | Flag `--save-predictions` (parquet per-rating) | `recommender/experiment/final_eval.py` | WP2 |
| 4 | Módulo `boundary_strata.py` (estratos, bootstrap, Wilcoxon) | `recommender/experiment/` | WP2 |
| 5 | Script de estabilidad de comunidades (+ dep. `cdlib`) | `scripts/`, `requirements.txt` | WP3 |
| 6 | Exportación de split a CSV + runner de baselines externos (+ dep. `cornac`) | `recommender/data.py`, `scripts/`, `requirements.txt` | WP4 |
| 7 | LPH dirigida/ponderada + validación LFR | `networks/communities/lph.py`, `scripts/` | WP5 |
| 8 | Tests unitarios de 1, 3, 4 y 7 | `tests/` | todos |

---

*Fin del protocolo. Cualquier desviación durante la ejecución debe documentarse en el findings doc del WP correspondiente con su justificación.*
