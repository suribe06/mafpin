# Auditoría de validez: LPH y redes densificadas

**Fecha:** 2026-10-25  
**Alcance:** investigación 2 (LPH, CMF social, core experiment y Route B), con investigación 1 solo como referencia de comparabilidad.  
**Método:** auditoría estática de documentación, código, scripts, comandos,
artefactos registrados y los dos papers primarios guardados en `articles/`.
No se ejecutaron comandos, tests ni experimentos.

**Actualización 2026-10-26:** se contrastaron las Ecuaciones (3)–(6) del paper
LPH y el repositorio oficial de reproducibilidad con la implementación local.
Se corrigieron `networks/communities/lph.py`, `docs/lph.md` y la formulación del
manuscrito; se añadió un test de regresión (3/3 OK).

**Pre-validación 2026-10-26 (minutos, sin reentrenar búsquedas):** se ejecutaron
el diagnóstico temporal C3, el bootstrap pareado sobre las predicciones
guardadas de ambos datasets, un refit congelado con semilla 42 (reproduce
exactamente el RMSE del core) y la ablación LPH con una semilla. Los
resultados se incorporan en C3, I2, I6 e I12. Todo lo pendiente quedó en
`scripts/run_validation_plan.sh` (ver «Plan de ejecución»).

## Veredicto ejecutivo

La decisión de **continuar el siguiente experimento con MovieLens densificado, preferentemente `k=4`, está razonablemente fundamentada como decisión operativa**: el gate pre-registrado de cobertura pasa en MovieLens (`75.0%` con `k=4`) y la estabilidad local también pasa (`ρ≈0.863`, Jaccard B10≈`0.554`). Ciao no cruza el umbral de cobertura ni replica el patrón beyond-accuracy de MovieLens. La evidencia está en `data/route_b_dense_gate_summary_20261008_132748.csv` y `docs/experiments/findings/route_b/route_b_next_steps_findings.md:20-29,90-116`.

La decisión **no está fundamentada si se formula como “MovieLens densificado ya mejora el recomendador”**. Solo se midió si el grafo denso produce comunidades suficientemente cubiertas y estables. No se ejecutaron el retuning, la nueva selección de red ni WP1/WP2 densos que el propio plan exige después de pasar el gate (`route_b_positives_and_refinement_plan.md:119-128`; `route_b_next_steps_findings.md:120-129`; `run_route_b_next_steps.sh:312-335`).

Sobre Ciao, la decisión está **mejor fundamentada de lo que parecía, pero por otra razón**. Bajo el protocolo per-user, M4c y M4d mejoran el RMSE global frente a M3 y M1, y lo hacen en **5 de 5 semillas** (≈−0.003, sd≈0.001). Pero el bootstrap pareado por usuario muestra que **ninguna diferencia se distingue de cero** (todos los CI 95% cruzan 0). Es decir, el efecto es estable frente a la inicialización pero muy pequeño frente a la heterogeneidad entre usuarios. Ciao no es un fracaso de implementación: es un **nulo** para la fusión LPH/BA y un efecto social **no generalizable entre usuarios** en accuracy (`core_experiment_ciao_findings.md:28-44,90-116`; `route_b_next_steps_findings.md:40-69`).

El contraste con el paper confirmó un problema metodológico: la implementación
del coeficiente global `h` no era invariante a la orientación arbitraria de las
aristas no dirigidas. **El código ya fue corregido y validado.** En MovieLens,
regenerar las comunidades de las redes seleccionadas y repetir `final_eval` con
HP congelados cambió el RMSE en menos de `1e-6` para M1, M3, M4c y M4d
(`data/movielens/validation/lph_h_fix_20261026_002518/`). Los resultados
históricos de MovieLens siguen siendo válidos respecto de este bug.

## Resultados de `quick` (2026-10-26, `data/validation_runs/20261026_005133/`)

Refit congelado (HP y red del manifest, sin retuning) con semillas
{42, 7, 123, 2024, 31337}. Δ = primera − segunda variante; negativo = primera mejor.

| Comparación | MovieLens Δ medio ± sd | Semillas a favor | Ciao Δ medio ± sd | Semillas a favor |
| --- | ---: | :---: | ---: | :---: |
| M2−M1 | **−0.0144 ± 0.0011** | 5/5 | — | — |
| M3−M1 | −0.0129 ± 0.0033 | 5/5 | +0.0011 ± 0.0008 | 0/5 |
| M3−M2 | +0.0015 ± 0.0034 | 2/5 | — | — |
| M4c−M3 | −0.0043 ± 0.0041 | 4/5 | −0.0030 ± 0.0011 | 5/5 |
| M4d−M3 | −0.0036 ± 0.0035 | 4/5 | −0.0033 ± 0.0012 | 5/5 |
| M4c−M1 | **−0.0172 ± 0.0035** | 5/5 | −0.0019 ± 0.0007 | 5/5 |
| M4d−M1 | **−0.0165 ± 0.0030** | 5/5 | −0.0022 ± 0.0009 | 5/5 |

**Qué cambia respecto de lo que afirmé antes:**

1. **M4c sobre M3 en MovieLens no es robusto.** El bootstrap por usuario
   (semilla 42) lo daba significativo, pero entre semillas gana solo 4/5 y
   |Δ| ≈ 1 sd. No pasa el criterio pre-fijado (5/5 y |Δ| > 2 sd). La semilla 42
   es una de las más favorables, y los HP se tunearon con esa semilla (riesgo
   de *winner's curse*).
2. **Lo robusto en MovieLens es «red inferida > CMF plano»:** M2, M4c y M4d
   ganan a M1 en 5/5 semillas con |Δ| ≥ 4 sd. Es la tesis de la
   investigación 1, no la de LPH.
3. **LPH no aporta sobre centralidad en ningún dataset:** M3−M2 en MovieLens es
   +0.0015 (peor, 2/5).

**Ablación LPH (3 semillas, red e HP de M3 congelados):**

| Conjunto | MovieLens RMSE medio (sd) | Δ vs centralidad | Ciao Δ vs centralidad |
| --- | ---: | ---: | ---: |
| centralidad pura | 0.91128 (0.0026) | — | — |
| + `pagerank_lph` (= M2) | 0.91106 (0.0035) | −0.0002 | −0.00001 |
| + \(\tilde h_v\) | 0.91162 (0.0021) | **+0.0003** | −0.00001 |
| + Jaccard-LPH | 0.90990 (0.0009) | −0.0014 | −0.00000 |
| + `s_v` | 0.90917 (0.0030) | −0.0021 | −0.00001 |
| M3 completo | 0.90865 (0.0006) | −0.0026 | −0.00003 |

- **\(\tilde h_v\), la métrica del paper, no aporta en ningún dataset** (confirmado con 3 semillas).
- Las demás diferencias en MovieLens (≤0.0026) están dentro de la variación
  entre semillas (sd 0.002–0.003); ninguna es concluyente.
- En Ciao todas las columnas dan lo mismo hasta 1e−5: con `w_user`≈0.04, el
  modelo prácticamente ignora la matriz lateral.

**Decisión antes de `dense`: no cambiar el conjunto de features.** Ningún
subconjunto mejora claramente a M3 completo, que además es el más estable
entre semillas. `dense` es precisamente la prueba que falta: si LPH no aporta
porque en `k=2` la mayoría de usuarios no tiene comunidad, con cobertura 75%
debería empezar a aportar. Si tampoco aporta con `k=4`, la conclusión es que
LPH no ayuda en este pipeline.

## Resultados de `dense` k=4 (2026-10-27, `data/validation_runs/20261027_110539/`)

Escalera M2/M3/M4c re-tuneada sobre las 300 redes densificadas, con red
re-seleccionada por CV (M2: exponential α33; M3: rayleigh α74; M4c: rayleigh
α22) y refit congelado con 5 semillas. La restauración de `k=2` es exacta
(`diff -r` vacío contra el snapshot).

**Dentro de la corrida densa (5 semillas):**

| Comparación | Δ medio ± sd | Semillas a favor | Bootstrap por usuario (semilla 42) |
| --- | ---: | :---: | --- |
| M3−M1 | −0.0168 ± 0.0035 | 5/5 | CI cruza 0 |
| M2−M1 | −0.0133 ± 0.0044 | 5/5 | CI cruza 0 |
| **M3−M2 (aporte LPH)** | −0.0035 ± 0.0035 | 4/5 | +0.0002, CI cruza 0 |
| **M4c−M3 (aporte social)** | **+0.0013** ± 0.0031 | 1/5 | CI cruza 0 |
| M4c−M1 | −0.0155 ± 0.0031 | 5/5 | excluye 0 (Holm p=0.005) |

**Denso vs `k=2` (medias de 5 semillas):** M3 mejora −0.0039 (≈2 sd); M2
empeora +0.0011; M4c empeora +0.0017.

**Beyond-accuracy y frontera (WP1/WP2 densos):**

- NDCG@10 cae en todas las variantes con red (M2 0.031, M3 0.036, M4c 0.022
  frente a M1 0.049): todas fallan la guardia pre-registrada (≥0.8×M1).
- M3 **baja** el CCE frente a M2 (−0.023, Holm p=1e−4), al revés que en `k=2`.
  La cobertura de M4c cae a la mitad.
- Estratos poblados (B10=65, B25=81), pero M3−M2 no mejora en la frontera. La
  única ganancia significativa de M3 sobre M2 está en **E75**, los usuarios
  *más integrados* (+0.0062, CI excluye 0): lo opuesto a la hipótesis B2.

**Veredicto (criterio pre-fijado: 5/5 semillas, |Δ| > 2 sd y CI por usuario que excluya 0):**

- **LPH no pasa.** M3−M2 mejora de dirección al densificar (de +0.0015 a
  −0.0035), pero queda en 4/5 con |Δ|≈1 sd y es nulo por usuario.
- **La regularización social guiada por frontera no pasa.** Con red densa,
  M4c es peor que M3.
- **Beyond-accuracy y frontera: refutados en denso.** Colapso de NDCG, CCE en
  contra y ganancia concentrada en usuarios integrados.
- **Lo robusto sigue siendo la red inferida frente al CMF plano** (M2/M3/M4c−M1:
  5/5 en `k=2` y en denso).

**Rama B: el paper se centra en «las propiedades de la red inferida mejoran
CMF», con LPH y la regularización por frontera como resultados negativos
reportados.** La mejora de M3 al densificar (≈2 sd) puede mencionarse como
observación, no como claim.

## Dónde centrar esfuerzos ahora

1. ~~`quick`~~ **hecho** (sección anterior).
2. ~~`dense`~~ **hecho: Rama B** (sección anterior). Siguiente: el control con
   red nula (permutar features y reconectar la red), imprescindible para el
   claim «la red inferida aporta estructura», y baselines externos sobre el
   mismo split.
3. **Atribución LPH: cerrada como negativa** en `k=2` y en `k=4`. Redactar el
   paper alrededor de lo robusto («red inferida > CMF plano»).
4. **Ciao:** reportarlo como nulo (BA y RMSE). No invertir en nuevas redes
   Ciao salvo que el claim lo requiera.
5. **Manuscrito:** los cambios de redacción pendientes (split, look-ahead,
   «influence», escalera M2, claims) están en
   [paper_TODO_2026-10-26.md](paper_TODO_2026-10-26.md).

## Alcance verificado

- Cascadas y red: `pipeline/steps/cascade.py`, `networks/cascades/generation.py`, `networks/inference/core.py`, `networks/network_io.py`.
- LPH y comunidades: `docs/lph.md`, `networks/communities/{lph,batch,boundary}.py`.
- Integración CMF: `recommender/enhanced/{features,model,social_regularization,tuning,social_search}.py`.
- Evaluación: `recommender/data.py`, `recommender/experiment/{final_eval,network_selection,manifest,variants}.py`, módulos Route B.
- Protocolos/comandos: `docs/experiments/`, `scripts/run_core_experiment.sh`, `scripts/run_route_b_next_steps.sh`.
- Evidencia registrada: manifests, `core_experiment_results.csv`, network selections, findings y resumen del gate denso.

## Hallazgos críticos

### C1. Confirmado y corregido: `h` no era invariante para el grafo no dirigido

**Qué encontré.** `compute_lph_paper` usa `_compute_network_homophily`. El numerador es simétrico, pero el denominador acumula solo el primer extremo `u` de cada arista:

- `networks/communities/lph.py:80-104`, especialmente `100-101`.
- El grafo se convierte antes a `nx.Graph` por unión de aristas dirigidas (`networks/communities/batch.py:70-96`; `networks/network_io.py:202-267`).

En un `Graph`, que un nodo aparezca como `u` o `v` en `G.edges()` depende del orden interno de nodos/aristas. Por tanto, el denominador y `h` pueden cambiar al reetiquetar o insertar el mismo grafo en otro orden. Un coeficiente de assortativity/Pearson sobre un grafo no dirigido no puede depender de esa orientación accidental.

**Por qué importa.** `h` entra en \(\lambda=(h+\sum_v\delta_v)/N\) y desplaza todos los \(\tilde h_v\). El ranking por \(\delta_v\) no cambia con un desplazamiento común, pero sí pueden cambiar:

- el signo de `lph_score`;
- la intensidad social, que recorta `-lph_score` en cero (`social_regularization.py:88-109`);
- la propiedad interpretada como “positivo/negativo = assortative/boundary”;
- la comparabilidad absoluta entre redes.

**Contraste con la fuente primaria.** El paper define \(s(v)\), \(\delta_v\),
\(\lambda=(h+\sum\delta)/N\) y \(\tilde h_v=\lambda-\delta_v\) en sus
Ecuaciones (3)–(6). La Eq. (5) exige que `h` sea el coeficiente de la Eq. (1)
calculado con el mismo atributo `s(.)`. El repositorio oficial del paper calcula
ese valor mediante assortativity no dirigida de igraph, es decir, con la
distribución marginal de **ambos extremos**.

**Corrección aplicada.** `networks/communities/lph.py:100-106` ahora usa la
semisuma de los cuadrados centrados de ambos extremos. Se añadió
`tests/test_lph.py` con invariancia al orden de inserción y la propiedad
\(\sum_v\tilde h_v=h\). `docs/lph.md` quedó alineado con las Ecuaciones (3)–(6).

**Impacto sobre resultados existentes.**

- `s(v)` y \(\delta_v\) no cambian.
- El nuevo `h` cambia \(\lambda\) y añade el mismo desplazamiento a todos los
  `lph_score`; rankings, percentiles B10/B25 e `is_boundary` permanecen iguales.
- La estandarización de `lph_score` en M3 elimina ese desplazamiento, por lo que
  no se espera cambio material por esa columna.
- `_boundary_intensity` sí depende de qué valores quedan por debajo de cero;
  M4c/M4d pueden cambiar. Esto debe medirse, no asumirse.

**Validación ejecutada (2026-10-26).** Tests de `tests/test_lph.py`: 3/3 OK,
incluida la igualdad con `networkx.numeric_assortativity_coefficient`, que
implementa la Eq. 1. MovieLens, redes seleccionadas `rayleigh_026`,
`powerlaw_043` y `rayleigh_068`, HP congelados, semilla 42:

| Variante | RMSE anterior | RMSE con fix | Delta |
| --- | ---: | ---: | ---: |
| M1 | 0.924770 | 0.924770 | −8.6e−08 |
| M3 | 0.908744 | 0.908744 | +2.4e−07 |
| M4c | 0.903866 | 0.903867 | +7.9e−07 |
| M4d | 0.902982 | 0.902981 | −4.4e−07 |

Todos los deltas son ruido numérico; el orden y las ganancias no cambian.
**C1 queda cerrado para MovieLens.** Ciao no se validó; por el mismo mecanismo
el riesgo es bajo, y basta regenerar sus comunidades en la próxima
reconstrucción.

### C2. El gate denso no es evidencia de mejora del modelo; el experimento que decidiría eso no se ejecutó

**Qué encontré.**

- El plan dice explícitamente que, tras pasar el gate, los HP y \(\alpha\) de `k=2` no aplican y hay que ejecutar la escalera corta M2/M3/M4c (`route_b_positives_and_refinement_plan.md:119-128`).
- El gate reutiliza la red M3 seleccionada en `k=2`, incluyendo la misma familia e índice \(\alpha\), y solo calcula cobertura (`run_route_b_next_steps.sh:103-139`).
- `RUN_RETUNE` estuvo fuera de la corrida y queda manual (`route_b_next_steps_findings.md:10-12,120-129`; `run_route_b_next_steps.sh:312-335`).
- No existen en la evidencia revisada los outputs planeados `route_b_gate_findings.md` y `route_b_dense_wp1_wp2_findings.md`.

**Por qué importa.** Densificar puede aumentar cobertura comunitaria por construcción sin mejorar RMSE, CCE, cobertura de recomendaciones ni el mecanismo de frontera. También puede cambiar cuál \(\alpha\) y qué hiperparámetros son óptimos.

**Cómo corregirlo.** Mantener la decisión como “abrir la siguiente fase en MovieLens k=4”, no como resultado positivo. Antes de cualquier claim:

1. retunear M2/M3/M4c en `k=4`;
2. volver a seleccionar `(diffusion_model, alpha_index)`;
3. ejecutar WP1 y WP2 densos con artefactos separados;
4. restaurar/verificar `k=2` con hashes, no solo copia de directorios;
5. hacer multi-semilla sobre el modelo ya congelado.

### C3. El split per-user tiene look-ahead masivo entre usuarios, pero no favorece al método (reclasificado: problema de encuadre, no sesgo)

**Qué encontré.** `split_data_temporal` retiene el último 20% **de cada usuario** (`recommender/data.py:196-240`) y las cascadas usan todo el train (`pipeline/steps/cascade.py:18-25`). Diagnóstico medido (`validation_checks.py temporal-overlap`):

| Métrica | MovieLens | Ciao |
| --- | ---: | ---: |
| Train posterior al primer rating de test | 99.9% | 99.96% |
| Test cuyo ítem tiene eventos train **posteriores** en su cascada | **82.4%** | 41.3% |
| Eventos de cascada posteriores a algún test del mismo ítem | **78.0%** | 25.8% |

El look-ahead es masivo: en MovieLens, para 4 de cada 5 ratings de test, NetInf vio adopciones posteriores del mismo ítem.

**Por qué no invalida la comparación.** Ese mismo train, con ratings futuros de otros usuarios, lo usa también el baseline M1 para factorizar. NetInf no recibe ninguna observación que M1 no tenga; solo las reorganiza como red. La ventaja temporal es **común a todas las variantes**, así que no explica las diferencias M1→M4. Tampoco hay fuga de la observación objetivo: la cascada del ítem nunca contiene el rating de test del propio usuario.

**Qué sí invalida.** Cualquier frase del paper que presente el método como **predicción del comportamiento futuro** o como una red de influencia temporalmente causal. Con un 82% de look-ahead, la red es una estructura de co-consumo de todo el periodo, no una red de difusión previa al test.

**Sugerencia (decisión recomendada).**

1. **Mantener per-user leave-last como protocolo principal.** Es el estándar en evaluación de recomendadores warm (equivale a un leave-last por usuario), mantiene el test sin cold users y es justo entre variantes.
2. **Cambiar la redacción:** «per-user chronological leave-last; transductive with respect to other users' interactions». Retirar del manuscrito «predict future behavior» y «causal/influence network» como claims; usar «behavioral co-adoption network».
3. **No correr ahora un cutoff global.** Ya se hizo en la campaña legacy: deja la mayoría del test en cold users y mide otra cosa. Solo hace falta si el paper quiere un claim de forecasting. En ese caso, la variante mínima es construir las cascadas solo con eventos anteriores al cutoff y mantener el CMF per-user, reportando `rmse_warm`.
4. Reportar la tabla de arriba en la sección de limitaciones.

## Hallazgos importantes

### I1. Resuelto: la formulación correcta es la de `docs/lph.md`, no la del borrador

**Qué encontré.**

- `docs/lph.md:120-146` define `h` como assortativity de `s(v)` y \(\lambda=(h+\sum\delta)/N\); el código sigue esa versión.
- `articles/draft_article.md:93-113` atribuye al mismo Algorithm 1 una referencia distinta: \(\bar h=|V|^{-1}\sum_v s(v)\) y usa \(\bar h\) en \(\lambda\).
- `docs/lph.md:173-187` además dice que `compute_lph_paper` retorna un diccionario, pero el código retorna la tupla `(lph_scores, s, delta)` (`lph.py:108-160`).

**Resolución con el paper.** Las Ecuaciones (3)–(6) mostradas en la fuente
primaria confirman `docs/lph.md` y el flujo del código:
\(s(v)\rightarrow\delta_v\rightarrow\lambda=(h+\sum\delta)/N
\rightarrow\tilde h_v=\lambda-\delta_v\). La fórmula con \(\bar h\) del
borrador era incorrecta.

**Corrección aplicada.** `articles/draft_article.md:107-113` ahora usa el
coeficiente `h` de la Eq. (1). También se corrigió allí la descripción de los
pesos sociales para que coincida con el código. Este hallazgo ya no requiere
trabajo experimental.

### I2. M3 introduce varias copias o transformaciones de la misma señal; no identifica el efecto de LPH

**Qué encontré.** Están activas simultáneamente `local_pluralistic_hom`, `lph_score`, `s_v`, `delta_v`, `is_boundary` y `pagerank_lph` (`config.py:229-249`). Pero:

- \(\tilde h_v=\lambda-\delta_v\), así que `lph_score` y `delta_v` son perfectamente afines dentro de una red (`lph.py:156-160`);
- `is_boundary` es un umbral de `lph_score` (`boundary.py:13-43`);
- `pagerank_lph` también deriva de `lph_score` (softmax de \(\tilde h_v\), `networks/centrality/pagerank_lph.py:116-130`);
- todas se estandarizan y entran juntas en `U` (`features.py:69-132`; `model.py:53-103`).

**Agravante: M2 no es “solo centralidad”.** `pagerank_lph` se guarda en el CSV de centralidad y está activo por defecto, así que **M2 ya contiene una transformación de \(\tilde h_v\)**. La comparación M3 vs M2 no aísla LPH.

**Por qué importa.** M3 vs M2 prueba un paquete redundante, no “LPH”. La colinealidad dificulta interpretar coeficientes/SHAP y puede favorecer o perjudicar la reconstrucción lateral sin aportar información nueva. Empíricamente M3≈M2 en RMSE global en MovieLens y M3 es peor que M1 en Ciao (`core_experiment_movielens_findings.md:50-58,108-112`; `core_experiment_ciao_findings.md:28-44,90-97`).

**Ablación preliminar (MovieLens, red e HP congelados de M3, semilla 42):**

| Conjunto de columnas | RMSE test | Δ vs centralidad |
| --- | ---: | ---: |
| centralidad pura | 0.91379 | — |
| + `pagerank_lph` (= M2 actual) | 0.91453 | +0.00074 |
| + `lph_score` (\(\tilde h_v\)) | 0.91386 | +0.00007 |
| + Jaccard-LPH | 0.91007 | −0.00372 |
| + `s_v` | 0.91040 | −0.00339 |
| M3 completo | **0.90874** | −0.00505 |

**Actualización con 3 semillas (ver «Resultados de `quick`»):** se confirma que \(\tilde h_v\) no aporta en ningún dataset. Las mejoras aparentes de Jaccard-LPH y `s_v` con la semilla 42 quedan dentro del ruido entre semillas. La lectura de una sola semilla sobrestimaba la señal.

**Cómo corregirlo.** Ablación mínima, en la red seleccionada y con HP congelados:

1. centralidad solamente;
2. + Jaccard-LPH;
3. + solo \(\tilde h_v\);
4. + solo `s_v` (si hay justificación);
5. paquete completo.

No incluir a la vez `lph_score`, `delta_v` e `is_boundary` en el modelo principal.

### I3. El indicador de frontera puede clasificar más del 20% e incluso aislamientos como frontera

**Qué encontré.** `compute_boundary_indicator` calcula el percentil sobre **todos** los nodos y usa `score <= threshold` (`boundary.py:13-43`). No excluye nodos sin comunidades ni resuelve empates. En una red con muchos valores iguales —caso plausible cuando `C(v)=∅`— puede marcar mucho más del 20%; si todos valen cero, marca a todos.

**Por qué importa.** En redes dispersas, “sin comunidad” no equivale a “puente entre comunidades”. Route B corrigió parcialmente esto para estratos, pero la matriz lateral y los pesos sociales siguen consumiendo los CSV originales.

**Cómo corregirlo.** Definir elegibilidad `num_communities >= 2` para frontera pluralista, asignar `NA/ISO` al resto y aplicar percentiles solo entre elegibles. Documentar cómo se tratan empates. Añadir un test con muchos ceros y empates.

### I4. La fórmula social publicada en el borrador no es la implementada

**Qué encontré.**

- El borrador describe `boundary_downweight` con `b_u` binario y `bridge_preserve = J + γ·b` (`draft_article.md:120-139`).
- El código usa intensidad continua derivada de la parte negativa de `lph_score` y `bridge_preserve = sigmoid(γJ-βb)` (`social_regularization.py:88-143`).
- `docs/social_regularization.md:50-68` sí coincide con el código.

Además, si existe al menos un `lph_score` negativo, `_boundary_intensity` ignora `is_boundary` para los demás y recorta todos los valores no negativos a cero (`social_regularization.py:88-100`).

**Por qué importa.** El manuscrito atribuye resultados a otro modelo. `β` no modula un indicador percentil como afirma el borrador, sino una intensidad dependiente del cero absoluto de una métrica cuyo desplazamiento depende de `h`.

**Cómo corregirlo.** Elegir una formulación primaria, fijarla antes de nuevas corridas y alinear manuscrito, documentación, tests y nombres de parámetros. Si se mantiene intensidad continua, normalizar por cuantiles/rango entre usuarios elegibles, no por `max(-h̃,0)` dependiente del signo.

### I5. El constructo de frontera de WP2 no coincide con el gate pre-registrado

**Qué encontré.**

- El plan denso exige `num_communities >= 2` (`route_b_positives_and_refinement_plan.md:110-128`).
- `assign_lph_strata` considera elegible `n_communities > 0`, incluyendo usuarios con una sola comunidad (`boundary_strata.py:20-52`).
- El gate llama `n_b25_eligible` al total con ≥2 comunidades y verifica solo que sea ≥30 (`run_route_b_next_steps.sh:120-136`). Sin embargo, para garantizar B25≥30 harían falta aproximadamente 120 elegibles, no 30.

**Por qué importa.** Una frontera entre comunidades requiere pluralidad de membresías según la propia motivación. El código de análisis puede evaluar una población distinta de la que abrió el gate.

**Cómo corregirlo.** Usar `>=2` de forma uniforme y hacer que el gate calcule los estratos reales y sus N después de percentiles/empates. En los artefactos densos actuales hay suficiente masa total, pero el PASS formal no fue implementado exactamente como se pre-registró.

### I6. El negativo de Ciao beyond-accuracy está contaminado por un ladder incompleto y artefactos mezclados

**Qué encontré.**

- La corrida fresca de Ciao incluye M1/M3/M4c/M4d; M2/M4a/M4b son filas stale (`core_experiment_ciao_findings.md:10-14`; `route_b_next_steps_findings.md:31-36`).
- El bootstrap BA quedó vacío porque las comparaciones pre-registradas exigen M2 (`route_b_next_steps_findings.md:40-50`).
- El script evita correctamente `--all-variants`, pero `route_b_commands.md:60-69` aún lo prescribe para Ciao; la ejecución real está en `run_route_b_next_steps.sh:225-250`.

**Por qué importa.** El patrón puntual CCE/cobertura de Ciao es evidencia negativa útil, pero no es una réplica completa de B1 ni una prueba estadística equivalente a MovieLens.

**Resuelto con las predicciones guardadas (bootstrap 1000, Wilcoxon + Holm).** Δ = primera − segunda variante; en RMSE, negativo = primera mejor.

| Comparación | Ciao Δ RMSE [CI 95%] | MovieLens Δ RMSE [CI 95%] | MovieLens Holm p |
| --- | --- | --- | ---: |
| M3−M1 | +0.0005 [−0.0027, 0.0034] | **−0.0049 [−0.0088, −0.0013]** | 0.008 |
| M3−M2 | — (M2 stale) | −0.0024 [−0.0045, −0.0006] | 0.33 |
| M4c−M3 | −0.0003 [−0.0021, 0.0015] | **−0.0042 [−0.0064, −0.0019]** | 1e−4 |
| M4d−M3 | −0.0009 [−0.0027, 0.0012] | −0.0007 [−0.0038, 0.0025] | 0.33 |
| M4c−M1 | +0.0002 [−0.0022, 0.0025] | **−0.0091 [−0.0125, −0.0055]** | 1e−8 |
| M4d−M1 | −0.0004 [−0.0026, 0.0019] | **−0.0056 [−0.0105, −0.0008]** | 0.003 |

En Ciao, CCE@10 tampoco separa ningún par (todos los CI cruzan 0, n=464).

**Conclusión.** Ciao no es un negativo contaminado: es un **nulo limpio** en BA y un efecto social estable entre semillas pero no significativo entre usuarios. En MovieLens, M4c−M3 es significativo por usuario **solo con la semilla 42**: entre semillas gana 4/5 con |Δ|≈1 sd (ver «Resultados de `quick`»). Lo robusto en ambas fuentes de variación es M4c/M4d sobre M1.

### I7. La comparación de tuning favorece distintos niveles de estabilidad

**Qué encontré.**

- Enhanced usa por defecto tres splits en la campaña (`tuning.py:90-110`).
- Social Optuna usa un único split random warm (`social_search.py:60-110,180-220`).
- Ambos hacen Stage A en la red índice 0, aunque el plan prefería varios cuantiles de \(\alpha\) (`tuning.py:70-88,160-188`; `core_experiment_plan.md:80-105`).
- El core final tiene una sola semilla/partición; la multi-semilla pre-registrada no se ejecutó para RMSE (`route_b_preregistration.md:20-27`; `route_b_positives_and_refinement_plan.md:80-90`).

**Por qué importa.** Los efectos principales de Ciao son ~0.3–0.4% y las diferencias M4a/M4c/M4d en MovieLens son del orden de 0.0005–0.001 RMSE. Sin distribución entre semillas, no se puede distinguir señal estable de variación de inicialización/selección.

**Cómo corregirlo.** Igualar el esquema de validación para M2/M3/M4, y luego repetir solo el fit final con red+HP congelados en 5–10 semillas. No volver a seleccionar HP por semilla.

### I8. Las métricas de ranking y diversidad no sostienen todavía un claim fuerte

**Qué encontré.**

- `evaluate_ranking` considera relevante todo ítem de test si no se pasa umbral (`recommender/data.py:326-420`), y `final_eval` no pasa umbral (`final_eval.py:213-219`).
- Ciao tiene NDCG@10≈0.001–0.004 (`core_experiment_ciao_findings.md:120-128`).
- La ILD primaria de contenido pre-registrada no está implementada; se usa `ild_latent`, que depende del propio modelo y no es comparable entre variantes (`route_b_protocol.md:60-69`; `route_b_commands.md:70-76`; `beyond_accuracy.py:70-93`).

**Por qué importa.** El nulo de Ciao puede reflejar el protocolo de ranking, no ausencia de utilidad. Y una ILD construida con factores distintos por modelo no permite afirmar que una lista es más diversa que otra.

**Cómo corregirlo.** Definir relevancia explícita, candidate set y tratamiento de cold items; implementar géneros/categorías fijos para ILD o retirar ILD de la hipótesis primaria.

### I9. No hay fuga clásica test→NetInf, pero el protocolo es transductivo

**Qué encontré.**

- Cascadas usan solo `train_df` (`pipeline/steps/cascade.py:18-25`).
- El scaler se ajusta sobre usuarios de train (`model.py:53-70`; `social_regularization.py:400-420`).
- Social Optuna recibe `train_df` externo (`social_search.py:60-110,212-243`).
- El header de cascadas declara todos los IDs, incluso test-only, pero no incluye sus interacciones (`generation.py:42-50,70-87,112-123`).

**Por qué importa.** No se encontró fuga de ratings o etiquetas de test. Sin embargo, conocer de antemano el universo completo de usuarios y evaluar leave-last es un protocolo transductivo, no cold-start inductivo ni forecast global.

**Cómo corregirlo.** Mantener esa distinción en el paper; usar el track cold-start separado para claims inductivos.

### I10. El paper de investigación 1 ya está disponible, pero la campaña actual sigue sin ser una réplica

**Qué encontré.**

- El paper publicado está ahora en `articles/ws-dmaa.pdf` (DOI
  `10.1142/S1793830923500520`) y confirma el contexto MovieLens de
  aproximadamente 100 000 ratings, 671 usuarios y 9066 películas.
- La investigación 1 reportada promedia 100 \(\alpha\) por modelo; la actual selecciona una red por variante mediante CV (`draft_article.md:270-280`; `network_selection.py:97-151`).
- El borrador reconoce que M2 no replica exactamente el paper porque falta farness (`draft_article.md:402-428`).
- En la configuración actual `hub_score` y `auth_score` están desactivados, igual que `num_communities` (`config.py:229-249`).
- El split actual es leave-last per-user, mientras los documentos históricos hablan de random o protocolos distintos.

**Por qué importa.** M2 es una continuación conceptual de MAFPIN, no una réplica del paper. Los RMSE 0.7867 y 0.90x no deben ponerse en una misma línea de progreso.

**Cómo corregirlo.** Declarar que investigación 2 usa un nuevo estimando y
comparar solo dentro de su ladder. Si se quiere una réplica estricta, todavía
hacen falta los scripts/manifiestos originales que fijen el split, HP y
agregación operacional exacta; disponer del PDF elimina la duda bibliográfica,
pero no reconstruye artefactos ejecutables.

### I11. Hay artefactos canónicos que pertenecen a campañas antiguas

**Qué encontré.**

- `data/movielens/canonical_baseline.json:344-346` conserva `global_test_rmse=1.041`, mientras el M1 per-user fresco es 0.9248.
- `data/ciao/core_experiment_results.csv:2-5` contiene filas stale sin columnas warm, mezcladas con filas frescas.
- Los findings indican explícitamente no mezclar campañas (`core_experiment_movielens_findings.md:8-18`; `core_experiment_ciao_findings.md:8-18`).

**Por qué importa.** Un análisis automatizado o una tabla futura puede consumir filas incompatibles y producir conclusiones falsas.

**Cómo corregirlo.** Versionar artefactos por `split_strategy + git_commit + stamp`, prohibir upsert entre campañas y validar que todas las filas de una tabla compartan manifest/hash.

### I12. El modo `retune-dense` de `run_route_b_next_steps.sh` no ejecuta la escalera que promete

**Qué encontré.**

- `run_route_b_next_steps.sh:328-332` corre `--steps recommend --model-variant M2|M3|M4c`. Pero `recommend` **ignora `--model-variant`** (`pipeline/_cli.py:232-236`; `pipeline/steps/recommend.py:65-84`): las variantes se activan con `--no-communities` y `--social-regularization --social-mode …`. Las tres corridas serían el mismo M3, sin regularización social.
- Los logs van a `logs/route_b/dense_k4_*`, pero `import_manifest` lee nombres fijos `logs/m*_recommend.log` (`manifest.py:105-113`). Además el `--log-file` hace *append* y el parser toma la primera coincidencia, así que reutilizaría HP de `k=2`.
- `--steps inference centrality communities` se ejecuta en el orden dado (`pipeline/_runner.py:106-116`): `pagerank_lph` se calcula con las comunidades **anteriores**. Esto no afecta la métrica del gate (solo cobertura), pero sí cualquier modelo entrenado sobre esos CSV de centralidad.

**Por qué importa.** Si se hubiera lanzado `RUN_RETUNE=1`, se habrían perdido 8–24 h y, peor, se habría reportado un M4c denso que no era social.

**Corrección.** `scripts/run_validation_plan.sh dense` usa los flags correctos, el orden `inference communities centrality`, logs con nombres canónicos y limpieza de artefactos `k=2` antes de importar. No usar `retune-dense`.

## Hallazgos menores

### M1. Resuelto: `docs/lph.md` tenía rutas y API obsoletas

`docs/lph.md:77-88,173-187` apuntaba a `networks/communities.py` y
`recommender/enhanced.py`, y describía un retorno incorrecto.

**Corrección aplicada 2026-10-26:** rutas, retorno de `compute_lph_paper`,
separación entre proxy Jaccard y \(\tilde h_v\), ecuaciones y convención para
aislados quedaron actualizados.

### M2. El nombre `lph_score` oculta que hay dos métricas diferentes

`local_pluralistic_hom` es el promedio Jaccard, mientras `lph_score` es \(\tilde h_v\) (`boundary.py:60-110`). Renombrar a `lph_jaccard` y `h_tilde`, o añadir un glosario obligatorio en resultados/figuras.

### M3. Los defaults de `lambda_social` no son únicos

CLI/tuning usan `0.001` (`pipeline/_cli.py:130-135`; `tuning.py:90-105`), mientras las funciones de fit usan `0.01` (`social_regularization.py:425-438,493-506`). Los runs core pasan valores Optuna y no dependen normalmente de esos defaults, pero scripts parciales sí pueden divergir. Centralizar el default.

### M4. El paralelismo documentado y el script difieren

`core_experiment_commands.md:31-39` muestra `--n-jobs 1`; `run_core_experiment.sh:170-180` usa el valor del batch, por defecto `-1`. No cambia el modelo si el determinismo se mantiene, pero debe registrarse para reproducibilidad y estabilidad del entorno.

## Ciao vs. MovieLens: lectura correcta

### MovieLens

- No existe red social nativa. La red se infiere de co-adopción temporal: cada película es una cascada, NetInf infiere aristas y luego se simetrizan (`generation.py:89-110`; `batch.py:70-96`).
- El gate denso pasa claramente en `k=4`; `k=6` aumenta cobertura pero empeora la estabilidad B10, por lo que `k=4` es la opción más defendible.
- En el core per-user, M3≈M2 (entre semillas M3 es peor en 3/5). Lo robusto es que la red inferida mejora al CMF plano: M2, M4c y M4d ganan a M1 en 5/5 semillas. M4c sobre M3 no es robusto entre semillas (4/5).

### Ciao

- La red NetInf se construye del mismo modo a partir de ratings, no del trust explícito. El trust es un track separado.
- Ciao replica solo la **dirección** de la mejora social en RMSE global; por usuario, ningún par es significativo y CCE tampoco separa variantes. Es un nulo, no un problema de implementación.
- Su look-ahead temporal es la mitad que el de MovieLens (41% vs 82%), y aun así no hay señal: la ausencia de efecto no se explica por fuga.
- Su cobertura comunitaria densa llega a 58.1% en `k=6`, justo por debajo de un umbral de 60% elegido previamente. Esto es un FAIL formal, no una discontinuidad científica.
- El negativo BA debe conservarse en el paper como resultado, pero no extrapolarse a “LPH/social no funciona”.

### ¿Tiene sentido llamar homofilia a la señal?

Solo con una salvedad fuerte. NetInf se alimenta de usuarios que califican los mismos ítems en secuencia; las comunidades y LPH se calculan después sobre esa misma red. La señal puede representar **homofilia conductual endógena al consumo**, no homofilia social independiente. La estabilidad entre \(\alpha\) vecinos demuestra robustez algorítmica local, no validez externa del constructo.

La validación más barata disponible es Ciao: correlacionar vecindad/comunidades/\(\tilde h_v\) inferidos con trust explícito, manteniendo ese análisis fuera del entrenamiento. Para MovieLens haría falta una señal externa de contenido o atributos de usuario.

## Riesgos de validez

1. **Validez de constructo:** “frontera” se deriva de comunidades del mismo grafo inferido de ratings; puede ser circular respecto del comportamiento que luego se predice.
2. **Validez interna:** el split per-user tiene look-ahead entre usuarios del 82% (MovieLens) y 41% (Ciao); es común a todas las variantes, pero impide claims de forecasting o causalidad temporal.
3. **Validez estadística:** core usa una sola semilla y no reporta CI para diferencias subporcentuales.
4. **Validez de atribución:** M3 agrupa seis señales correlacionadas y M4 cambia además la regularización; no hay una ablación limpia de \(\tilde h_v\).
5. **Validez de medición:** el indicador percentil trata mal empates/aislamientos (`h` ya corregido, impacto nulo).
6. **Validez externa:** dos datasets no equivalen a dos redes sociales comparables; MovieLens no tiene trust y Ciao NetInf no usa su trust.
7. **Validez de selección:** red, \(\alpha\), HP y variante se seleccionan con protocolos de validación no idénticos.
8. **Validez de conclusión:** pasar cobertura/estabilidad es una precondición, no evidencia de ganancia recomendadora.
9. **Validez documental:** manuscrito, `lph.md`, protocolos históricos y artefactos mezclan varias eras del experimento.
10. **Reproducibilidad:** la restauración bit-a-bit de redes `k=2`, los logs completos y el estado exacto del código de cada artefacto son **no verificables sin hashes o ejecución**.

## Qué verificar o corregir primero

1. **Corrección de `h`: cerrada.**  
   Validada en MovieLens con impacto `<1e-6` en RMSE. Ya no bloquea.

2. **Alinear elegibilidad de frontera y pesos sociales.**  
   Usar `num_communities>=2`, excluir ISO y decidir intensidad binaria vs continua. Recalcular solo CSVs de comunidades seleccionados antes de entrenar.

3. **Limpiar la atribución de LPH.**  
   Ablación preliminar hecha (I2); confirmar con 3 semillas (`quick`). Si se
   confirma, quitar `pagerank_lph` de M2 en futuras corridas para que M2 sea
   realmente «solo centralidad».

4. **Ciao: cerrado como nulo** (I6). No requiere más corridas.

5. **Ejecutar la fase densa que realmente decide la hipótesis.**  
   `run_validation_plan.sh dense` (I12 explica por qué no usar `retune-dense`).

6. **Cuantificar varianza entre semillas.**  
   `run_validation_plan.sh quick` (5 semillas, red+HP congelados).

7. **Look-ahead temporal: medido** (C3). Acción: cambio de redacción, no
   nuevas corridas.

8. **Validar el constructo externamente.**  
   En Ciao, comparar LPH/comunidades NetInf con trust. En MovieLens, usar géneros para ILD y coherencia comunitaria.

9. **Separar investigación 1 de investigación 2.**  
   El paper ya está disponible, pero M2 no reproduce sus features ni su
   agregación. Reportar continuidad conceptual, no progreso numérico.

## Plan de ejecución (`scripts/run_validation_plan.sh`)

Todo escribe en `data/validation_runs/<stamp>/`; ningún artefacto canónico
queda modificado. Análisis en `scripts/validation_checks.py`.

| Pregunta | Estado | Cómo se resuelve |
| --- | --- | --- |
| ¿La corrección de `h` cambia M4? | **Resuelto: no** | — |
| ¿Hay look-ahead temporal? | **Resuelto: sí, común a todas las variantes** | — (C3) |
| ¿Ciao fue un falso negativo? | **Resuelto: nulo real** | — (I6) |
| ¿M4c gana de forma estable entre semillas? | **Resuelto: sobre M1 sí (5/5); sobre M3 no en MovieLens (4/5)** | — |
| ¿LPH aporta algo distinto de centralidad (k=2)? | **Resuelto: no** | — |
| ¿La densificación mejora el recomendador? | **Resuelto: no para LPH ni social; M3 mejora ≈2 sd** | — |
| ¿La mejora frente a M1 viene de la estructura de la red? | Pendiente | control con red nula (por preparar) |
| ¿La “homofilia” es social? | Pendiente, no scriptado | overlap NetInf vs trust en Ciao |

```bash
./scripts/run_validation_plan.sh quick
nohup ./scripts/run_validation_plan.sh dense > data/validation_dense.nohup.log 2>&1 &
```

**Criterios de lectura:**

- `seeds_*/frozen_seeds_deltas.csv`: claim «M4c > M3» solo si `seeds_first_better` = 5/5 y `mean_delta` > 2·`sd_delta` en valor absoluto.
- `lph_ablation_*.csv`: «LPH aporta» solo si `centrality+h_tilde` o `centrality+jaccard_lph` mejora a `centrality` en las 3 semillas.
- `dense`: la tabla final compara medias de 5 semillas, densa vs `k=2`. La densificación tiene sentido solo si M3 o M4c densos mejoran a su versión `k=2` por más de 2 sd y, dentro de la corrida densa, **M3−M2** (aporte de LPH) o **M4c−M3** pasan el criterio de 5/5 semillas. Si no, LPH queda descartado como contribución.

## Tres preguntas antes de seguir

1. ¿El claim principal será **mejora de recomendación por LPH**, **mejora por regularización social sobre una red inferida**, o solo **viabilidad estructural de comunidades densificadas**? Los resultados actuales apoyan de forma distinta cada claim.
2. ¿Existen scripts, manifests o logs originales de `ws-dmaa` que permitan
   reproducir su split, features y agregación, o se tratará M2 únicamente como
   continuidad conceptual?
3. Si la ablación con 3 semillas confirma que \(\tilde h_v\) solo no aporta,
   ¿el paper se reformula alrededor de la superposición comunitaria vecinal
   (Jaccard/`s_v`) y la regularización social M4c, en vez de \(\tilde h_v\)?

## No verificable sin ejecutar

- Integridad bit-a-bit de la restauración `k=2`.
- Estado real de todos los logs y correspondencia de cada artefacto con el commit actual.
- Resultado recomendador de MovieLens densificado.
- Equivalencia ejecutable del protocolo actual con investigación 1 sin sus
  scripts/manifiestos históricos.
