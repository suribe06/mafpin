# TODO del manuscrito tras la auditoría LPH (2026-10-26)

Pendientes de redacción para `articles/draft_article.md`, derivados de
[lph_dense_network_decision_audit_2026-10-25.md](lph_dense_network_decision_audit_2026-10-25.md).
Los números de línea corresponden al borrador del 2026-10-26.

**Ya corregido en el borrador:** fórmula de \(\lambda\) con `h` de la Eq. 1
(§3.3), fórmulas de pesos sociales (§3.4) y autores de la referencia [2].

**Por qué el resto no se editó directamente:** el borrador todavía reporta la
campaña preliminar (split aleatorio, submuestra de 5000 ratings). Cambiar el
método sin reemplazar los resultados dejaría el paper inconsistente. Estos
cambios deben hacerse **junto con** la reescritura de §5–§7 con la campaña
per-user.

---

## A. Split y look-ahead temporal (hallazgo C3) — obligatorio

- [ ] **L168 — descripción del split.** Hoy dice «split at the user level with a
  fixed random seed» y que los test «represent future behavior». Reemplazar por:

  > The ratings are split per user in chronological order: for each user, the
  > last 20% of their ratings form the test set (per-user chronological
  > leave-last). Cascades and all network artifacts are built from training
  > interactions only, so no test rating is ever visible to NETINF. The protocol
  > is transductive with respect to other users: training interactions of one
  > user may occur after test interactions of another. This look-ahead is
  > shared by every model in the ladder, including the plain CMF baseline.

- [ ] **L432 — limitación «Random train/test split».** Ya no es cierta. Reemplazar
  por una limitación de look-ahead con la tabla medida:

  | | MovieLens | Ciao |
  | --- | ---: | ---: |
  | Test ratings whose item cascade contains later training events | 82.4% | 41.3% |
  | Cascade events posterior to some test rating of the same item | 78.0% | 25.8% |

  Texto sugerido: «Because the look-ahead is common to all variants it does not
  bias the comparison, but the inferred network should be read as a co-adoption
  structure over the whole period, not as a temporally causal diffusion
  network. Results do not support forecasting claims.»

- [ ] **Retirar claims de forecasting.** Buscar «future behavior», «predict
  future» y equivalentes; no deben quedar como claim del método.

## B. Lenguaje de «influencia» (C3) — obligatorio

`influence` aparece 14 veces. Mantenerlo solo al describir el modelo de NETINF
o el título de la referencia [4]; en el resto usar «behavioral co-adoption» o
«inferred co-adoption network».

- [ ] L21, L23: «people influence each other» / «network that mediates
  influence» → presentar la red como proximidad conductual inferida, no como
  influencia observada.
- [ ] L166: «B influenced A's decision» → «consistent with a diffusion process
  in the statistical sense formalized by NETINF; we do not claim causal
  influence».
- [ ] L172: «underlying influence structure» → «underlying co-adoption structure».
- [ ] L420, L438, L446: «influence dynamic», «inferred influence graph»,
  «asymmetric influence» → «co-adoption dynamic», «inferred directed graph»,
  «asymmetric co-adoption».

## C. Escalera de modelos y features — obligatorio

- [ ] **L218 — «M2 — MAFPIN (replicating [1])».** M2 no replica [1]: falta
  farness, hub/authority están desactivados y la red se selecciona por CV en vez
  de promediar 100 α. Cambiar a «inspired by [1]» y mantener la advertencia de
  L430.
- [ ] **L197 — lista de centralidades.** No coincide con `config.SideUserFeatures`:
  hub/authority están **desactivados** y falta `pagerank_lph`. Si se mantiene
  `pagerank_lph` en M2, declarar que deriva de \(\tilde h_v\) y que M2 no es
  «solo centralidad». **Recomendado:** quitar `pagerank_lph` de M2 en la
  próxima campaña.
- [ ] **L199 — cascade statistics.** Están desactivadas en la configuración
  actual; quitarlas de la lista o activarlas.
- [ ] **L180 — usuarios sin comunidad «are assigned \(\tilde h_v = 0\)».** Es
  falso en el código: un nodo aislado recibe \(\delta_v=0\) y por tanto
  \(\tilde h_v=\lambda\); un nodo con vecinos pero sin comunidades recibe
  \(s(v)=0\) y un \(\tilde h_v\) calculado normalmente. Describir el
  comportamiento real o cambiar el código y regenerar.
- [ ] **L198 — boundary features.** No coincide con la configuración:
  \(m(v)\) (`num_communities`) está **desactivado** y la Jaccard-LPH
  (`local_pluralistic_hom`) está activa pero no aparece listada.
- [ ] **L190 — indicador binario por percentil 20.** Declarar que los empates y
  usuarios sin comunidad pueden inflar la fracción marcada (hallazgo I3), o
  corregir la elegibilidad a `num_communities ≥ 2` antes de la próxima campaña.

## D. Resultados y claims — tras correr `run_validation_plan.sh`

- [ ] **§5.2–§5.4 (L236, L257) y §6.** Reemplazar el protocolo submuestreado
  (k-fold, 5000 ratings) por la campaña per-user: red seleccionada por CV en
  train, HP congelados y una evaluación en test, reportando `rmse_warm`.
- [ ] **L31 — «demonstrates consistent improvements».** No sostenible. Con
  `quick` (5 semillas + bootstrap por usuario):
  - **Robusto (MovieLens):** la red inferida mejora al CMF plano. M2, M4c y M4d
    ganan a M1 en 5/5 semillas (Δ −0.014 a −0.017, |Δ| ≥ 4 sd), y M4c/M4d−M1
    también son significativos por usuario.
  - **No robusto (MovieLens):** M4c/M4d sobre M3 (4/5 semillas, |Δ| ≈ 1 sd) y
    M3 sobre M2 (M3 peor en 3/5).
  - **Ciao:** M4c/M4d−M3 estables entre semillas (5/5, Δ ≈ −0.003) pero no
    significativos por usuario; M3 peor que M1 en 5/5. Detalle en la sección G.
- [ ] **Estadística.** Reportar dos fuentes de variación: entre semillas (5) y
  entre usuarios (bootstrap pareado + Wilcoxon/Holm). Criterio de claim: 5/5
  semillas, |Δ medio| > 2 sd y CI por usuario que excluya 0.
- [ ] **Atribución de LPH — resultado de `k=2`.** \(\tilde h_v\) no aporta en
  ningún dataset (ablación con 3 semillas), y ningún otro subconjunto LPH
  supera el ruido entre semillas. Mientras `dense` no diga otra cosa, el paper
  **no puede** afirmar que LPH mejora la recomendación. Contribución
  defendible hoy: «propiedades de la red inferida mejoran CMF» (continuidad
  con la investigación 1), con LPH reportado como resultado negativo. Afecta
  resumen, L27–L31, §3.3 y conclusiones.
- [ ] **Winner's curse.** Los HP se tunearon con la semilla 42, que resultó de
  las más favorables para M4c. Declararlo, o reportar medias de semillas en vez
  de la semilla 42.
- [ ] **Densificación — resultado (Rama B).** Con k=4, LPH (M3−M2: 4/5,
  |Δ|≈1 sd, nulo por usuario) y la regularización social (M4c peor que M3, 1/5)
  no pasan el criterio. NDCG colapsa, el CCE va en contra y la ganancia de M3
  sobre M2 se concentra en usuarios integrados (E75), no en la frontera.
  **Tesis del paper:** «las propiedades de la red inferida mejoran CMF»
  (M2/M3/M4c−M1: 5/5 semillas en k=2 y k=4). LPH, la regularización por
  frontera y beyond-accuracy van como resultados negativos pre-registrados.
  Detalle en el reporte, sección «Resultados de `dense`».
- [ ] **Antes de escribir los resultados:** control con red nula (features
  permutadas entre usuarios y red reconectada preservando grados, con HP
  congelados) y baselines externos sobre el mismo split. Sin el control, el
  claim central no distingue «estructura de la red» de «cualquier matriz
  lateral regulariza».

## G. Ciao como resultado nulo: por qué el paper se centra en MovieLens — obligatorio

Si el paper queda centrado en MovieLens, **Ciao no se omite**: se reporta como
réplica negativa y se explica. Omitirlo sería selección de resultados, y el
pre-registro de Route B comprometía reportar todos los resultados.

- [ ] **Reportar el nulo con su evidencia** (sección de resultados o apéndice):
  - Bootstrap pareado por usuario (1000 remuestras, Wilcoxon + Holm): ningún
    par M3−M1, M4c−M3, M4d−M3, M4c−M1 ni M4d−M1 se distingue de cero en RMSE
    (todos los CI 95% cruzan 0; Holm p = 1.0). CCE@10 tampoco (n=464).
  - Las mejoras de RMSE global de M4c/M4d sobre M3 (~0.3%) son estables entre
    5 semillas (5/5, sd ≈ 0.001), pero no se distinguen de cero entre usuarios:
    el efecto es reproducible y demasiado pequeño frente a la heterogeneidad
    de usuarios para generalizar.
  - Los atributos (M3) empeoran a M1 en 5/5 semillas, y en la ablación todas las
    columnas dan lo mismo hasta 1e−5: con `w_user` ≈ 0.04, el modelo ignora la
    matriz lateral.
  - Contraste: en MovieLens, con el mismo pipeline y el mismo análisis, M4c−M3
    y M4c−M1 sí son significativos. Que el mismo código produzca un efecto en
    un dataset y no en otro descarta un fallo de implementación como
    explicación principal.

- [ ] **Explicar por qué Ciao no es un terreno adecuado para el método**
  (características medidas del dataset, no supuestos):

  | | MovieLens | Ciao |
  | --- | ---: | ---: |
  | Usuarios / ítems / ratings | 671 / 9066 / 100 004 | 2248 / 16 861 / 36 065 |
  | Densidad de la matriz | 1.64% | 0.095% |
  | Ratings por usuario (media) | 149 | 16 |
  | Ratings por ítem (media) | 11.0 | 2.1 |
  | Fracción warm del test (usuario e ítem vistos en train) | 92.5% | 56.6% |
  | Usuarios con ≥1 comunidad, red densificada k=4 / k=6 | 75% / 90% | 40% / 58% |

  Lectura: con ~2 ratings por ítem, las cascadas de Ciao son muy cortas.
  NetInf infiere pocas aristas confiables, DEMON deja sin comunidad a la
  mayoría de usuarios incluso densificando, y las features comunitarias (y
  por tanto LPH) quedan vacías para buena parte de la población. Además, casi
  la mitad del test involucra ítems fríos, donde ni la regularización social
  ni los atributos de usuario pueden ayudar.

- [ ] **Matizar como hipótesis lo que no está demostrado.** La tabla muestra
  condiciones desfavorables; no prueba que sean la causa del nulo. Redactar
  como «consistent with», no «caused by». Factores adicionales a mencionar
  como posibles:
  - timestamps de Ciao con granularidad diaria (muchos empates dentro de una
    cascada), que degradan la inferencia de NetInf
    (`docs/experiments/ciao_trust_network_future_work.md:39`);
  - la red NetInf de Ciao casi no coincide con su trust declarado (~1.5% de
    aristas; medido en una red k=2 anterior, exponential α83, no en la red
    congelada actual).

- [ ] **Qué NO afirmar:** que Ciao «falló por implementación», que el método
  «no funciona» en general, ni que el nulo se explica por fuga temporal (el
  look-ahead de Ciao es la mitad que el de MovieLens: 41% vs 82%).

- [ ] **Chequeo opcional (barato, sin reentrenar) para respaldar la
  explicación:** desde las predicciones guardadas, repetir el bootstrap de Ciao
  solo con usuarios con muchos ratings de train (p. ej. ≥20) y solo sobre el
  test warm. Si aparece efecto en ese subgrupo, la explicación por escasez queda
  respaldada; si sigue siendo nulo, decirlo así.

- [ ] **Redacción sugerida para el paper:**

  > We replicated the full pipeline on Ciao under the same protocol. RMSE gains
  > of social regularization (~0.3%) were stable across five random seeds but
  > did not survive per-user paired testing, and neither did beyond-accuracy
  > effects. Ciao is two orders
  > of magnitude sparser than MovieLens (0.095% vs 1.64% density; 2.1 vs 11.0
  > ratings per item), which yields short cascades, sparse inferred networks
  > and community coverage below 60% even after densification. We therefore
  > report Ciao as a negative replication and treat MovieLens as the setting in
  > which the inferred-network signal is identifiable, while noting that the
  > method's applicability depends on interaction density.

## E. Comparabilidad con la investigación 1

- [ ] **Tabla 1 (L270–L280).** Mantener la nota de no comparabilidad y añadir
  explícitamente: distinto split, selección de red vs promedio sobre α, feature
  set distinto. No trazar una línea de progreso 0.7867 → 0.90x.

## F. Constructo de homofilia

- [ ] En discusión/limitaciones: la «homofilia» medida es conductual y endógena
  al consumo (comunidades y LPH se calculan sobre la misma red inferida de
  ratings). La estabilidad entre α vecinos es robustez algorítmica, no validez
  externa. Si se puede, validar contra el trust explícito de Ciao.
