# Propuesta de experimento: cold-start de usuarios

**Estado:** propuesta metodologica, no ejecutada  
**Relacion con el articulo:** experimento adicional para probar si la capa de atributos de red/comunidad/frontera ayuda especialmente cuando el usuario tiene poco historial de ratings.  
**Modelos objetivo:** M1, M2, M3 y, solo como extension condicionada, M4c/M4d.

## 1. Motivacion

Los resultados core muestran que M3, el modelo que agrega centralidades y atributos de comunidad/frontera al CMF, mejora el RMSE frente al baseline en MovieLens y CiaoDVD. Sin embargo, esos resultados estan calculados sobre el test global, mezclando usuarios con mucho historial y usuarios con poco historial.

El experimento de cold-start pregunta algo mas especifico:

> Cuando un usuario tiene pocas calificaciones disponibles en entrenamiento, las senales estructurales de la red inferida y de frontera comunitaria ayudan mas que en usuarios warm?

Esta pregunta es importante porque CMF puede usar informacion lateral del usuario. Si el usuario tiene pocos ratings, su vector latente aprendido desde la matriz usuario-item es debil; por tanto, los atributos de red, comunidad y frontera deberian tener mas valor relativo.

## 2. Pregunta cientifica

La pregunta principal es:

> La extension MAFPIN-LPH mejora la recomendacion para usuarios con pocos ratings en entrenamiento, en comparacion con un CMF baseline sin atributos?

Preguntas secundarias:

- La mejora de M3 frente a M1 es mayor en usuarios `1-3` y `4-10` ratings que en usuarios `>10`?
- La informacion de frontera/comunidad de M3 aporta mas que las centralidades solas de M2?
- Las variantes sociales M4c/M4d aportan algo adicional en cold-start, o el beneficio principal sigue estando en M3?
- En CiaoDVD, el grafo explicito de confianza permite evaluar un caso de cold-start puro que MovieLens no puede cubrir?

## 3. Hipotesis

| ID | Hipotesis | Evidencia esperada |
| --- | --- | --- |
| H1 | M3 mejora mas que M1 en usuarios con pocos ratings. | Delta RMSE M3-M1 mas grande en `1-3` y `4-10` que en `>10`. |
| H2 | M3 supera a M2 en cold-start moderado. | La capa comunidad/frontera aporta mas que centralidad sola. |
| H3 | M4c/M4d no deben asumirse ganadoras. | Solo se reportan como utiles si superan a M3 por estrato con evidencia estadistica. |
| H4 | El cold-start puro de 0 ratings requiere atributos externos o sociales. | MovieLens no lo puede resolver con NetInf si el usuario no aparece en train; Ciao podria evaluarlo con trust graph. |

## 4. Estratos de usuarios

El experimento separa usuarios segun el numero de ratings disponibles en entrenamiento:

| Grupo de usuarios | Definicion | Evaluacion |
| --- | --- | --- |
| `0` ratings en train | Usuario con ratings en test, pero ninguno disponible para entrenar su perfil | Cold-start puro; solo viable si hay atributos externos, sociales o de confianza disponibles antes de observar ratings. |
| `1-3` ratings en train | Usuario con muy poca evidencia individual | Few-shot cold-start fuerte. |
| `4-10` ratings en train | Usuario con historial pequeno pero no nulo | Cold-start moderado. |
| `>10` ratings en train | Usuario con historial suficiente | Usuario warm; sirve como control. |

La comparacion clave no es solo "quien gana globalmente", sino si el beneficio de M3 se concentra donde deberia: en `1-3` y `4-10`.

## 5. Modelos a comparar

| Modelo | Descripcion | Rol en el experimento |
| --- | --- | --- |
| M1 | CMF baseline sin atributos de red | Punto de comparacion: que pasa si solo usamos ratings. |
| M2 | CMF + centralidades de red inferida | Prueba si la estructura topologica clasica ayuda en cold-start. |
| M3 | CMF + centralidades + comunidad/frontera LPH | Modelo principal de la fusion; deberia ser el foco. |
| M4c | M3 + regularizacion social boundary-downweight | Variante social; incluir solo si se quiere probar si ayuda en cold-start. |
| M4d | M3 + regularizacion social bridge-preserve | Variante social; incluir especialmente en MovieLens, donde fue mejor por RMSE global. |

Recomendacion: el paper debe tratar M3 como modelo principal para cold-start. M4c/M4d deben ser analisis secundarios, porque los resultados core ya mostraron que la regularizacion social no es una mejora universal.

## 6. Diseno experimental recomendado

Hay dos formas posibles de construir el experimento. La segunda es la mas fuerte cientificamente.

### 6.1. Diagnostico sobre el split global actual

Usar el split temporal 80/20 ya existente y agrupar usuarios por su numero de ratings en train. Luego evaluar las predicciones del test por estrato.

Ventajas:

- Bajo costo.
- Reutiliza resultados y artefactos del experimento core.
- Sirve para una primera lectura rapida.

Limitaciones:

- Puede haber pocos usuarios en algunos estratos.
- No controla explicitamente cuantos ratings se dejan visibles por usuario.
- No produce necesariamente un buen grupo `0 ratings`.

### 6.2. Split controlado por usuario y tiempo

Construir un split especifico de cold-start:

1. Ordenar cronologicamente las interacciones de cada usuario.
2. Para cada usuario, dejar visibles en train solo `k` ratings iniciales, donde `k` define el estrato:
   - `k = 0`
   - `k = 1, 2, 3`
   - `k = 4, ..., 10`
   - `k > 10` para warm users
3. Usar los ratings posteriores del mismo usuario como test.
4. Construir cascadas y redes solo con informacion permitida en train.
5. Calcular atributos de red/comunidad/frontera solo desde la informacion de train.
6. Entrenar M1/M2/M3/M4c/M4d y evaluar por estrato.

Esta opcion es mas lenta, pero responde mejor a la pregunta cientifica. Evita que un usuario clasificado como cold-start haya contribuido con informacion futura a la red inferida.

## 7. Caso especial: usuarios con 0 ratings

El grupo `0 ratings en train` debe tratarse con cuidado.

Con la arquitectura actual basada en NetInf, si un usuario no tiene ningun rating en train:

- no aparece en cascadas de entrenamiento;
- no contribuye a la red inferida;
- no tiene centralidades NetInf;
- no tiene comunidades NetInf;
- no tiene LPH calculable desde la red inferida.

Por eso, en MovieLens el cold-start puro de usuarios con `0` ratings no es resoluble por M1/M2/M3 usando solo la informacion actual. Para evaluarlo seriamente se necesita informacion externa previa al rating:

- perfil declarado del usuario;
- atributos demograficos;
- grafo social externo;
- o, en CiaoDVD, grafo explicito de confianza.

En CiaoDVD si podria plantearse un experimento adicional:

| Variante | Fuente de atributos para usuario 0-rating |
| --- | --- |
| M1 | fallback global, item mean o sesgo de item |
| M2-trust | centralidades en trust graph |
| M3-trust | comunidades/frontera sobre trust graph |
| M3-NetInf | no aplicable si el usuario no aparece en cascadas de train |

Conclusion: el grupo `0 ratings` debe reportarse como un experimento separado de cold-start puro, no mezclarse con el resultado principal de NetInf.

## 8. Metricas

Metricas primarias por estrato:

- RMSE
- MAE
- numero de usuarios evaluados
- numero de ratings evaluados
- cobertura de prediccion

Metricas secundarias:

- NDCG@10, solo si cada usuario tiene suficientes items candidatos y suficientes positivos en test;
- Precision@10 y Recall@10, con la misma advertencia;
- MRR, si se mantiene el protocolo de ranking actual.

Metricas de robustez recomendadas:

- delta RMSE M3-M1 por usuario;
- delta RMSE M3-M2 por usuario;
- intervalo de confianza bootstrap por usuario;
- prueba pareada por usuario o por rating para M3 vs M1 y M3 vs M2.

## 9. Tabla de resultados esperada

La tabla principal deberia verse asi para cada dataset:

| Estrato | Usuarios | Ratings test | M1 RMSE | M2 RMSE | M3 RMSE | M4c RMSE | M4d RMSE | Mejor modelo | Delta M3-M1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | | | | | | | | | |
| 1-3 | | | | | | | | | |
| 4-10 | | | | | | | | | |
| >10 | | | | | | | | | |

Y una segunda tabla resumida:

| Comparacion | 1-3 | 4-10 | >10 | Interpretacion |
| --- | ---: | ---: | ---: | --- |
| M3 vs M1 | | | | Prueba si la fusion ayuda en cold-start. |
| M3 vs M2 | | | | Prueba si LPH aporta mas que centralidad. |
| Mejor M4 vs M3 | | | | Prueba si la regularizacion social agrega valor. |

## 10. Criterios de exito

El experimento apoyaria la narrativa cold-start si:

1. M3 mejora a M1 en `1-3` y/o `4-10` con mayor margen que en `>10`.
2. M3 mejora a M2 en los estratos cold-start, mostrando que frontera/comunidad aporta mas que centralidad sola.
3. Los intervalos de confianza de M3-M1 y M3-M2 no son compatibles con una diferencia trivial.
4. M4c/M4d se reportan solo si mejoran M3; si no, quedan como resultado negativo informativo.

El experimento no apoyaria la narrativa cold-start si:

1. M3 solo mejora en usuarios warm.
2. M2 explica todo el beneficio y M3 no agrega nada.
3. Las mejoras en `1-3` y `4-10` son numericamente pequenas y estadisticamente inciertas.
4. La cobertura cae demasiado, especialmente en CiaoDVD.

## 11. Artefactos a producir

Para que otro autor pueda reproducir y auditar el experimento, se deben generar:

```text
data/<dataset>/cold_start/
|-- split_manifest.json
|-- user_strata.csv
|-- train.csv
|-- test.csv
|-- cold_start_results.csv
|-- cold_start_user_deltas.csv
|-- bootstrap_confidence_intervals.csv
`-- README.md
```

Campos minimos de `user_strata.csv`:

| Campo | Descripcion |
| --- | --- |
| `user_id` | Identificador del usuario. |
| `n_train_ratings` | Numero de ratings visibles en train. |
| `n_test_ratings` | Numero de ratings usados para evaluar. |
| `stratum` | `0`, `1-3`, `4-10`, `>10`. |
| `appears_in_netinf_graph` | Si el usuario existe en la red inferida. |
| `has_centrality_features` | Si M2 puede construir atributos para el usuario. |
| `has_lph_features` | Si M3 puede construir atributos de comunidad/frontera. |
| `has_trust_features` | Solo CiaoDVD/Epinions; si existe en grafo de confianza. |

## 12. Comandos propuestos

Una interfaz deseable seria:

```bash
python -m recommender.experiment.cold_start \
  --dataset movielens \
  --split per_user_chrono \
  --strata 0 1-3 4-10 warm \
  --variants M1 M2 M3 M4c M4d \
  --seed 42 \
  --output-dir data/movielens/cold_start
```

Para CiaoDVD:

```bash
python -m recommender.experiment.cold_start \
  --dataset ciao \
  --split per_user_chrono \
  --strata 0 1-3 4-10 warm \
  --variants M1 M2 M3 M4c M4d M2_trust M3_trust \
  --seed 42 \
  --output-dir data/ciao/cold_start
```

Estas funciones aun no existen; son una propuesta de implementacion para que el experimento quede incorporado al pipeline.

## 13. Riesgos metodologicos

**Fuga de informacion.** El riesgo principal es calcular la red inferida, comunidades o LPH usando ratings que luego se usan como test. La regla debe ser estricta: todo atributo de usuario usado por el modelo debe calcularse solo con informacion disponible en train.

**Usuarios invisibles en NetInf.** Un usuario con `0` ratings en train no puede tener atributos NetInf. No se debe forzar un valor artificial de centralidad/LPH y venderlo como cold-start resuelto. Si se usa un valor default, debe reportarse como fallback.

**Estratos pequenos.** Si un estrato tiene pocos usuarios o pocos ratings, se debe reportar `N` y evitar conclusiones fuertes.

**Ranking en CiaoDVD.** CiaoDVD tiene catalogo grande y ranking muy disperso. Para cold-start, RMSE/MAE por estrato deben ser primarios; ranking solo secundario.

**Comparabilidad con el experimento core.** Si se cambia el split, los numeros no seran comparables uno a uno con las tablas core. Se debe reportar como experimento adicional, no como reemplazo.

## 14. Redaccion sugerida para el articulo

Si el experimento se ejecuta y M3 mejora en usuarios cold-start:

> The strongest gains of the boundary-aware attribute layer appear in users with limited training history. This supports the interpretation that LPH-derived community-boundary attributes act as useful side information when rating-derived user factors are underdetermined.

Si el experimento no muestra mejora clara:

> The cold-start stratification indicates that the current NetInf-derived attributes improve global rating prediction but do not yet solve user cold-start. This is expected for pure zero-rating users, who cannot appear in the inferred cascade network without external side information.

## 15. Decision editorial

Este experimento no debe formularse como "ya resolvimos cold-start". La formulacion correcta es:

> Proponemos una evaluacion cold-start para determinar si la capa de atributos de red, comunidad y frontera de MAFPIN-LPH aporta mas valor precisamente cuando el historial de ratings del usuario es escaso.

Si los resultados son positivos, se vuelve una contribucion fuerte del articulo. Si son mixtos, sigue siendo una buena seccion de limitaciones y trabajo futuro, especialmente para justificar el uso del trust graph en CiaoDVD.
