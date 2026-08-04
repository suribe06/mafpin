# Comandos CLI del experimento cold-start

Comandos para el plan en [cold_start_experiment_proposal.md](cold_start_experiment_proposal.md).
Implementación: `python -m recommender.experiment.cold_start`.
**Findings (interpretación):** [cold_start_findings.md](cold_start_findings.md).

Ejecutar desde la raíz del repo:

```bash
cd /home/suribe06/Documents/Workspaces/GitHub/research/mafpin
```

Prefijo recomendado:

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start ...
```

## Prerrequisitos (experimento core)

El cold-start **reusa** hiperparámetros y la red seleccionada del core:

- `data/<dataset>/experiment_manifest.json`
- `data/<dataset>/canonical_baseline.json`
- Para `--mode diagnostic`: también las features NetInf en `data/<dataset>/centrality_metrics/` y `communities/`

Si faltan, corre primero la Fase 2 del core ([core_experiment_commands.md](core_experiment_commands.md)):

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps import_manifest canonical_baseline network_selection final_eval \
  --dataset movielens \
  --all-variants \
  --log-file data/movielens/logs/phase2_coldstart_prereq.log
```

(Repite con `--dataset ciao` si vas a evaluar Ciao.)

## Variantes

| Modo | Default variants |
| --- | --- |
| `diagnostic` / `controlled` | `M1 M2 M3 M4c M4d` |
| `zero_shot_trust` | `M1 M2_trust M3_trust` |

## Fase 1 — Diagnóstico (§6.1)

Usa el split global temporal 80/20 del core y agrupa usuarios por `n_train_ratings`.
**Limitación:** las features NetInf vienen del core (posible fuga relativa a etiquetas cold-start).

### MovieLens

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset movielens \
  --mode diagnostic \
  --variants M1 M2 M3 M4c M4d \
  --seed 42 \
  --bootstrap-samples 1000 \
  --output-dir data/movielens/cold_start
```

### Ciao

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset ciao \
  --mode diagnostic \
  --variants M1 M2 M3 M4c M4d \
  --seed 42 \
  --bootstrap-samples 1000 \
  --output-dir data/ciao/cold_start
```

Salidas: `user_strata.csv`, `cold_start_results.csv`, `cold_start_user_deltas.csv`,
`bootstrap_confidence_intervals.csv`, `split_manifest.json`, `success_summary.md`.

**Interpretación Fase 1:** el split global temporal suele dejar estratos `1-3`/`4-10`
casi vacíos en MovieLens (muchos usuarios solo en test = estrato `0`). Si
`success_summary.md` marca warnings de N bajo, **no uses H1/H2 de diagnostic como
evidencia**; pasa a Fase 2 (`controlled`).

Si re-corriste variantes parciales (p.ej. solo M1–M3), M4c/M4d pueden quedar de un
run anterior en el CSV — re-ejecuta con `--variants M1 M2 M3 M4c M4d` para alinear.

## Fase 2 — Split controlado anti-fuga (§6.2)

Dos protocolos de split:

| `--split` | Protocolo | Cuándo |
| --- | --- | --- |
| `leave_last` (default) | Por usuario, hold-out del último 20% chrono | Ciao (estratos cold naturales) |
| `leave_k` | Tras leave-last, cap de train en `{0, 2, 7, all}` (round-robin) | MovieLens (todos ≥20 ratings → leave-last solo da `>10`) |

Luego rebuild cascade → NetInf → centrality → communities bajo `data/<dataset>/cold_start/`.
Dedup de filas exactas `(UserId, ItemId, timestamp)` y tie-break estable en timestamps empatados.

**Aviso:** NetInf con `n_alphas=100` puede tardar horas. Empieza con un grid pequeño si solo quieres smoke-test.

**Importante:** pasa `--output-dir` relativo o absoluto; el código lo resuelve a ruta absoluta.
NetInf corre con `cwd=networks/`, así que cascadas relativas fallaban antes (FAILED rc=0 sin redes).

### MovieLens — leave-k (estratos cold artificiales)

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset movielens \
  --mode controlled \
  --split leave_k \
  --variants M1 M2 M3 M4c M4d \
  --seed 42 \
  --bootstrap-samples 1000 \
  --output-dir data/movielens/cold_start
```

Smoke (grid pequeño):

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset movielens \
  --mode controlled \
  --split leave_k \
  --variants M1 M2 M3 \
  --seed 42 \
  --n-alphas 5 \
  --max-iter 500 \
  --output-dir data/movielens/cold_start
```

### MovieLens / Ciao — leave-last (default)

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset movielens \
  --mode controlled \
  --variants M1 M2 M3 M4c M4d \
  --seed 42 \
  --bootstrap-samples 1000 \
  --output-dir data/movielens/cold_start
```

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset ciao \
  --mode controlled \
  --variants M1 M2 M3 M4c M4d \
  --seed 42 \
  --bootstrap-samples 1000 \
  --output-dir data/ciao/cold_start
```

Si el rebuild ya terminó y solo quieres re-evaluar modelos:

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset movielens \
  --mode controlled \
  --split leave_k \
  --skip-rebuild \
  --variants M1 M2 M3 M4c M4d \
  --seed 42 \
  --output-dir data/movielens/cold_start
```

> **Nota:** `--skip-rebuild` con leave-k solo es válido si el rebuild previo usó el
> **mismo** leave-k train. Un NetInf de leave-last no sirve para leave-k (fuga /
> features de ratings que leave-k dropea).
## Fase 3 — Cold-start puro 0-ratings (Ciao trust)

Usuarios en el grafo de confianza tienen **todos** sus ratings en test (`n_train=0`).
Atributos: trust centrality (M2_trust) + comunidades/frontera simples en trust (M3_trust).
**No** usa NetInf.

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset ciao \
  --mode zero_shot_trust \
  --variants M1 M2_trust M3_trust \
  --seed 42 \
  --bootstrap-samples 1000 \
  --output-dir data/ciao/cold_start
```

Artefactos en `data/ciao/cold_start/zero_shot_trust/`.

## Fase 4 — Reporte de criterios de éxito

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset movielens \
  --mode report \
  --output-dir data/movielens/cold_start
```

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset ciao \
  --mode report \
  --output-dir data/ciao/cold_start
```

Genera / refresca `success_summary.md` (y el de `zero_shot_trust/` si existe).

## Cómo leer H1–H4

| Hipótesis | Qué mirar |
| --- | --- |
| H1-gain | M3 mejora M1 en cold (`Δ>0` en `1-3`/`4-10`, N≥10) — criterio principal |
| H1-stronger | ganancia cold > ganancia warm; si FAIL con H1-gain PASS → beneficio general de side-info, no “más en cold” |
| H2 | M3 RMSE < M2 RMSE en estratos cold |
| H3 | M4c/M4d solo si mejoran M3 por estrato con CI no trivial |
| H4 | Track `zero_shot_trust`: cobertura `has_trust_features` y RMSE de M2_trust/M3_trust vs M1 en estrato `0` |

## Tabla de artefactos

| Path | Contenido |
| --- | --- |
| `data/<ds>/cold_start/split_manifest.json` | Modo, tamaños de split, red seleccionada |
| `data/<ds>/cold_start/user_strata.csv` | Estrato y cobertura de features por usuario |
| `data/<ds>/cold_start/train.csv` / `test.csv` | Split usado (audit / controlled) |
| `data/<ds>/cold_start/cold_start_results.csv` | RMSE/MAE por variante × estrato |
| `data/<ds>/cold_start/cold_start_user_deltas.csv` | RMSE por usuario + deltas |
| `data/<ds>/cold_start/bootstrap_confidence_intervals.csv` | CIs bootstrap de deltas |
| `data/<ds>/cold_start/success_summary.md` | Checks §10 |
| `data/<ds>/cold_start/inferred_networks/` | Solo modo `controlled` (rebuild) |
| `data/<ds>/cold_start/zero_shot_trust/` | Solo Ciao Fase 3 |

## Tests unitarios

```bash
conda run --no-capture-output -n mafpin python -m unittest \
  tests.test_cold_start tests.test_cold_start_trust -v
```
