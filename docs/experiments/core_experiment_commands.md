# Comandos CLI del experimento principal (Fase 6)

Comandos exactos con `pipeline.py` para el plan en [core_experiment_plan.md](core_experiment_plan.md).
Sin scripts auxiliares: solo flags del CLI.
Referencia de flags: [cli/experiments.md](../cli/experiments.md), [cli/pipeline.md](../cli/pipeline.md).

Ejecutar desde la raíz del repo:

```bash
cd /home/suribe06/Documents/Workspaces/GitHub/research/mafpin
```

Prefijo recomendado (stream en terminal):

```bash
conda run --no-capture-output -n mafpin python pipeline.py ...
```

## Variantes → flags CLI

| ID | Comando base |
| --- | --- |
| M1 | Baseline: corre dentro de todo `--steps recommend` (no hay flag aparte) |
| M2 | `--no-communities` |
| M3 | sin flags extra (comunidades incluidas por defecto) |
| M4a | `--social-regularization --social-mode uniform` |
| M4b | `--social-regularization --social-mode community_jaccard` |
| M4c | `--social-regularization --social-mode boundary_downweight` |
| M4d | `--social-regularization --social-mode bridge_preserve` |

Flags comunes en casi todos los runs de recomendación:

```text
--cmf-method lbfgs --cmf-maxiter 25 --n-jobs 1 --seed 42
```

L-BFGS en `cmfrec` debe correr con **1 hilo BLAS por fit** (multi-hilo provoca segfault).
El pipeline lo fuerza automáticamente; usa `--n-jobs -1` si quieres paralelizar la evaluación
por red, no `--cmf-nthreads` alto.

Flags comunes en runs sociales (M4):

```text
--social-normalization mean_weight --social-search-max-ratings 0 --social-n-trials 200
```

## Trazabilidad (lo que ya da el CLI)

Cada invocación del pipeline:

1. **Log tee** — por defecto append a `data/<dataset>/pipeline.log`. Para no mezclar experimentos, pasa `--log-file` distinto en cada run (ver ejemplos).
2. **MLflow** — `recommend`, `hypertune` y `shap` registran params, métricas y JSON en `mlruns/`. Inspección:

   ```bash
   mlflow ui --backend-store-uri mlruns/
   ```

3. **Artefactos en disco** — escribe en `data/<dataset>/` (ver tabla al final). **Importante:** `baseline_search_results.json`, `enhanced_search_results.json` y `social_hyperparam_search_results.json` se sobrescriben en cada run del mismo dataset; el historial queda en MLflow y en cada `--log-file` que uses.

Monitoreo de un run en curso:

```bash
tail -f data/movielens/logs/m3_recommend.log
```

---

## 1. Prerrequisitos (una vez por dataset)

### MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps cascade delta inference centrality communities \
  --dataset movielens \
  --log-file data/movielens/logs/00_prerequisites.log
```

### Ciao

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps cascade delta inference centrality communities \
  --dataset ciao \
  --log-file data/ciao/logs/00_prerequisites.log
```

---

## 2. Pre-registro de redes (opcional)

### MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --preregister-networks \
  --dataset movielens \
  --seed 42 \
  --log-file data/movielens/logs/01_preregister.log
```

### Ciao

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --preregister-networks \
  --dataset ciao \
  --seed 42 \
  --log-file data/ciao/logs/01_preregister.log
```

Salida: `data/<dataset>/preregistered_network_sample.json`

---

## 3. Etapa A — solo búsqueda Optuna (`hypertune`)

Usa red representativa `exponential` índice 000. No evalúa las 300 redes.

### M1 — baseline

No hay paso `hypertune` para M1. La búsqueda baseline ocurre dentro de `recommend` (sección 4).

### M2 — MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps hypertune \
  --dataset movielens \
  --no-communities \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --seed 42 \
  --log-file data/movielens/logs/m2_hypertune.log
```

### M2 — Ciao

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps hypertune \
  --dataset ciao \
  --no-communities \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --seed 42 \
  --log-file data/ciao/logs/m2_hypertune.log
```

Salida: `data/<dataset>/enhanced_search_results.json`

### M3 — MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps hypertune \
  --dataset movielens \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --seed 42 \
  --log-file data/movielens/logs/m3_hypertune.log
```

### M3 — Ciao

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps hypertune \
  --dataset ciao \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --seed 42 \
  --log-file data/ciao/logs/m3_hypertune.log
```

### M4a — MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps hypertune \
  --dataset movielens \
  --social-regularization \
  --social-mode uniform \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --seed 42 \
  --log-file data/movielens/logs/m4a_hypertune.log
```

### M4b — MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps hypertune \
  --dataset movielens \
  --social-regularization \
  --social-mode community_jaccard \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --seed 42 \
  --log-file data/movielens/logs/m4b_hypertune.log
```

### M4c — MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps hypertune \
  --dataset movielens \
  --social-regularization \
  --social-mode boundary_downweight \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --seed 42 \
  --log-file data/movielens/logs/m4c_hypertune.log
```

### M4d — MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps hypertune \
  --dataset movielens \
  --social-regularization \
  --social-mode bridge_preserve \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --seed 42 \
  --log-file data/movielens/logs/m4d_hypertune.log
```

Repetir M4a–M4d cambiando `--dataset ciao` y `--log-file data/ciao/logs/m4*_hypertune.log`.

Salida: `data/<dataset>/social_hyperparam_search_results.json`

---

## 4. Etapa B — evaluación completa (`recommend --all-networks`)

Cada comando: búsqueda Optuna + baseline M1 en test global + evaluación en 300 redes.
Métricas por red en `data/<dataset>/inferred_networks/<model>/inferred_edges_<short>.csv`.

Orden sugerido del plan: **M3 y M4c primero**, luego ablaciones.

### 4.1 MovieLens — núcleo

**M3:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --all-networks \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/movielens/logs/m3_recommend.log
```

**M4c:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --all-networks \
  --social-regularization \
  --social-mode boundary_downweight \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/movielens/logs/m4c_recommend.log
```

M1: métricas baseline en el log y en `data/movielens/baseline_search_results.json` de cada run `recommend` (mismo archivo, último run gana en disco; ver MLflow para historial).

### 4.2 Ciao — núcleo

**M3:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset ciao \
  --all-networks \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/ciao/logs/m3_recommend.log
```

**M4c:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset ciao \
  --all-networks \
  --social-regularization \
  --social-mode boundary_downweight \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/ciao/logs/m4c_recommend.log
```

### 4.3 MovieLens — ablaciones

**M2:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --all-networks \
  --no-communities \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/movielens/logs/m2_recommend.log
```

**M4a:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --all-networks \
  --social-regularization \
  --social-mode uniform \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/movielens/logs/m4a_recommend.log
```

**M4b:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --all-networks \
  --social-regularization \
  --social-mode community_jaccard \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/movielens/logs/m4b_recommend.log
```

**M4d:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --all-networks \
  --social-regularization \
  --social-mode bridge_preserve \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/movielens/logs/m4d_recommend.log
```

### 4.4 Ciao — ablaciones

**M2:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset ciao \
  --all-networks \
  --no-communities \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/ciao/logs/m2_recommend.log
```

**M4a:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset ciao \
  --all-networks \
  --social-regularization \
  --social-mode uniform \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/ciao/logs/m4a_recommend.log
```

**M4b:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset ciao \
  --all-networks \
  --social-regularization \
  --social-mode community_jaccard \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/ciao/logs/m4b_recommend.log
```

**M4d:**

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset ciao \
  --all-networks \
  --social-regularization \
  --social-mode bridge_preserve \
  --social-normalization mean_weight \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/ciao/logs/m4d_recommend.log
```

---

## 5. Robustez — M4c con otra normalización

### MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --all-networks \
  --social-regularization \
  --social-mode boundary_downweight \
  --social-normalization normalized_laplacian \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/movielens/logs/m4c_robustness_laplacian.log
```

### Ciao

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset ciao \
  --all-networks \
  --social-regularization \
  --social-mode boundary_downweight \
  --social-normalization normalized_laplacian \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --seed 42 \
  --log-file data/ciao/logs/m4c_robustness_laplacian.log
```

---

## 6. Submuestra (si `--all-networks` es demasiado lento)

### MovieLens — 30 redes por modelo

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --sample-networks 30 \
  --social-regularization \
  --social-mode boundary_downweight \
  --social-normalization mean_weight \
  --social-n-trials 100 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1 \
  --log-file data/movielens/logs/m4c_sample30.log
```

Promover a `--all-networks` cuando el runtime lo permita (comando en §4.1 M4c).

---

## 7. SHAP (al final, modelos elegidos)

### M3 — MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps shap \
  --dataset movielens \
  --all-networks \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --seed 42 \
  --log-file data/movielens/logs/shap_m3.log
```

### M4c — MovieLens

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps shap \
  --dataset movielens \
  --all-networks \
  --social-regularization \
  --social-mode boundary_downweight \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --seed 42 \
  --log-file data/movielens/logs/shap_m4c.log
```

Repetir con `--dataset ciao` y logs en `data/ciao/logs/shap_*.log`.

Salida: `data/<dataset>/shap_results.json`

---

## 8. Salidas del CLI por paso

| Paso | Archivos principales |
| --- | --- |
| `cascade` … `communities` | `data/<dataset>/cascades.txt`, `inferred_networks/`, `centrality_metrics/`, `communities/`, `artifact_manifest.json` |
| `--preregister-networks` | `data/<dataset>/preregistered_network_sample.json` |
| `hypertune` (M2/M3) | `data/<dataset>/enhanced_search_results.json` |
| `hypertune` (M4) | `data/<dataset>/social_hyperparam_search_results.json` |
| `recommend` | `baseline_search_results.json`, `enhanced_search_results.json` o `social_hyperparam_search_results.json`, columnas `*_rmse_mean` / ranking en `inferred_networks/*/*.csv`, plots en `plots/<dataset>/` |
| `shap` | `data/<dataset>/shap_results.json` |
| Cualquier run | log en `--log-file` o `data/<dataset>/pipeline.log`; run en `mlruns/` |

## 9. Comparaciones del plan

```text
M2 vs M1  → logs m2_recommend vs baseline en cualquier recommend
M3 vs M2  → m3_recommend vs m2_recommend
M4a vs M3 → m4a_recommend vs m3_recommend
M4c vs M4a → m4c_recommend vs m4a_recommend
M4c vs M4b → m4c_recommend vs m4b_recommend
M4d vs M4c → m4d_recommend vs m4c_recommend
M4c vs M1  → m4c_recommend vs baseline en el mismo log/MLflow
```

Selección de `(diffusion_model, alpha_index)`: mejor `enhanced_rmse_mean` o `social_rmse_mean` en los CSV de `inferred_networks/`, solo sobre validación CV (no test global).

---

## 10. Fase 2 — evaluación correcta en test global (sin re-correr `--all-networks`)

Tras los runs de Etapa A/B, usa los **nuevos pasos** del pipeline para congelar hiperparámetros/redes y evaluar **una vez** en el test hold-out global.

### Artefactos nuevos

| Paso | Salida |
| --- | --- |
| `import_manifest` | `data/<dataset>/experiment_manifest.json` |
| `canonical_baseline` | `data/<dataset>/canonical_baseline.json` |
| `network_selection` | `data/<dataset>/network_selection_results.json` |
| `final_eval` | `data/<dataset>/core_experiment_results.csv` |

### MovieLens — secuencia completa (reutiliza logs existentes)

```bash
cd /home/suribe06/Documents/Workspaces/GitHub/research/mafpin

# 1) Importar hiperparámetros y mejores α por familia desde logs
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps import_manifest \
  --dataset movielens \
  --all-variants \
  --log-file data/movielens/logs/phase2_01_import_manifest.log

# 2) Baseline canónico M1 (una sola vez; reutilizado por todas las variantes)
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps canonical_baseline \
  --dataset movielens \
  --cmf-method lbfgs --cmf-maxiter 25 --seed 42 \
  --log-file data/movielens/logs/phase2_02_canonical_baseline.log

# 3) Congelar red (diffusion_model, alpha_index) por variante
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps network_selection \
  --dataset movielens \
  --all-variants \
  --log-file data/movielens/logs/phase2_03_network_selection.log

# 4) Evaluación final en test global — M1…M4d
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps final_eval \
  --dataset movielens \
  --all-variants \
  --cmf-method lbfgs --cmf-maxiter 25 --seed 42 \
  --log-file data/movielens/logs/phase2_04_final_eval.log
```

### Variante individual (p. ej. solo M4c)

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps import_manifest network_selection final_eval \
  --dataset movielens \
  --model-variant M4c \
  --cmf-method lbfgs --cmf-maxiter 25 --seed 42 \
  --log-file data/movielens/logs/phase2_m4c_final_eval.log
```

### Robustez M4c (Laplacian)

Incluida con `--all-variants` en `import_manifest` si existe `m4c_robustness_laplacian.log`:

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps import_manifest network_selection final_eval \
  --dataset movielens \
  --model-variant M4c_robustness \
  --cmf-method lbfgs --cmf-maxiter 25 --seed 42 \
  --log-file data/movielens/logs/phase2_m4c_robustness_final_eval.log
```

### Runs futuros de `recommend` — archivar por variante

Pasa `--run-id` para guardar snapshots bajo `data/<dataset>/runs/<run-id>/` (evita depender solo de CSVs compartidos):

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --all-networks --n-jobs 1 \
  --social-regularization --social-mode boundary_downweight \
  --social-normalization mean_weight --social-search-max-ratings 0 --social-n-trials 200 \
  --cmf-method lbfgs --cmf-maxiter 25 --seed 42 \
  --run-id m4c_recommend \
  --log-file data/movielens/logs/m4c_recommend.log
```

### Comparaciones válidas para el paper

Usa **`core_experiment_results.csv`** (columnas `rmse`, `mae`, `r2`, `ndcg_at_10`, `rmse_delta_vs_baseline`, `rmse_delta_vs_m3`):

```text
M4c vs M1  → rmse en core_experiment_results.csv (misma baseline canónica)
M3 vs M2   → idem
M4c vs M4a → idem
```

Los logs de Etapa B siguen siendo útiles para CV y diagnóstico; las **afirmaciones de test** deben salir de `final_eval`.
