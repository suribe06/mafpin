# Comandos — Ruta B (+ soft assignment)

Protocolo: [route_b_protocol.md](route_b_protocol.md)  
Pre-registro: [route_b_preregistration.md](route_b_preregistration.md)  
Branch de implementación: `feat/route-b-experiments`

Ejecutar desde la raíz del repo:

```bash
cd /home/suribe06/Documents/Workspaces/GitHub/research/mafpin
conda run --no-capture-output -n mafpin ...
```

**Orden (prioridad):** pre-registro → **WP3 ∥ WP1** → **WP2** → gate → WP4 (solo si GO) → WP5 condicional. Soft assignment en paralelo o tras WP2.

Artefactos bajo `data/<dataset>/route_b/` (no pisan el core).

---

## 0 — Pre-registro

1. Edita `docs/experiments/route_b_preregistration.md`
2. Pon el hash: `git rev-parse HEAD`
3. No cambies umbrales después de ver resultados

---

## WP3 — Estabilidad de comunidades / LPH (barato, primero o en paralelo)

```bash
conda run --no-capture-output -n mafpin python scripts/route_b_wp3_community_stability.py \
  --dataset movielens --neighbors 2 --detectors demon aslpaw --seed 42

conda run --no-capture-output -n mafpin python scripts/route_b_wp3_community_stability.py \
  --dataset ciao --neighbors 2 --detectors demon aslpaw --seed 42
```

Salida: `data/<ds>/route_b/community_stability.csv`  
Criterio: media Spearman(α vecinos) ≥ 0.7 interpretable; &lt; 0.4 → ruido.

---

## WP1 — Beyond-accuracy (CCE, ILD latente, novelty, coverage, Gini)

Re-evalúa variantes congeladas del manifest. **Re-entrena** CMF (HP/red fijos).

Smoke (una variante):

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps final_eval --model-variant M3 --beyond-accuracy \
  --dataset movielens --seed 42 \
  --log-file data/movielens/logs/route_b/wp1_m3.log
```

Paper run:

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps final_eval --all-variants --beyond-accuracy \
  --dataset movielens --seed 42 \
  --log-file data/movielens/logs/route_b/wp1_final_eval.log

conda run --no-capture-output -n mafpin python pipeline.py \
  --steps final_eval --all-variants --beyond-accuracy \
  --dataset ciao --seed 42 \
  --log-file data/ciao/logs/route_b/wp1_final_eval.log
```

Salidas:

- `data/<ds>/route_b/beyond_accuracy_results.csv`
- `data/<ds>/route_b/beyond_accuracy_per_user.parquet` (o `.csv` si no hay engine parquet)

Nota: ILD **primaria por géneros** no está cableada (no hay `movies.csv` en el repo); se reporta `ild_latent` como proxy diagnóstico. CCE usa la partición DEMON congelada de **M3**.

---

## WP2 — Estratos frontera + predicciones

### 2a — Guardar predicciones per-rating

```bash
conda run --no-capture-output -n mafpin python pipeline.py \
  --steps final_eval --all-variants --save-predictions \
  --dataset movielens --seed 42 \
  --log-file data/movielens/logs/route_b/wp2_preds.log

conda run --no-capture-output -n mafpin python pipeline.py \
  --steps final_eval --all-variants --save-predictions \
  --dataset ciao --seed 42 \
  --log-file data/ciao/logs/route_b/wp2_preds.log
```

(Puedes combinar: `--beyond-accuracy --save-predictions` en una sola pasada.)

Predicciones: `data/<ds>/route_b/predictions/<variant>.parquet` (fallback `.csv`).

### 2b — Análisis por $\tilde{h}_v$ / cross-community

```bash
conda run --no-capture-output -n mafpin python -m recommender.experiment.route_b.boundary_strata \
  --dataset movielens --bootstrap-samples 1000 --seed 42

conda run --no-capture-output -n mafpin python -m recommender.experiment.route_b.boundary_strata \
  --dataset ciao --bootstrap-samples 1000 --seed 42
```

Salidas:

- `boundary_strata_results.csv`
- `boundary_strata_bootstrap.csv`
- `cross_community_items_results.csv`

---

## Soft assignment (M3_soft) — cold-start

Membresía blanda: overlap de ítems train del usuario con perfiles ítem-de-comunidad (usuarios warm en DEMON). Solo few-shot (`1–10` ratings); estrato `0` sigue en ceros.

Tras rebuild controlled (leave-k en MovieLens):

```bash
# MovieLens leave-k + soft
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset movielens --mode controlled --split leave_k \
  --variants M1 M2 M3 M3_soft \
  --seed 42 --bootstrap-samples 1000 \
  --output-dir data/movielens/cold_start

# Si el NetInf cold-start ya existe:
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset movielens --mode controlled --split leave_k --skip-rebuild \
  --variants M1 M2 M3 M3_soft \
  --seed 42 --output-dir data/movielens/cold_start

# Ciao leave-last + soft
conda run --no-capture-output -n mafpin python -m recommender.experiment.cold_start \
  --dataset ciao --mode controlled \
  --variants M1 M2 M3 M3_soft \
  --seed 42 --output-dir data/ciao/cold_start
```

`M3_soft` reutiliza HP + red de M3 del manifest; quita one-hots `community_*` duros y añade `soft_community_*`.

---

## WP4 — Solo si WP1 o WP2 dieron GO (aún no implementado en código)

Baselines Cornac + Epinions + multi-semilla: ver protocolo §7.  
No ejecutar hasta el gate documentado en `route_b_wp1_findings.md` / `route_b_wp2_findings.md`.

---

## WP5 — Condicional (aún no implementado)

LPH dirigida + LFR: protocolo §8. Disparadores: WP3 inestable o GO débil en WP1/WP2.

---

## Tests unitarios (rápido)

```bash
conda run --no-capture-output -n mafpin python -m unittest \
  tests.test_route_b tests.test_cold_start -v
```

---

## Checklist de veredictos

Tras WP1+WP2+WP3, escribir:

- `docs/experiments/route_b_wp1_findings.md`
- `docs/experiments/route_b_wp2_findings.md`
- `docs/experiments/route_b_wp3_findings.md`

y decidir GO / NO-GO antes de WP4.
