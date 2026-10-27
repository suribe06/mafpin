# Route B next steps — hallazgos (Ciao per-user + Frente C densificación)

**Fecha:** 2026-10-08  
**Script:** `scripts/run_route_b_next_steps.sh all`  
**Log:** `data/logs/route_b_next_steps.nohup.log`  
**Wall-clock:** 13:27 → 15:31 (~2 h 4 min)  
**Stamp:** `20261008_132748`  
**Split:** per-user leave-last (canónico post-auditoría)  
**Plan:** [route_b_positives_and_refinement_plan.md](route_b_positives_and_refinement_plan.md)

**Alcance de esta corrida:** Ciao Route B WP1–3 + densificar k=4/6 (ML+Ciao) + gate + **restaurar redes k=2**.  
**No incluido:** retune ladder densa (`RUN_RETUNE`), WP4, multi-seed BA nuevo.

---

## Veredicto ejecutivo

| Pregunta | Respuesta |
|---|---|
| ¿Terminó limpio? | **Sí** (`FINISHED mode=all` 15:31:23); k=2 restaurado en ambos datasets |
| ¿Gate cobertura (≥60 % usuarios con ≥1 comunidad en ≥1 dataset)? | **PASS** vía **MovieLens** (k4 75 %, k6 90 %) |
| ¿Ciao llega a 60 %? | **No** (k4 40 %, k6 **58 %** — cerca, no cruza) |
| ¿Pool frontera (`num_communities≥2`, N≥30)? | **PASS** en las 4 celdas densas |
| ¿Estabilidad α-vecinos (ρ≥0.7, Jaccard B10≥0.5)? | **Ciao densa: fuerte PASS**; **ML k4: PASS**; **ML k6: mixto** (ρ media ~0.74 pero un vecino &lt;0.7; Jaccard media ~0.49) |
| ¿Ciao replica el patrón BA de ML (M3↑CCE, M4c↑cov)? | **No.** CCE ≈ flat; cobertura máxima en **M3**, no M4c |
| ¿WP2 frontera Ciao GO? | **No** — deltas M4\*/M3 vs M1 con CI que cruzan 0 |
| ¿Disparar retune denso overnight? | **Condicional:** sí solo en **MovieLens k=4** (cobertura+estabilidad más limpias). Ciao densificado **no** abre por sí solo el claim de fusión |

**Lectura paper:** densificar **sí** recupera sustrato comunitario usable en MovieLens. En Ciao la cobertura se acerca pero no cumple el umbral pre-registrado. El beyond-accuracy “M3=CCE / M4c=cobertura” **no se replica** en Ciao per-user → el hilo fusión sigue **débil cross-dataset**; lo más fuerte sigue siendo **RMSE social (core)** + **trust zero-shot (A1)**.

---

## 1. Ciao Route B (per-user)

Variantes frescas: **M1, M3, M4c, M4d** (manifest actual). Filas M2/M4a/M4b en CSV BA/preds son **stale** (cutoff viejo / N_CCE=32) — **ignorar** para claims.

### WP1 — Beyond-accuracy

Fuente: `data/ciao/route_b/beyond_accuracy_results.csv`

| Var | RMSE | NDCG@10 | CCE@10 | Coverage | N ranked | N CCE |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.9190 | 0.0036 | 0.439 | 0.0023 | 2248 | 464 |
| M3 | 0.9196 | 0.0022 | 0.437 | **0.0047** | 2248 | 464 |
| M4c | 0.9163 | 0.0012 | 0.436 | 0.0011 | 2248 | 464 |
| M4d | **0.9156** | 0.0013 | **0.443** | 0.0009 | 2248 | 464 |

Bootstrap pareado CCE/ILD/novelty: **vacío** (`n_users=0`) porque las comparaciones pre-registradas incluyen M2, ausente en el ladder fresco.

**Lectura:** NDCG sigue degenerado. CCE no separa M3 de M1. Cobertura: **M3 gana**, M4c/M4d pierden vs M1. Accuracy social (RMSE) sí favorece M4d/M4c — coherente con core Ciao, no con el titular BA de ML.

### WP2 — Estratos + cross-community

Fuente: `boundary_strata_*.csv`, `cross_community_items_results.csv`

Estratos poblados (M1/M3/M4c/M4d): B10=47, B25=70, MID=231, E75=116 ✅ (≥30).

Bootstrap (mean Δ; + ⇒ primera variante mejor en RMSE según convención del CSV):

| Estrato | M3−M1 | M4c−M3 | M4d−M3 |
|---|---|---|---|
| B10 | CI cruza 0 | cruza 0 | cruza 0 |
| B25 | cruza 0 | cruza 0 | cruza 0 |
| MID | ~−0.007 (borde) | cruza 0 | cruza 0 |
| E75 | cruza 0 | borde | borde |

Cross-community (n=273 ratings frescos): M1 0.944 &lt; M4c 0.947 &lt; M4d 0.949 &lt; M3 0.953 — **sin ventaja social** en ese nicho (contraste con ML per-user).

**Veredicto WP2 Ciao:** **NO-GO / nulo** para concentración frontera.

### WP3 — Estabilidad (k=2 restaurado)

Fuente: `data/ciao/route_b/community_stability.csv` (re-corrido post-restore)

| Eje | ρ media | Jaccard B10 media |
|---|---:|---:|
| α vecinos (DEMON) | **0.985** | **0.715** |

**SOPORTADA** (muy por encima del umbral). Estático: mean `#comunidades` ≈ 0.72; frac. frontera ≈ 0.21.

---

## 2. Frente C — densificación y gate

Snapshot k=2: `data/<ds>/route_b/dense_gate/k2_frozen_20261008_132748/`  
Métricas por k: `data/<ds>/route_b/dense_gate/k{4,6}/`  
Resumen: `data/route_b_dense_gate_summary_20261008_132748.csv`  
Redes canónicas **restauradas a k=2** al final.

### Cobertura comunitaria (red M3 congelada del core)

| Dataset | k | Frac. ≥1 com. | Frac. ≥2 com. | N≥2 | Cobertura≥60 % |
|---|---:|---:|---:|---:|---|
| MovieLens | 4 | **0.750** | 0.683 | 458 | **PASS** |
| MovieLens | 6 | **0.902** | 0.854 | 573 | **PASS** |
| Ciao | 4 | 0.403 | 0.377 | 847 | FAIL |
| Ciao | 6 | **0.581** | 0.567 | 1275 | FAIL (cerca) |

### Estabilidad WP3 sobre redes densas

| Dataset | k | ρ media α-vecinos | Jaccard B10 media | Lectura |
|---|---:|---:|---:|---|
| ML | 4 | 0.863 | 0.554 | interpretable |
| ML | 6 | 0.735 | 0.492 | ρ OK en media; Jaccard bajo el umbral 0.5 |
| Ciao | 4 | 0.961 | 0.733 | fuerte |
| Ciao | 6 | 0.935 | 0.744 | fuerte |

### Gate pre-registrado (plan)

| Condición | Umbral | Resultado |
|---|---|---|
| Cobertura | ≥60 % en ≥1 dataset | **PASS (ML)** |
| Pool frontera | N(`num_communities≥2`)≥30 | **PASS** (todas) |
| Estabilidad | ρ≥0.7 y Jaccard(B10)≥0.5 | **PASS Ciao**; **PASS ML k4**; **parcial ML k6** |

**Veredicto gate:** **PASS condicional en MovieLens densificado (preferir k=4 por estabilidad)**. Ciao densificado mejora cobertura (40→58 %) pero **no** abre el umbral; no usar Ciao k=6 como terreno “justo” para titular fusión aún.

---

## 3. Implicaciones y siguientes pasos

1. **No WP4** todavía (BA no limpio cross-dataset).
2. **Retune denso (opcional, overnight):** solo si se decide perseguir el paper de método en ML:
   ```bash
   RUN_RETUNE=1 RETUNE_DATASET=movielens RETUNE_K=4 \
     ./scripts/run_route_b_next_steps.sh retune-dense
   ```
   Luego: `import_manifest` (M2+M3+M4c juntos) → `network_selection` → WP1+WP2 sobre redes densas → `route_b_dense_wp1_wp2_findings.md`.
3. **Si no se retunea:** narrativa candidata = paper **empírico / diagnóstico**:
   - social RMSE revive bajo leave-last (core);
   - trust zero-shot fuerte (A1);
   - densificar desbloquea cobertura en ML pero **no** garantiza win BA de fusión en Ciao;
   - señales frontera / BA siguen débiles o dataset-específicas.
4. Actualizar a mano (si se quiere alinear docs antiguos): secciones Ciao en `route_b_wp{1,2,3}_findings.md` → apuntar aquí.

---

## 4. Artefactos clave

| Artefacto | Path |
|---|---|
| Log corrida | `data/logs/route_b_next_steps.nohup.log` |
| Gate summary | `data/route_b_dense_gate_summary_20261008_132748.csv` |
| Ciao BA / strata / WP3 | `data/ciao/route_b/` |
| Dense archives | `data/<ds>/route_b/dense_gate/k{4,6}/` |
| Backup pre-run | `data/ciao/route_b_backup_before_next_steps_20261008_132748/` |
| Core CSV backup | `data/*/core_experiment_results.pre_next_steps_20261008_132748.csv` |
