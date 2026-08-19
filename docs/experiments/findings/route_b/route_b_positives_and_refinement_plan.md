# Los positivos del proyecto y el plan para afinarlos

**Fecha:** 2026-08-19 (reconciliación post per-user)  
**Rama:** `feat/route-b-experiments`  
**Split canónico:** per-user chronological leave-last (`Split.STRATEGY=temporal`). El cutoff global previo dejó ~83 % cold-user en MovieLens y sesgó social/M4\* hacia “falla”; absolutos RMSE **no** se comparan entre protocolos.  
**Auditoría:** [experiment_validity_audit_2026-08-06.md](../../../reports/experiment_validity_audit_2026-08-06.md)  
**Core:** [MovieLens](../core_experiment/core_experiment_movielens_findings.md) · [Ciao](../core_experiment/core_experiment_ciao_findings.md)  
**Route B (ML fresco):** [WP1](route_b_wp1_findings.md) · [WP2](route_b_wp2_findings.md) · [WP3](route_b_wp3_findings.md)  
**Cierres auxiliares:** [trust A1](trust_consolidation_findings.md) · [M4c B1/B2](route_b_m4c_verification_findings.md) (multi-seed bajo criterio *pre* reformulación WP1)

---

## Foto honesta (2026-08-19)

La mayoría de hipótesis “fusión LPH → win claro en todo” siguen sin GO fuerte. Lo que **sí** cambió tras arreglar el split:

1. **Social RMSE revive** en core (ML ~2 % vs M1; Ciao ~0.3 %). M4c/M4d dejan de estar descartados por accuracy.
2. **Beyond-accuracy se reformula:** ya no hay un “único limpio M4c” multi-métrica. En ML per-user, **M3 gana CCE**; **M4c gana cobertura** (+ RMSE/NDCG OK). Soft assignment sigue **NO-GO**.
3. **Trust zero-shot (Ciao)** sigue siendo el efecto más grande y ya está **blindado multi-semilla** — pero **no** valida LPH.
4. **Ciao Route B (WP1–3)** aún no se regeneró bajo per-user; no mezclar tablas BA/estratos Ciao del cutoff global con ML fresco.
5. **Frente C (densificar)** sigue siendo la compuerta del paper de método; aún no corrida.

Un positivo sin verificación de robustez sigue siendo anécdota. Lo que sigue abajo fija claims vivos, lo cerrado, y el trabajo pendiente.

---

## Positivo 1 — Centralidad de confianza en zero-shot (Ciao) · lateral

**Estado:** **A1 PASS** → [trust_consolidation_findings.md](trust_consolidation_findings.md).

Usuarios del grafo trust sin ratings (estrato 0). M2_trust vs M1: ~10 % RMSE, Δ pooled ≈ 0.122 ± 0.002 en **10/10** semillas; CI bootstrap excluye 0 en todas. M3_trust **empeora** en las 10 → la señal es centralidad, no comunidades sobre trust.

**Framing:** auxiliar o paper corto trust. **No** es el pegamento narrativo MAFPIN+LPH (grafo explícito ≠ NetInf/LPH).

**Pendiente A:** A2 Epinions aparcado; A3 ablación solo si manuscript trust aparte.

---

## Positivo 2 — Social CMF baja RMSE (core, per-user) · hilo accuracy

**Estado:** fresco en core ML + ladder mínima Ciao (M1/M3/M4c/M4d). No sustituye Route B; cambia el prior del paper.

| Dataset | Winner(s) | Δ vs M1 (test) | vs M3 | Nota |
|---|---|---|---|---|
| MovieLens | M4d ≈ M4a ≈ M4c | ~−2.3 % | ~−0.5–0.6 % | M4c mejor `rmse_warm` entre M4; M3≈M2 |
| Ciao | **M4d ≥ M4c** | ~−0.3 % | ~−0.4 % | M3 **no** bate M1 |

**Lectura:** la regularización social (incluida boundary / bridge) **sí** transfiere a test cuando los usuarios son warm. Los atributos comunitarios solos (M3) son débiles o nulos. Esto **reabre** M4c/M4d como diseños de accuracy; no prueba by itself beyond-accuracy ni “LPH funciona”.

**Pendiente accuracy:** multi-semilla del ladder social (opcional, alineado con higiene del plan: ≥8/10) si se titula RMSE social; Ciao M2/M4a/M4b solo si se necesita ladder completa.

---

## Positivo 3 — Beyond-accuracy / frontera (Route B) · reformulado, débil

**Estado WP1–3 MovieLens (per-user):** ver findings enlazados. **Ciao Route B: pendiente.**

### Qué creíamos (cutoff global / plan 08-06)

M4c era el único punto “BA ↑ sin peaje” (NDCG + cobertura + CCE direccional).

### Qué hay ahora (ML per-user)

| Claim | Evidencia | Veredicto |
|---|---|---|
| M3 ↑ CCE@10 vs M1/M2, guardias OK | Bootstrap N=281, Holm limpio | **GO débil** (titular CCE) |
| M4c ↑ cobertura + RMSE↓ + NDCG OK | WP1 | **GO débil** (titular cobertura); **no** CCE↑ vs M3 |
| Una variante gana CCE∩cov∩NDCG | — | **No** |
| M4 social ayuda en B25 (y M4c en MID) | WP2 CI | **PARCIAL / débil**; no “solo frontera”; M3−M2 null |
| Estabilidad $\tilde{h}_v$ (α vecinos) | WP3 ρ≈0.865 | **SOPORTADA** (detector ASLPAw no corrido) |
| Soft assignment cold | deltas ~0–0.005 | **NO-GO** |
| M4c multi-seed BA (criterio viejo: NDCG∧cov ≥8/10) | [verificación](route_b_m4c_verification_findings.md) 7/10 | **NO-GO** como claim titular BA |

**B2 mecanismo (edge stats):** el regularizador `boundary_downweight` **sí** toca fronteras en el grafo (~25 % de aristas con overlap se des-ponderan). Correlación per-user ΔRMSE vs `n_communities` débil (N pequeño con comunidad). No fuerza renombrar a “selectivo incidental”; tampoco prueba causalidad del BA.

**Implicación para el paper de fusión:** no titular “M4c beyond-accuracy blindado”. Titulares candidatas, si se refuerzan: (i) **CCE vía M3**, (ii) **cobertura vía M4c**, (iii) **RMSE social M4\***, siempre con el alcance del dataset/split. WP4 **sigue cerrado** hasta un GO más limpio (idealmente post Frente C o Ciao Route B alineado).

---

## Mapa de claims (no son el mismo paper)

| Frente | Qué verifica | ¿Habla de LPH / fusión? | Estado |
|---|---|---|---|
| **A** trust zero-shot | Centralidad grafo **explícito** | **No** | **A1 cerrado PASS** |
| **B** BA / frontera / mecanismo | Beyond-accuracy + estratos + regularizador | **Sí** | WP1–3 ML hechos (débil); Ciao Ruta B **pendiente**; multi-seed M4c BA **FAIL** criterio viejo |
| **Acc** social RMSE (core) | M4\* vs M3/M1 | Parcial (social sí; LPH attributes no) | **Fresco**; multi-seed opcional |
| **C** densificar + constructo | ¿LPH puede operar con cobertura decente? | **Sí** — **compuerta** | **Pendiente** |

Reglas de higiene: [protocolo](route_b_protocol.md) §2 (pre-registro, artefactos core intactos salvo snapshot C, CIs bootstrap, Wilcoxon+Holm, todo se reporta). Semillas multi-seed: `42 7 123 2024 31337 101 999 12345 271828 314159` (criterio ≥ 8/10).

---

## Trabajo pendiente (alineado)

### 0. Higiene de narrativa (este doc)

Hecho aquí: Positivo 2 (BA M4c único) → **Positivo 2 accuracy social + Positivo 3 BA reformulado**; A1 y B1/B2 marcados con estado real; Ciao Route B y Frente C como siguientes.

### 1. Ciao Route B bajo per-user (bloquea réplica BA/estratos)

Tras prereqs + Phase 2 Ciao ya hechos para M1/M3/M4c/M4d:

1. WP1 `final_eval --beyond-accuracy` (mismas variantes que ML en la medida de lo posible; NDCG Ciao seguirá débil → lead CCE/cobertura/RMSE).
2. WP2 `boundary_strata` + cross-community (constructo ISO ya en código).
3. WP3 estabilidad α-vecinos si se quiere gate C con ambos datasets.
4. Actualizar `route_b_wp*_findings.md` secciones Ciao; no editar tablas ML.

Comandos: [route_b_commands.md](route_b_commands.md).

### 2. Frente C — compuerta de densificación (prioridad paper método)

WP3: con k≈2N mucha masa sin comunidad. Antes de “la fusión no funciona”, prueba justa.

**C1. Densificación.** Snapshot de `inferred_networks` / `centrality_metrics` / `communities` (k=2 congelado). Regenerar `--k-avg-degree 4` y `6` (`inference centrality communities` only). Restaurar k=2 para reproducir core.

**C2. Constructo frontera.** Estrato frontera: `num_communities ≥ 2`; usuarios 0 comunidades fuera de percentiles $\tilde{h}_v$ (ya parcialmente en `boundary_strata` / ISO). DEMON `seed=42`.

**C3. Gate (pre-registrado; no mirar y luego inventar):**

| Condición | Umbral |
|---|---|
| Cobertura comunitaria | ≥ 60 % usuarios con ≥ 1 comunidad, en ≥ 1 dataset |
| Población frontera | B25 con `num_communities ≥ 2` y N ≥ 30 |
| Estabilidad (WP3 rápido) | ρ Spearman α-vecinos ≥ 0.7 y Jaccard(B10) ≥ 0.5 |

- **Gate NO** en k=4 ni 6 → artículo diagnóstico (frontera no obtenible de forma segura en NetInf disperso).
- **Gate SÍ** → HP/α del core k=2 no aplican. Escalera corta M2/M3/M4c: hypertune → `recommend --all-networks` → `network_selection` → WP1+WP2. M1 canónico se reutiliza. Criterios protocolo §4.5 / §5.5. **Después** (fuera de fase): WP4 baselines / dataset grande.

**Salida C:** `route_b_gate_findings.md`; si cruza, `route_b_dense_wp1_wp2_findings.md`. B3 ranking sano (réplica BA) espera densificado ML — Ciao NDCG degenerado no sirve.

### 3. Re-pre-registro BA si se insiste en M4c titular

El FAIL de [route_b_m4c_verification_findings.md](route_b_m4c_verification_findings.md) aplica al criterio **NDCG∧cobertura ≥8/10** sobre el claim viejo. Tras WP1 per-user, opciones honestas:

- **No** re-titular M4c BA sin nuevo pre-registro.
- Si se pre-registra de nuevo: p.ej. (a) cobertura(M4c)>cobertura(M1) ≥8/10 **o** (b) CCE(M3)>CCE(M1) ≥8/10 — **una** métrica primaria, no el conjunción que ya falló.
- Correr multi-seed **después** del pre-registro, idealmente sobre artefactos densos (C) si el gate pasa.

### 4. No hacer todavía

- WP4 (baselines Cornac / Epinions / multi-dataset) — sin GO limpio post-C o réplica Ciao.
- A2 Epinions; A3 ablación trust (salvo paper trust).
- Soft assignment follow-ups.
- Mezclar cifras cutoff global con per-user en tablas publicables.

---

## Orden de decisión (actualizado)

1. **Ciao Route B (WP1–2, opcional WP3)** en paralelo o justo antes de C — cierra si el patrón M3-CCE / M4c-cov / social-RMSE se ve o se diluye fuera de ML.
2. **Frente C (C1–C3)** — compuerta del hilo fusión. Snapshot k=2 obligatorio.
3. **Sesión de decisión:** con Acc (core), B (WP ML+Ciao), gate C, y A1 (auxiliar) → paper de **método** (si C pasa y hay efecto), **diagnóstico** (si C falla), o **empírico honesto** (social RMSE + trust auxiliar + nulos BA documentados).
4. Trust **no** carga la tesis LPH. Ninguna cifra titular sin multi-semilla (10) cuando el claim es el headline.

Alcance de fase: **Ciao + MovieLens**. Regla de oro: umbrales antes de mirar; todo se reporta; core congelado no se pisa salvo snapshot explícito en C.
