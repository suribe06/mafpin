#!/usr/bin/env bash
# =============================================================================
# Route B — siguientes pasos (Ciao WP1–3 + Frente C densificación / gate)
#
# Plan: docs/experiments/findings/route_b/route_b_positives_and_refinement_plan.md
# Comandos base: docs/experiments/findings/route_b/route_b_commands.md
#
# Máquina de referencia (estimaciones): 16 cores, ~60 GB RAM, conda env `mafpin`,
#   --n-jobs -1 --cpu-fraction 0.6  → ~9 workers NetInf/paralelismo de redes.
#   L-BFGS CMF sigue en 1 hilo BLAS (cmfrec); el paralelismo no acelera final_eval
#   dentro de una variante, solo NetInf / sweeps de redes.
#
# TIEMPOS ESTIMADOS (wall-clock, no dejar la máquina suspendida):
#   §1  Backup + higiene                         ~2–5 min
#   §2  Ciao Route B WP1 (BA + preds)            ~25–45 min
#   §3  Ciao Route B WP2 (boundary strata)       ~5–15 min
#   §4  Ciao Route B WP3 (estabilidad DEMON)     ~5–15 min
#   §5  Frente C snapshot k=2 (ML + Ciao)        ~2–5 min
#   §6  Densificar MovieLens k=4 y k=6           ~45–90 min  (k=2 baseline ~7–15 min)
#   §7  Densificar Ciao k=4 y k=6                ~45–90 min  (k=2 baseline ~15 min)
#   §8  Gate summary + restaurar k=2             ~2–5 min
#   ─────────────────────────────────────────────
#   TOTAL (fase ejecutable, SIN retune post-gate) ~2.5–4.5 h
#   Típico en esta máquina:                      ~3–3.5 h
#
#   §9  OPCIONAL — retune ladder densa (OFF)     +8–24 h+  (NO incluido en el total)
#       Solo si el gate pasa; hypertune + recommend --all-networks M2/M3/M4c.
#
# Uso:
#   chmod +x scripts/run_route_b_next_steps.sh
#   # noche / bloque de ~4 h:
#   nohup ./scripts/run_route_b_next_steps.sh all > data/logs/route_b_next_steps.nohup.log 2>&1 &
#
#   ./scripts/run_route_b_next_steps.sh ciao-rb      # solo §1–4
#   ./scripts/run_route_b_next_steps.sh frente-c     # solo §5–8 (pisa redes in-place; restaura k=2)
#   ./scripts/run_route_b_next_steps.sh all
#   RUN_RETUNE=1 ./scripts/run_route_b_next_steps.sh retune-dense  # solo tras GO manual
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-mafpin}"
N_JOBS="${N_JOBS:--1}"
CPU_FRACTION="${CPU_FRACTION:-0.6}"
SEED="${SEED:-42}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MODE="${1:-all}"

RUN() {
  # shellcheck disable=SC2086
  conda run --no-capture-output -n "$CONDA_ENV" "$@"
}

PIPE() {
  local ds="$1"; shift
  local log="$1"; shift
  mkdir -p "$(dirname "$log")"
  echo ""
  echo ">>> [$(date -Iseconds)] pipeline.py --dataset $ds $*"
  echo "    log: $log"
  RUN python pipeline.py \
    --dataset "$ds" \
    --n-jobs "$N_JOBS" \
    --cpu-fraction "$CPU_FRACTION" \
    --seed "$SEED" \
    --log-file "$log" \
    "$@"
}

section() {
  local name="$1" est="$2"
  echo ""
  echo "================================================================"
  echo "§ $name"
  echo "  ETA: $est"
  echo "  start: $(date -Iseconds)"
  echo "================================================================"
  SECTION_T0=$(date +%s)
}

section_done() {
  local t1
  t1=$(date +%s)
  echo "  end: $(date -Iseconds)  (wall $((t1 - SECTION_T0))s)"
}

backup_tree() {
  local src="$1" dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst"
    cp -a "$src" "$dst"
    echo "  backup: $src → $dst"
  else
    echo "  skip backup (missing): $src"
  fi
}

# Coverage / frontera gate on the frozen M3 community CSV (current in-place nets).
gate_metrics() {
  local ds="$1" tag="$2" out="$3"
  RUN python - <<PY
import json
from pathlib import Path
import pandas as pd

ds = "$ds"
tag = "$tag"
out = Path("$out")
root = Path("$ROOT")
ns = json.loads((root / f"data/{ds}/network_selection_results.json").read_text())
m3 = ns["variants"]["M3"]
diff = m3["diffusion_model"]
idx = int(m3["alpha_index"])
com_path = root / f"data/{ds}/communities/{diff}/communities_{diff}_{idx:03d}.csv"
df = pd.read_csv(com_path)
n = len(df)
cov1 = float((df["num_communities"] >= 1).mean()) if n else float("nan")
cov2 = float((df["num_communities"] >= 2).mean()) if n else float("nan")
n_b25_eligible = int((df["num_communities"] >= 2).sum())
row = {
    "dataset": ds,
    "tag": tag,
    "diffusion_model": diff,
    "alpha_index": idx,
    "n_users": n,
    "frac_users_ge1_community": cov1,
    "frac_users_ge2_community": cov2,
    "n_users_ge2_community": n_b25_eligible,
    "gate_coverage_ge_0.60": bool(cov1 >= 0.60),
    "gate_frontier_pool_ge_30": bool(n_b25_eligible >= 30),
    "communities_csv": str(com_path),
}
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([row]).to_csv(out, index=False)
print(json.dumps(row, indent=2))
PY
}

snapshot_k2() {
  local ds="$1"
  local snap="data/${ds}/route_b/dense_gate/k2_frozen_${STAMP}"
  mkdir -p "$snap"
  for d in inferred_networks centrality_metrics communities; do
    backup_tree "data/${ds}/${d}" "${snap}/${d}"
  done
  echo "$snap" > "data/${ds}/route_b/dense_gate/LATEST_K2_SNAPSHOT.txt"
  echo "  k2 snapshot root: $snap"
}

restore_k2() {
  local ds="$1"
  local ptr="data/${ds}/route_b/dense_gate/LATEST_K2_SNAPSHOT.txt"
  if [[ ! -f "$ptr" ]]; then
    echo "ERROR: no snapshot pointer $ptr" >&2
    exit 1
  fi
  local snap
  snap="$(cat "$ptr")"
  echo "  restoring k=2 for $ds from $snap"
  for d in inferred_networks centrality_metrics communities; do
    rm -rf "data/${ds}/${d}"
    cp -a "${snap}/${d}" "data/${ds}/${d}"
  done
}

densify_one() {
  local ds="$1" k="$2"
  local tag="k${k}"
  local outdir="data/${ds}/route_b/dense_gate/${tag}"
  mkdir -p "$outdir" "data/${ds}/logs/route_b"

  section "Densify ${ds} k=${k} (inference→communities)" "~20–45 min (más si k=6)"
  PIPE "$ds" "data/${ds}/logs/route_b/c1_density_${tag}.log" \
    --steps inference centrality communities \
    --k-avg-degree "$k"
  section_done

  gate_metrics "$ds" "$tag" "${outdir}/gate_metrics.csv"
  # Archive community/centrality snapshot for this k (paths will be overwritten next)
  backup_tree "data/${ds}/communities" "${outdir}/communities"
  backup_tree "data/${ds}/centrality_metrics" "${outdir}/centrality_metrics"
  # Networks are large; keep a pointer + optional shallow list
  ls "data/${ds}/inferred_networks" > "${outdir}/inferred_networks_listing.txt" || true

  section "WP3 stability on ${ds} @ ${tag}" "~5–15 min"
  # ASLPAw often missing deps; DEMON only (same as ML per-user re-run)
  RUN python scripts/route_b_wp3_community_stability.py \
    --dataset "$ds" --neighbors 2 --detectors demon --seed "$SEED" \
    2>&1 | tee "data/${ds}/logs/route_b/wp3_${tag}.log"
  cp -f "data/${ds}/route_b/community_stability.csv" \
        "${outdir}/community_stability.csv"
  section_done
}

run_backup() {
  section "Backup artefactos que se van a regenerar" "2–5 min"
  mkdir -p data/ciao/route_b_backup_before_next_steps_${STAMP}
  mkdir -p data/logs
  # Ciao Route B (tablas cutoff global / previas)
  for f in beyond_accuracy_results.csv beyond_accuracy_per_user.parquet \
           beyond_accuracy_bootstrap.csv boundary_strata_results.csv \
           boundary_strata_bootstrap.csv cross_community_items_results.csv \
           community_stability.csv; do
    if [[ -e "data/ciao/route_b/$f" ]]; then
      cp -a "data/ciao/route_b/$f" \
        "data/ciao/route_b_backup_before_next_steps_${STAMP}/$f"
    fi
  done
  if [[ -d data/ciao/route_b/predictions ]]; then
    cp -a data/ciao/route_b/predictions \
      "data/ciao/route_b_backup_before_next_steps_${STAMP}/predictions"
  fi
  # Core CSV (final_eval hace upsert por variant y pisa filas)
  cp -a data/ciao/core_experiment_results.csv \
    "data/ciao/core_experiment_results.pre_next_steps_${STAMP}.csv"
  cp -a data/movielens/core_experiment_results.csv \
    "data/movielens/core_experiment_results.pre_next_steps_${STAMP}.csv" 2>/dev/null || true
  section_done
}

run_ciao_rb() {
  # Manifest Ciao fresco solo tiene M3/M4c/M4d; M1 usa canonical_baseline.
  # NO usar --all-variants (intentaría M2/M4a/M4b y dejaría filas BA viejas confusas).
  local variants=(M1 M3 M4c M4d)

  section "Ciao WP1 — final_eval + beyond-accuracy + predictions" "25–45 min"
  # Una sola invocación por variante → upsert BA acumula; luego restore core CSV.
  for v in "${variants[@]}"; do
    PIPE ciao "data/ciao/logs/route_b/wp1_${v}_per_user.log" \
      --steps final_eval \
      --model-variant "$v" \
      --beyond-accuracy \
      --save-predictions
  done
  # Restaurar filas core pre-WP1 (WP1 no debe ser la fuente canónica del core)
  if [[ -f "data/ciao/core_experiment_results.pre_next_steps_${STAMP}.csv" ]]; then
    cp -a "data/ciao/core_experiment_results.pre_next_steps_${STAMP}.csv" \
      data/ciao/core_experiment_results.csv
    echo "  restored ciao core_experiment_results.csv from pre-WP1 backup"
  fi
  section_done

  section "Ciao WP2 — boundary strata + cross-community" "5–15 min"
  RUN python -m recommender.experiment.route_b.boundary_strata \
    --dataset ciao --bootstrap-samples 1000 --seed "$SEED" \
    2>&1 | tee data/ciao/logs/route_b/wp2_boundary_strata_per_user.log
  section_done

  section "Ciao WP3 — community stability (DEMON)" "5–15 min"
  RUN python scripts/route_b_wp3_community_stability.py \
    --dataset ciao --neighbors 2 --detectors demon --seed "$SEED" \
    2>&1 | tee data/ciao/logs/route_b/wp3_community_stability_per_user.log
  section_done

  echo ""
  echo "Ciao Route B listo. Revisar:"
  echo "  data/ciao/route_b/beyond_accuracy_results.csv"
  echo "  data/ciao/route_b/boundary_strata_*.csv"
  echo "  data/ciao/route_b/community_stability.csv"
  echo "  Luego actualizar docs/experiments/findings/route_b/route_b_wp{1,2,3}_findings.md (sección Ciao)."
}

run_frente_c() {
  section "Frente C — snapshot k=2 (ML + Ciao)" "2–5 min"
  mkdir -p data/movielens/route_b/dense_gate data/ciao/route_b/dense_gate
  snapshot_k2 movielens
  snapshot_k2 ciao
  section_done

  # Por dataset: k=4 luego k=6 (in-place), archivando métricas de gate en route_b/dense_gate/
  densify_one movielens 4
  densify_one movielens 6
  densify_one ciao 4
  densify_one ciao 6

  section "Gate summary + restaurar redes k=2" "2–5 min"
  restore_k2 movielens
  restore_k2 ciao

  local summary="data/route_b_dense_gate_summary_${STAMP}.csv"
  RUN python - <<PY
from pathlib import Path
import pandas as pd
root = Path("$ROOT")
frames = []
for ds in ("movielens", "ciao"):
    for k in (4, 6):
        p = root / f"data/{ds}/route_b/dense_gate/k{k}/gate_metrics.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
if not frames:
    raise SystemExit("no gate_metrics.csv found")
out = pd.concat(frames, ignore_index=True)
out_path = root / "$summary"
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
print("wrote", out_path)
# Pre-registered gate (plan): coverage>=0.60 on >=1 dataset; frontier pool>=30
any_cov = bool(out["gate_coverage_ge_0.60"].any())
print()
print("GATE coverage (>=60% users with >=1 community on >=1 row):", "PASS" if any_cov else "FAIL")
print("Inspect frontier_pool_ge_30 and WP3 ρ in each dense_gate/k*/community_stability.csv")
print("If PASS → decide manually before RUN_RETUNE=1 (overnight). If FAIL → diagnostic paper path.")
PY
  section_done
}

run_retune_dense() {
  # DISABLED by default. Enormous. Only after reading gate summary + human GO.
  if [[ "${RUN_RETUNE:-0}" != "1" ]]; then
    echo "retune-dense requires RUN_RETUNE=1 (and a densify snapshot you intend to use)."
    echo "This step is overnight (8–24h+). Not part of the default ~3h budget."
    exit 1
  fi
  local ds="${RETUNE_DATASET:-movielens}"
  local k="${RETUNE_K:-4}"
  echo "WARNING: will densify $ds to k=$k in-place, then Stage B ladder M2/M3/M4c."
  echo "Core k=2 must already be snapshotted. Ctrl+C within 10s to abort."
  sleep 10

  snapshot_k2 "$ds"
  densify_one "$ds" "$k"

  section "Retune hypertune M2/M3/M4c on dense $ds k=$k" "several hours"
  for v in M2 M3 M4c; do
    PIPE "$ds" "data/${ds}/logs/route_b/dense_k${k}_${v}_recommend.log" \
      --steps recommend --model-variant "$v" --all-networks
  done
  section_done

  echo "Next (manual): import_manifest (all three in one call) → network_selection → WP1+WP2."
  echo "Then restore k=2: snapshot listed in data/${ds}/route_b/dense_gate/LATEST_K2_SNAPSHOT.txt"
}

print_banner() {
  cat <<EOF
================================================================
Route B next steps  mode=$MODE  stamp=$STAMP
env=$CONDA_ENV  n_jobs=$N_JOBS  cpu_fraction=$CPU_FRACTION  seed=$SEED
repo=$ROOT

ETA total (all, sin retune): ~2.5–4.5 h  |  típico ~3–3.5 h
  Arranca cuando tengas un bloque de ~4 h (o overnight con nohup).
================================================================
EOF
}

print_banner

case "$MODE" in
  ciao-rb)
    run_backup
    run_ciao_rb
    ;;
  frente-c)
    run_backup
    run_frente_c
    ;;
  all)
    run_backup
    run_ciao_rb
    run_frente_c
    ;;
  retune-dense)
    run_retune_dense
    ;;
  *)
    echo "Usage: $0 {all|ciao-rb|frente-c|retune-dense}" >&2
    exit 2
    ;;
esac

echo ""
echo "================================================================"
echo "FINISHED mode=$MODE at $(date -Iseconds)"
echo "Hallazgos docs: aún manuales (wp*_findings.md, route_b_gate_findings.md)."
echo "================================================================"
