#!/usr/bin/env bash
# =============================================================================
# Validation plan from docs/reports/lph_dense_network_decision_audit_2026-10-25.md
#
# Every output goes to data/validation_runs/<STAMP>/. Canonical artifacts
# (manifests, core CSVs, k=2 networks) are never modified permanently.
#
# Modes:
#   quick  (~20–60 min)  C3 diagnostic, bootstrap on saved predictions,
#                        frozen multi-seed refit, LPH column ablation.
#   dense  (~8–24 h)     MovieLens k=4: densify → recommend M2/M3/M4c over all
#                        networks → import_manifest → network_selection →
#                        final_eval (+BA, predictions) → WP2 → seeds/bootstrap.
#                        Runs in-place on data/movielens, then moves the dense
#                        state to the output dir and restores the k=2 snapshot.
#   all    quick, then dense.
#
# Usage:
#   ./scripts/run_validation_plan.sh quick
#   nohup ./scripts/run_validation_plan.sh dense > data/validation_dense.nohup.log 2>&1 &
#
# Env overrides: CONDA_ENV, N_JOBS, CPU_FRACTION, DENSE_K, SEEDS
#
# If a dense run is killed with SIGKILL (no trap), restore manually with the
# commands written to <OUT>/RESTORE_K2.txt before doing anything else.
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-mafpin}"
N_JOBS="${N_JOBS:--1}"
CPU_FRACTION="${CPU_FRACTION:-0.6}"
DENSE_K="${DENSE_K:-4}"
SEEDS="${SEEDS:-42 7 123 2024 31337}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MODE="${1:-quick}"
OUT="data/validation_runs/${STAMP}"
mkdir -p "$OUT/logs"

RUN() { conda run --no-capture-output -n "$CONDA_ENV" "$@"; }
CHECK() { RUN python scripts/validation_checks.py "$@"; }

PIPE() {
  local log="$1"; shift
  mkdir -p "$(dirname "$log")"
  echo ">>> [$(date -Iseconds)] pipeline.py --dataset movielens $*  (log: $log)"
  RUN python pipeline.py \
    --dataset movielens \
    --n-jobs "$N_JOBS" \
    --cpu-fraction "$CPU_FRACTION" \
    --cmf-method lbfgs \
    --cmf-maxiter 25 \
    --seed 42 \
    --log-file "$log" \
    "$@"
}

section() { echo; echo "=== § $1  ($(date -Iseconds))"; }

# -----------------------------------------------------------------------------
# quick
# -----------------------------------------------------------------------------
run_quick() {
  section "C3 — temporal look-ahead diagnostic"
  for ds in movielens ciao; do
    CHECK temporal-overlap --dataset "$ds" --out "$OUT/c3_temporal_${ds}.csv"
  done

  section "Paired bootstrap on existing k=2 predictions"
  CHECK prediction-bootstrap --dataset movielens \
    --pred-dir data/movielens/route_b/predictions \
    --variants M1 M2 M3 M4c M4d \
    --per-user-ba data/movielens/route_b/beyond_accuracy_per_user.parquet \
    --out "$OUT/bootstrap_movielens_k2.csv"
  # Ciao: only the per-user-fresh variants (M2/M4a/M4b files are stale).
  CHECK prediction-bootstrap --dataset ciao \
    --pred-dir data/ciao/route_b/predictions \
    --variants M1 M3 M4c M4d \
    --per-user-ba data/ciao/route_b/beyond_accuracy_per_user.parquet \
    --out "$OUT/bootstrap_ciao_k2.csv"

  section "Frozen multi-seed refit (HP + network from manifest, no retuning)"
  # shellcheck disable=SC2086
  CHECK frozen-seeds --dataset movielens --variants M1 M2 M3 M4c M4d \
    --seeds $SEEDS --out-dir "$OUT/seeds_movielens_k2" \
    2>&1 | tee "$OUT/logs/seeds_movielens.log"
  # shellcheck disable=SC2086
  CHECK frozen-seeds --dataset ciao --variants M1 M3 M4c M4d \
    --seeds $SEEDS --out-dir "$OUT/seeds_ciao_k2" \
    2>&1 | tee "$OUT/logs/seeds_ciao.log"

  section "LPH column ablation (frozen M3 HP + network)"
  for ds in movielens ciao; do
    CHECK lph-ablation --dataset "$ds" --seeds 42 7 123 \
      --out "$OUT/lph_ablation_${ds}.csv" \
      2>&1 | tee "$OUT/logs/ablation_${ds}.log"
  done
}

# -----------------------------------------------------------------------------
# dense (MovieLens k=DENSE_K)
# -----------------------------------------------------------------------------
SNAP="$OUT/movielens_k2_snapshot"
DENSE_STATE="$OUT/movielens_dense_k${DENSE_K}_state"
RESTORED=0

restore_k2() {
  [[ $RESTORED -eq 1 || ! -d "$SNAP" ]] && return 0
  echo "=== Restoring MovieLens k=2 from $SNAP"
  rm -rf "$DENSE_STATE"
  mv data/movielens "$DENSE_STATE"
  cp -a "$SNAP" data/movielens
  if [[ -d "$OUT/plots_movielens_k2" ]]; then
    rm -rf plots/movielens
    cp -a "$OUT/plots_movielens_k2" plots/movielens
  fi
  RESTORED=1
  echo "    dense state kept at $DENSE_STATE"
}

run_dense() {
  section "Dense — snapshot MovieLens k=2 (~345 MB)"
  cp -a data/movielens "$SNAP"
  [[ -d plots/movielens ]] && cp -a plots/movielens "$OUT/plots_movielens_k2"
  cat > "$OUT/RESTORE_K2.txt" <<EOF
cd $ROOT
rm -rf data/movielens && cp -a $SNAP data/movielens
rm -rf plots/movielens && cp -a $OUT/plots_movielens_k2 plots/movielens
EOF
  trap restore_k2 EXIT

  # Stale k=2 artifacts would otherwise leak into manifest/selection/BA:
  # import_manifest parses logs/m*_recommend.log (appended, first match wins),
  # network_selection prefers runs/<run_id>/, BA/WP2 upsert into route_b/.
  rm -f data/movielens/logs/m*_recommend*.log data/movielens/logs/m*_hypertune.log
  rm -rf data/movielens/runs data/movielens/route_b
  rm -f data/movielens/experiment_manifest.json \
        data/movielens/network_selection_results.json \
        data/movielens/core_experiment_results.csv

  local logs="data/movielens/logs"
  section "Dense — NetInf k=${DENSE_K} + communities (fixed h) + centrality"
  # communities before centrality: pagerank_lph reads the community CSVs.
  PIPE "$logs/dense_00_prereq.log" \
    --steps inference communities centrality --k-avg-degree "$DENSE_K"

  section "Dense — recommend M2 / M3 / M4c over all networks (overnight)"
  PIPE "$logs/m2_recommend.log" --steps recommend --all-networks \
    --no-communities --run-id m2_recommend
  PIPE "$logs/m3_recommend.log" --steps recommend --all-networks \
    --run-id m3_recommend
  PIPE "$logs/m4c_recommend.log" --steps recommend --all-networks \
    --social-regularization --social-mode boundary_downweight \
    --social-normalization mean_weight \
    --social-search-max-ratings 0 --social-n-trials 200 \
    --run-id m4c_recommend

  section "Dense — Phase 2 (M1 reuses canonical_baseline.json)"
  PIPE "$logs/dense_01_import_manifest.log" --steps import_manifest
  PIPE "$logs/dense_02_network_selection.log" --steps network_selection --all-variants
  PIPE "$logs/dense_03_final_eval.log" --steps final_eval --all-variants \
    --beyond-accuracy --save-predictions

  section "Dense — WP2 strata + bootstrap + frozen seeds"
  RUN python -m recommender.experiment.route_b.boundary_strata \
    --dataset movielens --bootstrap-samples 1000 --seed 42 \
    2>&1 | tee "$logs/dense_04_wp2.log"
  CHECK prediction-bootstrap --dataset movielens \
    --pred-dir data/movielens/route_b/predictions \
    --variants M1 M2 M3 M4c \
    --per-user-ba data/movielens/route_b/beyond_accuracy_per_user.parquet \
    --out "$OUT/bootstrap_movielens_dense_k${DENSE_K}.csv"
  # shellcheck disable=SC2086
  CHECK frozen-seeds --dataset movielens --variants M1 M2 M3 M4c \
    --seeds $SEEDS --out-dir "$OUT/seeds_movielens_dense_k${DENSE_K}"

  cp data/movielens/core_experiment_results.csv "$OUT/core_dense_k${DENSE_K}.csv"
  cp data/movielens/network_selection_results.json \
     "$OUT/network_selection_dense_k${DENSE_K}.json"

  restore_k2
  trap - EXIT

  # k=2 seeds from the latest `quick` run; seed variance (~0.003) is as large
  # as the effects, so the decision uses seed means, not seed 42 alone.
  local k2_seeds="${K2_SEEDS:-$(ls -d data/validation_runs/*/seeds_movielens_k2 2>/dev/null | sort | tail -1)}"

  section "Dense vs k=2 (test RMSE, same split, M1 shared)"
  RUN python - "$OUT/core_dense_k${DENSE_K}.csv" "$SNAP/core_experiment_results.csv" \
    "$OUT/seeds_movielens_dense_k${DENSE_K}/frozen_seeds_raw.csv" \
    "${k2_seeds}/frozen_seeds_raw.csv" <<'PY'
import sys
from pathlib import Path
import pandas as pd
cols = ["model_variant", "rmse", "rmse_warm", "diffusion_model", "alpha_index"]
dense = pd.read_csv(sys.argv[1])[cols].set_index("model_variant")
k2 = pd.read_csv(sys.argv[2])[cols].set_index("model_variant")
table = dense.join(k2, lsuffix="_dense", rsuffix="_k2", how="left")
table["delta_rmse"] = table["rmse_dense"] - table["rmse_k2"]
print("Seed 42 (core):")
print(table.to_string())

dense_seeds, k2_seeds = Path(sys.argv[3]), Path(sys.argv[4])
if dense_seeds.exists() and k2_seeds.exists():
    stats = {
        name: pd.read_csv(p).groupby("model_variant")["rmse"].agg(["mean", "std"])
        for name, p in (("dense", dense_seeds), ("k2", k2_seeds))
    }
    seeds = stats["dense"].join(stats["k2"], lsuffix="_dense", rsuffix="_k2", how="left")
    seeds["delta_mean"] = seeds["mean_dense"] - seeds["mean_k2"]
    print("\nSeed means (5 seeds each):")
    print(seeds.to_string())
else:
    print(f"\nSeed comparison skipped (missing {dense_seeds} or {k2_seeds})")
PY
}

case "$MODE" in
  quick) run_quick ;;
  dense) run_dense ;;
  all)   run_quick; run_dense ;;
  *) echo "Usage: $0 {quick|dense|all}" >&2; exit 1 ;;
esac

echo
echo "FINISHED mode=$MODE  outputs: $OUT"
