#!/usr/bin/env bash
# Core experiment batch — any dataset (full ladder + Phase 2 final_eval).
# Source: docs/core_experiment_commands.md
#
# Usage:
#   ./scripts/run_core_experiment.sh --dataset ciao
#   ./scripts/run_core_experiment.sh --dataset movielens
#   nohup ./scripts/run_core_experiment.sh --dataset ciao &
#
# Continues on failure (set +e). Review when done:
#   column -t -s $'\t' data/<dataset>/logs/run_summary.tsv
#   awk -F'\t' 'NR>1 && $2!="0"{print}' data/<dataset>/logs/run_summary.tsv

set +e
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

VALID_DATASETS=(movielens ciao epinions)
DATASET=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --dataset DATASET

Run the full core experiment ladder for one dataset:
  prerequisites → preregister → Stage A (hypertune) → Stage B (recommend)
  → Phase 2 (import_manifest, canonical_baseline, network_selection, final_eval)

Options:
  --dataset DATASET   Required. One of: ${VALID_DATASETS[*]}
  -h, --help          Show this help

Examples:
  $(basename "$0") --dataset ciao
  nohup $(basename "$0") --dataset movielens &

Logs:  data/<dataset>/logs/
Output: data/<dataset>/core_experiment_results.csv

See docs/core_experiment_commands.md for individual step commands.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "Error: --dataset is required." >&2
  usage >&2
  exit 1
fi

valid=0
for d in "${VALID_DATASETS[@]}"; do
  if [[ "$DATASET" == "$d" ]]; then
    valid=1
    break
  fi
done
if [[ $valid -eq 0 ]]; then
  echo "Error: invalid dataset '$DATASET'. Choose: ${VALID_DATASETS[*]}" >&2
  exit 1
fi

LOG_DIR="data/${DATASET}/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/run_all_${STAMP}.log"
SUMMARY="${LOG_DIR}/run_summary.tsv"

echo -e "step\texit_code\tfinished_at" >"$SUMMARY"

PIPE=(conda run --no-capture-output -n mafpin python pipeline.py)

CMF_FLAGS=(--cmf-method lbfgs --cmf-maxiter 25 --seed 42)
SOCIAL_FLAGS=(
  --social-normalization mean_weight
  --social-search-max-ratings 0
  --social-n-trials 200
)
RECOMMEND_FLAGS=(--all-networks --n-jobs 1 "${CMF_FLAGS[@]}")

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$MASTER_LOG"
}

run_step() {
  local name="$1"
  shift
  log "START $name"
  log "CMD: $*"
  "$@" >>"$MASTER_LOG" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    log "OK   $name (exit=0)"
  else
    log "FAIL $name (exit=$rc) — continuing"
  fi
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(date -Iseconds)" >>"$SUMMARY"
  return 0
}

log "=== Core experiment batch started (dataset=$DATASET) ==="
log "ROOT=$ROOT"
log "MASTER_LOG=$MASTER_LOG"
log "SUMMARY=$SUMMARY"

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
run_step prerequisites "${PIPE[@]}" \
  --steps cascade delta inference centrality communities \
  --dataset "$DATASET" \
  --log-file "${LOG_DIR}/00_prerequisites.log"

# ---------------------------------------------------------------------------
# 2. Pre-register networks (optional; cheap)
# ---------------------------------------------------------------------------
run_step preregister "${PIPE[@]}" \
  --preregister-networks \
  --dataset "$DATASET" \
  --seed 42 \
  --log-file "${LOG_DIR}/01_preregister.log"

# ---------------------------------------------------------------------------
# 3. Stage A — hypertune (representative network exponential_000)
# M1: no hypertune step (baseline search runs inside recommend)
# ---------------------------------------------------------------------------
run_step m2_hypertune "${PIPE[@]}" \
  --steps hypertune \
  --dataset "$DATASET" \
  --no-communities \
  "${CMF_FLAGS[@]}" \
  --log-file "${LOG_DIR}/m2_hypertune.log"

run_step m3_hypertune "${PIPE[@]}" \
  --steps hypertune \
  --dataset "$DATASET" \
  "${CMF_FLAGS[@]}" \
  --log-file "${LOG_DIR}/m3_hypertune.log"

run_step m4a_hypertune "${PIPE[@]}" \
  --steps hypertune \
  --dataset "$DATASET" \
  --social-regularization \
  --social-mode uniform \
  "${SOCIAL_FLAGS[@]}" \
  "${CMF_FLAGS[@]}" \
  --log-file "${LOG_DIR}/m4a_hypertune.log"

run_step m4b_hypertune "${PIPE[@]}" \
  --steps hypertune \
  --dataset "$DATASET" \
  --social-regularization \
  --social-mode community_jaccard \
  "${SOCIAL_FLAGS[@]}" \
  "${CMF_FLAGS[@]}" \
  --log-file "${LOG_DIR}/m4b_hypertune.log"

run_step m4c_hypertune "${PIPE[@]}" \
  --steps hypertune \
  --dataset "$DATASET" \
  --social-regularization \
  --social-mode boundary_downweight \
  "${SOCIAL_FLAGS[@]}" \
  "${CMF_FLAGS[@]}" \
  --log-file "${LOG_DIR}/m4c_hypertune.log"

run_step m4d_hypertune "${PIPE[@]}" \
  --steps hypertune \
  --dataset "$DATASET" \
  --social-regularization \
  --social-mode bridge_preserve \
  "${SOCIAL_FLAGS[@]}" \
  "${CMF_FLAGS[@]}" \
  --log-file "${LOG_DIR}/m4d_hypertune.log"

# ---------------------------------------------------------------------------
# 4. Stage B — recommend --all-networks (core first, then ablations)
# Each run archives artifacts under data/<dataset>/runs/<run-id>/
# ---------------------------------------------------------------------------
run_step m3_recommend "${PIPE[@]}" \
  --steps recommend \
  --dataset "$DATASET" \
  "${RECOMMEND_FLAGS[@]}" \
  --run-id m3_recommend \
  --log-file "${LOG_DIR}/m3_recommend.log"

run_step m4c_recommend "${PIPE[@]}" \
  --steps recommend \
  --dataset "$DATASET" \
  "${RECOMMEND_FLAGS[@]}" \
  --social-regularization \
  --social-mode boundary_downweight \
  "${SOCIAL_FLAGS[@]}" \
  --run-id m4c_recommend \
  --log-file "${LOG_DIR}/m4c_recommend.log"

run_step m2_recommend "${PIPE[@]}" \
  --steps recommend \
  --dataset "$DATASET" \
  "${RECOMMEND_FLAGS[@]}" \
  --no-communities \
  --run-id m2_recommend \
  --log-file "${LOG_DIR}/m2_recommend.log"

run_step m4a_recommend "${PIPE[@]}" \
  --steps recommend \
  --dataset "$DATASET" \
  "${RECOMMEND_FLAGS[@]}" \
  --social-regularization \
  --social-mode uniform \
  "${SOCIAL_FLAGS[@]}" \
  --run-id m4a_recommend \
  --log-file "${LOG_DIR}/m4a_recommend.log"

run_step m4b_recommend "${PIPE[@]}" \
  --steps recommend \
  --dataset "$DATASET" \
  "${RECOMMEND_FLAGS[@]}" \
  --social-regularization \
  --social-mode community_jaccard \
  "${SOCIAL_FLAGS[@]}" \
  --run-id m4b_recommend \
  --log-file "${LOG_DIR}/m4b_recommend.log"

run_step m4d_recommend "${PIPE[@]}" \
  --steps recommend \
  --dataset "$DATASET" \
  "${RECOMMEND_FLAGS[@]}" \
  --social-regularization \
  --social-mode bridge_preserve \
  "${SOCIAL_FLAGS[@]}" \
  --run-id m4d_recommend \
  --log-file "${LOG_DIR}/m4d_recommend.log"

run_step m4c_robustness_laplacian "${PIPE[@]}" \
  --steps recommend \
  --dataset "$DATASET" \
  "${RECOMMEND_FLAGS[@]}" \
  --social-regularization \
  --social-mode boundary_downweight \
  --social-normalization normalized_laplacian \
  --social-search-max-ratings 0 \
  --social-n-trials 200 \
  --run-id m4c_robustness_laplacian \
  --log-file "${LOG_DIR}/m4c_robustness_laplacian.log"

# ---------------------------------------------------------------------------
# 5. Phase 2 — global test evaluation (canonical M1 + frozen networks)
# ---------------------------------------------------------------------------
run_step phase2_import_manifest "${PIPE[@]}" \
  --steps import_manifest \
  --dataset "$DATASET" \
  --all-variants \
  --log-file "${LOG_DIR}/phase2_01_import_manifest.log"

run_step phase2_canonical_baseline "${PIPE[@]}" \
  --steps canonical_baseline \
  --dataset "$DATASET" \
  "${CMF_FLAGS[@]}" \
  --log-file "${LOG_DIR}/phase2_02_canonical_baseline.log"

run_step phase2_network_selection "${PIPE[@]}" \
  --steps network_selection \
  --dataset "$DATASET" \
  --all-variants \
  --log-file "${LOG_DIR}/phase2_03_network_selection.log"

run_step phase2_final_eval "${PIPE[@]}" \
  --steps final_eval \
  --dataset "$DATASET" \
  --all-variants \
  "${CMF_FLAGS[@]}" \
  --log-file "${LOG_DIR}/phase2_04_final_eval.log"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
FAILED="$(awk -F'\t' 'NR>1 && $2!="0"{print $1}' "$SUMMARY" | tr '\n' ' ')"
log "=== Core experiment batch finished (dataset=$DATASET) ==="
log "Summary: $SUMMARY"
if [[ -n "${FAILED// /}" ]]; then
  log "FAILED steps: $FAILED"
  log "Re-run failed steps individually (see docs/core_experiment_commands.md)"
else
  log "All steps exited 0"
fi
log "Primary results: data/${DATASET}/core_experiment_results.csv"

exit 0
