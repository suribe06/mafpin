#!/usr/bin/env bash
# Core experiment batch — any dataset (full ladder + Phase 2 final_eval).
# Source: docs/core_experiment_commands.md
#
# Usage:
#   ./scripts/run_core_experiment.sh --dataset ciao
#   ./scripts/run_core_experiment.sh --dataset ciao --from preregister
#
# Continues on failure (set +e). Review when done:
#   column -t -s $'\t' data/<dataset>/logs/run_summary.tsv

set +e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

VALID_DATASETS=(movielens ciao epinions)
ALL_STEPS=(
  prerequisites preregister
  m2_hypertune m3_hypertune m4a_hypertune m4b_hypertune m4c_hypertune m4d_hypertune
  m3_recommend m4c_recommend m2_recommend m4a_recommend m4b_recommend m4d_recommend
  m4c_robustness_laplacian
  phase2_import_manifest phase2_canonical_baseline phase2_network_selection phase2_final_eval
)

DATASET=""
FROM_STEP=""
DRY_RUN=0
N_JOBS=-1
CPU_FRACTION=0.4
BATCH_DONE=0
RUN_ACTIVE=0
FAILED_STEPS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") --dataset DATASET [OPTIONS]

Options:
  --dataset DATASET   Required. One of: ${VALID_DATASETS[*]}
  --from STEP         Resume from STEP (skip earlier steps). Example: --from preregister
  --dry-run           Print planned steps without running pipeline.py
  --n-jobs N          Worker processes for recommend network eval (default: -1 = auto)
  --cpu-fraction F    Core fraction when --n-jobs -1 (default: 0.4)
  -h, --help          Show this help

Steps (--from values):
  ${ALL_STEPS[*]}

Run detached (recommended):
  cd $ROOT
  tmux new -s ciao
  ./scripts/run_core_experiment.sh --dataset ciao --from preregister
  # Ctrl-b d to detach

  nohup ./scripts/run_core_experiment.sh --dataset ciao --from preregister &
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    --from)
      FROM_STEP="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --n-jobs)
      N_JOBS="${2:-}"
      shift 2
      ;;
    --cpu-fraction)
      CPU_FRACTION="${2:-}"
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

if [[ -n "$FROM_STEP" ]]; then
  found=0
  for s in "${ALL_STEPS[@]}"; do
    if [[ "$s" == "$FROM_STEP" ]]; then
      found=1
      break
    fi
  done
  if [[ $found -eq 0 ]]; then
    echo "Error: unknown step '$FROM_STEP'." >&2
    echo "Valid steps: ${ALL_STEPS[*]}" >&2
    exit 1
  fi
  RUN_ACTIVE=0
else
  RUN_ACTIVE=1
fi

LOG_DIR="data/${DATASET}/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/run_all_${STAMP}.log"
SUMMARY="${LOG_DIR}/run_summary.tsv"

if [[ $DRY_RUN -eq 1 ]]; then
  if [[ ! -f "$SUMMARY" ]]; then
    echo -e "step\texit_code\tfinished_at" >"$SUMMARY"
    echo "# session dry-run at $(date -Iseconds)" >>"$SUMMARY"
  fi
elif [[ -n "$FROM_STEP" && -f "$SUMMARY" ]]; then
  echo >>"$SUMMARY"
  echo "# session --from $FROM_STEP at $(date -Iseconds)" >>"$SUMMARY"
else
  echo -e "step\texit_code\tfinished_at" >"$SUMMARY"
  echo "# session full batch at $(date -Iseconds)" >>"$SUMMARY"
fi

# ponytail: tmux/nohup often break `conda run`; use env python directly
MAFPIN_PYTHON="${MAFPIN_PYTHON:-}"
if [[ -z "$MAFPIN_PYTHON" && -x "${HOME}/anaconda3/envs/mafpin/bin/python" ]]; then
  MAFPIN_PYTHON="${HOME}/anaconda3/envs/mafpin/bin/python"
fi
if [[ -z "$MAFPIN_PYTHON" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  MAFPIN_PYTHON="$(conda run -n mafpin which python 2>/dev/null | tail -1)"
fi
if [[ ! -x "$MAFPIN_PYTHON" ]] && command -v python >/dev/null 2>&1; then
  MAFPIN_PYTHON="$(command -v python)"
fi
if [[ ! -x "$MAFPIN_PYTHON" ]]; then
  echo "Error: mafpin python not found. Set MAFPIN_PYTHON=/path/to/mafpin/bin/python" >&2
  exit 1
fi

PIPE=("$MAFPIN_PYTHON" pipeline.py)

CMF_FLAGS=(--cmf-method lbfgs --cmf-maxiter 25 --seed 42)
SOCIAL_FLAGS=(
  --social-normalization mean_weight
  --social-search-max-ratings 0
  --social-n-trials 200
)
RECOMMEND_FLAGS=(
  --all-networks
  --n-jobs "$N_JOBS"
  --cpu-fraction "$CPU_FRACTION"
  "${CMF_FLAGS[@]}"
)

log() {
  # ponytail: avoid pipefail/teardown killing the batch on tee SIGPIPE
  echo "[$(date -Iseconds)] $*" | tee -a "$MASTER_LOG" || true
}

on_exit() {
  local code=$?
  if [[ $BATCH_DONE -eq 0 ]]; then
    log "BATCH INTERRUPTED (shell exit=$code) — tmux closed, SSH drop, or kill?"
    if [[ -n "$FROM_STEP" ]]; then
      log "Resume: ./scripts/run_core_experiment.sh --dataset $DATASET --from $FROM_STEP"
    else
      log "Check run_summary.tsv for last OK step, then --from NEXT_STEP"
    fi
  fi
}
trap on_exit EXIT

run_step() {
  local name="$1"
  shift

  if [[ $RUN_ACTIVE -eq 0 ]]; then
    if [[ "$name" == "$FROM_STEP" ]]; then
      RUN_ACTIVE=1
      log "RESUME from $name"
    else
      log "SKIP $name (before --from $FROM_STEP)"
      return 0
    fi
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN $name"
    log "CMD: $*"
    return 0
  fi

  log "START $name"
  log "CMD: $*"
  "$@" >>"$MASTER_LOG" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    log "OK   $name (exit=0)"
  else
    log "FAIL $name (exit=$rc) — continuing"
    FAILED_STEPS+=("$name")
  fi
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(date -Iseconds)" >>"$SUMMARY"
  return 0
}

log "=== Core experiment batch started (dataset=$DATASET) ==="
[[ -n "$FROM_STEP" ]] && log "Resuming from step: $FROM_STEP"
[[ $DRY_RUN -eq 1 ]] && log "DRY-RUN mode enabled"
log "ROOT=$ROOT"
log "PYTHON=$MAFPIN_PYTHON"
log "RECOMMEND_PARALLEL: n_jobs=$N_JOBS cpu_fraction=$CPU_FRACTION"
log "MASTER_LOG=$MASTER_LOG"
log "SUMMARY=$SUMMARY"

if [[ $DRY_RUN -eq 0 ]]; then
  if ! "$MAFPIN_PYTHON" -c "import pandas; from pipeline import main" >>"$MASTER_LOG" 2>&1; then
    log "PREFLIGHT FAILED: pandas/pipeline import — fix mafpin env before batch"
    exit 1
  fi
  log "PREFLIGHT OK"
fi

run_step prerequisites "${PIPE[@]}" \
  --steps cascade delta inference centrality communities \
  --dataset "$DATASET" \
  --log-file "${LOG_DIR}/00_prerequisites.log"

run_step preregister "${PIPE[@]}" \
  --preregister-networks \
  --dataset "$DATASET" \
  --seed 42 \
  --log-file "${LOG_DIR}/01_preregister.log"

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

BATCH_DONE=1
log "=== Core experiment batch finished (dataset=$DATASET) ==="
if [[ $DRY_RUN -eq 1 ]]; then
  log "DRY-RUN complete — no pipeline commands executed; run without --dry-run to start"
elif [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FAILED steps this session: ${FAILED_STEPS[*]}"
  log "Summary: $SUMMARY"
else
  log "All steps in this session exited 0"
  log "Summary: $SUMMARY"
fi
log "Primary results: data/${DATASET}/core_experiment_results.csv"

exit 0
