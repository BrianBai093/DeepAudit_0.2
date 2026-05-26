#!/usr/bin/env bash
# DeepAudit batch pipeline runner for paper_with_code-style datasets.
#
# Expected case layout:
#   paper_with_code/<run_id>/code
#   paper_with_code/<run_id>/paper/full.md  (created automatically from paper.pdf when missing)
#   paper_with_code/<run_id>/paper.pdf
#
# Usage:
#   ./scripts/run_batch_audit.sh [phases] [paper_with_code_dir]
#
# Examples:
#   ./scripts/run_batch_audit.sh
#   ./scripts/run_batch_audit.sh 1,2,3
#   ./scripts/run_batch_audit.sh 2 /path/to/paper_with_code
#
# Useful env vars:
#   RUN_IDS=case_a,case_b       only run selected folders
#   CONTINUE_ON_ERROR=0         stop batch at the first failed case
#   DRY_RUN=1                   print planned runs without executing p2c
#   PYTHON_BIN=python           choose the Python executable
#   BUDGET_MINUTES=180          phase budget passed through to p2c
#   P2C_MIN_EXEC_TIMEOUT_SEC=10800
#                               minimum phase2 executor session timeout, default 3h
#   P2C_EXECUTION_MODE=full     ask phase2 to attempt direct full-fidelity runs

set -euo pipefail

# ---------- DEFAULTS ----------
PROJECT_ROOT="${PROJECT_ROOT:-/home/yb2636_columbia_edu/DeepAudit_0.2}"
PAPER_WITH_CODE_DIR="${2:-${PAPER_WITH_CODE_DIR:-${PROJECT_ROOT}/paper_with_code}}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/batch_papers}"
BUDGET_MINUTES="${BUDGET_MINUTES:-180}"
P2C_MIN_EXEC_TIMEOUT_SEC="${P2C_MIN_EXEC_TIMEOUT_SEC:-10800}"
P2C_ATOMIC_LLM_SENTENCE_BUDGET="${P2C_ATOMIC_LLM_SENTENCE_BUDGET:-32}"
P2C_ATOMIC_LLM_TABLE_BUDGET="${P2C_ATOMIC_LLM_TABLE_BUDGET:-20}"
OPENAI_TIMEOUT_SEC="${OPENAI_TIMEOUT_SEC:-300}"
OPENAI_VISION_TIMEOUT_SEC="${OPENAI_VISION_TIMEOUT_SEC:-360}"
P2C_EXECUTION_MODE="${P2C_EXECUTION_MODE:-standard}"
PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_IDS="${RUN_IDS:-}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif [[ -x "/home/yb2636_columbia_edu/miniconda3/envs/agent/bin/python" ]]; then
    PYTHON_BIN="/home/yb2636_columbia_edu/miniconda3/envs/agent/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
# ------------------------------

usage() {
  echo "Usage: $0 [phases] [paper_with_code_dir]"
  echo "  phases: comma-separated list, default '1,2,3'"
  echo "  paper_with_code_dir: default '${PROJECT_ROOT}/paper_with_code'"
}

if [[ $# -gt 2 ]]; then
  usage
  exit 1
fi

PHASES="${1:-1,2,3}"

cd "$PROJECT_ROOT"

[[ -d "$PAPER_WITH_CODE_DIR" ]] || { echo "paper_with_code dir not found: $PAPER_WITH_CODE_DIR"; exit 1; }

export P2C_MIN_EXEC_TIMEOUT_SEC
export P2C_ATOMIC_LLM_SENTENCE_BUDGET
export P2C_ATOMIC_LLM_TABLE_BUDGET
export OPENAI_TIMEOUT_SEC
export OPENAI_VISION_TIMEOUT_SEC
export P2C_EXECUTION_MODE
export PIP_NO_CACHE_DIR

BATCH_STARTED_AT="$(date '+%Y%m%d_%H%M%S')"
BATCH_LOG_DIR="${ARTIFACTS_DIR}/_batch_logs/${BATCH_STARTED_AT}"
SUMMARY_TSV="${BATCH_LOG_DIR}/summary.tsv"
RUN_LOG="${BATCH_LOG_DIR}/batch.log"
mkdir -p "$BATCH_LOG_DIR" "$OUTPUT_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1
printf "run_id\tstatus\tfailed_phase\treport\n" > "$SUMMARY_TSV"

IFS=',' read -ra PHASE_LIST <<< "$PHASES"
IFS=',' read -ra SELECTED_RUN_IDS <<< "$RUN_IDS"

if [[ "$DRY_RUN" != "1" ]]; then
  NEED_OPENAI=0
  NEED_ANTHROPIC=0
  for phase in "${PHASE_LIST[@]}"; do
    [[ "$phase" == "1" || "$phase" == "3" ]] && NEED_OPENAI=1
    [[ "$phase" == "2" ]] && NEED_ANTHROPIC=1
  done
  if [[ "$NEED_OPENAI" == "1" ]]; then
    : "${OPENAI_API_KEY:?OPENAI_API_KEY not set}"
  fi
  if [[ "$NEED_ANTHROPIC" == "1" ]]; then
    : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not set}"
  fi
fi

selected_run_id() {
  local run_id="$1"
  local selected
  [[ -z "$RUN_IDS" ]] && return 0
  for selected in "${SELECTED_RUN_IDS[@]}"; do
    [[ "$run_id" == "$selected" ]] && return 0
  done
  return 1
}

phase_selected() {
  local needle="$1"
  local phase
  for phase in "${PHASE_LIST[@]}"; do
    [[ "$phase" == "$needle" ]] && return 0
  done
  return 1
}

find_paper_md() {
  local case_dir="$1"
  local preferred="${case_dir}/paper/full.md"
  local found
  if [[ -f "$preferred" ]]; then
    echo "$preferred"
    return 0
  fi
  found="$(find "${case_dir}/paper" -maxdepth 1 -type f \( -name '*.md' -o -name '*.markdown' \) 2>/dev/null | sort | head -n 1 || true)"
  [[ -n "$found" ]] || return 1
  echo "$found"
}

find_paper_pdf() {
  local case_dir="$1"
  if [[ -f "${case_dir}/paper.pdf" ]]; then
    echo "${case_dir}/paper.pdf"
    return 0
  fi
  if [[ -f "${case_dir}/paper/paper.pdf" ]]; then
    echo "${case_dir}/paper/paper.pdf"
    return 0
  fi
  return 1
}

run_case() {
  local case_dir="$1"
  local run_id
  local paper_md
  local paper_pdf
  local repo_dir
  local paper_md_out
  local phase

  run_id="$(basename "$case_dir")"
  repo_dir="${case_dir}/code"
  paper_pdf="$(find_paper_pdf "$case_dir" || true)"
  paper_md_out="${OUTPUT_ROOT}/${run_id}/paper.md"

  [[ -d "$repo_dir" ]] || { echo "Repo not found for ${run_id}: $repo_dir"; return 2; }
  if phase_selected "1" && [[ -z "$paper_pdf" ]]; then
    echo "Paper PDF not found for ${run_id}"
    return 2
  fi
  if ! paper_md="$(find_paper_md "$case_dir")"; then
    paper_md="${case_dir}/paper/full.md"
    if ! phase_selected "1"; then
      echo "Paper markdown not found for ${run_id}; run phase 1 with paper.pdf first"
      return 2
    fi
  fi
  mkdir -p "$(dirname "$paper_md_out")"
  if [[ -f "$paper_md" && ( ! -f "$paper_md_out" || "$paper_md" -nt "$paper_md_out" ) ]]; then
    cp "$paper_md" "$paper_md_out"
  fi

  echo ""
  echo "================================================================"
  echo " DeepAudit batch case"
  echo " run_id     : $run_id"
  echo " phases     : $PHASES"
  echo " paper      : $paper_md"
  echo " paper_pdf  : $paper_pdf"
  echo " repo       : $repo_dir"
  echo " artifacts  : ${ARTIFACTS_DIR}/${run_id}"
  echo " paper_out  : $paper_md_out"
  echo " python     : $PYTHON_BIN"
  echo " exec mode  : ${P2C_EXECUTION_MODE}"
  echo " pip cache  : PIP_NO_CACHE_DIR=${PIP_NO_CACHE_DIR}"
  echo "================================================================"

  for phase in "${PHASE_LIST[@]}"; do
    echo ""
    echo ">>> ${run_id}: Phase ${phase} starting at $(date '+%Y-%m-%d %H:%M:%S')"
    local pdf_args=()
    if [[ "$phase" == "1" ]]; then
      pdf_args=(--paper_pdf "$paper_pdf")
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] $PYTHON_BIN -m p2c.main --phase $phase --paper_md $paper_md --paper_md_out $paper_md_out --repo_dir $repo_dir --run_id $run_id --artifacts_dir $ARTIFACTS_DIR --budget_minutes $BUDGET_MINUTES ${pdf_args[*]}"
      continue
    fi

    if "$PYTHON_BIN" -m p2c.main \
      --phase "$phase" \
      --paper_md "$paper_md" \
      --paper_md_out "$paper_md_out" \
      --repo_dir "$repo_dir" \
      --run_id "$run_id" \
      --artifacts_dir "$ARTIFACTS_DIR" \
      --budget_minutes "$BUDGET_MINUTES" \
      "${pdf_args[@]}"; then
      echo "<<< ${run_id}: Phase ${phase} finished at $(date '+%Y-%m-%d %H:%M:%S')"
    else
      local status=$?
      echo "!!! ${run_id}: Phase ${phase} failed with status ${status} at $(date '+%Y-%m-%d %H:%M:%S')"
      printf "%s\tFAILED\t%s\t%s\n" "$run_id" "$phase" "${ARTIFACTS_DIR}/${run_id}/results/report.md" >> "$SUMMARY_TSV"
      return "$status"
    fi
  done

  printf "%s\tDONE\t\t%s\n" "$run_id" "${ARTIFACTS_DIR}/${run_id}/results/report.md" >> "$SUMMARY_TSV"
  return 0
}

mapfile -t CASE_DIRS < <(find "$PAPER_WITH_CODE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#CASE_DIRS[@]} -eq 0 ]]; then
  echo "No case directories found under: $PAPER_WITH_CODE_DIR"
  exit 1
fi

echo "================================================================"
echo " DeepAudit batch pipeline"
echo " cases root : $PAPER_WITH_CODE_DIR"
echo " phases     : $PHASES"
echo " artifacts  : $ARTIFACTS_DIR"
echo " output     : $OUTPUT_ROOT"
echo " budget     : ${BUDGET_MINUTES} minutes per phase"
echo " min timeout: ${P2C_MIN_EXEC_TIMEOUT_SEC}s"
echo " exec mode  : ${P2C_EXECUTION_MODE}"
echo " batch log  : $BATCH_LOG_DIR"
echo " run log    : $RUN_LOG"
echo " continue   : $CONTINUE_ON_ERROR"
echo " dry run    : $DRY_RUN"
echo " selected   : ${RUN_IDS:-all}"
echo "================================================================"

batch_status=0
for case_dir in "${CASE_DIRS[@]}"; do
  run_id="$(basename "$case_dir")"
  if ! selected_run_id "$run_id"; then
    echo "Skipping ${run_id} because RUN_IDS is set."
    continue
  fi

  if run_case "$case_dir"; then
    :
  else
    status=$?
    batch_status=$status
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "Stopping batch because CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}."
      exit "$status"
    fi
  fi
done

echo ""
echo "================================================================"
echo " Batch complete."
echo " Summary: $SUMMARY_TSV"
echo " Reports: ${ARTIFACTS_DIR}/<run_id>/results/report.md"
echo "================================================================"

exit "$batch_status"
