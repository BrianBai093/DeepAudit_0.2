#!/usr/bin/env bash
# Overnight batch runner for paper_with_code/16..23
#
# - Cleans any prior conda env named "<run_id>_executor" before each paper.
# - Runs phases 1 -> 2 -> 3 sequentially per paper.
# - Continues to the next paper even if one fails.
# - Per-paper stdout/stderr goes to artifacts/_overnight/<timestamp>/<run_id>.log
# - Final summary at artifacts/_overnight/<timestamp>/SUMMARY.tsv
#
# Usage:
#   ./scripts/run_overnight_16_23.sh
#
# Env vars (optional):
#   BUDGET_MINUTES                  default 180 (per phase)
#   P2C_MIN_EXEC_TIMEOUT_SEC        default 10800 (3h cap on phase2 executor session)
#   PYTHON_BIN                      default agent conda env python

set -uo pipefail

PROJECT_ROOT="/home/yb2636_columbia_edu/DeepAudit_0.2"
cd "$PROJECT_ROOT"

PAPER_ROOT="${PROJECT_ROOT}/paper_with_code"
ARTIFACTS_DIR="artifacts"
OUTPUT_ROOT="output/batch_papers"

BUDGET_MINUTES="${BUDGET_MINUTES:-180}"
P2C_MIN_EXEC_TIMEOUT_SEC="${P2C_MIN_EXEC_TIMEOUT_SEC:-10800}"
OPENAI_TIMEOUT_SEC="${OPENAI_TIMEOUT_SEC:-300}"
OPENAI_VISION_TIMEOUT_SEC="${OPENAI_VISION_TIMEOUT_SEC:-360}"
P2C_ATOMIC_LLM_SENTENCE_BUDGET="${P2C_ATOMIC_LLM_SENTENCE_BUDGET:-32}"
P2C_ATOMIC_LLM_TABLE_BUDGET="${P2C_ATOMIC_LLM_TABLE_BUDGET:-20}"

# API keys live in the `agent` conda env vars; activating it exports them.
CONDA_BASE="/home/yb2636_columbia_edu/miniconda3"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate agent

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CONDA_BIN="$(command -v mamba || command -v conda || true)"

export P2C_MIN_EXEC_TIMEOUT_SEC OPENAI_TIMEOUT_SEC OPENAI_VISION_TIMEOUT_SEC
export P2C_ATOMIC_LLM_SENTENCE_BUDGET P2C_ATOMIC_LLM_TABLE_BUDGET

: "${OPENAI_API_KEY:?OPENAI_API_KEY not in agent env — run: conda env config vars set OPENAI_API_KEY=... -n agent}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not in agent env — run: conda env config vars set ANTHROPIC_API_KEY=... -n agent}"
[[ -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN not executable: $PYTHON_BIN"; exit 1; }
[[ -n "$CONDA_BIN" ]] || { echo "Neither mamba nor conda found in PATH"; exit 1; }

PAPERS=(
  "16_Neural_Controlled_Differential_Equations"
  "17_Learning_to_Simulate_Complex_Physics_GNS"
  "18_ScoreBased_Generative_Modeling_via_SDEs"
  "19_SINDy_Autoencoder"
  "20_Implicit_Neural_Representations_with_Periodic_Activation_Functions"
  "21_KAN_KolmogorovArnold_Networks"
  "22_Spikformer_When_Spiking_Neural_Network_Meets_Transformer"
  "23_Next_Generation_Reservoir_Computing"
)

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${ARTIFACTS_DIR}/_overnight/${TIMESTAMP}"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
SUMMARY="${LOG_DIR}/SUMMARY.tsv"
BATCH_LOG="${LOG_DIR}/batch.log"
printf "run_id\tstatus\tfailed_phase\tstart\tend\tduration_sec\n" > "$SUMMARY"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$BATCH_LOG"; }

clean_envs_for() {
  local run_id="$1"
  local env_name="${run_id}_executor"
  # Remove the executor env if it exists; ignore failure.
  if "$CONDA_BIN" env list 2>/dev/null | awk '{print $1}' | grep -Fxq "$env_name"; then
    log "  cleanup: removing conda env '$env_name'"
    "$CONDA_BIN" env remove -n "$env_name" -y --quiet >>"$BATCH_LOG" 2>&1 || true
  fi
  # Sweep any leftover snapshots like <env>__snap_*
  while read -r leftover; do
    [[ -z "$leftover" ]] && continue
    log "  cleanup: removing leftover snapshot env '$leftover'"
    "$CONDA_BIN" env remove -n "$leftover" -y --quiet >>"$BATCH_LOG" 2>&1 || true
  done < <("$CONDA_BIN" env list 2>/dev/null | awk '{print $1}' | grep -E "^${env_name}__snap_" || true)
  # Drop any cached venv fallback.
  rm -rf "/tmp/p2c_venv_${env_name}" 2>/dev/null || true
}

run_phase() {
  local run_id="$1" phase="$2" paper_md="$3" paper_md_out="$4" repo_dir="$5" paper_pdf="$6" per_log="$7"
  local pdf_args=()
  [[ "$phase" == "1" ]] && pdf_args=(--paper_pdf "$paper_pdf")
  log "  phase ${phase} starting"
  if "$PYTHON_BIN" -m p2c.main \
      --phase "$phase" \
      --paper_md "$paper_md" \
      --paper_md_out "$paper_md_out" \
      --repo_dir "$repo_dir" \
      --run_id "$run_id" \
      --artifacts_dir "$ARTIFACTS_DIR" \
      --budget_minutes "$BUDGET_MINUTES" \
      "${pdf_args[@]}" >>"$per_log" 2>&1; then
    log "  phase ${phase} OK"
    return 0
  else
    local rc=$?
    log "  phase ${phase} FAILED rc=${rc}"
    return "$rc"
  fi
}

log "================================================================"
log " Overnight batch starting"
log " papers      : ${#PAPERS[@]}"
log " budget_min  : $BUDGET_MINUTES per phase"
log " min_timeout : ${P2C_MIN_EXEC_TIMEOUT_SEC}s"
log " python      : $PYTHON_BIN"
log " conda bin   : $CONDA_BIN"
log " log dir     : $LOG_DIR"
log "================================================================"

for run_id in "${PAPERS[@]}"; do
  case_dir="${PAPER_ROOT}/${run_id}"
  paper_md="${case_dir}/paper/full.md"
  paper_pdf="${case_dir}/paper.pdf"
  repo_dir="${case_dir}/code"
  paper_md_out="${OUTPUT_ROOT}/${run_id}/paper.md"
  per_log="${LOG_DIR}/${run_id}.log"

  log ""
  log "============================================================"
  log " >>> ${run_id}"
  log "============================================================"

  if [[ ! -f "$paper_md" || ! -f "$paper_pdf" || ! -d "$repo_dir" ]]; then
    log "  SKIP: missing inputs (md=$paper_md pdf=$paper_pdf repo=$repo_dir)"
    printf "%s\tSKIP\tinputs\t-\t-\t0\n" "$run_id" >> "$SUMMARY"
    continue
  fi

  mkdir -p "$(dirname "$paper_md_out")"
  cp -f "$paper_md" "$paper_md_out"

  start_ts=$(date '+%s')
  start_iso=$(date '+%Y-%m-%dT%H:%M:%S')

  clean_envs_for "$run_id"

  failed_phase=""
  for phase in 1 2 3; do
    if ! run_phase "$run_id" "$phase" "$paper_md" "$paper_md_out" "$repo_dir" "$paper_pdf" "$per_log"; then
      failed_phase="$phase"
      break
    fi
  done

  end_ts=$(date '+%s')
  end_iso=$(date '+%Y-%m-%dT%H:%M:%S')
  dur=$((end_ts - start_ts))

  # Always cleanup after, so the next paper starts with no leftover env.
  clean_envs_for "$run_id"

  if [[ -z "$failed_phase" ]]; then
    log "  RESULT: DONE  duration=${dur}s"
    printf "%s\tDONE\t\t%s\t%s\t%d\n" "$run_id" "$start_iso" "$end_iso" "$dur" >> "$SUMMARY"
  else
    log "  RESULT: FAILED at phase ${failed_phase}  duration=${dur}s"
    printf "%s\tFAILED\t%s\t%s\t%s\t%d\n" "$run_id" "$failed_phase" "$start_iso" "$end_iso" "$dur" >> "$SUMMARY"
  fi
done

log ""
log "================================================================"
log " Overnight batch complete."
log " Summary : $SUMMARY"
log " Per-run : $LOG_DIR/<run_id>.log"
log "================================================================"

cat "$SUMMARY"
