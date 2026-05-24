#!/usr/bin/env bash
# DeepAudit v0.5 — one-shot full pipeline runner
#
# Usage:
#   ./scripts/run_audit.sh <run_id> [phase]
#
# Examples:
#   ./scripts/run_audit.sh my_audit          # run all 3 phases
#   ./scripts/run_audit.sh my_audit 2        # run only phase 2
#   ./scripts/run_audit.sh my_audit 1,2      # run phases 1 and 2
#
# Edit the DEFAULTS block below to point at your paper / repo / artifacts paths.

set -euo pipefail

# ---------- DEFAULTS (edit these once) ----------
PROJECT_ROOT="/home/yb2636_columbia_edu/DeepAudit_0.2"
PAPER_MD="Target/paper/full.md"
PAPER_MD_OUT="output/paper.md"
PAPER_PDF="Target/paper.pdf"
REPO_DIR="${PROJECT_ROOT}/Target/code"
ARTIFACTS_DIR="artifacts"
BUDGET_MINUTES="${BUDGET_MINUTES:-180}"  # default 3h, override with env var
P2C_MIN_EXEC_TIMEOUT_SEC="${P2C_MIN_EXEC_TIMEOUT_SEC:-7200}"
P2C_ATOMIC_LLM_SENTENCE_BUDGET="${P2C_ATOMIC_LLM_SENTENCE_BUDGET:-32}"
P2C_ATOMIC_LLM_TABLE_BUDGET="${P2C_ATOMIC_LLM_TABLE_BUDGET:-20}"
OPENAI_TIMEOUT_SEC="${OPENAI_TIMEOUT_SEC:-300}"
OPENAI_VISION_TIMEOUT_SEC="${OPENAI_VISION_TIMEOUT_SEC:-360}"
P2C_EXECUTION_MODE="${P2C_EXECUTION_MODE:-standard}"  # standard or full
# -------------------------------------------------

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [phases]"
  echo "  phases: comma-separated list, default '1,2,3'"
  exit 1
fi

RUN_ID="$1"
PHASES="${2:-1,2,3}"

cd "$PROJECT_ROOT"

# Sanity checks
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not set}"
[[ -d "$REPO_DIR" ]] || { echo "Repo not found: $REPO_DIR"; exit 1; }
if [[ ",$PHASES," == *",1,"* ]]; then
  [[ -f "$PAPER_PDF" ]] || { echo "Paper PDF not found: $PAPER_PDF"; exit 1; }
elif [[ ! -f "$PAPER_MD" ]]; then
  echo "Paper markdown not found: $PAPER_MD (run phase 1 with paper.pdf first)"
  exit 1
fi
mkdir -p "$(dirname "$PAPER_MD_OUT")"

echo "================================================================"
echo " DeepAudit v0.5 Pipeline"
echo " run_id     : $RUN_ID"
echo " phases     : $PHASES"
echo " paper      : $PAPER_MD"
echo " paper_pdf  : $PAPER_PDF"
echo " repo       : $REPO_DIR"
echo " artifacts  : $ARTIFACTS_DIR/$RUN_ID"
echo " min timeout: ${P2C_MIN_EXEC_TIMEOUT_SEC}s"
echo " exec mode  : ${P2C_EXECUTION_MODE}"
echo " atomic LLM : sentences=${P2C_ATOMIC_LLM_SENTENCE_BUDGET}, tables=${P2C_ATOMIC_LLM_TABLE_BUDGET}"
echo " OpenAI wait: text/json=${OPENAI_TIMEOUT_SEC}s, vision=${OPENAI_VISION_TIMEOUT_SEC}s"
echo "================================================================"

export P2C_MIN_EXEC_TIMEOUT_SEC
export P2C_ATOMIC_LLM_SENTENCE_BUDGET
export P2C_ATOMIC_LLM_TABLE_BUDGET
export OPENAI_TIMEOUT_SEC
export OPENAI_VISION_TIMEOUT_SEC
export P2C_EXECUTION_MODE

IFS=',' read -ra PHASE_LIST <<< "$PHASES"
for phase in "${PHASE_LIST[@]}"; do
  echo ""
  echo ">>> Phase $phase starting at $(date '+%Y-%m-%d %H:%M:%S')"
  PDF_ARGS=()
  if [[ "$phase" == "1" ]]; then
    PDF_ARGS=(--paper_pdf "$PAPER_PDF")
  fi
  python -m p2c.main \
    --phase "$phase" \
    --paper_md "$PAPER_MD" \
    --paper_md_out "$PAPER_MD_OUT" \
    --repo_dir "$REPO_DIR" \
    --run_id "$RUN_ID" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --budget_minutes "$BUDGET_MINUTES" \
    "${PDF_ARGS[@]}"
  echo "<<< Phase $phase finished at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "================================================================"
echo " All requested phases complete."
echo " Verdict : $ARTIFACTS_DIR/$RUN_ID/results/verdict.json"
echo " Report  : $ARTIFACTS_DIR/$RUN_ID/results/report.md"
echo "================================================================"
