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
PAPER_MD="${PROJECT_ROOT}/Target/paper/full.md"
PAPER_MD_OUT="${PROJECT_ROOT}/output/paper.md"
REPO_DIR="${PROJECT_ROOT}/Target/code"
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts"
BUDGET_MINUTES="${BUDGET_MINUTES:-180}"  # default 3h, override with env var
# -------------------------------------------------

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [phases]"
  echo "  phases: comma-separated list, default '1,2,3'"
  exit 1
fi

RUN_ID="$1"
PHASES="${2:-1,2,3}"

# Sanity checks
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not set}"
[[ -f "$PAPER_MD" ]] || { echo "Paper not found: $PAPER_MD"; exit 1; }
[[ -d "$REPO_DIR" ]] || { echo "Repo not found: $REPO_DIR"; exit 1; }

cd "$PROJECT_ROOT"

echo "================================================================"
echo " DeepAudit v0.5 Pipeline"
echo " run_id     : $RUN_ID"
echo " phases     : $PHASES"
echo " paper      : $PAPER_MD"
echo " repo       : $REPO_DIR"
echo " artifacts  : $ARTIFACTS_DIR/$RUN_ID"
echo "================================================================"

IFS=',' read -ra PHASE_LIST <<< "$PHASES"
for phase in "${PHASE_LIST[@]}"; do
  echo ""
  echo ">>> Phase $phase starting at $(date '+%Y-%m-%d %H:%M:%S')"
  python -m p2c.main \
    --phase "$phase" \
    --paper_md "$PAPER_MD" \
    --paper_md_out "$PAPER_MD_OUT" \
    --repo_dir "$REPO_DIR" \
    --run_id "$RUN_ID" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --budget_minutes "$BUDGET_MINUTES"
  echo "<<< Phase $phase finished at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "================================================================"
echo " All requested phases complete."
echo " Verdict : $ARTIFACTS_DIR/$RUN_ID/results/verdict.json"
echo " Report  : $ARTIFACTS_DIR/$RUN_ID/results/report.md"
echo "================================================================"
