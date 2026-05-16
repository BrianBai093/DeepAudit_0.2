#!/usr/bin/env bash
# Lightweight overnight monitor for DeepAudit batch runs.
#
# Usage:
#   RUN_IDS=06_case,07_case HOURS=9 ./scripts/monitor_batch_audit.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/yb2636_columbia_edu/DeepAudit_0.2}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
RUN_IDS="${RUN_IDS:-}"
HOURS="${HOURS:-9}"
INTERVAL_SEC="${INTERVAL_SEC:-3600}"

cd "$PROJECT_ROOT"

IFS=',' read -ra SELECTED_RUN_IDS <<< "$RUN_IDS"

latest_batch_dir() {
  find "${ARTIFACTS_DIR}/_batch_logs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true
}

print_run_status() {
  local run_id="$1"
  local run_root="${ARTIFACTS_DIR}/${run_id}"
  local execution="${run_root}/execution"
  local results="${run_root}/results"

  echo "-- ${run_id}"
  [[ -d "$run_root" ]] || { echo "   no artifact directory yet"; return 0; }
  [[ -f "${execution}/context.json" ]] && echo "   context: present"
  [[ -f "${execution}/env_setup_result.json" ]] && echo "   env: $(grep -m1 -E 'validation_passed|valid|status' "${execution}/env_setup_result.json" || true)"
  [[ -f "${execution}/phase2_state.json" ]] && echo "   phase2_state: present"
  [[ -f "${execution}/executor_outputs/PHASE2_RESULTS.md" ]] && echo "   phase2 results: present"
  [[ -f "${results}/report.md" ]] && echo "   report: ${results}/report.md"
  find "${execution}/executor_outputs" -maxdepth 1 -type f -name '*.log' -printf '   log: %TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -n 5 || true
}

for ((i = 0; i <= HOURS; i++)); do
  echo "================================================================"
  echo "DeepAudit batch monitor check ${i}/${HOURS} at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "================================================================"
  echo "Active processes:"
  pgrep -af 'run_batch_audit|p2c.main|claude|python.*p2c' || echo "  none"

  batch_dir="$(latest_batch_dir)"
  if [[ -n "$batch_dir" ]]; then
    echo ""
    echo "Latest batch dir: $batch_dir"
    [[ -f "${batch_dir}/summary.tsv" ]] && { echo "Summary:"; cat "${batch_dir}/summary.tsv"; }
    [[ -f "${batch_dir}/batch.log" ]] && { echo ""; echo "Batch log tail:"; tail -n 80 "${batch_dir}/batch.log"; }
  fi

  echo ""
  echo "Selected run statuses:"
  if [[ ${#SELECTED_RUN_IDS[@]} -gt 0 && -n "${SELECTED_RUN_IDS[0]}" ]]; then
    for run_id in "${SELECTED_RUN_IDS[@]}"; do
      print_run_status "$run_id"
    done
  else
    echo "  RUN_IDS not set"
  fi

  [[ "$i" -lt "$HOURS" ]] || break
  sleep "$INTERVAL_SEC"
done
