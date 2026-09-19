#!/usr/bin/env bash
# Wait for confirmatory matrix jobs, then run robustness sweep (plan v3 WP-C).
#
# Usage (after starting both matrix jobs):
#   MATRIX_PID_HARD=<pid> MATRIX_PID_EASY=<pid> bash scripts/run_robustness_after_matrix.sh
#
# Or poll log files only:
#   bash scripts/run_robustness_after_matrix.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
LOG_HARD="${LOG_HARD:-$REV_SUP/matrix_hard_comm_skip_confirmatory.log}"
LOG_EASY="${LOG_EASY:-$REV_SUP/matrix_easy_label_flip_confirmatory.log}"
ROBUST_LOG="${ROBUST_LOG:-$REV_SUP/robustness_comm_skip_confirmatory.log}"

wait_pid() {
  local pid="$1" name="$2"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $name pid $pid not running (already done?)."
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Waiting for $name (pid $pid)..."
  while kill -0 "$pid" 2>/dev/null; do
    sleep 120
  done
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $name pid $pid exited."
}

wait_log_done() {
  local log="$1" name="$2"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Polling $name log: $log"
  while [[ ! -f "$log" ]] || ! grep -q '^Done\.$' "$log"; do
    sleep 120
  done
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $name log reports Done."
}

mkdir -p "$REV_SUP"

if [[ -n "${MATRIX_PID_HARD:-}" ]]; then
  wait_pid "$MATRIX_PID_HARD" "hard matrix (comm_skip)"
fi
if [[ -n "${MATRIX_PID_EASY:-}" ]]; then
  wait_pid "$MATRIX_PID_EASY" "easy matrix (label_flip)"
fi

wait_log_done "$LOG_HARD" "hard matrix"
wait_log_done "$LOG_EASY" "easy matrix"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Both matrix jobs complete — starting robustness sweep."
echo "robustness_queued_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$REV_SUP/ROBUSTNESS_QUEUE.log"

INCLUDE_B2=1 INCLUDE_B5=1 INCLUDE_B3=0 INCLUDE_B4=0 \
  SEEDS="${SEEDS:-43 44 45 46 47}" \
  FRACTIONS="${FRACTIONS:-0.1 0.2 0.3 0.4}" \
  MODES="${MODES:-comm_skip}" \
  COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}" \
  REV_SUP="$REV_SUP" \
  bash scripts/run_robustness_sweep.sh \
  2>&1 | tee "$ROBUST_LOG"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Robustness sweep finished. Log: $ROBUST_LOG"
