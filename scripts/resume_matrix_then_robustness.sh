#!/usr/bin/env bash
# Resume confirmatory matrix + robustness after a mid-run crash.
# Sequential to avoid racing on shared benign arms.
# Existing B0/B1 metrics are skipped by run_experiments.py; B3–B5 re-run if present.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
mkdir -p "$REV_SUP"
SEEDS="${SEEDS:-43 44 45 46 47}"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

LOG_HARD="$REV_SUP/matrix_hard_comm_skip_confirmatory.log"
LOG_EASY="$REV_SUP/matrix_easy_label_flip_confirmatory.log"
LOG_ROBUST="$REV_SUP/robustness_comm_skip_confirmatory.log"
LOG_PIPE="$REV_SUP/resume_pipeline.log"

{
  echo ""
  echo "===== RESUME pipeline start $STAMP ====="
  echo "seeds=[$SEEDS]"
} | tee -a "$LOG_PIPE" | tee -a "$LOG_HARD" | tee -a "$LOG_EASY"

echo "[$STAMP] Step 1/3: hard matrix (comm_skip), including shared benign" | tee -a "$LOG_PIPE"
INCLUDE_B2=1 SEEDS="$SEEDS" POISON_MODE=comm_skip FRAC=0.25 RUN_BENIGN=1 \
  bash scripts/run_iomt_natural_reduced.sh 2>&1 | tee -a "$LOG_HARD"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Hard matrix Done." | tee -a "$LOG_PIPE"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Step 2/3: easy matrix (label_flip), RUN_BENIGN=0 (shared)" | tee -a "$LOG_PIPE"
INCLUDE_B2=1 SEEDS="$SEEDS" POISON_MODE=label_flip FLIP_P=0.8 FRAC=0.25 RUN_BENIGN=0 \
  bash scripts/run_iomt_natural_reduced.sh 2>&1 | tee -a "$LOG_EASY"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Easy matrix Done." | tee -a "$LOG_PIPE"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Step 3/3: robustness sweep (comm_skip)" | tee -a "$LOG_PIPE"
echo "robustness_queued_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$REV_SUP/ROBUSTNESS_QUEUE.log"
INCLUDE_B2=1 INCLUDE_B5=1 INCLUDE_B3=0 INCLUDE_B4=0 \
  SEEDS="$SEEDS" FRACTIONS="${FRACTIONS:-0.1 0.2 0.3 0.4}" MODES=comm_skip \
  COMPROMISE_ROUND=20 REV_SUP="$REV_SUP" \
  bash scripts/run_robustness_sweep.sh 2>&1 | tee -a "$LOG_ROBUST"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Resume pipeline finished." | tee -a "$LOG_PIPE"
