#!/usr/bin/env bash
# Minimum-viable confirmatory runs for supervisor asks S1–S3 (plan v3 lean).
#
# Already done (do not redo):
#   H1 B1 vs B2 under comm_skip, seeds 43–47 (PASS)
#   Easy H0 label-flip B1/B3–B5 for seeds 42–44 (reuse)
#
# This script runs only what is still needed for a reviewable Results section:
#   1) Easy: B2 at f=0.25 label_flip (ablation completeness; fast)
#   2) Hard: B0 + B5 at f=0.25 comm_skip (H2 / F1 context; no B3/B4)
#   3) Robustness: B0/B1/B2 across f∈{0.1,0.2,0.3,0.4} (S3; fast — no B5 in sweep)
#
# B5 appears only at the gate fraction (step 2). Response/gov B3/B4 deferred.
# WP-D (second dataset) is separate.
#
# Usage:
#   bash scripts/run_mvp_confirmatory.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
SEEDS="${SEEDS:-43 44 45 46 47}"
mkdir -p "$REV_SUP"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG="$REV_SUP/mvp_confirmatory.log"

{
  echo ""
  echo "===== MVP confirmatory start $STAMP ====="
  echo "seeds=[$SEEDS]"
  echo "scope: easy B2; hard B0+B5; robustness B0/B1/B2 × f (no B3/B4; no B5 in sweep)"
} | tee -a "$LOG"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Step 1/3: easy B2 (label_flip)" | tee -a "$LOG"
INCLUDE_B0=0 INCLUDE_B1=0 INCLUDE_B2=1 INCLUDE_B3=0 INCLUDE_B4=0 INCLUDE_B5=0 \
  RUN_BENIGN=0 SEEDS="$SEEDS" POISON_MODE=label_flip FLIP_P=0.8 FRAC=0.25 \
  bash scripts/run_iomt_natural_reduced.sh 2>&1 | tee -a "$LOG"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Step 2/3: hard B0+B5 (comm_skip) @ f=0.25" | tee -a "$LOG"
INCLUDE_B0=1 INCLUDE_B1=0 INCLUDE_B2=0 INCLUDE_B3=0 INCLUDE_B4=0 INCLUDE_B5=1 \
  RUN_BENIGN=0 SEEDS="$SEEDS" POISON_MODE=comm_skip FRAC=0.25 \
  bash scripts/run_iomt_natural_reduced.sh 2>&1 | tee -a "$LOG"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Step 3/3: robustness B0/B1/B2 (comm_skip)" | tee -a "$LOG"
INCLUDE_B0=1 INCLUDE_B1=1 INCLUDE_B2=1 INCLUDE_B3=0 INCLUDE_B4=0 INCLUDE_B5=0 \
  SEEDS="$SEEDS" FRACTIONS="${FRACTIONS:-0.1 0.2 0.3 0.4}" MODES=comm_skip \
  bash scripts/run_robustness_sweep.sh 2>&1 | tee -a "$LOG"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] MVP confirmatory finished." | tee -a "$LOG"
echo "Done." | tee -a "$LOG"
