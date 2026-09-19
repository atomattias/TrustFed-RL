#!/usr/bin/env bash
# TrustFed-RL IoMT experiment matrix — master plan only
# (EXPERIMENT_PLAN §3: B0, B1, B3, B4, B5−S, B5, B5†, B4†).
# Resume: skips existing metric JSON. Optional BACKUP_DIR zip after each phase.
#
# Usage:
#   bash scripts/run_trustfed_rl_iomt_matrix.sh
#   PHASES="B0,B1,B5" bash scripts/run_trustfed_rl_iomt_matrix.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv312/bin/python" ]]; then
  PYTHON="$ROOT/.venv312/bin/python"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

METRICS_DIR="$ROOT/results/trustfed_agent/metrics"
LOG_DIR="${LOG_DIR:-$ROOT/results/trustfed_agent/logs}"
mkdir -p "$METRICS_DIR" "$LOG_DIR"

BACKUP_DIR="${BACKUP_DIR:-}"
MAIN_LOG="$LOG_DIR/trustfed_rl_iomt_matrix_$(date +%Y%m%d_%H%M%S).log"
PHASES_FILTER="${PHASES:-ALL}"

# Plain redirect (not process-substitution tee): survives nohup/background.
# Set MATRIX_TEE=1 to also mirror to an outer log via `tee -a`.
if [[ "${MATRIX_TEE:-0}" == "1" ]]; then
  exec > >(tee -a "$MAIN_LOG") 2>&1
else
  exec >>"$MAIN_LOG" 2>&1
fi
echo "MAIN_LOG=$MAIN_LOG"

echo "=== TrustFed-RL IoMT master matrix started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Python: $PYTHON"
echo "Backup dir: ${BACKUP_DIR:-'(none — set BACKUP_DIR to zip after each phase)'}"
echo "Phases: $PHASES_FILTER (master: B0,B1,B3,B4,B5S,B5,B5star,B4star + optional B5P,B1U)"
"$PYTHON" scripts/verify_install.py || true

backup_metrics() {
  if [[ -z "$BACKUP_DIR" ]]; then
    return 0
  fi
  mkdir -p "$BACKUP_DIR"
  local zip_latest="$BACKUP_DIR/iomt_metrics_latest.zip"
  local zip_stamp="$BACKUP_DIR/iomt_metrics_$(date +%Y%m%d_%H%M%S).zip"
  rm -f "$zip_latest"
  (
    cd "$METRICS_DIR"
    zip -q "$zip_latest" run_*.json 2>/dev/null || true
  )
  cp -f "$zip_latest" "$zip_stamp" 2>/dev/null || true
  echo "Backup: $zip_latest ($(ls "$METRICS_DIR"/run_*.json 2>/dev/null | wc -l | tr -d ' ') JSON files)"
}

count_run() {
  local pattern="$1"
  local n
  n=$(ls "$METRICS_DIR"/$pattern 2>/dev/null | wc -l | tr -d ' ') || true
  echo "${n:-0}"
}

should_run() {
  local phase="$1"
  if [[ "$PHASES_FILTER" == "ALL" ]]; then
    return 0
  fi
  [[ ",$PHASES_FILTER," == *",$phase,"* ]]
}

print_status() {
  echo "B0:    $(count_run 'run_trustfed_fedavg_iomt_seed_*.json')/12"
  echo "B1:    $(count_run 'run_trustfed_b1_iomt_seed_*.json')/12"
  echo "B1-U:  $(count_run 'run_trustfed_b1_uniform_iomt_seed_*.json')/12"
  echo "B3:    $(count_run 'run_trustfed_governance_iomt_seed_*.json')/12"
  echo "B4:    $(count_run 'run_trustfed_rl_only_iomt_seed_*.json')/12"
  echo "B5-S:  $(count_run 'run_trustfed_agent_no_cr_iomt_seed_*.json')/12"
  echo "B5:    $(count_run 'run_trustfed_agent_iomt_seed_*.json')/12"
  echo "B5†:   $(count_run 'run_trustfed_agent_co_adaptive_iomt_seed_*.json')/12"
  echo "B4†:   $(count_run 'run_trustfed_rl_only_co_adaptive_iomt_seed_*.json')/12"
  echo "B5-P:  $(count_run 'run_trustfed_agent_privacy_iomt_seed_*.json')/12"
}

echo ""
echo "=== Status before run ==="
print_status

# --- B0 FedAvg ---
if should_run "B0"; then
  echo ""
  echo "=== Phase B0: FedAvg equal-weight baseline ==="
  "$PYTHON" run_experiments.py regression --baseline fedavg --dataset iomt
  backup_metrics
fi

# --- B1 ---
if should_run "B1"; then
  echo ""
  echo "=== Phase B1: 6-signal trust-weighted FL ==="
  "$PYTHON" run_experiments.py regression --baseline b1 --dataset iomt
  backup_metrics
fi

# --- B1-U uniform b_i (opt-in; IoT-J trust-validity check) ---
if [[ ",$PHASES_FILTER," == *",B1U,"* ]]; then
  echo ""
  echo "=== Phase B1-U: B1 with uniform contextual prior b_i=0.10 ==="
  "$PYTHON" run_experiments.py regression --baseline b1 --uniform-b-prior --dataset iomt
  backup_metrics
fi

# --- B3 ---
if should_run "B3"; then
  echo ""
  echo "=== Phase B3: governance only ==="
  "$PYTHON" run_experiments.py agent --approach trustfed_governance --dataset iomt
  backup_metrics
fi

# --- B4 ---
if should_run "B4"; then
  echo ""
  echo "=== Phase B4: RL only, no governance ==="
  "$PYTHON" run_experiments.py agent --approach trustfed_rl_only --dataset iomt
  backup_metrics
fi

# --- B5−S (no C,R) — master ablation ---
if should_run "B5S"; then
  echo ""
  echo "=== Phase B5−S: full stack without C,R trust signals ==="
  "$PYTHON" run_experiments.py agent --approach trustfed_agent --no-cr-signals --dataset iomt
  backup_metrics
fi

# --- B5 ---
if should_run "B5"; then
  echo ""
  echo "=== Phase B5: full TrustFed-RL (static) ==="
  "$PYTHON" run_experiments.py agent --approach trustfed_agent --dataset iomt
  backup_metrics
fi

# --- B5-P weighted masking (opt-in; never part of default ALL / D1 freeze) ---
if [[ ",$PHASES_FILTER," == *",B5P,"* ]]; then
  echo ""
  echo "=== Phase B5-P: full stack + weighted update hiding ==="
  "$PYTHON" run_experiments.py agent --approach trustfed_agent_privacy --dataset iomt
  backup_metrics
fi

# --- B5† ---
if should_run "B5star"; then
  echo ""
  echo "=== Phase B5†: full + co-adaptive adversary ==="
  "$PYTHON" run_experiments.py agent \
    --approach trustfed_agent \
    --adversary co_adaptive \
    --dataset iomt
  backup_metrics
fi

# --- B4† ---
if should_run "B4star"; then
  echo ""
  echo "=== Phase B4†: RL-only + co-adaptive adversary ==="
  "$PYTHON" run_experiments.py agent \
    --approach trustfed_rl_only \
    --adversary co_adaptive \
    --dataset iomt
  backup_metrics
fi

echo ""
echo "=== Analyze ==="
"$PYTHON" run_experiments.py analyze
backup_metrics

echo ""
echo "=== Status after run ==="
print_status

echo ""
echo "=== TrustFed-RL IoMT master matrix complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Log: $MAIN_LOG"
echo "Summary: results/trustfed_agent/summary.json"
