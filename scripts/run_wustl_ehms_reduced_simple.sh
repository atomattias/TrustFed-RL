#!/usr/bin/env bash
# TrustFed-RL WUSTL-EHMS reduced matrix (P4): B0, B1, B4, B5.
#   bash scripts/run_wustl_ehms_reduced_simple.sh
#   SEEDS="42 43 44" bash scripts/run_wustl_ehms_reduced_simple.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON="${PYTHON:-$ROOT/.venv312/bin/python}"
# Force space-separated seeds (ignore comma-separated inherited SEEDS)
if [[ -z "${SEEDS:-}" || "$SEEDS" == *","* ]]; then
  SEEDS="42 43 44"
fi
read -r -a SEED_ARR <<< "$SEEDS"
LOG="$ROOT/results/trustfed_agent/logs/wustl_ehms_reduced_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
CLIENTS="$ROOT/data/CSVs/wustl_ehms_clients"
if [[ ! -d "$CLIENTS" ]] || [[ -z "$(ls -A "$CLIENTS" 2>/dev/null)" ]]; then
  echo "Converting WUSTL-EHMS clients..."
  "$PYTHON" scripts/convert_wustl_ehms_to_clients.py
fi
echo "LOG=$LOG" | tee "$LOG"
echo "SEEDS=${SEED_ARR[*]}" | tee -a "$LOG"
echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
run() { echo "+ $*" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1; echo "rc=$?" | tee -a "$LOG"; }
for s in "${SEED_ARR[@]}"; do run "$PYTHON" run_experiments.py regression --baseline fedavg --dataset wustl_ehms --seed "$s"; done
for s in "${SEED_ARR[@]}"; do run "$PYTHON" run_experiments.py regression --baseline b1 --dataset wustl_ehms --seed "$s"; done
for s in "${SEED_ARR[@]}"; do run "$PYTHON" run_experiments.py agent --approach trustfed_rl_only --dataset wustl_ehms --seed "$s"; done
for s in "${SEED_ARR[@]}"; do run "$PYTHON" run_experiments.py agent --approach trustfed_agent_privacy --dataset wustl_ehms --seed "$s"; done
echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
ls -1 results/trustfed_agent/metrics/run_trustfed_*_wustl_ehms_seed_*.json 2>/dev/null || true
