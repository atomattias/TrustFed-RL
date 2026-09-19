#!/usr/bin/env bash
# B1-U: trust-only FL with uniform contextual prior b_i=0.10 (IoT-J P1).
# Usage: bash scripts/run_b1u_uniform.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv312/bin/python" ]]; then
  PYTHON="$ROOT/.venv312/bin/python"
fi

LOG_DIR="$ROOT/results/trustfed_agent/logs"
METRICS_DIR="$ROOT/results/trustfed_agent/metrics"
mkdir -p "$LOG_DIR" "$METRICS_DIR"
MAIN_LOG="$LOG_DIR/b1u_uniform_$(date +%Y%m%d_%H%M%S).log"
exec >>"$MAIN_LOG" 2>&1

echo "MAIN_LOG=$MAIN_LOG"
echo "=== B1-U started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Python: $PYTHON"

SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53)
for seed in "${SEEDS[@]}"; do
  out="$METRICS_DIR/run_trustfed_b1_uniform_iomt_seed_${seed}.json"
  if [[ -f "$out" ]]; then
    echo "Skip seed $seed (exists)"
    continue
  fi
  echo ""
  echo "=== B1-U seed $seed ==="
  "$PYTHON" run_experiments.py regression \
    --baseline b1 \
    --uniform-b-prior \
    --dataset iomt \
    --seed "$seed"
done

echo ""
echo "=== B1-U complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Done: $(ls "$METRICS_DIR"/run_trustfed_b1_uniform_iomt_seed_*.json 2>/dev/null | wc -l | tr -d ' ')/12"
echo "Log: $MAIN_LOG"
