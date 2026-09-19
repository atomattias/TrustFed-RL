#!/usr/bin/env bash
# WP-A exploration (supervisor plan v3): seed 42 ONLY, B1 vs B2 under a candidate hard attack.
# Default candidate A1 = sign_flip @ f=0.25.
#
# Usage:
#   bash scripts/run_wp_a_explore.sh
#   POISON_MODE=sign_flip FRAC=0.25 bash scripts/run_wp_a_explore.sh
#   POISON_MODE=label_flip FLIP_P=0.3 FRAC=0.40 bash scripts/run_wp_a_explore.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="${PYTHON:-./.venv312/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

ROUNDS="${ROUNDS:-30}"
SEED="${SEED:-42}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
FLIP_P="${FLIP_P:-0.8}"
FRAC="${FRAC:-0.25}"
POISON_MODE="${POISON_MODE:-sign_flip}"
DATASET=iomt_natural
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
mkdir -p "$REV_SUP" results/trustfed_agent/metrics

if [[ ! -d data/CSVs/iomt_natural/clients ]]; then
  echo "Missing natural clients — running converter..."
  "$PY" scripts/convert_iomt_natural_clients.py
fi

echo "=== WP-A exploration seed=$SEED mode=$POISON_MODE f=$FRAC t_a=$COMPROMISE_ROUND rounds=$ROUNDS ==="
echo "WARNING: exploration seed only — do not use confirmatory seeds 43-47 for tuning."

extra=(--compromise-round "$COMPROMISE_ROUND" --flip-p "$FLIP_P"
       --adversary-fraction "$FRAC" --poison-mode "$POISON_MODE")

echo "---- B1 V-only ----"
"$PY" run_experiments.py regression --dataset "$DATASET" --baseline b1 \
  --seed "$SEED" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"

echo "---- B2 behavioural ----"
"$PY" run_experiments.py regression --dataset "$DATASET" --baseline b2 \
  --seed "$SEED" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"

echo "=== H1 preview gate (exploration) ==="
"$PY" scripts/eval_h1_gate.py \
  --metrics results/trustfed_agent/metrics \
  --seeds "$SEED" \
  --poison-mode "$POISON_MODE" \
  --adversary-fraction "$FRAC" \
  --min-delta 0.05 \
  --min-sign-frac 1.0 \
  --role exploration \
  --export "$REV_SUP/wp_a_explore_${POISON_MODE}_f${FRAC}_seed${SEED}.json" \
  || true

echo "Update $REV_SUP/ATTACK_SELECTION_LOG.md with the printed AUROCs."
echo "Done."
