#!/usr/bin/env bash
# Confirmatory H1 gate only: B1 vs B2 under frozen hard attack (seeds 43–47).
# Does not retune the attack. Full B0–B5 matrix is separate (run_iomt_natural_reduced.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="${PYTHON:-./.venv312/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

ROUNDS="${ROUNDS:-30}"
SEEDS="${SEEDS:-43 44 45 46 47}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
FLIP_P="${FLIP_P:-0.8}"
FRAC="${FRAC:-0.25}"
# Load frozen mode from hard config if present
POISON_MODE="${POISON_MODE:-comm_skip}"
DATASET=iomt_natural
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
mkdir -p "$REV_SUP"

echo "=== H1 confirmatory B1 vs B2 mode=$POISON_MODE f=$FRAC seeds=[$SEEDS] ==="

extra=(--compromise-round "$COMPROMISE_ROUND" --flip-p "$FLIP_P"
       --adversary-fraction "$FRAC" --poison-mode "$POISON_MODE")

for seed in $SEEDS; do
  echo "---- seed $seed : B1 ----"
  "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b1 \
    --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
  echo "---- seed $seed : B2 ----"
  "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b2 \
    --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
done

"$PY" scripts/eval_h1_gate.py \
  --metrics results/trustfed_agent/metrics \
  --seeds $SEEDS \
  --poison-mode "$POISON_MODE" \
  --adversary-fraction "$FRAC" \
  --role confirmatory \
  --export "$REV_SUP/h1_gate_confirmatory_${POISON_MODE}_f${FRAC}.json"

echo "Done. See $REV_SUP/h1_gate_confirmatory_${POISON_MODE}_f${FRAC}.json"
