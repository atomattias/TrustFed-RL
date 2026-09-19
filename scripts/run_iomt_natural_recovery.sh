#!/usr/bin/env bash
# §11 trust recovery: on_off late compromise (attack window then benign).
# Plan v3: B2 mandatory; confirmatory seeds 43–47 by default.
#
# Usage:
#   bash scripts/run_iomt_natural_recovery.sh
#   ROUNDS=12 COMPROMISE_ROUND=4 ON_OFF_ATTACK_ROUNDS=3 SEEDS="42" bash scripts/run_iomt_natural_recovery.sh
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
ON_OFF_ATTACK_ROUNDS="${ON_OFF_ATTACK_ROUNDS:-5}"
INCLUDE_B2="${INCLUDE_B2:-1}"
DATASET=iomt_natural
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
mkdir -p "$REV_SUP"

echo "=== recovery (on_off) rounds=$ROUNDS t_a=$COMPROMISE_ROUND attack_rounds=$ON_OFF_ATTACK_ROUNDS seeds=[$SEEDS] ==="

for seed in $SEEDS; do
  extra=(--compromise-round "$COMPROMISE_ROUND" --flip-p "$FLIP_P"
         --adversary-fraction "$FRAC" --poison-mode on_off
         --on-off-attack-rounds "$ON_OFF_ATTACK_ROUNDS")
  "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b1 \
    --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
  if [[ "$INCLUDE_B2" == "1" ]]; then
    "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b2 \
      --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
  fi
  "$PY" trustfed_agent_runner.py --dataset "$DATASET" --approach trustfed_agent \
    --random-state "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
done

"$PY" scripts/analyze_iomt_natural_trust.py \
  --input results/trustfed_agent/metrics \
  --export "$REV_SUP/iomt_natural_recovery_summary.json"

echo "Done: $REV_SUP/iomt_natural_recovery_summary.json"
