#!/usr/bin/env bash
# WP-D second-dataset replication skeleton (supervisor plan v3).
# Arms: B0/B1/B2/B5. Run only after hard attack is frozen (H1-go or documented H1-null).
#
# Default dataset: wustl_ehms (requires converter output under data/CSVs/wustl_ehms_clients).
#
# Usage:
#   bash scripts/run_wp_d_second_dataset.sh
#   DATASET=wustl_ehms SEEDS="43 44 45 46 47" POISON_MODE=sign_flip bash scripts/run_wp_d_second_dataset.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="${PYTHON:-./.venv312/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

DATASET="${DATASET:-wustl_ehms}"
ROUNDS="${ROUNDS:-30}"
SEEDS="${SEEDS:-43 44 45 46 47}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
FLIP_P="${FLIP_P:-0.8}"
FRAC="${FRAC:-0.25}"
POISON_MODE="${POISON_MODE:-sign_flip}"
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
mkdir -p "$REV_SUP"

if [[ "$DATASET" == "wustl_ehms" && ! -d data/CSVs/wustl_ehms_clients ]]; then
  echo "Missing wustl_ehms clients. Run: $PY scripts/convert_wustl_ehms_to_clients.py"
  echo "(WP-D is a skeleton until converter + late-compromise parity are verified.)"
  exit 1
fi

echo "=== WP-D D2 replication dataset=$DATASET mode=$POISON_MODE f=$FRAC seeds=[$SEEDS] ==="
echo "NOTE: Confirm late-compromise + shared val/test semantics match D1 before claiming H4."

extra=(--compromise-round "$COMPROMISE_ROUND" --flip-p "$FLIP_P"
       --adversary-fraction "$FRAC" --poison-mode "$POISON_MODE")

for seed in $SEEDS; do
  "$PY" run_experiments.py regression --dataset "$DATASET" --baseline fedavg \
    --seed "$seed" --num-rounds "$ROUNDS" "${extra[@]}"
  "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b1 \
    --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
  "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b2 \
    --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
  "$PY" trustfed_agent_runner.py --dataset "$DATASET" --approach trustfed_agent \
    --random-state "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
done

"$PY" scripts/analyze_iomt_natural_trust.py \
  --input results/trustfed_agent/metrics \
  --dataset "$DATASET" \
  --export "$REV_SUP/d2_${DATASET}_trust_summary_${POISON_MODE}.json" || true

"$PY" scripts/eval_h1_gate.py \
  --metrics results/trustfed_agent/metrics \
  --seeds $SEEDS \
  --poison-mode "$POISON_MODE" \
  --adversary-fraction "$FRAC" \
  --role confirmatory \
  --export "$REV_SUP/d2_${DATASET}_h1_gate_${POISON_MODE}.json" \
  || true

echo "Done. Review $REV_SUP/d2_* before claiming H4."
