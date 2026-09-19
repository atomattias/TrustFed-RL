#!/usr/bin/env bash
# Reduced natural-partition gate (supervisor plan v3).
# Defaults: B2 mandatory; seeds 43–47 confirmatory; poison mode overridable.
#
# Easy regime:  POISON_MODE=label_flip FLIP_P=0.8
# Hard regime:  POISON_MODE=sign_flip (or frozen hard mode)
#
# Usage:
#   bash scripts/run_iomt_natural_reduced.sh
#   SEEDS="43 44 45 46 47" POISON_MODE=sign_flip bash scripts/run_iomt_natural_reduced.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="${PYTHON:-./.venv312/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

ROUNDS="${ROUNDS:-30}"
# Plan v3 confirmatory default (exploration uses run_wp_a_explore.sh / seed 42)
SEEDS="${SEEDS:-43 44 45 46 47}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
FLIP_P="${FLIP_P:-0.8}"
FRAC="${FRAC:-0.25}"
POISON_MODE="${POISON_MODE:-label_flip}"
INCLUDE_B0="${INCLUDE_B0:-1}"
INCLUDE_B1="${INCLUDE_B1:-1}"
INCLUDE_B2="${INCLUDE_B2:-1}"
INCLUDE_B3="${INCLUDE_B3:-1}"
INCLUDE_B4="${INCLUDE_B4:-1}"
INCLUDE_B5="${INCLUDE_B5:-1}"
INCLUDE_B5R="${INCLUDE_B5R:-0}"
RUN_BENIGN="${RUN_BENIGN:-1}"
DATASET=iomt_natural
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
mkdir -p "$REV_SUP"

if [[ ! -d data/CSVs/iomt_natural/clients ]]; then
  echo "Missing natural clients — running converter..."
  "$PY" scripts/convert_iomt_natural_clients.py
fi

echo "=== iomt_natural reduced matrix rounds=$ROUNDS t_a=$COMPROMISE_ROUND mode=$POISON_MODE f=$FRAC seeds=[$SEEDS] B0=$INCLUDE_B0 B1=$INCLUDE_B1 B2=$INCLUDE_B2 B3=$INCLUDE_B3 B4=$INCLUDE_B4 B5=$INCLUDE_B5 ==="

run_agent() {
  local seed="$1" approach="$2" extra=("${@:3}")
  "$PY" trustfed_agent_runner.py --dataset "$DATASET" --approach "$approach" \
    --random-state "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
}

for seed in $SEEDS; do
  if [[ "$RUN_BENIGN" == "1" ]]; then
    echo "---- seed $seed : BENIGN ----"
    if [[ "$INCLUDE_B0" == "1" ]]; then
      "$PY" run_experiments.py regression --dataset "$DATASET" --baseline fedavg \
        --seed "$seed" --num-rounds "$ROUNDS" --no-late-compromise
    fi
    if [[ "$INCLUDE_B1" == "1" ]]; then
      "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b1 \
        --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior --no-late-compromise
    fi
    if [[ "$INCLUDE_B2" == "1" ]]; then
      "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b2 \
        --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior --no-late-compromise
    fi
    if [[ "$INCLUDE_B3" == "1" ]]; then
      run_agent "$seed" trustfed_governance --no-rl --no-late-compromise
    fi
    if [[ "$INCLUDE_B4" == "1" ]]; then
      run_agent "$seed" trustfed_rl_only --no-governance --no-late-compromise
    fi
    if [[ "$INCLUDE_B5" == "1" ]]; then
      run_agent "$seed" trustfed_agent --no-late-compromise
    fi
    if [[ "$INCLUDE_B5R" == "1" ]]; then
      run_agent "$seed" trustfed_agent --include-r-in-t --no-late-compromise
    fi
  fi

  echo "---- seed $seed : POISON mode=$POISON_MODE @${FRAC} t_a=$COMPROMISE_ROUND ----"
  local_extra=(--compromise-round "$COMPROMISE_ROUND" --flip-p "$FLIP_P"
               --adversary-fraction "$FRAC" --poison-mode "$POISON_MODE")
  if [[ "$INCLUDE_B0" == "1" ]]; then
    "$PY" run_experiments.py regression --dataset "$DATASET" --baseline fedavg \
      --seed "$seed" --num-rounds "$ROUNDS" "${local_extra[@]}"
  fi
  if [[ "$INCLUDE_B1" == "1" ]]; then
    "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b1 \
      --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${local_extra[@]}"
  fi
  if [[ "$INCLUDE_B2" == "1" ]]; then
    "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b2 \
      --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${local_extra[@]}"
  fi
  if [[ "$INCLUDE_B3" == "1" ]]; then
    run_agent "$seed" trustfed_governance --no-rl "${local_extra[@]}"
  fi
  if [[ "$INCLUDE_B4" == "1" ]]; then
    run_agent "$seed" trustfed_rl_only --no-governance "${local_extra[@]}"
  fi
  if [[ "$INCLUDE_B5" == "1" ]]; then
    run_agent "$seed" trustfed_agent "${local_extra[@]}"
  fi
  if [[ "$INCLUDE_B5R" == "1" ]]; then
    run_agent "$seed" trustfed_agent --include-r-in-t "${local_extra[@]}"
  fi
done

TAG="${POISON_MODE}_f${FRAC}"
echo "=== Analyzing → $REV_SUP ==="
"$PY" scripts/analyze_iomt_natural_trust.py \
  --input results/trustfed_agent/metrics \
  --export "$REV_SUP/iomt_natural_trust_summary_${TAG}.json"

if [[ "$INCLUDE_B2" == "1" ]]; then
  "$PY" scripts/eval_h1_gate.py \
    --metrics results/trustfed_agent/metrics \
    --seeds $SEEDS \
    --poison-mode "$POISON_MODE" \
    --adversary-fraction "$FRAC" \
    --role confirmatory \
    --export "$REV_SUP/h1_gate_${TAG}.json" \
    || true
fi

echo "Done."
